from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from .feature_engineering import (
    TeamFeatureEngineer,
    build_football_pairwise_features,
)
from .football_model import MatchPredictor


FOOTBALL_CLASS_NAMES = ["home", "draw", "away"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip_prob(value: float) -> float:
    return float(np.clip(value, 1e-4, 1.0 - 1e-4))


def _prediction_schema(
    model_name: str,
    model_label: str,
    params_used: dict[str, Any],
    training_rows: int,
    feature_count: int,
    validation_summary: dict[str, Any],
    team_a: str,
    team_b: str,
    p_team_a: float,
    p_draw_90: float,
    p_team_b: float,
    p_advance_a: float,
    p_advance_b: float,
    confidence: float,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_name": model_name,
        "model_label": model_label,
        "params_used": params_used,
        "training_rows": int(training_rows),
        "feature_count": int(feature_count),
        "validation_summary": validation_summary,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "team_a": team_a,
        "team_b": team_b,
        "p_team_a": round(float(p_team_a), 4),
        "p_draw_90": round(float(p_draw_90), 4),
        "p_team_b": round(float(p_team_b), 4),
        "p_advance_a": round(float(p_advance_a), 4),
        "p_advance_b": round(float(p_advance_b), 4),
        "advance_a": round(float(p_advance_a), 4),
        "advance_b": round(float(p_advance_b), 4),
        "confidence": round(float(confidence), 4),
    }
    if extras:
        payload.update(extras)
    return payload


def _teams_frame(teams_df: pd.DataFrame) -> pd.DataFrame:
    frame = teams_df.copy()
    if frame.empty:
        return frame
    if "team" in frame.columns:
        frame = frame.set_index("team")
    elif "name" in frame.columns:
        frame = frame.set_index("name")
    return frame


def _players_frame(players_df: pd.DataFrame) -> pd.DataFrame:
    frame = players_df.copy()
    if frame.empty:
        return frame
    for col in ("team", "goals", "assists", "minutes", "yellow_cards", "red_cards", "tackles", "interceptions", "saves", "rating"):
        if col not in frame.columns:
            frame[col] = 0
    return frame


def _player_aggregates(players_df: pd.DataFrame) -> pd.DataFrame:
    frame = _players_frame(players_df)
    if frame.empty:
        return pd.DataFrame()
    agg = frame.groupby("team", as_index=True).agg(
        player_goals=("goals", "sum"),
        player_assists=("assists", "sum"),
        player_minutes=("minutes", "sum"),
        player_yellow_cards=("yellow_cards", "sum"),
        player_red_cards=("red_cards", "sum"),
        player_tackles=("tackles", "sum"),
        player_interceptions=("interceptions", "sum"),
        player_saves=("saves", "sum"),
        player_rating=("rating", "mean"),
        player_count=("team", "size"),
    )
    top3 = frame.sort_values(["team", "rating"], ascending=[True, False]).groupby("team")["rating"].head(3)
    agg["top3_player_share"] = top3.groupby(level=0).sum()
    return agg.fillna(0.0)


class BaselinePoissonBlendModel:
    label = "Baseline Poisson Blend"
    name = "baseline_poisson_blend"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.features: TeamFeatureEngineer | None = None
        self.predictor: MatchPredictor | None = None
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._trained = False

    def fit(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        teams = _teams_frame(teams_df)
        players = _players_frame(players_df)
        self.features = TeamFeatureEngineer(teams.to_dict("index"), players.to_dict("records"))
        self.predictor = MatchPredictor(self.features)
        poisson_weight = float(self.params.get("poisson_weight", 0.55))
        logistic_weight = float(self.params.get("logistic_weight", 0.45))
        blend = poisson_weight + logistic_weight
        self.predictor.POISSON_WEIGHT = poisson_weight / blend if blend > 0 else 0.55
        for key, value in self.params.items():
            setattr(self.predictor, key.upper(), value) if hasattr(self.predictor, key.upper()) else setattr(self.predictor, key, value)
        self.training_rows = int(len(matches_df))
        self.feature_count = len(self.features.scores[next(iter(self.features.scores))]) if self.features.scores else 0
        self.validation_summary = {
            "log_loss": None,
            "brier_score": None,
            "holdout_accuracy": None,
            "monte_carlo_runs": int(self.params.get("mc_simulations", 10000)),
        }
        self._trained = True
        return self

    def predict_match(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained or not self.predictor:
            raise RuntimeError("BaselinePoissonBlendModel must be fit before predict_match")
        raw = self.predictor.predict(team_a, team_b)
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            team_a,
            team_b,
            raw["p_win_90"],
            raw["p_draw_90"],
            raw["p_loss_90"],
            raw["advance_a"],
            raw["advance_b"],
            raw["confidence"],
            extras={
                "xg_a": raw["xg_a"],
                "xg_b": raw["xg_b"],
                "drivers": raw["drivers"],
                "penalty_win_prob_a": raw["penalty_win_prob_a"],
            },
        )

    def advance_probability(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> float:
        return self.predict_match(team_a, team_b, context_dict)["p_advance_a"]


class EloBTModel:
    label = "Elo / Bradley-Terry"
    name = "elo_bt"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.features: TeamFeatureEngineer | None = None
        self.team_ratings: dict[str, float] = {}
        self.team_counts: dict[str, int] = {}
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._trained = False

    def fit(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        teams = _teams_frame(teams_df)
        players = _players_frame(players_df)
        self.features = TeamFeatureEngineer(teams.to_dict("index"), players.to_dict("records"))
        base = 1500.0
        self.team_ratings = {team: base + (scores["overall"] - 50.0) * 8.0 for team, scores in self.features.scores.items()}
        self.team_counts = {team: 0 for team in self.team_ratings}

        matches = matches_df.copy()
        if not matches.empty:
            matches["date"] = pd.to_datetime(matches["date"])
            matches = matches.sort_values(["date", "id"])
        k_factor = float(self.params.get("k_factor", 24))
        decay = float(self.params.get("decay", 0.02))
        mov_boost = float(self.params.get("mov_boost", 0.15))
        regularization_c = float(self.params.get("regularization_c", 1.0))
        draw_prior = float(self.params.get("draw_prior", 0.26))
        draws = 0
        total = 0
        for idx, match in matches.iterrows():
            if match.get("status") != "completed":
                continue
            home = match.get("home")
            away = match.get("away")
            if not home or not away:
                continue
            ra = self.team_ratings.get(home, base)
            rb = self.team_ratings.get(away, base)
            expected_home = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            home_score = int(match.get("home_score") or 0)
            away_score = int(match.get("away_score") or 0)
            if home_score > away_score:
                actual_home = 1.0
            elif home_score < away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5
                draws += 1
            total += 1
            margin = abs(home_score - away_score)
            boost = 1.0 + mov_boost * math.log1p(margin)
            age_factor = math.exp(-decay * idx)
            delta = k_factor * boost * age_factor * (actual_home - expected_home)
            self.team_ratings[home] = ra + delta - regularization_c * 0.02 * (ra - base)
            self.team_ratings[away] = rb - delta - regularization_c * 0.02 * (rb - base)
            self.team_counts[home] += 1
            self.team_counts[away] += 1
        self.training_rows = int(total)
        self.feature_count = 10
        self.validation_summary = {
            "log_loss": None,
            "brier_score": None,
            "holdout_accuracy": None,
            "monte_carlo_runs": int(self.params.get("mc_simulations", 10000)),
            "draw_rate": round(draws / total, 4) if total else None,
        }
        self._trained = True
        self._draw_prior = draw_prior
        return self

    def _score(self, team_a: str, team_b: str) -> tuple[float, float, float, float]:
        ra = self.team_ratings.get(team_a, 1500.0)
        rb = self.team_ratings.get(team_b, 1500.0)
        diff = ra - rb
        p_home_win = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        draw_prior = float(self.params.get("draw_prior", getattr(self, "_draw_prior", 0.26)))
        closeness = math.exp(-abs(diff) / 260.0)
        p_draw = np.clip(draw_prior + 0.18 * closeness, 0.05, 0.42)
        p_home = (1.0 - p_draw) * p_home_win
        p_away = (1.0 - p_draw) * (1.0 - p_home_win)
        return float(p_home), float(p_draw), float(p_away), float(diff)

    def predict_match(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("EloBTModel must be fit before predict_match")
        p_home, p_draw, p_away, diff = self._score(team_a, team_b)
        penalty_tilt = float(self.params.get("penalty_tilt", 0.55))
        p_pen_a = 0.5 + (p_home / max(1e-9, p_home + p_away) - 0.5) * penalty_tilt
        p_advance_a = p_home + p_draw * p_pen_a
        p_advance_b = 1.0 - p_advance_a
        confidence = 0.5 + min(0.45, abs(p_advance_a - 0.5) * 1.05)
        confidence *= 0.75 + 0.25 * min(1.0, (self.team_counts.get(team_a, 0) + self.team_counts.get(team_b, 0)) / 6.0)
        features = self.features.scores if self.features else {}
        extras = {
            "drivers": {
                team_a: features.get(team_a, {}),
                team_b: features.get(team_b, {}),
            },
            "elo_diff": round(float(diff), 2),
        }
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            team_a,
            team_b,
            p_home,
            p_draw,
            p_away,
            p_advance_a,
            p_advance_b,
            confidence,
            extras=extras,
        )

    def advance_probability(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> float:
        return self.predict_match(team_a, team_b, context_dict)["p_advance_a"]


class HistGBFootballModel:
    label = "HistGradientBoosting + Calibration"
    name = "histgb_classifier"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.features: TeamFeatureEngineer | None = None
        self.feature_names: list[str] = []
        self.model: Any = None
        self.calibrator: Any = None
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._trained = False

    def _frame(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        teams = _teams_frame(teams_df)
        players = _players_frame(players_df)
        self.features = TeamFeatureEngineer(teams.to_dict("index"), players.to_dict("records"))
        rows = []
        labels = []
        matches = matches_df.copy()
        if not matches.empty:
            matches["date"] = pd.to_datetime(matches["date"])
            matches = matches.sort_values(["date", "id"])
        for _, match in matches.iterrows():
            if match.get("status") != "completed":
                continue
            home = match.get("home")
            away = match.get("away")
            if not home or not away:
                continue
            context = {
                "match_date": match["date"],
                f"{home}_last_date": match["date"] - pd.Timedelta(days=3),
                f"{away}_last_date": match["date"] - pd.Timedelta(days=3),
            }
            feat = build_football_pairwise_features(home, away, self._teams_for_frame(self.features), players, context)
            rows.append(feat)
            home_score = int(match.get("home_score") or 0)
            away_score = int(match.get("away_score") or 0)
            labels.append(0 if home_score > away_score else 1 if home_score == away_score else 2)
        frame = pd.DataFrame(rows).fillna(0.0)
        self.feature_names = list(frame.columns)
        return frame, pd.Series(labels, dtype=int)

    @staticmethod
    def _teams_for_frame(features: TeamFeatureEngineer) -> pd.DataFrame:
        rows = []
        for team, scores in features.scores.items():
            row = dict(features.teams.get(team, {}))
            row.update(scores)
            row["team"] = team
            rows.append(row)
        return pd.DataFrame(rows)

    def fit(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        X, y = self._frame(matches_df, teams_df, players_df)
        self.training_rows = int(len(X))
        self.feature_count = int(X.shape[1]) if not X.empty else 0
        self.validation_summary = {
            "log_loss": None,
            "brier_score": None,
            "holdout_accuracy": None,
            "monte_carlo_runs": int(self.params.get("mc_simulations", 10000)),
        }
        if X.empty or y.nunique() < 2:
            self.model = None
            self.calibrator = None
            self._trained = True
            return self

        classifier = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(self.params.get("learning_rate", 0.07)),
            max_iter=int(self.params.get("max_iter", 250)),
            max_leaf_nodes=int(self.params.get("max_leaf_nodes", 31)),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
            l2_regularization=float(self.params.get("l2_regularization", 0.1)),
            random_state=42,
        )
        classifier.fit(X, y)
        self.model = classifier
        self.calibrator = None
        calib_method = str(self.params.get("calibration_method", "sigmoid"))
        try:
            if len(X) >= 6 and y.nunique() >= 3:
                self.calibrator = CalibratedClassifierCV(estimator=classifier, method=calib_method, cv=3)
                self.calibrator.fit(X, y)
        except Exception:  # noqa: BLE001
            self.calibrator = None
        self._trained = True
        return self

    def _predict_proba(self, features: dict[str, float]) -> np.ndarray:
        X = pd.DataFrame([features], columns=self.feature_names).fillna(0.0)
        model = self.calibrator or self.model
        if model is None:
            return np.array([0.42, 0.26, 0.32])
        probs = model.predict_proba(X)[0]
        if len(probs) == 2:
            probs = np.array([probs[0], 0.15, probs[1]])
        if len(probs) != 3:
            probs = np.resize(np.asarray(probs, dtype=float), 3)
            probs = probs / max(probs.sum(), 1e-9)
        return probs

    def predict_match(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("HistGBFootballModel must be fit before predict_match")
        features = build_football_pairwise_features(team_a, team_b, self._teams_frame(), self._players_frame(), context_dict)
        probs = self._predict_proba(features)
        p_home, p_draw, p_away = map(float, probs)
        penalty_tilt = float(self.params.get("penalty_tilt", 0.55))
        p_pen_a = 0.5 + (p_home / max(1e-9, p_home + p_away) - 0.5) * penalty_tilt
        p_advance_a = p_home + p_draw * p_pen_a
        p_advance_b = 1.0 - p_advance_a
        confidence = 0.5 + min(0.45, max(0.0, float(max(probs) - 1.0 / 3.0)))
        features_map = self.features.scores if self.features else {}
        extras = {
            "drivers": {
                team_a: features_map.get(team_a, {}),
                team_b: features_map.get(team_b, {}),
            },
            "feature_vector": features,
        }
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            team_a,
            team_b,
            p_home,
            p_draw,
            p_away,
            p_advance_a,
            p_advance_b,
            confidence,
            extras=extras,
        )

    def _teams_frame(self) -> pd.DataFrame:
        if not self.features:
            return pd.DataFrame()
        return self._teams_for_frame(self.features)

    def _players_frame(self) -> pd.DataFrame:
        if not self.features:
            return pd.DataFrame()
        rows = []
        for team, players in self.features.players_by_team.items():
            for p in players:
                row = dict(p)
                row["team"] = team
                rows.append(row)
        return pd.DataFrame(rows)

    def advance_probability(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> float:
        return self.predict_match(team_a, team_b, context_dict)["p_advance_a"]


class FootballEnsembleModel:
    label = "Manual Football Ensemble"
    name = "football_ensemble"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.baseline = BaselinePoissonBlendModel(self.params)
        self.elo = EloBTModel(self.params)
        self.histgb = HistGBFootballModel(self.params)
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._trained = False
        self.weights = {}

    def fit(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame):
        self.baseline.fit(matches_df, teams_df, players_df)
        self.elo.fit(matches_df, teams_df, players_df)
        self.histgb.fit(matches_df, teams_df, players_df)
        self.features = self.baseline.features
        self.weights = {
            "baseline": float(self.params.get("w_baseline", 0.40)),
            "elo_bt": float(self.params.get("w_elo_bt", 0.35)),
            "histgb": float(self.params.get("w_histgb", 0.25)),
        }
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.training_rows = max(self.baseline.training_rows, self.elo.training_rows, self.histgb.training_rows)
        self.feature_count = max(self.baseline.feature_count, self.elo.feature_count, self.histgb.feature_count)
        self.validation_summary = {
            "log_loss": None,
            "brier_score": None,
            "holdout_accuracy": None,
            "monte_carlo_runs": int(self.params.get("mc_simulations", 10000)),
            "weight_mode": self.params.get("weight_mode", "manual"),
        }
        self._trained = True
        return self

    def predict_match(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("FootballEnsembleModel must be fit before predict_match")
        preds = [
            ("baseline", self.baseline.predict_match(team_a, team_b, context_dict)),
            ("elo_bt", self.elo.predict_match(team_a, team_b, context_dict)),
            ("histgb", self.histgb.predict_match(team_a, team_b, context_dict)),
        ]
        weights = self.weights or {"baseline": 0.4, "elo_bt": 0.35, "histgb": 0.25}
        p_home = sum(weights[name] * pred["p_team_a"] for name, pred in preds)
        p_draw = sum(weights[name] * pred["p_draw_90"] for name, pred in preds)
        p_away = sum(weights[name] * pred["p_team_b"] for name, pred in preds)
        p_adv_a = sum(weights[name] * pred["p_advance_a"] for name, pred in preds)
        p_adv_b = 1.0 - p_adv_a
        confidence = sum(weights[name] * pred["confidence"] for name, pred in preds)
        extras = {
            "submodels": {name: pred for name, pred in preds},
            "stacking_ready": True,
            "weights": self.weights,
        }
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            team_a,
            team_b,
            p_home,
            p_draw,
            p_away,
            p_adv_a,
            p_adv_b,
            confidence,
            extras=extras,
        )

    def advance_probability(self, team_a: str, team_b: str, context_dict: dict[str, Any] | None = None) -> float:
        return self.predict_match(team_a, team_b, context_dict)["p_advance_a"]
