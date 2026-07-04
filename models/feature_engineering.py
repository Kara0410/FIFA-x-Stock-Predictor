"""
Team feature engineering.

Turns raw tournament statistics (teams.json + players.json) into five
normalized 0-100 component scores per team:

    attack, defense, discipline, form, player_impact

plus an overall strength rating. All components are min-max normalized
ACROSS the 32 knockout teams, so a score of 100 means "best in the
tournament on this dimension", not an absolute measure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def _minmax(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a {team: value} dict to 0..1 (0.5 if constant)."""
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class TeamFeatureEngineer:
    """Computes component scores for every team from raw stats."""

    # Weights of each component in the overall strength rating.
    OVERALL_WEIGHTS = {
        "attack": 0.30,
        "defense": 0.28,
        "player_impact": 0.20,
        "form": 0.15,
        "discipline": 0.07,
    }

    def __init__(self, teams: dict, players: list[dict]):
        self.teams = teams
        self.players_by_team: dict[str, list[dict]] = {}
        for p in players:
            self.players_by_team.setdefault(p["team"], []).append(p)
        self.scores = self._compute_all()

    # ------------------------------------------------------------------ #
    # Raw (unnormalized) component values                                #
    # ------------------------------------------------------------------ #
    def _attack_raw(self) -> dict[str, float]:
        """attack = goals/match*0.35 + shots*0.20 + SoT*0.25 + possession*0.10 + passing*0.10
        Each ingredient is normalized across teams first so units are comparable."""
        gpm = _minmax({n: t["goals_for"] / max(1, t["matches_played"]) for n, t in self.teams.items()})
        shots = _minmax({n: t["shots_per_match"] for n, t in self.teams.items()})
        sot = _minmax({n: t["shots_on_target_per_match"] for n, t in self.teams.items()})
        poss = _minmax({n: t["possession"] for n, t in self.teams.items()})
        pacc = _minmax({n: t["passing_accuracy"] for n, t in self.teams.items()})
        return {n: gpm[n] * 0.35 + shots[n] * 0.20 + sot[n] * 0.25 + poss[n] * 0.10 + pacc[n] * 0.10
                for n in self.teams}

    def _discipline_raw(self) -> dict[str, float]:
        """Fewer fouls / yellows / reds -> higher score. Reds weigh heaviest."""
        badness = {n: t["fouls_per_match"]
                   + 3.0 * (t["yellow_cards"] / max(1, t["matches_played"]))
                   + 12.0 * (t["red_cards"] / max(1, t["matches_played"]))
                   for n, t in self.teams.items()}
        norm = _minmax(badness)
        return {n: 1.0 - v for n, v in norm.items()}   # invert: low badness = high score

    def _defense_raw(self, discipline: dict[str, float]) -> dict[str, float]:
        """defense = clean sheets*0.25 + inverse conceded*0.35 + tackles/int proxy*0.15
                   + GK/defender contribution*0.15 + discipline*0.10"""
        cs = _minmax({n: t["clean_sheets"] / max(1, t["matches_played"]) for n, t in self.teams.items()})
        inv_ga = _minmax({n: 1.0 / (1.0 + t["goals_against"] / max(1, t["matches_played"]))
                          for n, t in self.teams.items()})
        duels = _minmax({n: t["tackles_per_match"] + t["interceptions_per_match"]
                         for n, t in self.teams.items()})

        # GK + defender contribution from player data: saves, tackles, interceptions
        def gk_def(n: str) -> float:
            total = 0.0
            for p in self.players_by_team.get(n, []):
                if p["position"] in ("GK", "DF"):
                    total += p.get("saves", 0) * 0.6 + p.get("tackles", 0) * 0.3 + p.get("interceptions", 0) * 0.3
            return total
        gkd = _minmax({n: gk_def(n) for n in self.teams})

        return {n: cs[n] * 0.25 + inv_ga[n] * 0.35 + duels[n] * 0.15 + gkd[n] * 0.15 + discipline[n] * 0.10
                for n in self.teams}

    def _form_raw(self) -> dict[str, float]:
        """Recent results (recency-weighted), group points and goal difference."""
        outcome_pts = {"W": 1.0, "D": 0.45, "L": 0.0}

        def recency_form(n: str) -> float:
            results = self.teams[n].get("recent_results", [])
            if not results:
                return 0.5
            # Newest result gets the largest weight.
            weights = [1.25 ** i for i in range(len(results))]
            score = sum(outcome_pts[r] * w for r, w in zip(results, weights))
            return score / sum(weights)

        recent = _minmax({n: recency_form(n) for n in self.teams})
        pts = _minmax({n: t["group_points"] for n, t in self.teams.items()})
        gd = _minmax({n: (t["goals_for"] - t["goals_against"]) / max(1, t["matches_played"])
                      for n, t in self.teams.items()})
        return {n: recent[n] * 0.45 + pts[n] * 0.30 + gd[n] * 0.25 for n in self.teams}

    def _player_impact_raw(self) -> dict[str, float]:
        """Aggregate key-player output: goals, assists, per-90 contribution,
        availability (minutes/appearances), minus a card penalty."""
        def impact(n: str) -> float:
            total = 0.0
            for p in self.players_by_team.get(n, []):
                minutes = max(1, p["minutes"])
                per90 = (p["goals"] * 4.0 + p["assists"] * 3.0) / minutes * 90.0
                availability = min(1.0, minutes / (self.teams[n]["matches_played"] * 90.0))
                defensive = (p.get("tackles", 0) + p.get("interceptions", 0)) * 0.08
                cards = p["yellow_cards"] * 0.4 + p["red_cards"] * 2.0
                total += (p["goals"] * 4.0 + p["assists"] * 3.0 + per90 * 1.5
                          + defensive) * (0.6 + 0.4 * availability) - cards
            return total
        return _minmax({n: impact(n) for n in self.teams})

    # ------------------------------------------------------------------ #
    def _compute_all(self) -> dict[str, dict[str, float]]:
        attack = self._attack_raw()
        discipline = self._discipline_raw()
        defense = self._defense_raw(discipline)
        form = self._form_raw()
        impact = self._player_impact_raw()

        scores = {}
        for n in self.teams:
            comp = {
                "attack": round(attack[n] * 100, 1),
                "defense": round(defense[n] * 100, 1),
                "discipline": round(discipline[n] * 100, 1),
                "form": round(form[n] * 100, 1),
                "player_impact": round(impact[n] * 100, 1),
            }
            comp["overall"] = round(sum(comp[k] * w for k, w in self.OVERALL_WEIGHTS.items()), 1)
            scores[n] = comp
        return scores

    # Public helpers ---------------------------------------------------- #
    def get(self, team: str) -> dict[str, float]:
        return self.scores[team]

    def rank_of(self, team: str, component: str = "overall") -> int:
        ordered = sorted(self.scores, key=lambda t: -self.scores[t][component])
        return ordered.index(team) + 1

    def top_players(self, team: str, k: int = 4) -> list[dict]:
        ps = sorted(self.players_by_team.get(team, []),
                    key=lambda p: -(p["goals"] * 4 + p["assists"] * 3 + p["rating"]))
        return ps[:k]


# --------------------------------------------------------------------------- #
# Model-ready feature builders
# --------------------------------------------------------------------------- #
def _team_row_from_teams_df(teams_df: pd.DataFrame, team: str) -> dict[str, Any]:
    if teams_df.empty:
        return {}
    df = teams_df.copy()
    if "team" in df.columns:
        df = df.set_index("team")
    elif "name" in df.columns:
        df = df.set_index("name")
    if team not in df.index:
        return {}
    row = df.loc[team]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()


def _player_team_frame(players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df.empty:
        return pd.DataFrame()
    cols = [
        "team", "goals", "assists", "minutes", "yellow_cards", "red_cards",
        "tackles", "interceptions", "saves", "rating",
    ]
    frame = players_df.copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = 0
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
    agg["top3_player_share"] = frame.sort_values(["team", "rating"], ascending=[True, False]) \
        .groupby("team")["rating"].head(3).groupby(level=0).sum()
    return agg.fillna(0.0)


def build_football_team_frame(matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe per-team snapshots keyed by match date.

    The returned frame uses only information known before the current match.
    It is intentionally lightweight because the sample corpus is small; the
    goal is to provide stable, model-ready rows rather than a full event log.
    """
    if matches_df.empty or teams_df.empty:
        return pd.DataFrame()

    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "id"]).reset_index(drop=True)
    player_agg = _player_team_frame(players_df)

    team_names = list(teams_df["team"] if "team" in teams_df.columns else teams_df["name"])
    state: dict[str, dict[str, Any]] = {
        team: {
            "games": 0,
            "goals_for_roll": 0.0,
            "goals_against_roll": 0.0,
            "points_roll": 0.0,
            "minutes_roll": 0.0,
            "last_date": None,
            "recent_results": [],
        }
        for team in team_names
    }

    rows: list[dict[str, Any]] = []
    for _, match in matches.iterrows():
        if match.get("status") != "completed":
            continue
        home = match.get("home")
        away = match.get("away")
        if not home or not away:
            continue
        for side, team, opp in (("home", home, away), ("away", away, home)):
            team_row = _team_row_from_teams_df(teams_df, team)
            opp_row = _team_row_from_teams_df(teams_df, opp)
            if not team_row or not opp_row:
                continue
            team_state = state[team]
            opp_state = state[opp]
            recent = team_state["recent_results"][-5:]
            result_score = {"W": 1.0, "D": 0.5, "L": 0.0}
            rows.append({
                "match_id": match["id"],
                "date": match["date"],
                "team": team,
                "opponent": opp,
                "side": side,
                "matches_played_before": team_state["games"],
                "rest_days": float((match["date"] - pd.Timestamp(team_state["last_date"])).days) if team_state["last_date"] is not None else 7.0,
                "rolling_points": team_state["points_roll"],
                "rolling_goal_diff": team_state["goals_for_roll"] - team_state["goals_against_roll"],
                "recent_form": float(np.mean([result_score[r] for r in recent])) if recent else 0.5,
                "cumulative_minutes": team_state["minutes_roll"],
                "player_minutes": float(player_agg.loc[team, "player_minutes"]) if team in player_agg.index else 0.0,
                "player_count": float(player_agg.loc[team, "player_count"]) if team in player_agg.index else 0.0,
                "top3_player_share": float(player_agg.loc[team, "top3_player_share"]) if team in player_agg.index else 0.0,
                "attack": float(team_row.get("shots_on_target_per_match", 0.0)),
                "defense": float(team_row.get("clean_sheets", 0.0)),
                "discipline": float(team_row.get("yellow_cards", 0.0) + team_row.get("red_cards", 0.0)),
                "form": float(team_row.get("group_points", 0.0)),
                "player_impact": float(player_agg.loc[team, "player_rating"]) if team in player_agg.index else 0.0,
                "rank_proxy": float(team_row.get("group_points", 0.0)),
            })

        home_score = int(match.get("home_score") or 0)
        away_score = int(match.get("away_score") or 0)
        winner = match.get("winner")
        for team, goals_for, goals_against in ((home, home_score, away_score), (away, away_score, home_score)):
            if not team:
                continue
            team_state = state[team]
            team_state["games"] += 1
            team_state["goals_for_roll"] += goals_for
            team_state["goals_against_roll"] += goals_against
            team_state["points_roll"] += 3.0 if winner == team else (1.0 if home_score == away_score else 0.0)
            team_state["minutes_roll"] += 90.0
            team_state["last_date"] = match["date"]
            team_state["recent_results"].append("W" if winner == team else ("D" if home_score == away_score else "L"))

    return pd.DataFrame(rows)


def build_football_pairwise_features(
    home_team: str,
    away_team: str,
    teams_df: pd.DataFrame,
    players_df: pd.DataFrame,
    context: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Construct leakage-safe pairwise football features for a single matchup."""
    context = context or {}
    home = _team_row_from_teams_df(teams_df, home_team)
    away = _team_row_from_teams_df(teams_df, away_team)
    player_agg = _player_team_frame(players_df)

    def get_player_metric(team: str, key: str) -> float:
        if team in player_agg.index and key in player_agg.columns:
            return float(player_agg.loc[team, key])
        return 0.0

    home_minutes = get_player_metric(home_team, "player_minutes")
    away_minutes = get_player_metric(away_team, "player_minutes")
    home_top3 = get_player_metric(home_team, "top3_player_share")
    away_top3 = get_player_metric(away_team, "top3_player_share")

    def recent_goal_diff(row: dict[str, Any]) -> float:
        goals_for = float(row.get("goals_for", 0.0))
        goals_against = float(row.get("goals_against", 0.0))
        played = max(1.0, float(row.get("matches_played", 1.0)))
        return (goals_for - goals_against) / played

    def rest_days(row: dict[str, Any]) -> float:
        last_date = context.get(f"{row.get('name', '')}_last_date")
        if last_date is None:
            return float(context.get("rest_days_default", 7.0))
        match_date = pd.Timestamp(context.get("match_date")) if context.get("match_date") else pd.Timestamp.today()
        return float(max(0.0, (match_date - pd.Timestamp(last_date)).days))

    features = {
        "attack_diff": float(home.get("shots_on_target_per_match", 0.0) - away.get("shots_on_target_per_match", 0.0)),
        "defense_diff": float(home.get("clean_sheets", 0.0) - away.get("clean_sheets", 0.0)),
        "player_impact_diff": float(get_player_metric(home_team, "player_rating") - get_player_metric(away_team, "player_rating")),
        "discipline_diff": float((away.get("yellow_cards", 0.0) + away.get("red_cards", 0.0)) -
                                 (home.get("yellow_cards", 0.0) + home.get("red_cards", 0.0))),
        "form_diff": float(home.get("group_points", 0.0) - away.get("group_points", 0.0)),
        "rank_diff": float(home.get("overall", home.get("group_points", 0.0)) - away.get("overall", away.get("group_points", 0.0))),
        "rest_days_diff": float(rest_days(home) - rest_days(away)),
        "cumulative_minutes_diff": float(home_minutes - away_minutes),
        "top3_player_share_diff": float(home_top3 - away_top3),
        "recent_goal_diff_diff": float(recent_goal_diff(home) - recent_goal_diff(away)),
    }
    return features


def build_stock_feature_frame(
    stock_df: pd.DataFrame,
    football_events_df: pd.DataFrame,
    sponsor_meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Merge stock history with football event channels on trading days."""
    if stock_df.empty:
        return pd.DataFrame()

    df = stock_df.copy()
    df = df.sort_index()
    if "close" not in df.columns:
        raise ValueError("stock_df must contain a 'close' column")

    sponsor_meta = sponsor_meta or {}
    exposure = float(sponsor_meta.get("exposure_score", 0.5))
    region = str(sponsor_meta.get("region", "")).lower()

    df["return"] = df["close"].pct_change()
    df["lag1_return"] = df["return"].shift(1)
    df["lag2_return"] = df["return"].shift(2)
    df["lag3_return"] = df["return"].shift(3)
    df["ma_gap_3"] = df["close"].rolling(3).mean() / df["close"] - 1.0
    df["ma_gap_5"] = df["close"].rolling(5).mean() / df["close"] - 1.0
    df["rolling_vol_5"] = df["return"].rolling(5).std()
    df["rolling_vol_10"] = df["return"].rolling(10).std()
    df["momentum_3"] = df["close"].pct_change(3)
    df["momentum_5"] = df["close"].pct_change(5)
    df["trend_since_start"] = df["close"] / df["close"].iloc[0] - 1.0

    events = football_events_df.copy()
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"]).dt.normalize()
        events = events.set_index("date").sort_index()
        event_cols = [
            "exposure_score", "match_day_intensity", "upset_score", "alive_relevance",
            "event_decay_3", "event_decay_5", "same_day_event", "lag1_event",
            "lag2_event", "region_relevance", "cum_stage_intensity",
        ]
        for col in event_cols:
            if col not in events.columns:
                events[col] = 0.0
        aligned = events.reindex(df.index, method=None).fillna(0.0)
        for col in event_cols:
            df[col] = aligned[col].astype(float)
    else:
        for col in [
            "exposure_score", "match_day_intensity", "upset_score", "alive_relevance",
            "event_decay_3", "event_decay_5", "same_day_event", "lag1_event",
            "lag2_event", "region_relevance", "cum_stage_intensity",
        ]:
            df[col] = 0.0

    df["exposure_score"] = df["exposure_score"].fillna(exposure if exposure else 0.5)
    df["region_relevance"] = df["region_relevance"].fillna(1.0 if region else 0.5)
    df["feature_target"] = df["return"].shift(-1)
    return df
