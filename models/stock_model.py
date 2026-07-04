"""
Experimental sponsor stock forecaster.

Combines standard market features (returns, moving averages, volatility)
with World-Cup-derived signals (tournament stage intensity, match-day flags,
knockout game count, upset score, sponsor exposure) and trains a
GradientBoostingRegressor to predict next-day returns, which are rolled
forward recursively for a 7-trading-day forecast.

NOT financial advice. The tournament window provides only ~15 trading days
of history, so this is a demonstration of the modelling pipeline, not a
production forecaster. If there is too little data the model degrades
gracefully to a drift + volatility estimate.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

FORECAST_DAYS = 7
MIN_TRAIN_ROWS = 8          # below this, fall back to a naive drift model
CONF_Z = 1.28               # ~80% confidence band


class FootballSignalBuilder:
    """Derives per-calendar-date football intensity features from the bracket."""

    STAGE_INTENSITY = {"GROUP": 1.0, "R32": 2.0, "R16": 3.0, "QF": 4.0, "SF": 5.0, "F": 6.0}

    def __init__(self, matches_doc: dict, tournament_start: str = "2026-06-11"):
        self.start = pd.Timestamp(tournament_start)
        self.matches = matches_doc.get("matches", [])
        self.upsets = {pd.Timestamp(u["date"]): u["magnitude"] for u in matches_doc.get("upsets", [])}

        self.match_days: dict[pd.Timestamp, list[str]] = {}
        for m in self.matches:
            day = pd.Timestamp(m["date"])
            self.match_days.setdefault(day, []).append(m["round"])

    def features_for(self, date: pd.Timestamp, exposure_score: float) -> dict[str, float]:
        date = pd.Timestamp(date).normalize()
        rounds_today = self.match_days.get(date, [])
        # Before the knockout schedule begins, every day is a group match day.
        in_group_stage = self.start <= date < pd.Timestamp("2026-06-28")
        if rounds_today:
            intensity = max(self.STAGE_INTENSITY[r] for r in rounds_today)
        elif in_group_stage:
            intensity = self.STAGE_INTENSITY["GROUP"]
        else:
            intensity = 0.0

        knockout_played = sum(
            1 for m in self.matches
            if m["status"] == "completed" and pd.Timestamp(m["date"]) <= date
        )
        upset = self.upsets.get(date, 0.0)
        # Global fan attention proxy: stage intensity scaled by games that day,
        # decaying to a floor on rest days.
        attention = intensity * max(1, len(rounds_today)) if intensity else 0.3

        return {
            "stage_intensity": intensity,
            "is_match_day": 1.0 if (rounds_today or in_group_stage) else 0.0,
            "n_matches_today": float(len(rounds_today)),
            "knockout_games_played": float(knockout_played),
            "upset_score": upset,
            "fan_attention": attention,
            "sponsor_exposure": exposure_score * (1.0 + 0.15 * intensity),
        }


class StockPredictor:
    """Trains on the tournament-window price history and forecasts 7 days."""

    def __init__(self, signal_builder: FootballSignalBuilder):
        self.signals = signal_builder

    # ------------------------------------------------------------------ #
    def _build_frame(self, history: pd.DataFrame, exposure: float) -> pd.DataFrame:
        df = history.copy()
        df["return"] = df["close"].pct_change()
        df["ma3"] = df["close"].rolling(3).mean() / df["close"] - 1.0
        df["ma5"] = df["close"].rolling(5).mean() / df["close"] - 1.0
        df["vol3"] = df["return"].rolling(3).std()
        df["momentum"] = df["close"].pct_change(3)
        # Simple market trend proxy: cumulative return since the window start.
        df["trend"] = df["close"] / df["close"].iloc[0] - 1.0

        fb = [self.signals.features_for(d, exposure) for d in df.index]
        for key in fb[0]:
            df[key] = [row[key] for row in fb]
        df["target"] = df["return"].shift(-1)   # next-day return
        return df

    FEATURES = ["return", "ma3", "ma5", "vol3", "momentum", "trend",
                "stage_intensity", "is_match_day", "n_matches_today",
                "knockout_games_played", "upset_score", "fan_attention",
                "sponsor_exposure"]

    def forecast(self, history: pd.DataFrame, exposure: float) -> dict:
        """history: DataFrame indexed by date with a 'close' column."""
        df = self._build_frame(history, exposure)
        train = df.dropna(subset=self.FEATURES + ["target"])

        daily_ret = df["return"].dropna()
        drift = float(daily_ret.mean()) if len(daily_ret) else 0.0
        vol = float(daily_ret.std()) if len(daily_ret) > 2 else 0.015

        model_used = "GradientBoostingRegressor"
        if len(train) >= MIN_TRAIN_ROWS:
            model = GradientBoostingRegressor(
                n_estimators=120, max_depth=2, learning_rate=0.05,
                subsample=0.9, random_state=7)
            model.fit(train[self.FEATURES], train["target"])
            resid = train["target"] - model.predict(train[self.FEATURES])
            resid_std = float(max(resid.std(), vol * 0.6, 1e-4))
        else:
            model = None
            model_used = "naive drift (not enough history)"
            resid_std = max(vol, 1e-4)

        # ---- recursive 7-business-day forecast --------------------------- #
        last_date = df.index[-1]
        last_close = float(df["close"].iloc[-1])
        future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

        closes = list(df["close"].values)
        preds, lower, upper = [], [], []
        price = last_close
        for h, day in enumerate(future_dates, start=1):
            ret_hist = pd.Series(closes).pct_change()
            row = {
                "return": float(ret_hist.iloc[-1]),
                "ma3": float(np.mean(closes[-3:]) / price - 1.0),
                "ma5": float(np.mean(closes[-5:]) / price - 1.0),
                "vol3": float(ret_hist.tail(3).std()) if len(ret_hist) > 3 else vol,
                "momentum": float(price / closes[-4] - 1.0) if len(closes) >= 4 else 0.0,
                "trend": float(price / closes[0] - 1.0),
            }
            row.update(self.signals.features_for(day, exposure))
            if model is not None:
                x = pd.DataFrame([[row[f] for f in self.FEATURES]], columns=self.FEATURES)
                r = float(model.predict(x)[0])
                # With ~2-3 weeks of history the tree model overfits badly, so
                # shrink its prediction toward the plain historical drift; the
                # model earns more weight as real data accumulates.
                shrink = min(1.0, len(train) / 40.0)
                r = shrink * r + (1.0 - shrink) * drift
                r = float(np.clip(r, -0.025, 0.025))    # keep daily moves sane
            else:
                r = drift
            price = price * (1.0 + r)
            closes.append(price)
            band = CONF_Z * resid_std * np.sqrt(h) * last_close
            preds.append(round(price, 2))
            lower.append(round(max(0.01, price - band), 2))
            upper.append(round(price + band, 2))

        return {
            "model": model_used,
            "train_rows": int(len(train)),
            "residual_std": round(resid_std, 5),
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "predicted": preds,
            "lower": lower,
            "upper": upper,
            "expected_7d_change_pct": round((preds[-1] / last_close - 1.0) * 100, 2),
            "disclaimer": "Experimental model on a tiny sample. Not financial advice.",
        }
