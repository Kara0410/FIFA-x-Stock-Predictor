from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .feature_engineering import build_stock_feature_frame


FORECAST_DAYS = 7


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _prediction_schema(
    model_name: str,
    model_label: str,
    params_used: dict[str, Any],
    training_rows: int,
    feature_count: int,
    validation_summary: dict[str, Any],
    history_df: pd.DataFrame,
    future_dates: list[pd.Timestamp],
    predicted: list[float],
    lower: list[float],
    upper: list[float],
    model_source: str,
    disclaimer: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_label": model_label,
        "params_used": params_used,
        "training_rows": int(training_rows),
        "feature_count": int(feature_count),
        "validation_summary": validation_summary,
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in history_df.index],
            "close": [round(float(v), 2) for v in history_df["close"].tolist()],
        },
        "forecast": {
            "model": model_source,
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "predicted": [round(float(v), 2) for v in predicted],
            "lower": [round(float(v), 2) for v in lower],
            "upper": [round(float(v), 2) for v in upper],
            "expected_7d_change_pct": round((predicted[-1] / float(history_df["close"].iloc[-1]) - 1.0) * 100, 2) if predicted else 0.0,
            "disclaimer": disclaimer,
        },
    }


def _safe_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2:
        return None
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _confidence_bands(predicted: list[float], last_close: float, resid_std: float) -> tuple[list[float], list[float]]:
    lower, upper = [], []
    for idx, price in enumerate(predicted, start=1):
        band = 1.28 * resid_std * np.sqrt(idx) * last_close
        lower.append(max(0.01, price - band))
        upper.append(price + band)
    return lower, upper


class BaselineGBRStockModel:
    label = "Baseline Gradient Boosting"
    name = "baseline_gbr"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.model: GradientBoostingRegressor | None = None
        self.feature_names: list[str] = []
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._frame: pd.DataFrame | None = None
        self._trained = False

    def fit(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None):
        frame = build_stock_feature_frame(stock_df, football_events_df, sponsor_meta)
        self._frame = frame
        feature_cols = [
            "return", "lag1_return", "lag2_return", "lag3_return", "ma_gap_3", "ma_gap_5",
            "rolling_vol_5", "rolling_vol_10", "momentum_3", "momentum_5", "trend_since_start",
            "exposure_score", "match_day_intensity", "upset_score", "alive_relevance",
            "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event",
            "region_relevance", "cum_stage_intensity",
        ]
        frame = frame.dropna(subset=["feature_target"]).copy()
        frame[feature_cols] = frame[feature_cols].fillna(0.0)
        self.feature_names = feature_cols
        self.training_rows = int(len(frame))
        self.feature_count = len(feature_cols)

        if len(frame) < 6:
            self.model = None
            self.validation_summary = {"mae": None, "rmse": None, "directional_accuracy": None, "backtest_windows": 0}
            self._trained = True
            return self

        model = GradientBoostingRegressor(
            n_estimators=int(self.params.get("n_estimators", 120)),
            max_depth=int(self.params.get("max_depth", 2)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
            subsample=float(self.params.get("subsample", 0.90)),
            random_state=42,
        )
        split = max(4, int(len(frame) * 0.8))
        train = frame.iloc[:split]
        holdout = frame.iloc[split:]
        model.fit(train[feature_cols], train["feature_target"])
        self.model = model
        if not holdout.empty:
            preds = model.predict(holdout[feature_cols])
            mae = mean_absolute_error(holdout["feature_target"], preds)
            rmse = _rmse(holdout["feature_target"], preds)
            da = _safe_directional_accuracy(holdout["feature_target"].to_numpy(), preds)
        else:
            preds = model.predict(train[feature_cols])
            mae = mean_absolute_error(train["feature_target"], preds)
            rmse = _rmse(train["feature_target"], preds)
            da = _safe_directional_accuracy(train["feature_target"].to_numpy(), preds)
        self.validation_summary = {
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "directional_accuracy": round(float(da), 4) if da is not None else None,
            "backtest_windows": int(len(holdout) if not holdout.empty else len(train)),
        }
        self._trained = True
        return self

    def _future_row(self, frame: pd.DataFrame, future_date: pd.Timestamp, forecasted_returns: list[float], sponsor_meta: dict[str, Any] | None, football_events_df: pd.DataFrame) -> pd.Series:
        sponsor_meta = sponsor_meta or {}
        returns = frame["return"] if "return" in frame.columns else frame["close"].pct_change()
        future = pd.DataFrame(index=[future_date])
        future["close"] = np.nan
        future["return"] = forecasted_returns[-1] if forecasted_returns else float(returns.dropna().iloc[-1])
        future["lag1_return"] = forecasted_returns[-1] if forecasted_returns else float(returns.dropna().iloc[-1])
        future["lag2_return"] = forecasted_returns[-2] if len(forecasted_returns) > 1 else float(returns.dropna().iloc[-2]) if len(returns.dropna()) > 1 else 0.0
        future["lag3_return"] = forecasted_returns[-3] if len(forecasted_returns) > 2 else float(returns.dropna().iloc[-3]) if len(returns.dropna()) > 2 else 0.0
        last_close = float(frame["close"].iloc[-1]) if not frame.empty else 1.0
        future["ma_gap_3"] = 0.0
        future["ma_gap_5"] = 0.0
        future["rolling_vol_5"] = float(returns.tail(5).std()) if returns.dropna().shape[0] > 1 else 0.0
        future["rolling_vol_10"] = float(returns.tail(10).std()) if returns.dropna().shape[0] > 2 else 0.0
        future["momentum_3"] = float(last_close / frame["close"].iloc[-4] - 1.0) if len(frame) >= 4 else 0.0
        future["momentum_5"] = float(last_close / frame["close"].iloc[-6] - 1.0) if len(frame) >= 6 else 0.0
        future["trend_since_start"] = float(last_close / frame["close"].iloc[0] - 1.0) if len(frame) else 0.0
        events = football_events_df.copy()
        if not events.empty:
            events["date"] = pd.to_datetime(events["date"]).dt.normalize()
            events = events.set_index("date").sort_index()
            if future_date in events.index:
                event_row = events.loc[future_date]
                if isinstance(event_row, pd.DataFrame):
                    event_row = event_row.iloc[0]
                for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                    future[col] = float(event_row.get(col, 0.0))
            else:
                for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                    future[col] = 0.0
        else:
            for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                future[col] = 0.0
        future["exposure_score"] = float(sponsor_meta.get("exposure_score", 0.5))
        future["region_relevance"] = float(sponsor_meta.get("region_relevance", 1.0))
        return future.iloc[0]

    def forecast(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            self.fit(stock_df, football_events_df, sponsor_meta)
        assert self._frame is not None
        frame = self._frame.copy()
        history_df = stock_df.copy().sort_index()
        future_dates = list(pd.bdate_range(history_df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS))
        last_close = float(history_df["close"].iloc[-1])
        daily_returns = frame["return"].dropna()
        drift = float(daily_returns.mean()) if len(daily_returns) else 0.0
        resid_std = float(max(daily_returns.std(), 1e-4)) if len(daily_returns) else 0.01

        predicted = []
        prices = [float(history_df["close"].iloc[-1])]
        forecasted_returns: list[float] = []
        for future_date in future_dates:
            if self.model is None:
                next_ret = drift
            else:
                row = self._future_row(history_df.join(frame[["return"]], how="left"), future_date, forecasted_returns, sponsor_meta, football_events_df)
                x = pd.DataFrame([row[self.feature_names].fillna(0.0)])
                next_ret = float(self.model.predict(x)[0])
            forecasted_returns.append(next_ret)
            next_price = prices[-1] * (1.0 + next_ret)
            prices.append(next_price)
            predicted.append(next_price)
        lower, upper = _confidence_bands(predicted, last_close, resid_std)
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            history_df,
            future_dates,
            predicted,
            lower,
            upper,
            "GradientBoostingRegressor",
            "Experimental model on a small local sample. Not financial advice.",
        )


class SarimaxStockModel:
    label = "SARIMAX + Football Signals"
    name = "sarimax_exog"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.results = None
        self.feature_names: list[str] = []
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._frame: pd.DataFrame | None = None
        self._trained = False

    def _feature_cols(self) -> list[str]:
        return [
            "lag1_return", "lag2_return", "lag3_return", "ma_gap_3", "ma_gap_5",
            "rolling_vol_5", "rolling_vol_10", "momentum_3", "momentum_5",
            "exposure_score", "match_day_intensity", "upset_score", "alive_relevance",
            "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event",
            "region_relevance", "cum_stage_intensity",
        ]

    def fit(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None):
        frame = build_stock_feature_frame(stock_df, football_events_df, sponsor_meta)
        self._frame = frame
        feature_cols = self._feature_cols()
        frame = frame.dropna(subset=["feature_target"]).copy()
        frame[feature_cols] = frame[feature_cols].fillna(0.0)
        exog_cols = [col for col in feature_cols if frame[col].nunique(dropna=False) > 1]
        if not exog_cols:
            exog_cols = [col for col in feature_cols if col != "region_relevance"] or feature_cols[:]
        self.feature_names = exog_cols
        self.feature_count = len(exog_cols)
        self.training_rows = int(len(frame))
        order = tuple(self.params.get("order", [1, 0, 1]))
        seasonal_order = tuple(self.params.get("seasonal_order", [0, 0, 0, 0]))
        trend = self.params.get("trend", "c")
        if len(frame) < 8:
            self.results = None
            self.validation_summary = {"mae": None, "rmse": None, "directional_accuracy": None, "backtest_windows": 0}
            self._trained = True
            return self
        split = max(5, int(len(frame) * 0.8))
        train = frame.iloc[:split]
        holdout = frame.iloc[split:]
        train_exog = train[self.feature_names].copy()
        if trend in {"c", "ct"} and any(train_exog[col].nunique(dropna=False) <= 1 for col in train_exog.columns):
            train_exog = train_exog.loc[:, train_exog.nunique(dropna=False) > 1]
            self.feature_names = list(train_exog.columns)
            self.feature_count = len(self.feature_names)
            if train_exog.empty:
                trend = "n"
        model = SARIMAX(
            train["feature_target"],
            exog=train_exog if not train_exog.empty else None,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=bool(self.params.get("enforce_stationarity", True)),
            enforce_invertibility=bool(self.params.get("enforce_invertibility", True)),
        )
        self.results = model.fit(disp=False, maxiter=200)
        if not holdout.empty:
            holdout_exog = holdout[self.feature_names].copy() if self.feature_names else None
            fc = self.results.get_forecast(steps=len(holdout), exog=holdout_exog if holdout_exog is not None and not holdout_exog.empty else None)
            preds = fc.predicted_mean.to_numpy()
            mae = mean_absolute_error(holdout["feature_target"], preds)
            rmse = _rmse(holdout["feature_target"], preds)
            da = _safe_directional_accuracy(holdout["feature_target"].to_numpy(), preds)
        else:
            preds = self.results.fittedvalues.to_numpy()
            mae = mean_absolute_error(train["feature_target"], preds)
            rmse = _rmse(train["feature_target"], preds)
            da = _safe_directional_accuracy(train["feature_target"].to_numpy(), preds)
        self.validation_summary = {
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "directional_accuracy": round(float(da), 4) if da is not None else None,
            "backtest_windows": int(len(holdout) if not holdout.empty else len(train)),
        }
        self._trained = True
        return self

    def _future_row(self, frame: pd.DataFrame, future_date: pd.Timestamp, forecasted_returns: list[float], sponsor_meta: dict[str, Any] | None, football_events_df: pd.DataFrame) -> pd.Series:
        sponsor_meta = sponsor_meta or {}
        returns = frame["return"] if "return" in frame.columns else frame["close"].pct_change()
        last_returns = list(returns.dropna().tail(3).to_numpy())
        while len(last_returns) < 3:
            last_returns.insert(0, 0.0)
        future = pd.Series(index=self.feature_names, dtype=float)
        future["lag1_return"] = forecasted_returns[-1] if forecasted_returns else last_returns[-1]
        future["lag2_return"] = forecasted_returns[-2] if len(forecasted_returns) > 1 else last_returns[-2]
        future["lag3_return"] = forecasted_returns[-3] if len(forecasted_returns) > 2 else last_returns[-3]
        last_close = float(frame["close"].iloc[-1])
        future["ma_gap_3"] = float(frame["close"].tail(3).mean() / last_close - 1.0) if len(frame) >= 3 else 0.0
        future["ma_gap_5"] = float(frame["close"].tail(5).mean() / last_close - 1.0) if len(frame) >= 5 else 0.0
        future["rolling_vol_5"] = float(returns.tail(5).std()) if returns.dropna().shape[0] > 1 else 0.0
        future["rolling_vol_10"] = float(returns.tail(10).std()) if returns.dropna().shape[0] > 2 else 0.0
        future["momentum_3"] = float(last_close / frame["close"].iloc[-4] - 1.0) if len(frame) >= 4 else 0.0
        future["momentum_5"] = float(last_close / frame["close"].iloc[-6] - 1.0) if len(frame) >= 6 else 0.0
        events = football_events_df.copy()
        if not events.empty:
            events["date"] = pd.to_datetime(events["date"]).dt.normalize()
            events = events.set_index("date").sort_index()
            row = events.loc[future_date] if future_date in events.index else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                future[col] = float(row.get(col, 0.0)) if row is not None else 0.0
        else:
            for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                future[col] = 0.0
        future["exposure_score"] = float(sponsor_meta.get("exposure_score", 0.5))
        future["region_relevance"] = float(sponsor_meta.get("region_relevance", 1.0))
        return future

    def forecast(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            self.fit(stock_df, football_events_df, sponsor_meta)
        assert self._frame is not None
        history_df = stock_df.copy().sort_index()
        future_dates = list(pd.bdate_range(history_df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS))
        daily_returns = self._frame["return"].dropna()
        resid_std = float(max(daily_returns.std(), 1e-4)) if len(daily_returns) else 0.01
        predicted = []
        lower = []
        upper = []
        price = float(history_df["close"].iloc[-1])
        forecasted_returns: list[float] = []
        results = self.results
        for future_date in future_dates:
            row = self._future_row(history_df, future_date, forecasted_returns, sponsor_meta, football_events_df)
            if results is None:
                next_ret = float(daily_returns.mean()) if len(daily_returns) else 0.0
                next_lower = next_ret - 1.28 * resid_std
                next_upper = next_ret + 1.28 * resid_std
            else:
                row_df = pd.DataFrame([row[self.feature_names].fillna(0.0)]) if self.feature_names else None
                fc = results.get_forecast(steps=1, exog=row_df if row_df is not None and not row_df.empty else None)
                next_ret = float(fc.predicted_mean.iloc[0])
                conf = fc.conf_int(alpha=0.20).iloc[0]
                next_lower = float(conf.iloc[0])
                next_upper = float(conf.iloc[1])
                try:
                    results = results.append([next_ret], exog=row_df if row_df is not None and not row_df.empty else None, refit=False)
                except Exception:  # noqa: BLE001
                    results = None
            forecasted_returns.append(next_ret)
            price = price * (1.0 + next_ret)
            predicted.append(price)
            lower.append(max(0.01, price * (1.0 + next_lower)))
            upper.append(price * (1.0 + next_upper))
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            history_df,
            future_dates,
            predicted,
            lower,
            upper,
            "SARIMAX",
            "Experimental local forecast using exogenous football signals. Not financial advice.",
        )


class ElasticNetStockModel:
    label = "ElasticNet Factor Model"
    name = "elasticnet_factor"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._frame: pd.DataFrame | None = None
        self._trained = False

    def _feature_cols(self) -> list[str]:
        lookback = int(self.params.get("lookback_lags", 10))
        cols = [f"lag_{i}" for i in range(1, lookback + 1)]
        cols += [
            "ma_gap_3", "ma_gap_5", "rolling_vol_5", "rolling_vol_10", "momentum_3", "momentum_5",
            "exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3",
            "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity",
        ]
        return cols

    def fit(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None):
        frame = build_stock_feature_frame(stock_df, football_events_df, sponsor_meta)
        self._frame = frame
        feature_cols = self._feature_cols()
        for i in range(1, int(self.params.get("lookback_lags", 10)) + 1):
            frame[f"lag_{i}"] = frame["return"].shift(i)
        frame = frame.dropna(subset=["feature_target"]).copy()
        frame[feature_cols] = frame[feature_cols].fillna(0.0)
        self.feature_names = feature_cols
        self.training_rows = int(len(frame))
        self.feature_count = len(feature_cols)
        if len(frame) < 6:
            self.pipeline = None
            self.validation_summary = {"mae": None, "rmse": None, "directional_accuracy": None, "backtest_windows": 0}
            self._trained = True
            return self

        split = max(4, int(len(frame) * 0.8))
        train = frame.iloc[:split]
        holdout = frame.iloc[split:]
        use_scaler = bool(self.params.get("feature_scaling", True))
        steps = []
        if use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "enet",
            ElasticNet(
                alpha=float(self.params.get("alpha", 0.01)),
                l1_ratio=float(self.params.get("l1_ratio", 0.30)),
                max_iter=int(self.params.get("max_iter", 3000)),
                tol=float(self.params.get("tol", 0.0001)),
                random_state=42,
            ),
        ))
        pipeline = Pipeline(steps)
        pipeline.fit(train[feature_cols], train["feature_target"])
        self.pipeline = pipeline
        if not holdout.empty:
            preds = pipeline.predict(holdout[feature_cols])
            mae = mean_absolute_error(holdout["feature_target"], preds)
            rmse = _rmse(holdout["feature_target"], preds)
            da = _safe_directional_accuracy(holdout["feature_target"].to_numpy(), preds)
        else:
            preds = pipeline.predict(train[feature_cols])
            mae = mean_absolute_error(train["feature_target"], preds)
            rmse = _rmse(train["feature_target"], preds)
            da = _safe_directional_accuracy(train["feature_target"].to_numpy(), preds)
        self.validation_summary = {
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "directional_accuracy": round(float(da), 4) if da is not None else None,
            "backtest_windows": int(len(holdout) if not holdout.empty else len(train)),
        }
        self._trained = True
        return self

    def _future_row(self, frame: pd.DataFrame, future_date: pd.Timestamp, forecasted_returns: list[float], sponsor_meta: dict[str, Any] | None, football_events_df: pd.DataFrame) -> pd.Series:
        sponsor_meta = sponsor_meta or {}
        lookback = int(self.params.get("lookback_lags", 10))
        future = pd.Series(index=self.feature_names, dtype=float)
        for i in range(1, lookback + 1):
            if len(forecasted_returns) >= i:
                future[f"lag_{i}"] = forecasted_returns[-i]
            else:
                hist = frame["return"].dropna() if "return" in frame.columns else frame["close"].pct_change().dropna()
                future[f"lag_{i}"] = float(hist.iloc[-i]) if len(hist) >= i else 0.0
        last_close = float(frame["close"].iloc[-1])
        future["ma_gap_3"] = float(frame["close"].tail(3).mean() / last_close - 1.0) if len(frame) >= 3 else 0.0
        future["ma_gap_5"] = float(frame["close"].tail(5).mean() / last_close - 1.0) if len(frame) >= 5 else 0.0
        returns = frame["return"] if "return" in frame.columns else frame["close"].pct_change()
        future["rolling_vol_5"] = float(returns.tail(5).std()) if returns.dropna().shape[0] > 1 else 0.0
        future["rolling_vol_10"] = float(returns.tail(10).std()) if returns.dropna().shape[0] > 2 else 0.0
        future["momentum_3"] = float(last_close / frame["close"].iloc[-4] - 1.0) if len(frame) >= 4 else 0.0
        future["momentum_5"] = float(last_close / frame["close"].iloc[-6] - 1.0) if len(frame) >= 6 else 0.0
        events = football_events_df.copy()
        if not events.empty:
            events["date"] = pd.to_datetime(events["date"]).dt.normalize()
            events = events.set_index("date").sort_index()
            row = events.loc[future_date] if future_date in events.index else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                future[col] = float(row.get(col, 0.0)) if row is not None else 0.0
        else:
            for col in ["exposure_score", "match_day_intensity", "upset_score", "alive_relevance", "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event", "region_relevance", "cum_stage_intensity"]:
                future[col] = 0.0
        future["exposure_score"] = float(sponsor_meta.get("exposure_score", 0.5))
        future["region_relevance"] = float(sponsor_meta.get("region_relevance", 1.0))
        return future

    def forecast(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            self.fit(stock_df, football_events_df, sponsor_meta)
        assert self._frame is not None
        history_df = stock_df.copy().sort_index()
        future_dates = list(pd.bdate_range(history_df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS))
        daily_returns = self._frame["return"].dropna()
        resid_std = float(max(daily_returns.std(), 1e-4)) if len(daily_returns) else 0.01
        predicted = []
        price = float(history_df["close"].iloc[-1])
        forecasted_returns: list[float] = []
        for future_date in future_dates:
            row = self._future_row(history_df, future_date, forecasted_returns, sponsor_meta, football_events_df)
            if self.pipeline is None:
                next_ret = float(daily_returns.mean()) if len(daily_returns) else 0.0
            else:
                next_ret = float(self.pipeline.predict(pd.DataFrame([row[self.feature_names].fillna(0.0)]))[0])
            forecasted_returns.append(next_ret)
            price = price * (1.0 + next_ret)
            predicted.append(price)
        lower, upper = _confidence_bands(predicted, float(history_df["close"].iloc[-1]), resid_std)
        return _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            history_df,
            future_dates,
            predicted,
            lower,
            upper,
            "ElasticNet",
            "Experimental factor model using lagged market and football-event features. Not financial advice.",
        )


class StockEnsembleModel:
    label = "Manual Stock Ensemble"
    name = "stock_ensemble"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.baseline = BaselineGBRStockModel(self.params)
        self.sarimax = SarimaxStockModel(self.params)
        self.elasticnet = ElasticNetStockModel(self.params)
        self.training_rows = 0
        self.feature_count = 0
        self.validation_summary: dict[str, Any] = {}
        self._trained = False
        self.weights: dict[str, float] = {}

    def fit(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None):
        self.baseline.fit(stock_df, football_events_df, sponsor_meta)
        self.sarimax.fit(stock_df, football_events_df, sponsor_meta)
        self.elasticnet.fit(stock_df, football_events_df, sponsor_meta)
        self.weights = {
            "baseline": float(self.params.get("w_baseline", 0.35)),
            "sarimax": float(self.params.get("w_sarimax", 0.40)),
            "elasticnet": float(self.params.get("w_elasticnet", 0.25)),
        }
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.training_rows = max(self.baseline.training_rows, self.sarimax.training_rows, self.elasticnet.training_rows)
        self.feature_count = max(self.baseline.feature_count, self.sarimax.feature_count, self.elasticnet.feature_count)
        self.validation_summary = {
            "mae": None,
            "rmse": None,
            "directional_accuracy": None,
            "backtest_windows": max(self.baseline.validation_summary.get("backtest_windows", 0),
                                    self.sarimax.validation_summary.get("backtest_windows", 0),
                                    self.elasticnet.validation_summary.get("backtest_windows", 0)),
            "weight_mode": self.params.get("weight_mode", "manual"),
        }
        self._trained = True
        return self

    def forecast(self, stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._trained:
            self.fit(stock_df, football_events_df, sponsor_meta)
        baseline = self.baseline.forecast(stock_df, football_events_df, sponsor_meta)
        sarimax = self.sarimax.forecast(stock_df, football_events_df, sponsor_meta)
        elastic = self.elasticnet.forecast(stock_df, football_events_df, sponsor_meta)

        def wavg(values: list[list[float]], weights: list[float]) -> list[float]:
            arr = np.average(np.array(values, dtype=float), axis=0, weights=np.array(weights, dtype=float))
            return [float(v) for v in arr]

        weights = [self.weights["baseline"], self.weights["sarimax"], self.weights["elasticnet"]]
        predicted = wavg(
            [baseline["forecast"]["predicted"], sarimax["forecast"]["predicted"], elastic["forecast"]["predicted"]],
            weights,
        )
        lower = wavg(
            [baseline["forecast"]["lower"], sarimax["forecast"]["lower"], elastic["forecast"]["lower"]],
            weights,
        )
        upper = wavg(
            [baseline["forecast"]["upper"], sarimax["forecast"]["upper"], elastic["forecast"]["upper"]],
            weights,
        )
        out = _prediction_schema(
            self.name,
            self.label,
            dict(self.params),
            self.training_rows,
            self.feature_count,
            self.validation_summary,
            stock_df.copy().sort_index(),
            [pd.Timestamp(d) for d in baseline["forecast"]["dates"]],
            predicted,
            lower,
            upper,
            "ensemble",
            "Manual weighted ensemble of baseline, SARIMAX and ElasticNet forecasts. Not financial advice.",
        )
        out["forecast"]["submodels"] = {
            "baseline_gbr": baseline["forecast"],
            "sarimax_exog": sarimax["forecast"],
            "elasticnet_factor": elastic["forecast"],
            "weights": self.weights,
            "stacking_ready": True,
        }
        return out
