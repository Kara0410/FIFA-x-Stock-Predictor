"""
Stock price history access.

Primary source: yfinance (Yahoo Finance), from the World Cup start date
(2026-06-11) until today. Downloads are cached as CSV in data/raw/ for a few
hours to avoid hammering the API on every page load.

Fallback: if the download fails (offline, delisted, rate-limit), a
deterministic synthetic price series is generated and clearly flagged with
source="demo" so the UI can label it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .fifa_data_service import FifaDataService
from .sponsor_service import DEMO_BASE_PRICES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TOURNAMENT_START = "2026-06-11"
CACHE_MAX_AGE_SECONDS = 4 * 3600


class StockDataService:
    def __init__(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._fifa = FifaDataService()

    # ------------------------------------------------------------------ #
    def get_history(self, ticker: str) -> tuple[pd.DataFrame, str]:
        """Returns (DataFrame indexed by date with 'close' column, source).

        source is 'yfinance' for real data or 'demo' for synthetic fallback.
        """
        cached = self._read_cache(ticker)
        if cached is not None:
            return cached

        df = self._download(ticker)
        if df is not None and len(df) >= 5:
            self._write_cache(ticker, df, "yfinance")
            return df, "yfinance"

        demo = self._demo_series(ticker)
        self._write_cache(ticker, demo, "demo")
        return demo, "demo"

    def get_stock_history_df(self, ticker: str) -> pd.DataFrame:
        history, _ = self.get_history(ticker)
        return history

    def get_football_event_df(self) -> pd.DataFrame:
        matches = self._fifa.get_matches_doc().get("matches", [])
        upsets = {pd.Timestamp(item["date"]).normalize(): float(item.get("magnitude", 0.0))
                  for item in self._fifa.get_matches_doc().get("upsets", [])}
        if not matches:
            return pd.DataFrame(columns=[
                "date", "exposure_score", "match_day_intensity", "upset_score", "alive_relevance",
                "event_decay_3", "event_decay_5", "same_day_event", "lag1_event", "lag2_event",
                "region_relevance", "cum_stage_intensity",
            ])

        stage_weights = {"GROUP": 1.0, "R32": 2.0, "R16": 3.0, "QF": 4.0, "SF": 5.0, "F": 6.0}
        start = pd.Timestamp(TOURNAMENT_START)
        dates = pd.bdate_range(start, dt.date.today())
        by_date = defaultdict(list)
        for match in matches:
            by_date[pd.Timestamp(match["date"]).normalize()].append(match)

        cumulative = 0.0
        rows = []
        for date in dates:
            todays = by_date.get(date, [])
            stage_intensity = max((stage_weights[m["round"]] for m in todays), default=0.0)
            match_day_intensity = float(len(todays) * stage_intensity)
            upset_score = float(upsets.get(date, 0.0))
            cumulative += stage_intensity
            alive_relevance = float(max(0.0, 1.0 - 0.03 * len([m for m in matches if pd.Timestamp(m["date"]).normalize() <= date and m.get("status") == "completed"])))
            same_day_event = 1.0 if todays else 0.0
            recent_intensity = [
                max((stage_weights[m["round"]] for m in by_date.get(date - pd.Timedelta(days=offset), [])), default=0.0)
                for offset in (1, 2, 3, 5)
            ]
            rows.append({
                "date": date,
                "exposure_score": float(min(1.0, 0.20 + 0.15 * stage_intensity + 0.10 * len(todays))),
                "match_day_intensity": match_day_intensity,
                "upset_score": upset_score,
                "alive_relevance": alive_relevance,
                "event_decay_3": float(sum(recent_intensity[:3]) / 3.0 if recent_intensity[:3] else 0.0),
                "event_decay_5": float(sum(recent_intensity) / 4.0 if recent_intensity else 0.0),
                "same_day_event": same_day_event,
                "lag1_event": float(recent_intensity[0] if recent_intensity else 0.0),
                "lag2_event": float(recent_intensity[1] if len(recent_intensity) > 1 else 0.0),
                "region_relevance": 1.0,
                "cum_stage_intensity": cumulative,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def _download(self, ticker: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
            raw = yf.download(ticker, start=TOURNAMENT_START, end=end,
                              progress=False, auto_adjust=True, threads=False)
            if raw is None or raw.empty:
                return None
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):        # multi-index columns
                close = close.iloc[:, 0]
            df = pd.DataFrame({"close": close.astype(float)})
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "date"
            return df.dropna()
        except Exception as exc:                        # noqa: BLE001
            print(f"[stock] yfinance download failed for {ticker}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    def _demo_series(self, ticker: str) -> pd.DataFrame:
        """Deterministic geometric-random-walk demo prices per ticker."""
        seed = int(hashlib.sha1(ticker.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        base = DEMO_BASE_PRICES.get(ticker.upper(), 100.0)

        dates = pd.bdate_range(TOURNAMENT_START, dt.date.today())
        drift = rng.uniform(-0.0008, 0.0014)
        vol = rng.uniform(0.008, 0.018)
        prices, price = [], base
        for _ in dates:
            price *= 1.0 + rng.gauss(drift, vol)
            prices.append(round(price, 2))
        df = pd.DataFrame({"close": prices}, index=dates)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------ #
    # Tiny CSV cache in data/raw/                                          #
    # ------------------------------------------------------------------ #
    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace(".", "_").replace("/", "_")
        return RAW_DIR / f"{safe}.csv"

    def _write_cache(self, ticker: str, df: pd.DataFrame, source: str):
        try:
            out = df.copy()
            out["source"] = source
            out.to_csv(self._cache_path(ticker))
        except OSError as exc:
            print(f"[stock] cache write failed for {ticker}: {exc}")

    def _read_cache(self, ticker: str) -> tuple[pd.DataFrame, str] | None:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > CACHE_MAX_AGE_SECONDS:
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            source = str(df["source"].iloc[0])
            return df[["close"]], source
        except Exception:                               # noqa: BLE001
            return None
