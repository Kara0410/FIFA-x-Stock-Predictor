"""
FIFA data access layer.

Currently reads the sample JSON files in data/sample/. To plug in real data:

  * Option A - replace the JSON files with real exports that follow the same
    schema (see data/sample/generate_sample_data.py for the contract).
  * Option B - implement fetch_live_* below against a real API/scraper and
    flip USE_LIVE_DATA to True. Responses are normalized into the same shape,
    so nothing downstream changes.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

USE_LIVE_DATA = False   # flip when fetch_live_* are implemented


class FifaDataService:
    def __init__(self):
        self._teams = None
        self._players = None
        self._matches = None

    # ------------------------------------------------------------------ #
    # Sample-data loaders (default path)                                  #
    # ------------------------------------------------------------------ #
    def _load(self, filename: str):
        path = SAMPLE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run 'python data/sample/generate_sample_data.py' first")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_teams(self) -> dict:
        """{team_name: stats-dict} for all 32 knockout teams."""
        if self._teams is None:
            self._teams = self.fetch_live_teams() if USE_LIVE_DATA else self._load("teams.json")
        return self._teams

    def get_players(self) -> list[dict]:
        if self._players is None:
            self._players = self.fetch_live_players() if USE_LIVE_DATA else self._load("players.json")
        return self._players

    def get_matches_doc(self) -> dict:
        """Full bracket document: matches list + metadata + upsets."""
        if self._matches is None:
            self._matches = self.fetch_live_matches() if USE_LIVE_DATA else self._load("matches.json")
        return self._matches

    def get_matches(self) -> list[dict]:
        return self.get_matches_doc()["matches"]

    def team_meta(self, name: str) -> dict | None:
        return self.get_teams().get(name)

    def invalidate(self):
        """Drop caches so the next call re-reads the data files/API."""
        self._teams = self._players = self._matches = None

    # ------------------------------------------------------------------ #
    # Live-data hooks (implement these for real FIFA 2026 data)           #
    # ------------------------------------------------------------------ #
    def fetch_live_teams(self) -> dict:
        """Example skeleton:
            resp = requests.get("https://your-football-api/worldcup2026/teams", timeout=10)
            return normalize_teams(resp.json())
        Must return the same schema as data/sample/teams.json."""
        raise NotImplementedError("Wire a real FIFA data API here")

    def fetch_live_players(self) -> list[dict]:
        raise NotImplementedError("Wire a real FIFA data API here")

    def fetch_live_matches(self) -> dict:
        raise NotImplementedError("Wire a real FIFA data API here")
