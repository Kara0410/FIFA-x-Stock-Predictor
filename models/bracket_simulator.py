"""
Monte Carlo bracket simulator.

Simulates every remaining knockout match N times (default 10,000) using the
MatchPredictor's advance probabilities and tallies, per team:

    probability to reach R16 / QF / SF / Final / win the tournament

It also records, for matches whose participants are not yet known, how often
each team appeared in that slot (used by the UI to show "most likely
participants" on TBD match cards).
"""
from __future__ import annotations

import random
from collections import defaultdict

from .football_model import MatchPredictor

ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]
# Reaching round X means "appeared in a match of round X"; winning the final
# is tracked separately as "champion".
NEXT_ROUND_KEY = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "F"}


class BracketSimulator:
    def __init__(self, matches: list[dict], predictor: MatchPredictor, seed: int = 42):
        self.matches = sorted(matches, key=lambda m: ROUND_ORDER.index(m["round"]))
        self.by_id = {m["id"]: m for m in self.matches}
        self.predictor = predictor
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _participants(self, match: dict, sim_winners: dict[str, str]) -> tuple[str, str]:
        """Resolve the two teams of a match inside one simulation run."""
        home = match["home"] or sim_winners.get(match["home_from"])
        away = match["away"] or sim_winners.get(match["away_from"])
        return home, away

    def simulate(self, n_simulations: int = 10000) -> dict:
        rng = random.Random(self.seed)
        reach = defaultdict(lambda: defaultdict(int))       # team -> round -> count
        champion = defaultdict(int)                          # team -> titles
        slot_appearances = defaultdict(lambda: defaultdict(int))   # match -> team -> count
        slot_wins = defaultdict(lambda: defaultdict(int))           # match -> team -> wins

        for _ in range(n_simulations):
            sim_winners: dict[str, str] = {}
            for m in self.matches:
                if m["status"] == "completed":
                    sim_winners[m["id"]] = m["winner"]
                    continue
                home, away = self._participants(m, sim_winners)
                # Tally who showed up in this match (interesting for TBD cards).
                slot_appearances[m["id"]][home] += 1
                slot_appearances[m["id"]][away] += 1
                reach[home][m["round"]] += 1
                reach[away][m["round"]] += 1

                p_home = self.predictor.advance_probability(home, away)
                win = home if rng.random() < p_home else away
                sim_winners[m["id"]] = win
                slot_wins[m["id"]][win] += 1
                if m["round"] == "F":
                    champion[win] += 1

        return self._summarize(n_simulations, reach, champion, slot_appearances, slot_wins)

    # ------------------------------------------------------------------ #
    def _summarize(self, n, reach, champion, slot_appearances, slot_wins) -> dict:
        # Teams already eliminated (lost a completed match) get zeros;
        # teams that already reached a round via completed matches get 1.0.
        completed_reach = defaultdict(set)
        eliminated = set()
        for m in self.matches:
            if m["status"] != "completed":
                continue
            for t in (m["home"], m["away"]):
                completed_reach[t].add(m["round"])
                if t != m["winner"]:
                    eliminated.add(t)

        all_teams = set(self.predictor.features.teams.keys())
        table = {}
        for team in all_teams:
            row = {}
            for rnd in ROUND_ORDER:
                if rnd in completed_reach[team]:
                    row[rnd] = 1.0                       # already played this round
                else:
                    row[rnd] = reach[team][rnd] / n      # simulated frequency
            row["champion"] = champion[team] / n
            if team in eliminated:
                # Keep historical rounds at 1.0 but zero out everything ahead.
                seen_last = max((ROUND_ORDER.index(r) for r in completed_reach[team]), default=-1)
                for i, rnd in enumerate(ROUND_ORDER):
                    if i > seen_last:
                        row[rnd] = 0.0
                row["champion"] = 0.0
            table[team] = {k: round(v, 4) for k, v in row.items()}

        # Per-match slot candidates for TBD matches.
        match_outlook = {}
        for mid, apps in slot_appearances.items():
            candidates = []
            for team, count in sorted(apps.items(), key=lambda kv: -kv[1]):
                candidates.append({
                    "team": team,
                    "p_appear": round(count / n, 4),
                    "p_win_if_played": round(slot_wins[mid][team] / count, 4) if count else 0.0,
                    "p_win_match": round(slot_wins[mid][team] / n, 4),
                })
            match_outlook[mid] = candidates

        return {
            "n_simulations": n,
            "advancement": table,
            "match_outlook": match_outlook,
            "title_ranking": sorted(
                [{"team": t, "p_champion": table[t]["champion"],
                  "p_final": table[t]["F"], "p_sf": table[t]["SF"], "p_qf": table[t]["QF"]}
                 for t in table if table[t]["champion"] > 0 or table[t]["F"] > 0],
                key=lambda r: -r["p_champion"]),
        }

    # ------------------------------------------------------------------ #
    def predicted_path(self) -> tuple[str | None, set[str]]:
        """Greedy most-likely walk through the bracket.

        Returns (most likely champion, set of match ids on that champion's
        path) for highlighting in the UI.
        """
        winners: dict[str, str] = {}
        for m in self.matches:
            if m["status"] == "completed":
                winners[m["id"]] = m["winner"]
                continue
            home = m["home"] or winners.get(m["home_from"])
            away = m["away"] or winners.get(m["away_from"])
            if not home or not away:
                return None, set()
            p = self.predictor.advance_probability(home, away)
            winners[m["id"]] = home if p >= 0.5 else away

        final = next((m for m in self.matches if m["round"] == "F"), None)
        if not final:
            return None, set()
        champ = winners[final["id"]]

        path = set()
        for m in self.matches:
            home = m["home"] or winners.get(m["home_from"])
            away = m["away"] or winners.get(m["away_from"])
            if champ in (home, away):
                path.add(m["id"])
        return champ, path
