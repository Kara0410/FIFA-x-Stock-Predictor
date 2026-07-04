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
