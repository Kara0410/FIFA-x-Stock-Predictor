"""
Match outcome model.

Combines two views of a knockout tie and blends them:

1. Poisson expected-goals view
   expected_goals_A = BASE_GOAL_RATE * attack_A / defense_B
   expected_goals_B = BASE_GOAL_RATE * attack_B / defense_A
   A full Poisson score grid converts the two xG values into
   P(win) / P(draw) / P(loss) over 90 minutes.

2. Logistic strength view
   P(A beats B) = 1 / (1 + exp(-k * (overall_A - overall_B)))

For knockout football the 90-minute draw probability is redistributed via a
penalty-shootout model: mostly a coin flip, tilted slightly toward the
stronger side.
"""
from __future__ import annotations

import math

from .feature_engineering import TeamFeatureEngineer


class MatchPredictor:
    BASE_GOAL_RATE = 1.30      # average goals per team per knockout match
    LOGISTIC_K = 0.060         # steepness of the strength -> probability curve
    POISSON_WEIGHT = 0.55      # blend: Poisson view vs logistic view
    MAX_GOALS = 9              # Poisson grid size
    PENALTY_TILT = 0.30        # 0 = pure coin flip in shootouts, 1 = pure strength

    def __init__(self, features: TeamFeatureEngineer):
        self.features = features
        self._cache: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------ #
    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        fa, fb = self.features.get(team_a), self.features.get(team_b)
        # Component scores are 0-100; shift so ratios stay in a sane range
        # (a score of 0 must not mean "concedes infinite goals").
        atk_a, atk_b = fa["attack"] + 45.0, fb["attack"] + 45.0
        def_a, def_b = fa["defense"] + 45.0, fb["defense"] + 45.0
        xg_a = self.BASE_GOAL_RATE * atk_a / def_b
        xg_b = self.BASE_GOAL_RATE * atk_b / def_a
        clamp = lambda x: max(0.25, min(3.6, x))
        return clamp(xg_a), clamp(xg_b)

    @staticmethod
    def _poisson_pmf(lam: float, k: int) -> float:
        return math.exp(-lam) * lam ** k / math.factorial(k)

    def _poisson_outcome(self, xg_a: float, xg_b: float) -> tuple[float, float, float]:
        """P(A wins), P(draw), P(B wins) over 90 minutes from the score grid."""
        pa = [self._poisson_pmf(xg_a, k) for k in range(self.MAX_GOALS + 1)]
        pb = [self._poisson_pmf(xg_b, k) for k in range(self.MAX_GOALS + 1)]
        win = draw = loss = 0.0
        for i in range(self.MAX_GOALS + 1):
            for j in range(self.MAX_GOALS + 1):
                p = pa[i] * pb[j]
                if i > j:
                    win += p
                elif i == j:
                    draw += p
                else:
                    loss += p
        total = win + draw + loss
        return win / total, draw / total, loss / total

    # ------------------------------------------------------------------ #
    def predict(self, team_a: str, team_b: str) -> dict:
        """Full prediction for a knockout tie between two named teams."""
        key = (team_a, team_b)
        if key in self._cache:
            return self._cache[key]

        fa, fb = self.features.get(team_a), self.features.get(team_b)
        xg_a, xg_b = self.expected_goals(team_a, team_b)
        p_win, p_draw, p_loss = self._poisson_outcome(xg_a, xg_b)

        # Logistic strength view (two-way, no draw).
        diff = fa["overall"] - fb["overall"]
        p_logistic = 1.0 / (1.0 + math.exp(-self.LOGISTIC_K * diff))

        # Blend the two-way Poisson probability with the logistic one.
        p_poisson_2way = p_win / max(1e-9, p_win + p_loss)
        p_2way = self.POISSON_WEIGHT * p_poisson_2way + (1 - self.POISSON_WEIGHT) * p_logistic

        # Rescale the 90-minute win/loss split to match the blended view,
        # keeping the Poisson draw probability.
        p_win_adj = p_2way * (1 - p_draw)
        p_loss_adj = (1 - p_2way) * (1 - p_draw)

        # Knockout: draws go to extra time + penalties. Shootouts are mostly
        # luck, so tilt only PENALTY_TILT of the way toward the stronger team.
        p_pen_a = 0.5 + (p_2way - 0.5) * self.PENALTY_TILT
        advance_a = p_win_adj + p_draw * p_pen_a
        advance_b = 1.0 - advance_a

        # Confidence: how far from a coin flip the model is, tempered by the
        # tiny sample size of tournament data (max ~4 matches per team).
        matches = min(self.features.teams[team_a]["matches_played"],
                      self.features.teams[team_b]["matches_played"])
        data_factor = min(1.0, matches / 4.0)
        confidence = 0.50 + min(0.45, abs(advance_a - 0.5) * 0.9) * data_factor

        result = {
            "team_a": team_a, "team_b": team_b,
            "advance_a": round(advance_a, 4), "advance_b": round(advance_b, 4),
            "p_win_90": round(p_win_adj, 4), "p_draw_90": round(p_draw, 4), "p_loss_90": round(p_loss_adj, 4),
            "xg_a": round(xg_a, 2), "xg_b": round(xg_b, 2),
            "penalty_win_prob_a": round(p_pen_a, 3),
            "confidence": round(confidence, 3),
            "drivers": {
                team_a: {k: fa[k] for k in ("attack", "defense", "form", "discipline", "player_impact", "overall")},
                team_b: {k: fb[k] for k in ("attack", "defense", "form", "discipline", "player_impact", "overall")},
            },
        }
        self._cache[key] = result
        return result

    def advance_probability(self, team_a: str, team_b: str) -> float:
        """Convenience: P(team_a advances past team_b) including penalties."""
        return self.predict(team_a, team_b)["advance_a"]
