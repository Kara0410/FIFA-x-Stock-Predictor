import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from app import app
from models.bracket_simulator import BracketSimulator


ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = ROOT / "data" / "sample" / "matches.json"


class RecordingPredictor:
    def __init__(self, teams):
        self.features = SimpleNamespace(teams={team: {} for team in teams})
        self.calls = []

    def advance_probability(self, home, away):
        self.calls.append((home, away))
        return 0.5


class CompletedMatchRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(MATCHES_PATH.read_text(encoding="utf-8"))
        cls.matches = cls.document["matches"]

    def test_round_of_32_contains_only_verified_completed_results(self):
        completed = [match for match in self.matches if match["status"] == "completed"]

        self.assertEqual(16, len(completed))
        self.assertTrue(all(match["round"] == "R32" for match in completed))
        self.assertTrue(all(match["winner"] for match in completed))
        self.assertTrue(all(match["home_score"] is not None for match in completed))
        self.assertTrue(all(match["away_score"] is not None for match in completed))

    def test_round_of_16_pairings_are_actual_winners(self):
        expected = {
            "M89": ("Paraguay", "France"),
            "M90": ("Canada", "Morocco"),
            "M91": ("Brazil", "Norway"),
            "M92": ("Mexico", "England"),
            "M93": ("Portugal", "Spain"),
            "M94": ("USA", "Belgium"),
            "M95": ("Argentina", "Egypt"),
            "M96": ("Switzerland", "Colombia"),
        }
        actual = {
            match["id"]: (match["home"], match["away"])
            for match in self.matches
            if match["round"] == "R16"
        }

        self.assertEqual(expected, actual)

    def test_simulator_never_predicts_a_completed_match(self):
        teams = {
            team
            for match in self.matches
            for team in (match["home"], match["away"])
            if team
        }
        predictor = RecordingPredictor(teams)
        simulator = BracketSimulator(self.matches, predictor)

        simulator.simulate(n_simulations=2)

        completed_pairs = {
            (match["home"], match["away"])
            for match in self.matches
            if match["status"] == "completed"
        }
        self.assertTrue(predictor.calls)
        self.assertTrue(completed_pairs.isdisjoint(predictor.calls))

    def test_api_exposes_predictions_only_for_unplayed_matches(self):
        with app.test_client() as client:
            response = client.get("/api/bracket")

        self.assertEqual(200, response.status_code)
        matches = response.get_json()["matches"]
        completed = [match for match in matches if match["status"] == "completed"]
        upcoming = [match for match in matches if match["status"] == "upcoming"]

        self.assertTrue(completed)
        self.assertTrue(upcoming)
        self.assertTrue(all(match["prediction"] is None for match in completed))
        self.assertTrue(all(match["prediction"] is not None for match in upcoming))


if __name__ == "__main__":
    unittest.main()
