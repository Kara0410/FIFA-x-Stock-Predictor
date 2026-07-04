import tempfile
import unittest
from pathlib import Path

from app import app
from models import persistence
from models.model_registry import FOOTBALL_MODELS, STOCK_MODELS


class ApiModelTests(unittest.TestCase):
    def setUp(self):
        self._orig_last = persistence.LAST_CONFIG_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        persistence.LAST_CONFIG_PATH = Path(self._tmpdir.name) / "last_model_config.json"

    def tearDown(self):
        persistence.LAST_CONFIG_PATH = self._orig_last
        self._tmpdir.cleanup()

    def test_model_defaults_schema(self):
        with app.test_client() as client:
            response = client.get("/api/model_defaults")

        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertIn("football", data)
        self.assertIn("stock", data)
        self.assertIn("baseline_poisson_blend", data["football"])
        self.assertIn("baseline_gbr", data["stock"])

    def test_last_config_returns_200_when_missing(self):
        with app.test_client() as client:
            response = client.get("/api/last_config")

        self.assertEqual(200, response.status_code)
        data = response.get_json()
        self.assertIn("football", data)
        self.assertIn("stock", data)
        self.assertIn("stock_ticker", data)

    def test_recalculate_supports_requested_model_combinations(self):
        cases = [
            ("baseline_poisson_blend", "baseline_gbr"),
            ("elo_bt", "sarimax_exog"),
            ("histgb_classifier", "elasticnet_factor"),
            ("football_ensemble", "stock_ensemble"),
        ]
        with app.test_client() as client:
            for football_model, stock_model in cases:
                payload = {
                    "football": {
                        "model": football_model,
                        "params": {**FOOTBALL_MODELS[football_model]["default_params"], "mc_simulations": 250},
                    },
                    "stock": {
                        "model": stock_model,
                        "params": STOCK_MODELS[stock_model]["default_params"],
                    },
                    "stock_ticker": "KO",
                    "persist": False,
                }
                response = client.post("/api/recalculate", json=payload)
                self.assertEqual(200, response.status_code, football_model)
                data = response.get_json()
                self.assertIn("bracket", data)
                self.assertIn("football_model_info", data)
                self.assertIn("stock_model_info", data)
                self.assertIn("stock_history", data)
                self.assertIn("stock_forecast", data)


if __name__ == "__main__":
    unittest.main()
