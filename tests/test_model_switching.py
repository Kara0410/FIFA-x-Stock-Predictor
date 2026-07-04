import json
import unittest

from models.model_registry import (
    FOOTBALL_MODELS,
    STOCK_MODELS,
    get_football_model,
    get_stock_model,
    normalize_weights,
)
from models.persistence import load_last_config, save_last_config


class ModelSwitchingTests(unittest.TestCase):
    def test_default_params_load_correctly(self):
        self.assertIn("baseline_poisson_blend", FOOTBALL_MODELS)
        self.assertIn("baseline_gbr", STOCK_MODELS)
        self.assertEqual(0.55, FOOTBALL_MODELS["baseline_poisson_blend"]["default_params"]["poisson_weight"])
        self.assertEqual(120, STOCK_MODELS["baseline_gbr"]["default_params"]["n_estimators"])

    def test_can_select_each_football_and_stock_model(self):
        for name, meta in FOOTBALL_MODELS.items():
            model = get_football_model(name, meta["default_params"])
            self.assertEqual(name, model.name)

        for name, meta in STOCK_MODELS.items():
            model = get_stock_model(name, meta["default_params"])
            self.assertEqual(name, model.name)

    def test_ensemble_weights_normalize_correctly(self):
        weights = normalize_weights({"a": 2, "b": 1, "c": 1})
        self.assertAlmostEqual(0.5, weights["a"])
        self.assertAlmostEqual(0.25, weights["b"])
        self.assertAlmostEqual(0.25, weights["c"])

    def test_persisted_config_can_be_saved_and_loaded(self):
        config = {
            "football": {"model": "elo_bt", "params": {"k_factor": 30}},
            "stock": {"model": "elasticnet_factor", "params": {"alpha": 0.02}},
            "stock_ticker": "KO",
            "persist": True,
        }
        save_last_config(config)
        loaded = load_last_config()
        self.assertIsNotNone(loaded)
        self.assertEqual("elo_bt", loaded["football"]["model"])
        self.assertEqual("elasticnet_factor", loaded["stock"]["model"])


if __name__ == "__main__":
    unittest.main()
