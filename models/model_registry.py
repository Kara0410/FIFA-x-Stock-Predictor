from __future__ import annotations

from typing import Any

from .football_alt_models import (
    BaselinePoissonBlendModel,
    EloBTModel,
    FootballEnsembleModel,
    HistGBFootballModel,
)
from .stock_alt_models import (
    BaselineGBRStockModel,
    ElasticNetStockModel,
    SarimaxStockModel,
    StockEnsembleModel,
)


def normalize_weights(weights_dict: dict[str, float]) -> dict[str, float]:
    weights = {key: max(0.0, float(value)) for key, value in (weights_dict or {}).items()}
    total = sum(weights.values())
    if total <= 0:
        n = max(1, len(weights))
        return {key: 1.0 / n for key in weights} if weights else {}
    return {key: value / total for key, value in weights.items()}


FOOTBALL_MODELS: dict[str, dict[str, Any]] = {
    "baseline_poisson_blend": {
        "name": "baseline_poisson_blend",
        "label": "Baseline Poisson Blend",
        "description": "Existing scoreline blend model for 90-minute results and penalty advancement.",
        "class": BaselinePoissonBlendModel,
        "default_params": {
            "poisson_weight": 0.55,
            "logistic_weight": 0.45,
            "penalty_tilt": 0.55,
            "mc_simulations": 10000,
        },
        "controls": {
            "poisson_weight": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Poisson weight"},
            "logistic_weight": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Logistic weight"},
            "penalty_tilt": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Penalty tilt"},
            "mc_simulations": {"type": "number", "min": 1000, "max": 50000, "step": 1000, "label": "Monte Carlo runs"},
        },
        "supports_train": True,
        "supports_predict_proba": True,
        "pros": "Fast, familiar, and stable on the current data shape.",
        "cons": "Limited adaptivity when the bracket or form signal shifts.",
    },
    "elo_bt": {
        "name": "elo_bt",
        "label": "Elo / Bradley-Terry",
        "description": "Dynamic rating model with draw handling and penalty tilt.",
        "class": EloBTModel,
        "default_params": {
            "k_factor": 24,
            "decay": 0.02,
            "mov_boost": 0.15,
            "draw_prior": 0.26,
            "penalty_tilt": 0.55,
            "regularization_c": 1.0,
        },
        "controls": {
            "k_factor": {"type": "number", "min": 8, "max": 64, "step": 1, "label": "K-factor"},
            "decay": {"type": "slider", "min": 0.0, "max": 0.1, "step": 0.005, "label": "Rating decay"},
            "mov_boost": {"type": "slider", "min": 0.0, "max": 0.5, "step": 0.01, "label": "Margin boost"},
            "draw_prior": {"type": "slider", "min": 0.05, "max": 0.45, "step": 0.01, "label": "Draw prior"},
            "penalty_tilt": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Penalty tilt"},
            "regularization_c": {"type": "number", "min": 0.0, "max": 4.0, "step": 0.1, "label": "Regularization"},
        },
        "supports_train": True,
        "supports_predict_proba": True,
        "pros": "Very interpretable and easy to tune.",
        "cons": "Needs enough match history to stabilize ratings.",
    },
    "histgb_classifier": {
        "name": "histgb_classifier",
        "label": "HistGB Classifier",
        "description": "Gradient-boosted multiclass model with calibrated probabilities.",
        "class": HistGBFootballModel,
        "default_params": {
            "learning_rate": 0.07,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 10,
            "l2_regularization": 0.10,
            "calibration_method": "sigmoid",
        },
        "controls": {
            "learning_rate": {"type": "slider", "min": 0.01, "max": 0.2, "step": 0.01, "label": "Learning rate"},
            "max_iter": {"type": "number", "min": 50, "max": 1000, "step": 25, "label": "Max iterations"},
            "max_leaf_nodes": {"type": "number", "min": 8, "max": 127, "step": 1, "label": "Max leaf nodes"},
            "max_depth": {"type": "number", "min": 2, "max": 12, "step": 1, "label": "Max depth", "allow_null": True},
            "min_samples_leaf": {"type": "number", "min": 1, "max": 50, "step": 1, "label": "Min samples / leaf"},
            "l2_regularization": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.01, "label": "L2 regularization"},
            "calibration_method": {"type": "select", "label": "Calibration", "options": ["sigmoid", "isotonic"]},
        },
        "supports_train": True,
        "supports_predict_proba": True,
        "pros": "Learns nonlinear interactions and outputs calibrated class probabilities.",
        "cons": "Can overfit on very small tournament slices.",
    },
    "football_ensemble": {
        "name": "football_ensemble",
        "label": "Football Ensemble",
        "description": "Manual weighted blend of the baseline, Elo and HistGB models.",
        "class": FootballEnsembleModel,
        "default_params": {
            "w_baseline": 0.40,
            "w_elo_bt": 0.35,
            "w_histgb": 0.25,
            "weight_mode": "manual",
        },
        "controls": {
            "w_baseline": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Baseline weight"},
            "w_elo_bt": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Elo weight"},
            "w_histgb": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "HistGB weight"},
            "weight_mode": {"type": "select", "label": "Weight mode", "options": ["manual", "stacking-ready"]},
        },
        "supports_train": True,
        "supports_predict_proba": True,
        "pros": "Best way to blend interpretability and calibrated probabilities.",
        "cons": "Still limited by the same local data slice as its members.",
    },
}


STOCK_MODELS: dict[str, dict[str, Any]] = {
    "baseline_gbr": {
        "name": "baseline_gbr",
        "label": "Baseline Gradient Boosting",
        "description": "Existing local tree model for short-term sponsor stock prediction.",
        "class": BaselineGBRStockModel,
        "default_params": {
            "n_estimators": 120,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.90,
        },
        "controls": {
            "n_estimators": {"type": "number", "min": 20, "max": 500, "step": 10, "label": "Estimators"},
            "max_depth": {"type": "number", "min": 1, "max": 10, "step": 1, "label": "Tree depth"},
            "learning_rate": {"type": "slider", "min": 0.01, "max": 0.3, "step": 0.01, "label": "Learning rate"},
            "subsample": {"type": "slider", "min": 0.5, "max": 1.0, "step": 0.01, "label": "Subsample"},
        },
        "supports_train": True,
        "supports_predict_proba": False,
        "pros": "Fast and resilient when the sample window is tiny.",
        "cons": "Does not explicitly model event timing.",
    },
    "sarimax_exog": {
        "name": "sarimax_exog",
        "label": "SARIMAX + Football Signals",
        "description": "ARIMA-style forecast with exogenous tournament regressors.",
        "class": SarimaxStockModel,
        "default_params": {
            "order": [1, 0, 1],
            "seasonal_order": [0, 0, 0, 0],
            "trend": "c",
            "exog_lag_days": 1,
            "rolling_window": 5,
            "enforce_stationarity": True,
            "enforce_invertibility": True,
        },
        "controls": {
            "order": {"type": "text", "label": "Order", "placeholder": "[1,0,1]"},
            "seasonal_order": {"type": "text", "label": "Seasonal order", "placeholder": "[0,0,0,0]"},
            "trend": {"type": "select", "label": "Trend", "options": ["n", "c", "t", "ct"]},
            "exog_lag_days": {"type": "number", "min": 0, "max": 7, "step": 1, "label": "Exog lag days"},
            "rolling_window": {"type": "number", "min": 2, "max": 21, "step": 1, "label": "Rolling window"},
            "enforce_stationarity": {"type": "toggle", "label": "Enforce stationarity"},
            "enforce_invertibility": {"type": "toggle", "label": "Enforce invertibility"},
        },
        "supports_train": True,
        "supports_predict_proba": False,
        "pros": "Brings in event regressors and uncertainty bands.",
        "cons": "Can be fragile when the history window is very short.",
    },
    "elasticnet_factor": {
        "name": "elasticnet_factor",
        "label": "ElasticNet Factor Model",
        "description": "Standardized lag-factor model with recursive 7-day forecasts.",
        "class": ElasticNetStockModel,
        "default_params": {
            "alpha": 0.01,
            "l1_ratio": 0.30,
            "max_iter": 3000,
            "tol": 0.0001,
            "lookback_lags": 10,
            "feature_scaling": True,
        },
        "controls": {
            "alpha": {"type": "slider", "min": 0.0001, "max": 0.1, "step": 0.0005, "label": "Alpha"},
            "l1_ratio": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "L1 ratio"},
            "max_iter": {"type": "number", "min": 500, "max": 10000, "step": 250, "label": "Max iterations"},
            "tol": {"type": "number", "min": 0.00001, "max": 0.01, "step": 0.00001, "label": "Tolerance"},
            "lookback_lags": {"type": "number", "min": 3, "max": 30, "step": 1, "label": "Lag count"},
            "feature_scaling": {"type": "toggle", "label": "Standardize inputs"},
        },
        "supports_train": True,
        "supports_predict_proba": False,
        "pros": "Simple, transparent, and stable with lagged inputs.",
        "cons": "Linear structure misses sharp event-driven breaks.",
    },
    "stock_ensemble": {
        "name": "stock_ensemble",
        "label": "Stock Ensemble",
        "description": "Manual weighted blend of the three stock forecasters.",
        "class": StockEnsembleModel,
        "default_params": {
            "w_baseline": 0.35,
            "w_sarimax": 0.40,
            "w_elasticnet": 0.25,
            "weight_mode": "manual",
        },
        "controls": {
            "w_baseline": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Baseline weight"},
            "w_sarimax": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "SARIMAX weight"},
            "w_elasticnet": {"type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "label": "ElasticNet weight"},
            "weight_mode": {"type": "select", "label": "Weight mode", "options": ["manual", "stacking-ready"]},
        },
        "supports_train": True,
        "supports_predict_proba": False,
        "pros": "Balances trend, events, and lag structure.",
        "cons": "Adds complexity without adding new data sources.",
    },
}


def get_football_model(name: str, params: dict[str, Any] | None = None):
    entry = FOOTBALL_MODELS[name]
    merged = dict(entry["default_params"])
    if params:
        merged.update(params)
    return entry["class"](merged)


def get_stock_model(name: str, params: dict[str, Any] | None = None):
    entry = STOCK_MODELS[name]
    merged = dict(entry["default_params"])
    if params:
        merged.update(params)
    return entry["class"](merged)
