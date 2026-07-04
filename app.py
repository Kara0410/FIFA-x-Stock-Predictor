from __future__ import annotations

import ast
import hashlib
import json
import threading
from copy import deepcopy
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from models.bracket_simulator import BracketSimulator
from models.model_registry import (
    FOOTBALL_MODELS,
    STOCK_MODELS,
    get_football_model,
    get_stock_model,
    normalize_weights,
)
from models.persistence import config_signature, load_last_config, load_model, save_last_config, save_model
from services.fifa_data_service import FifaDataService
from services.sponsor_service import SponsorService
from services.stock_data_service import StockDataService

app = Flask(__name__)

fifa = FifaDataService()
sponsors = SponsorService()
stocks = StockDataService()

DEFAULT_FOOTBALL = "baseline_poisson_blend"
DEFAULT_STOCK = "baseline_gbr"
DEFAULT_TICKER = "KO"
DEFAULT_SIMULATIONS = 10_000

STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {}


def _defaults_for(section: str, name: str) -> dict[str, Any]:
    registry = FOOTBALL_MODELS if section == "football" else STOCK_MODELS
    return deepcopy(registry[name]["default_params"])


def _current_defaults() -> dict[str, Any]:
    return {
        "football": {
            "model": DEFAULT_FOOTBALL,
            "params": _defaults_for("football", DEFAULT_FOOTBALL),
        },
        "stock": {
            "model": DEFAULT_STOCK,
            "params": _defaults_for("stock", DEFAULT_STOCK),
        },
        "stock_ticker": DEFAULT_TICKER,
        "persist": True,
    }


def _model_defaults_payload() -> dict[str, Any]:
    def serialize(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
        payload = {}
        for name, meta in registry.items():
            payload[name] = {
                "label": meta["label"],
                "description": meta["description"],
                "defaults": deepcopy(meta["default_params"]),
                "controls": deepcopy(meta["controls"]),
                "supports_train": bool(meta["supports_train"]),
                "supports_predict_proba": bool(meta["supports_predict_proba"]),
                "pros": meta.get("pros", ""),
                "cons": meta.get("cons", ""),
            }
        return payload

    return {"football": serialize(FOOTBALL_MODELS), "stock": serialize(STOCK_MODELS)}


def _normalize_param_value(value: Any, default: Any) -> Any:
    if value is None:
        return None
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return default
    if isinstance(default, list):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else default
            except Exception:  # noqa: BLE001
                return default
    if default is None:
        return value
    return value


def _merge_model_config(section: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    registry = FOOTBALL_MODELS if section == "football" else STOCK_MODELS
    payload = payload or {}
    default_name = DEFAULT_FOOTBALL if section == "football" else DEFAULT_STOCK
    model_name = payload.get("model") or default_name
    if model_name not in registry:
        model_name = default_name
    defaults = deepcopy(registry[model_name]["default_params"])
    incoming = payload.get("params") or {}
    params = {}
    for key, default in defaults.items():
        params[key] = _normalize_param_value(incoming.get(key, default), default)
    for key, value in incoming.items():
        if key not in params:
            params[key] = value
    if section == "football" and model_name in {"football_ensemble"}:
        params = {**defaults, **params}
        params = {
            **params,
            "w_baseline": float(params.get("w_baseline", 0.4)),
            "w_elo_bt": float(params.get("w_elo_bt", 0.35)),
            "w_histgb": float(params.get("w_histgb", 0.25)),
        }
        weights = normalize_weights({
            "baseline": params["w_baseline"],
            "elo_bt": params["w_elo_bt"],
            "histgb": params["w_histgb"],
        })
        params["w_baseline"] = weights["baseline"]
        params["w_elo_bt"] = weights["elo_bt"]
        params["w_histgb"] = weights["histgb"]
    if section == "stock" and model_name in {"stock_ensemble"}:
        params = {**defaults, **params}
        weights = normalize_weights({
            "baseline": params.get("w_baseline", 0.35),
            "sarimax": params.get("w_sarimax", 0.40),
            "elasticnet": params.get("w_elasticnet", 0.25),
        })
        params["w_baseline"] = weights["baseline"]
        params["w_sarimax"] = weights["sarimax"]
        params["w_elasticnet"] = weights["elasticnet"]
    return {"model": model_name, "params": params}


def _merge_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    defaults = _current_defaults()
    payload = payload or {}
    football = _merge_model_config("football", payload.get("football") if isinstance(payload.get("football"), dict) else None)
    stock = _merge_model_config("stock", payload.get("stock") if isinstance(payload.get("stock"), dict) else None)
    ticker = str(payload.get("stock_ticker") or defaults["stock_ticker"]).upper()
    persist = bool(payload.get("persist", True))
    return {
        "football": football,
        "stock": stock,
        "stock_ticker": ticker,
        "persist": persist,
    }


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "empty"
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    except Exception:  # noqa: BLE001
        hashed = frame.to_json(date_format="iso", orient="split").encode("utf-8")
    return hashlib.sha1(hashed).hexdigest()


def _football_data_hash(matches_df: pd.DataFrame, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> str:
    return hashlib.sha1(("|".join([_frame_hash(matches_df), _frame_hash(teams_df), _frame_hash(players_df)])).encode("utf-8")).hexdigest()


def _stock_data_hash(stock_df: pd.DataFrame, football_events_df: pd.DataFrame, sponsor_meta: dict[str, Any]) -> str:
    return hashlib.sha1(("|".join([
        _frame_hash(stock_df),
        _frame_hash(football_events_df),
        json.dumps(sponsor_meta, sort_keys=True, ensure_ascii=False),
    ])).encode("utf-8")).hexdigest()


def _team_blob(name: str | None, teams: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not name:
        return None
    team = teams.get(name)
    if not team:
        return None
    return {"name": name, "code": team.get("code", name[:3].upper()), "flag": team.get("flag", "")}


def _result_to_ui_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    xg_home = pred.get("xg_a", pred.get("xg_home"))
    xg_away = pred.get("xg_b", pred.get("xg_away"))
    if xg_home is None:
        xg_home = round(1.0 + float(pred.get("p_team_a", 0.5)) * 1.75, 2)
    if xg_away is None:
        xg_away = round(1.0 + float(pred.get("p_team_b", 0.5)) * 1.75, 2)
    return {
        "home_advance": round(float(pred.get("p_advance_a", pred.get("advance_a", 0.5))), 4),
        "away_advance": round(float(pred.get("p_advance_b", pred.get("advance_b", 0.5))), 4),
        "p_draw_90": round(float(pred.get("p_draw_90", 0.25)), 4),
        "xg_home": round(float(xg_home), 2),
        "xg_away": round(float(xg_away), 2),
        "confidence": round(float(pred.get("confidence", 0.5)), 4),
        "drivers": pred.get("drivers", {}),
        "p_team_a": round(float(pred.get("p_team_a", pred.get("p_advance_a", 0.5))), 4),
        "p_team_b": round(float(pred.get("p_team_b", pred.get("p_advance_b", 0.5))), 4),
    }


def _selected_model_info(model_name: str, params: dict[str, Any], training_rows: int, feature_count: int, validation_summary: dict[str, Any], prediction_timestamp: str) -> dict[str, Any]:
    registry = FOOTBALL_MODELS if model_name in FOOTBALL_MODELS else STOCK_MODELS
    return {
        "model_name": model_name,
        "model_label": registry[model_name]["label"],
        "params_used": params,
        "training_rows": int(training_rows),
        "feature_count": int(feature_count),
        "validation_summary": validation_summary,
        "prediction_timestamp": prediction_timestamp,
    }


def _build_football_model(config: dict[str, Any], persist: bool) -> Any:
    matches_df = fifa.get_matches_df()
    teams_df = fifa.get_teams_df()
    players_df = fifa.get_players_df()
    data_hash = _football_data_hash(matches_df, teams_df, players_df)
    model_name = config["model"]
    params = config["params"]
    cached = load_model(model_name, params, data_hash, ticker=None) if persist else None
    if cached is not None:
        return cached, data_hash
    model = get_football_model(model_name, params)
    model.fit(matches_df, teams_df, players_df)
    if persist:
        save_model(model, model_name, params, data_hash)
    return model, data_hash


def _build_stock_model(config: dict[str, Any], ticker: str, persist: bool) -> tuple[Any, str, dict[str, Any], pd.DataFrame, str]:
    stock_df, source = stocks.get_history(ticker)
    football_events_df = stocks.get_football_event_df()
    sponsor_meta = sponsors.get(ticker) or {"ticker": ticker, "name": ticker, "currency": "USD", "exposure_score": 0.5, "region": "Global"}
    sponsor_meta = dict(sponsor_meta)
    sponsor_meta["exposure_score"] = sponsors.exposure_for(ticker, teams_still_alive())
    sponsor_meta["region_relevance"] = 1.0 if sponsor_meta.get("region") else 0.75
    data_hash = _stock_data_hash(stock_df, football_events_df, sponsor_meta)
    model_name = config["model"]
    params = config["params"]
    cached = load_model(model_name, params, data_hash, ticker=ticker) if persist else None
    if cached is not None:
        return cached, data_hash, sponsor_meta, stock_df, source
    model = get_stock_model(model_name, params)
    model.fit(stock_df, football_events_df, sponsor_meta)
    if persist:
        save_model(model, model_name, params, data_hash, ticker=ticker)
    return model, data_hash, sponsor_meta, stock_df, source


def _build_bracket_response(football_model: Any, football_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str | None, set[str]]:
    teams = fifa.get_teams()
    matches = fifa.get_matches()
    simulator = BracketSimulator(matches, football_model)
    sim = simulator.simulate(int(football_config["params"].get("mc_simulations", DEFAULT_SIMULATIONS)))
    champ, path = simulator.predicted_path()

    def predict_for_match(match: dict[str, Any]) -> dict[str, Any] | None:
        if match["status"] == "upcoming" and match.get("home") and match.get("away"):
            context = {"match_date": match.get("date"), "venue": match.get("venue")}
            pred = football_model.predict_match(match["home"], match["away"], context)
            return _result_to_ui_prediction(pred)
        return None

    matches_out = []
    for match in matches:
        entry = {
            "id": match["id"],
            "round": match["round"],
            "date": match["date"],
            "venue": match["venue"],
            "status": match["status"],
            "home": _team_blob(match.get("home"), teams),
            "away": _team_blob(match.get("away"), teams),
            "home_from": match.get("home_from"),
            "away_from": match.get("away_from"),
            "home_score": match.get("home_score"),
            "away_score": match.get("away_score"),
            "penalties": match.get("penalties"),
            "winner": match.get("winner"),
            "on_predicted_path": match["id"] in path,
            "prediction": None,
            "slot_candidates": None,
        }
        if match["status"] == "upcoming":
            entry["prediction"] = predict_for_match(match)
        elif match["status"] == "scheduled":
            outlook = sim["match_outlook"].get(match["id"], [])
            entry["slot_candidates"] = [
                {**candidate, "flag": teams[candidate["team"]]["flag"], "code": teams[candidate["team"]]["code"]}
                for candidate in outlook[:6]
                if candidate["team"] in teams
            ]
        matches_out.append(entry)

    ranking = [
        {**row, "flag": teams[row["team"]]["flag"], "code": teams[row["team"]]["code"]}
        for row in sim["title_ranking"][:10]
        if row["team"] in teams
    ]
    champion = champ or next((row["team"] for row in ranking[:1]), None)
    return {
        "tournament": "FIFA World Cup 2026",
        "as_of": fifa.get_matches_doc().get("note", ""),
        "n_simulations": sim["n_simulations"],
        "matches": matches_out,
        "title_ranking": ranking,
        "predicted_champion": champion,
        "model": {
            "components": getattr(getattr(football_model, "features", None), "OVERALL_WEIGHTS", {}),
            "base_goal_rate": getattr(football_model, "BASE_GOAL_RATE", getattr(getattr(football_model, "predictor", None), "BASE_GOAL_RATE", None)),
            "poisson_weight": getattr(football_model, "POISSON_WEIGHT", getattr(getattr(football_model, "predictor", None), "POISSON_WEIGHT", None)),
            "penalty_tilt": getattr(football_model, "PENALTY_TILT", getattr(getattr(football_model, "predictor", None), "PENALTY_TILT", None)),
            "selected_model": football_config["model"],
        },
    }, sim, champ, path


def _build_stock_payload(ticker: str, stock_model: Any, sponsor_meta: dict[str, Any], stock_df: pd.DataFrame) -> dict[str, Any]:
    football_events_df = stocks.get_football_event_df()
    forecast_blob = stock_model.forecast(stock_df, football_events_df, sponsor_meta)
    return {
        "sponsor": sponsors.get(ticker) or {"ticker": ticker, "name": ticker, "currency": "USD"},
        "source": "demo" if "DEMO" in forecast_blob["forecast"]["disclaimer"].upper() else ("yfinance" if ticker in {"ADS.DE", "KO", "V", "BAC", "MCD", "PEP", "VZ", "HD", "VVV", "LNVGY", "2222.SR"} else "yfinance"),
        "is_demo": False,
        "exposure_score": round(float(sponsor_meta.get("exposure_score", 0.5)), 3),
        "history": forecast_blob["history"],
        "forecast": forecast_blob["forecast"],
        "model_info": _selected_model_info(
            stock_model.name,
            stock_model.params,
            getattr(stock_model, "training_rows", 0),
            getattr(stock_model, "feature_count", 0),
            getattr(stock_model, "validation_summary", {}),
            getattr(stock_model, "validation_summary", {}).get("prediction_timestamp", ""),
        ),
    }


def _compute_dashboard(config: dict[str, Any]) -> dict[str, Any]:
    football_model, football_hash = _build_football_model(config["football"], config["persist"])
    stock_model, stock_hash, sponsor_meta, stock_df, stock_source = _build_stock_model(config["stock"], config["stock_ticker"], config["persist"])
    bracket, sim, champ, path = _build_bracket_response(football_model, config["football"])
    stock_payload = _build_stock_payload(config["stock_ticker"], stock_model, sponsor_meta, stock_df)
    stock_payload["source"] = stock_source
    stock_payload["is_demo"] = stock_source == "demo"

    state = {
        "ready": True,
        "config": config,
        "football_model": football_model,
        "football_hash": football_hash,
        "stock_model": stock_model,
        "stock_hash": stock_hash,
        "sponsor_meta": sponsor_meta,
        "stock_df": stock_df,
        "bracket": bracket,
        "simulation": sim,
        "predicted_champion": champ,
        "predicted_path": path,
        "stock_payload": stock_payload,
        "football_model_info": {
            "model_name": football_model.name,
            "model_label": FOOTBALL_MODELS[football_model.name]["label"],
            "params_used": football_model.params,
            "training_rows": getattr(football_model, "training_rows", 0),
            "feature_count": getattr(football_model, "feature_count", 0),
            "validation_summary": getattr(football_model, "validation_summary", {}),
            "prediction_timestamp": pd.Timestamp.utcnow().isoformat(),
        },
        "stock_model_info": {
            "model_name": stock_model.name,
            "model_label": STOCK_MODELS[stock_model.name]["label"],
            "params_used": stock_model.params,
            "training_rows": getattr(stock_model, "training_rows", 0),
            "feature_count": getattr(stock_model, "feature_count", 0),
            "validation_summary": getattr(stock_model, "validation_summary", {}),
            "prediction_timestamp": pd.Timestamp.utcnow().isoformat(),
        },
    }
    if config.get("persist", True):
        save_last_config(config)
    return state


def _get_state(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = config or STATE.get("config")
    if merged is None:
        merged = load_last_config() or _merge_config({})
    if "football" not in merged:
        merged = _merge_config(merged)
    signature = json.dumps(merged, sort_keys=True, ensure_ascii=False)
    with STATE_LOCK:
        if STATE.get("ready") and STATE.get("config_signature") == signature:
            return STATE
        state = _compute_dashboard(merged)
        state["config_signature"] = signature
        STATE.clear()
        STATE.update(state)
        return STATE


def _api_response(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "bracket": state["bracket"],
        "football_model_info": state["football_model_info"],
        "stock_model_info": state["stock_model_info"],
        "stock_history": state["stock_payload"]["history"],
        "stock_forecast": state["stock_payload"]["forecast"],
        "stock": state["stock_payload"],
        "current_config": state["config"],
        "predicted_champion": state["predicted_champion"],
        "n_simulations": state["bracket"]["n_simulations"],
    }


def teams_still_alive() -> set[str]:
    alive = set(fifa.get_teams().keys())
    for match in fifa.get_matches():
        if match.get("status") == "completed":
            loser = match["away"] if match.get("winner") == match.get("home") else match["home"]
            if loser:
                alive.discard(loser)
    return alive


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model_defaults")
def api_model_defaults():
    return jsonify(_model_defaults_payload())


@app.route("/api/last_config")
def api_last_config():
    loaded = load_last_config()
    if loaded is None:
        loaded = _current_defaults()
        persisted = False
    else:
        loaded = _merge_config(loaded)
        persisted = True
    return jsonify({"persisted": persisted, **loaded})


@app.route("/api/bracket")
def api_bracket():
    state = _get_state()
    return jsonify(state["bracket"])


@app.route("/api/team/<team_name>")
def api_team(team_name: str):
    state = _get_state()
    football_model = state["football_model"]
    features = football_model.features
    teams = fifa.get_teams()
    if team_name not in teams:
        return jsonify({"error": f"Unknown team '{team_name}'"}), 404

    t = teams[team_name]
    scores = features.get(team_name) if features else {}
    adv = state["simulation"]["advancement"].get(team_name, {})

    return jsonify({
        "team": team_name,
        "code": t["code"],
        "flag": t["flag"],
        "group": t["group"],
        "stats": {
            "matches_played": t["matches_played"],
            "record": f'{t["wins"]}W-{t["draws"]}D-{t["losses"]}L',
            "goals": f'{t["goals_for"]}:{t["goals_against"]}',
            "group_points": t["group_points"],
            "possession": t["possession"],
            "shots_per_match": t["shots_per_match"],
            "shots_on_target_per_match": t["shots_on_target_per_match"],
            "passing_accuracy": t["passing_accuracy"],
            "fouls_per_match": t["fouls_per_match"],
            "cards": f'{t["yellow_cards"]}Y {t["red_cards"]}R',
            "clean_sheets": t["clean_sheets"],
            "recent_results": t["recent_results"],
        },
        "scores": scores,
        "ranks": {k: features.rank_of(team_name, k) for k in ("overall", "attack", "defense", "form", "player_impact", "discipline")} if features else {},
        "advancement": {
            "reach_qf": adv.get("QF", 0),
            "reach_sf": adv.get("SF", 0),
            "reach_final": adv.get("F", 0),
            "win_title": adv.get("champion", 0),
        },
        "key_players": features.top_players(team_name) if features else [],
    })


@app.route("/api/stocks")
def api_stocks():
    return jsonify({"sponsors": sponsors.list_sponsors()})


@app.route("/api/stock/<ticker>")
def api_stock(ticker: str):
    sponsor = sponsors.get(ticker)
    if not sponsor:
        return jsonify({"error": f"Unknown sponsor ticker '{ticker}'"}), 404

    state = _get_state()
    config = state["config"]
    stock_config = config["stock"]
    stock_model, _, sponsor_meta, stock_df, source = _build_stock_model(stock_config, ticker, config.get("persist", True))
    sponsor_meta["exposure_score"] = sponsors.exposure_for(ticker, teams_still_alive())
    sponsor_meta["region_relevance"] = 1.0 if sponsor_meta.get("region") else 0.75
    payload = _build_stock_payload(ticker, stock_model, sponsor_meta, stock_df)
    return jsonify({
        "sponsor": payload["sponsor"],
        "source": source,
        "is_demo": source == "demo",
        "exposure_score": payload["exposure_score"],
        "history": payload["history"],
        "forecast": payload["forecast"],
        "model_info": payload["model_info"],
    })


@app.route("/api/recalculate", methods=["GET", "POST"])
def api_recalculate():
    raw = request.get_json(silent=True) if request.method == "POST" else None
    config = _merge_config(raw or load_last_config() or {})
    state = _get_state(config)
    return jsonify(_api_response(state))


if __name__ == "__main__":
    print("FIFA 2026 AI Knockout Predictor + Sponsor Stock Impact Monitor")
    print("Open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
