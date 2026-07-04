"""
FIFA 2026 AI Knockout Predictor + Sponsor Stock Impact Monitor
--------------------------------------------------------------
Run locally:

    pip install -r requirements.txt
    python app.py

then open http://localhost:5000

Routes:
    GET /                        dashboard page
    GET /api/bracket             bracket + probabilities + Monte Carlo summary
    GET /api/team/<team_name>    team explanation (scores, players, advancement)
    GET /api/stocks              sponsor list
    GET /api/stock/<ticker>      price history + 7-day prediction
    GET /api/recalculate         drop caches and rerun the simulation
"""
from __future__ import annotations

import threading

from flask import Flask, jsonify, render_template

from models.bracket_simulator import BracketSimulator
from models.feature_engineering import TeamFeatureEngineer
from models.football_model import MatchPredictor
from models.stock_model import FootballSignalBuilder, StockPredictor
from services.fifa_data_service import FifaDataService
from services.sponsor_service import SponsorService
from services.stock_data_service import StockDataService

app = Flask(__name__)

fifa = FifaDataService()
sponsors = SponsorService()
stocks = StockDataService()

N_SIMULATIONS = 10_000

# ---------------------------------------------------------------------------
# Model state - built lazily, guarded by a lock, invalidated by /api/recalculate
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: dict = {}


def get_state() -> dict:
    """Build (or return cached) features, predictor and simulation results."""
    with _state_lock:
        if _state.get("ready"):
            return _state

        teams = fifa.get_teams()
        players = fifa.get_players()
        matches = fifa.get_matches()

        features = TeamFeatureEngineer(teams, players)
        predictor = MatchPredictor(features)
        simulator = BracketSimulator(matches, predictor)
        simulation = simulator.simulate(N_SIMULATIONS)
        champ, path = simulator.predicted_path()

        _state.update({
            "ready": True,
            "features": features,
            "predictor": predictor,
            "simulation": simulation,
            "predicted_champion": champ,
            "predicted_path": path,
            "signal_builder": FootballSignalBuilder(fifa.get_matches_doc()),
        })
        return _state


def invalidate_state():
    with _state_lock:
        _state.clear()
    fifa.invalidate()


def teams_still_alive() -> set[str]:
    """Teams that have not lost a completed knockout match."""
    alive = set(fifa.get_teams().keys())
    for m in fifa.get_matches():
        if m["status"] == "completed":
            loser = m["away"] if m["winner"] == m["home"] else m["home"]
            alive.discard(loser)
    return alive


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Bracket API
# ---------------------------------------------------------------------------
@app.route("/api/bracket")
def api_bracket():
    state = get_state()
    features: TeamFeatureEngineer = state["features"]
    predictor: MatchPredictor = state["predictor"]
    sim = state["simulation"]
    teams = fifa.get_teams()

    def team_blob(name: str | None) -> dict | None:
        if not name:
            return None
        t = teams[name]
        return {"name": name, "code": t["code"], "flag": t["flag"]}

    out_matches = []
    for m in fifa.get_matches():
        entry = {
            "id": m["id"], "round": m["round"], "date": m["date"], "venue": m["venue"],
            "status": m["status"],
            "home": team_blob(m["home"]), "away": team_blob(m["away"]),
            "home_from": m["home_from"], "away_from": m["away_from"],
            "home_score": m["home_score"], "away_score": m["away_score"],
            "penalties": m["penalties"], "winner": m["winner"],
            "on_predicted_path": m["id"] in state["predicted_path"],
            "prediction": None, "slot_candidates": None,
        }
        if m["status"] == "upcoming" and m["home"] and m["away"]:
            p = predictor.predict(m["home"], m["away"])
            entry["prediction"] = {
                "home_advance": p["advance_a"], "away_advance": p["advance_b"],
                "p_draw_90": p["p_draw_90"], "xg_home": p["xg_a"], "xg_away": p["xg_b"],
                "confidence": p["confidence"], "drivers": p["drivers"],
            }
        elif m["status"] == "scheduled":
            # Participants unknown: show the most likely candidates per slot.
            outlook = sim["match_outlook"].get(m["id"], [])
            entry["slot_candidates"] = [
                {**c, "flag": teams[c["team"]]["flag"], "code": teams[c["team"]]["code"]}
                for c in outlook[:6]
            ]
        out_matches.append(entry)

    ranking = [
        {**r, "flag": teams[r["team"]]["flag"], "code": teams[r["team"]]["code"]}
        for r in sim["title_ranking"][:10]
    ]

    return jsonify({
        "tournament": "FIFA World Cup 2026",
        "as_of": fifa.get_matches_doc().get("note", ""),
        "n_simulations": sim["n_simulations"],
        "matches": out_matches,
        "title_ranking": ranking,
        "predicted_champion": state["predicted_champion"],
        "model": {
            "components": TeamFeatureEngineer.OVERALL_WEIGHTS,
            "base_goal_rate": MatchPredictor.BASE_GOAL_RATE,
            "poisson_weight": MatchPredictor.POISSON_WEIGHT,
            "penalty_tilt": MatchPredictor.PENALTY_TILT,
        },
    })


@app.route("/api/team/<team_name>")
def api_team(team_name: str):
    state = get_state()
    features: TeamFeatureEngineer = state["features"]
    teams = fifa.get_teams()
    if team_name not in teams:
        return jsonify({"error": f"Unknown team '{team_name}'"}), 404

    t = teams[team_name]
    scores = features.get(team_name)
    adv = state["simulation"]["advancement"].get(team_name, {})

    return jsonify({
        "team": team_name, "code": t["code"], "flag": t["flag"], "group": t["group"],
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
        "ranks": {k: features.rank_of(team_name, k)
                  for k in ("overall", "attack", "defense", "form", "player_impact", "discipline")},
        "advancement": {
            "reach_qf": adv.get("QF", 0), "reach_sf": adv.get("SF", 0),
            "reach_final": adv.get("F", 0), "win_title": adv.get("champion", 0),
        },
        "key_players": features.top_players(team_name),
    })


@app.route("/api/recalculate")
def api_recalculate():
    invalidate_state()
    state = get_state()   # rebuild immediately
    return jsonify({
        "status": "ok",
        "n_simulations": state["simulation"]["n_simulations"],
        "predicted_champion": state["predicted_champion"],
    })


# ---------------------------------------------------------------------------
# Stocks API
# ---------------------------------------------------------------------------
@app.route("/api/stocks")
def api_stocks():
    return jsonify({"sponsors": sponsors.list_sponsors()})


@app.route("/api/stock/<ticker>")
def api_stock(ticker: str):
    sponsor = sponsors.get(ticker)
    if not sponsor:
        return jsonify({"error": f"Unknown sponsor ticker '{ticker}'"}), 404

    history, source = stocks.get_history(ticker)
    state = get_state()
    exposure = sponsors.exposure_for(ticker, teams_still_alive())

    predictor = StockPredictor(state["signal_builder"])
    forecast = predictor.forecast(history, exposure)

    return jsonify({
        "sponsor": sponsor,
        "source": source,                       # 'yfinance' (real) or 'demo'
        "is_demo": source == "demo",
        "exposure_score": round(exposure, 3),
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in history.index],
            "close": [round(float(c), 2) for c in history["close"]],
        },
        "forecast": forecast,
    })


if __name__ == "__main__":
    print("FIFA 2026 AI Knockout Predictor + Sponsor Stock Impact Monitor")
    print("Open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
