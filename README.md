# FIFA 2026 AI Knockout Predictor + Sponsor Stock Impact Monitor

A fully local Python dashboard that

1. draws the **FIFA World Cup 2026 knockout bracket** (Round of 32 → Final),
2. predicts **win/advance probabilities** for every not-yet-played knockout match using
   only tournament data (results, goals, possession, shots, passing, discipline, player stats),
3. runs a **10,000-run Monte Carlo simulation** of the remaining bracket, and
4. monitors **FIFA sponsor stocks** (Adidas, Coca-Cola, Visa, …) since kickoff
   (2026-06-11) with an **experimental 7-day forecast** that mixes market features
   with World-Cup signals (stage intensity, match days, upsets, sponsor exposure).

No Docker, no Streamlit, no cloud — Flask + vanilla HTML/CSS/JS on localhost.

---

## Installation

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**.

The sample data files are pre-generated and committed; regenerate them any time with:

```bash
python data/sample/generate_sample_data.py
```

## Project structure

```
fifa_2026_ai_knockout_stock_dashboard/
├── app.py                      # Flask app + all API routes
├── requirements.txt
├── data/
│   ├── raw/                    # cached yfinance downloads (CSV, auto-created)
│   ├── processed/              # reserved for future feature exports
│   └── sample/                 # sample tournament data + its generator
├── models/
│   ├── feature_engineering.py  # attack/defense/form/discipline/player-impact scores
│   ├── football_model.py       # Poisson xG + logistic blend, penalty model
│   ├── bracket_simulator.py    # 10,000-run Monte Carlo of the remaining bracket
│   └── stock_model.py          # GradientBoosting stock forecaster + football signals
├── services/
│   ├── fifa_data_service.py    # data access (sample JSON now, real API later)
│   ├── stock_data_service.py   # yfinance + CSV cache + demo fallback
│   └── sponsor_service.py      # sponsor catalogue & exposure scores
├── static/css/style.css        # dark glassmorphism theme
├── static/js/dashboard.js      # bracket renderer, modals, Plotly stock chart
└── templates/index.html
```

## API routes

| Route                    | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `GET /`                  | dashboard page                                      |
| `GET /api/bracket`       | bracket, match predictions, Monte Carlo summary     |
| `GET /api/team/<name>`   | team profile: scores, ranks, players, advancement   |
| `GET /api/stocks`        | sponsor list                                        |
| `GET /api/stock/<ticker>`| price history + 7-day forecast + confidence band    |
| `GET /api/recalculate`   | drop caches, rebuild features, re-simulate          |

## How the football model works

1. **Feature engineering** (`models/feature_engineering.py`) — five 0–100 scores per
   team, min-max normalized across the 32 knockout teams:
   * *Attack* = goals/match ·0.35 + shots ·0.20 + shots on target ·0.25 + possession ·0.10 + passing ·0.10
   * *Defense* = clean sheets ·0.25 + inverse conceded ·0.35 + tackles/interceptions ·0.15 + GK/DF contribution ·0.15 + discipline ·0.10
   * *Discipline* = inverse of fouls + 3·yellows + 12·reds (per match)
   * *Form* = recency-weighted W/D/L + group points + goal difference
   * *Player impact* = key-player goals/assists/per-90/minutes, minus card penalty
2. **Match probability** (`models/football_model.py`) —
   `xG_A = 1.30 · attack_A / defense_B` feeds a full Poisson score grid
   (win/draw/loss over 90'), blended 55/45 with a logistic curve over the overall
   strength difference. Knockout draws are resolved by a penalty-shootout model that
   is 70% coin flip, 30% strength-tilted.
3. **Monte Carlo** (`models/bracket_simulator.py`) — simulates all remaining matches
   10,000× and reports each team's probability of reaching the QF/SF/Final and
   winning the title, plus most-likely participants for TBD matches.

## How the stock model works (experimental!)

* History via **yfinance** from 2026-06-11 to today, cached in `data/raw/` for 4h.
* If the download fails (offline/rate-limited) a deterministic synthetic series is
  generated and the UI shows a **DEMO DATA** badge instead of the blue REAL badge.
* Features: daily returns, 3/5-day MAs, rolling volatility, momentum, trend, plus
  football signals per date — stage intensity (group=1 … final=6), match-day flag,
  knockout games played, upset score, fan-attention proxy, sponsor exposure
  (boosted while brand-related teams are alive).
* A `GradientBoostingRegressor` predicts next-day returns, rolled forward 7 trading
  days; the band is ±1.28σ of training residuals (~80%).
* With ~3 weeks of prices the model auto-falls back to drift+volatility when there
  are fewer than 8 training rows.

**This is a modelling demo, not financial advice.**

## Limitations

* Tournament data is **synthetic sample data** (deterministic, seeded) shaped like
  the real thing; the bracket state pretends today is 2026-07-04 (Round of 16).
* Tiny samples everywhere: 3–4 matches per team, ~15 trading days per stock.
  Confidence values are deliberately modest.
* yfinance is unofficial and can rate-limit; the app degrades to demo data.
* One-off penalties/red cards/injuries are not modelled.

## Replacing sample FIFA data with real data

The schema contract lives in `data/sample/generate_sample_data.py`.

* **Easiest:** overwrite `data/sample/teams.json`, `players.json`, `matches.json`
  with real exports in the same shape, then hit **Recalculate** in the UI
  (or `GET /api/recalculate`).
* **Cleaner:** implement `fetch_live_teams/players/matches()` in
  `services/fifa_data_service.py` against your football API of choice
  (e.g. api-football, SportMonks, a FIFA scraper) and set `USE_LIVE_DATA = True`.
  Everything downstream — features, predictions, simulation, UI — is unchanged.

As real knockout results come in, set each match's `status` to `"completed"`,
fill `home_score`/`away_score`/`penalties`/`winner`, and the simulator
automatically locks those results and re-simulates only the remaining games.
