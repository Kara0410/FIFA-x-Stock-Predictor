# FIFA 2026 AI Knockout Predictor + Sponsor Stock Impact Monitor

A local Flask dashboard that:

1. renders the FIFA World Cup 2026 knockout bracket from the Round of 32 to the final;
2. predicts advance probabilities only for matches that have not been played;
3. runs 10,000 Monte Carlo simulations of the remaining bracket; and
4. produces an experimental seven-trading-day forecast for selected sponsor stocks.

The dashboard uses Flask, vanilla HTML/CSS/JavaScript, scikit-learn, pandas, and
Yahoo Finance data through `yfinance`.

> This project is an educational modelling demonstration. Its football
> probabilities are not betting advice, and its stock forecasts are not
> financial advice.

## Installation

Python 3.11 or newer is recommended.

```bash
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

The committed JSON files can be regenerated with:

```bash
python data/sample/generate_sample_data.py
```

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
fifa_2026_ai_knockout_stock_dashboard/
|-- app.py                         Flask application and API routes
|-- data/
|   |-- raw/                       Cached stock-price CSV files
|   |-- processed/                 Reserved for generated feature exports
|   `-- sample/                    Tournament JSON files and generator
|-- models/
|   |-- model_registry.py          Model catalogue, defaults, and UI metadata
|   |-- feature_engineering.py     Football and stock feature builders
|   |-- football_model.py          Original Poisson/logistic predictor
|   |-- football_alt_models.py     Baseline, Elo, HistGB, and ensemble wrappers
|   |-- bracket_simulator.py       Monte Carlo bracket simulation
|   |-- stock_model.py             Original stock predictor
|   |-- stock_alt_models.py        GBR, SARIMAX, ElasticNet, and ensemble models
|   `-- persistence.py             joblib model and JSON configuration storage
|-- services/
|   |-- fifa_data_service.py       Tournament data access
|   |-- stock_data_service.py      yfinance access, cache, and fallback
|   `-- sponsor_service.py         Sponsor catalogue and exposure scores
|-- static/css/style.css           Dashboard layout and theme
|-- static/js/dashboard.js         Bracket and chart rendering
|-- templates/index.html           Main dashboard template
`-- tests/                          Regression tests
```

## API routes

| Route | Purpose |
|---|---|
| `GET /` | Render the dashboard |
| `GET /api/bracket` | Return fixtures, match probabilities, and tournament simulation |
| `GET /api/team/<name>` | Return team features, players, ranks, and advancement probabilities |
| `GET /api/stocks` | Return the configured sponsor catalogue |
| `GET /api/stock/<ticker>` | Return price history and the seven-day forecast |
| `GET /api/model_defaults` | Return available models, defaults, descriptions, and control metadata |
| `GET /api/last_config` | Return the last persisted model selection and ticker |
| `GET /api/recalculate` | Recalculate with the last/default configuration |
| `POST /api/recalculate` | Train or reload selected models and return a combined bracket and stock forecast |

## Model selection: which combination should you use?

The football and stock selectors are independent. A football model estimates
unplayed knockout matches and drives the Monte Carlo bracket. A stock model
forecasts seven trading days from price history plus date-aligned tournament
signals. Selecting a football model does **not** mechanically force a stock
price prediction up or down.

Use these combinations as practical starting points:

| Goal | Football model | Stock model | Why |
|---|---|---|---|
| Recommended general dashboard | `football_ensemble` | `stock_ensemble` | Blends the different modelling assumptions and reduces dependence on one estimator. It is the broadest comparison, but also the slowest. |
| Conservative choice for the current small dataset | `baseline_poisson_blend` | `elasticnet_factor` | The football baseline encodes scoring structure directly; regularized ElasticNet is less flexible than the tree and time-series models when stock history is short. |
| Fastest recalculation | `baseline_poisson_blend` | `baseline_gbr` | Both models train quickly and preserve the original dashboard behavior. |
| Most interpretable | `elo_bt` | `elasticnet_factor` | Elo exposes team ratings and rating differences; ElasticNet is a regularized linear factor model. |
| Event-timing experiment | `elo_bt` or `football_ensemble` | `sarimax_exog` | SARIMAX explicitly models returns as a time series with football-event regressors. Use only when enough clean price history is available. |
| Nonlinear experiment | `histgb_classifier` | `baseline_gbr` | Both can learn nonlinear feature interactions, but both can overfit a small local sample. Compare holdout metrics before trusting them. |

The recommended combinations are engineering defaults, not evidence that one
pair is universally more accurate. Compare the validation fields returned by
the API and prefer simpler models when the sample is small or unstable.

## Football models

All football models expose the same prediction contract:

- 90-minute probabilities for team A win, draw, and team B win;
- knockout advancement probabilities after draw/penalty handling;
- a confidence value and model metadata;
- the number of training rows, feature count, parameters, and validation
  summary.

Completed matches are never predicted again. Their recorded winner is locked
into the bracket. Models run only for matches that are still unplayed.

### `baseline_poisson_blend`

The original interpretable model combines:

- a Poisson score grid derived from expected goals;
- a logistic curve based on the difference in overall team strength;
- a penalty tilt that slightly favors the stronger side after a 90-minute
  draw.

This is the safest compatibility choice and works well with very small data
because its football structure is encoded explicitly rather than learned
entirely from match rows.

Default parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `poisson_weight` | 0.55 | Contribution of the expected-goals score model |
| `logistic_weight` | 0.45 | Contribution of the overall-strength model |
| `penalty_tilt` | 0.55 | Strength influence when resolving a knockout draw |
| `mc_simulations` | 10,000 | Number of remaining-bracket simulations |

### `elo_bt`

The Elo/Bradley-Terry model initializes every team around 1500 using the
engineered overall-strength score, then updates ratings chronologically from
completed matches. Rating updates include margin-of-victory adjustment,
time/sequence decay, and shrinkage toward the 1500 baseline. A draw prior is
increased when ratings are close.

Use it when you want transparent dynamic ratings. Its weakness is that ratings
need enough completed match history to stabilize.

Default parameters:

| Parameter | Default |
|---|---:|
| `k_factor` | 24 |
| `decay` | 0.02 |
| `mov_boost` | 0.15 |
| `draw_prior` | 0.26 |
| `penalty_tilt` | 0.55 |
| `regularization_c` | 1.0 |

### `histgb_classifier`

This model trains a multiclass
`HistGradientBoostingClassifier(loss="log_loss")` for home/team-A win, draw,
or away/team-B win. It uses ten pairwise feature differences and attempts
probability calibration with `CalibratedClassifierCV`. Sigmoid calibration is
the default; calibration is skipped if the sample does not contain enough rows
and outcome classes.

Use it to explore nonlinear interactions. Treat its output cautiously because
the local tournament dataset is small relative to the flexibility of a
boosted-tree classifier.

Default parameters:

| Parameter | Default |
|---|---:|
| `learning_rate` | 0.07 |
| `max_iter` | 250 |
| `max_leaf_nodes` | 31 |
| `max_depth` | `null` |
| `min_samples_leaf` | 10 |
| `l2_regularization` | 0.10 |
| `calibration_method` | `sigmoid` |

### `football_ensemble`

The football ensemble trains all three football models and calculates a
weighted average of their win/draw/loss, advancement, and confidence values.
Weights are normalized automatically. `stacking-ready` is currently a hook for
future learned stacking; the implemented prediction is still a manual weighted
average.

Default weights:

| Member | Weight |
|---|---:|
| Baseline Poisson blend | 0.40 |
| Elo/Bradley-Terry | 0.35 |
| HistGB classifier | 0.25 |

This is the recommended general-purpose view because it exposes less model
specific variance. It cannot correct weak source data: all members are trained
from the same local dataset.

## Stock models

Every stock model predicts the next-day return recursively for seven business
days and returns price history, forecast dates, predicted closes, lower/upper
bands, model information, and validation statistics. If too few usable rows
remain after feature construction, individual models fall back to historical
drift rather than failing.

### `baseline_gbr`

The original `GradientBoostingRegressor` uses 22 market and football-event
features. Shallow trees learn nonlinear interactions without feature scaling.
An 80/20 chronological split provides MAE, RMSE, and directional accuracy when
enough rows are available.

| Parameter | Default |
|---|---:|
| `n_estimators` | 120 |
| `max_depth` | 2 |
| `learning_rate` | 0.05 |
| `subsample` | 0.90 |

Use it for fast local experimentation. It does not explicitly represent
time-series error structure.

### `sarimax_exog`

SARIMAX models daily returns with autoregressive/moving-average terms and
exogenous football-event regressors. It produces model-derived 80% confidence
intervals. Constant exogenous columns are removed when they conflict with an
explicit trend.

| Parameter | Default |
|---|---|
| `order` | `[1, 0, 1]` |
| `seasonal_order` | `[0, 0, 0, 0]` |
| `trend` | `c` |
| `exog_lag_days` | 1 |
| `rolling_window` | 5 |
| `enforce_stationarity` | `true` |
| `enforce_invertibility` | `true` |

Use it when event timing and uncertainty intervals are the focus. Very short
or nearly constant histories can make SARIMAX estimates fragile.

`exog_lag_days` and `rolling_window` are exposed for configuration
compatibility, but the current feature builder uses fixed lag-1/lag-2 event
channels and fixed 5/10-day volatility windows. Changing those two controls
does not yet rebuild the feature windows.

### `elasticnet_factor`

ElasticNet predicts next-day return from standardized lagged returns, market
factors, and football-event factors. L1 regularization can remove weak factors;
L2 regularization stabilizes correlated factors. The model recursively feeds
predicted returns into the next forecast step.

| Parameter | Default |
|---|---:|
| `alpha` | 0.01 |
| `l1_ratio` | 0.30 |
| `max_iter` | 3000 |
| `tol` | 0.0001 |
| `lookback_lags` | 10 |
| `feature_scaling` | `true` |

This is the preferred conservative stock model for the current small sample.
Its linear form is easier to regularize, but it can miss abrupt nonlinear
market moves.

### `stock_ensemble`

The stock ensemble trains all three stock models and averages their predicted
prices and confidence bounds. Weights are normalized automatically.
`stacking-ready` is metadata for a future learned combiner; current behavior is
manual averaging.

| Member | Weight |
|---|---:|
| Baseline GBR | 0.35 |
| SARIMAX | 0.40 |
| ElasticNet | 0.25 |

Use it for the broadest experimental view. It is not automatically more
accurate, because all members share the same short history and event data.

## Why the football prediction models are designed this way

World Cup knockout data is small: each team has only a few tournament matches.
A large black-box model would overfit that sample and produce probabilities that
look precise without being reliable. The project therefore offers several
different statistical views:

- a Poisson expected-goals model, which represents football scoring explicitly;
- a logistic strength model, which stabilizes noisy recent scorelines;
- dynamic Elo ratings for interpretable form updates;
- a calibrated boosted-tree classifier for nonlinear relationships; and
- an ensemble that averages their different assumptions.

Monte Carlo simulation then handles uncertainty from future opponents and
later bracket rounds.

## FIFA data used by the model

The application reads three files through `FifaDataService`:

- `data/sample/matches.json`: dates, rounds, teams, scores, penalties, winners,
  bracket dependencies, and match status;
- `data/sample/teams.json`: matches played, wins/draws/losses, goals, group
  points, possession, shots, passing, cards, clean sheets, tackles,
  interceptions, saves, and recent results;
- `data/sample/players.json`: position, appearances, minutes, goals, assists,
  cards, defensive actions, saves, and rating for selected players.

Fixtures marked `completed` use the winner recorded in the local match file.
The current team and player performance fields are deterministic estimates
generated by `data/sample/generate_sample_data.py`; they should be replaced by
a verified, licensed statistics feed before treating the probabilities as
production output. The application trusts the status and scores in the local
dataset, so those fields must be kept accurate.

Match status is part of the model contract:

- `completed`: preserve the recorded winner and never generate a prediction;
- `upcoming`: both participants are known, so expose a direct match prediction;
- `scheduled`: participants depend on earlier matches, so expose simulated slot
  candidates instead of pretending the teams are known.

This prevents already-played games from being overwritten by the model.

## Football feature engineering

`TeamFeatureEngineer` min-max normalizes every input across the 32 knockout
teams. A score of 100 means best in this tournament dataset, not best in
football generally.

The five component scores are:

- **Attack**: goals per match 35%, shots 20%, shots on target 25%,
  possession 10%, and passing accuracy 10%.
- **Defense**: clean-sheet rate 25%, inverse goals conceded 35%,
  tackles/interceptions 15%, goalkeeper/defender contribution 15%, and
  discipline 10%.
- **Discipline**: inverse of fouls plus three times yellow cards and twelve
  times red cards per match.
- **Form**: recency-weighted results 45%, group points 30%, and goal difference
  per match 25%. Newer results receive exponentially greater weight.
- **Player impact**: goals, assists, per-90 output, minutes/availability, and
  defensive contribution, minus card penalties.

Overall strength combines those components using:

| Component | Weight |
|---|---:|
| Attack | 30% |
| Defense | 28% |
| Player impact | 20% |
| Form | 15% |
| Discipline | 7% |

The HistGB model additionally uses pairwise differences:

- attack, defense, player impact, discipline, form, and rank;
- rest days and cumulative player minutes;
- top-three-player share; and
- recent goal-difference rate.

The model-ready rolling team builder processes completed matches in date order
and records each snapshot before updating the state with the current result.
That shift prevents the current match outcome from leaking into its own
features.

## Match probabilities and bracket simulation

For teams A and B, the model calculates expected goals:

```text
xG_A = 1.30 * (attack_A + 45) / (defense_B + 45)
xG_B = 1.30 * (attack_B + 45) / (defense_A + 45)
```

Each xG value is constrained to 0.25-3.60. A Poisson grid covering scores from
0 through 9 goals produces 90-minute win, draw, and loss probabilities.

A second probability comes from the overall-strength difference:

```text
P(A beats B) = 1 / (1 + exp(-0.060 * (overall_A - overall_B)))
```

The final two-way estimate is 55% Poisson and 45% logistic. The Poisson draw
probability is retained. For a knockout draw, the advancement probability is
mostly a 50/50 outcome with a 30% tilt toward the stronger team.

`BracketSimulator` runs the remaining tournament 10,000 times with a fixed
seed. Every completed result is locked. Each unplayed match samples a winner
from the model probability, and that winner advances through the correct
bracket dependency. The output includes:

- probability of reaching each later round;
- probability of winning the tournament;
- likely participants in matches whose teams are still unknown;
- a single most-likely projected path for highlighting the diagram.

## Why football data is included in the stock forecast

The stock model tests a narrow hypothesis: tournament activity can act as a
short-lived attention and exposure signal for public companies connected to
the event. For example, a sponsor may receive more visibility on match days or
while a related national team remains active.

Football data does not replace market data. It is added as a small contextual
feature set beside returns, momentum, and volatility. The model demonstrates
how an event stream can be joined to a financial time series by calendar date;
it does not establish that World Cup events cause stock-price movements.

## Stock data and features

`StockDataService` requests adjusted daily closing prices from Yahoo Finance
from 2026-06-11 through the current date. Downloads are cached in `data/raw/`
for four hours. If the download fails or returns fewer than five rows, the
service creates a deterministic demo series and labels it `demo` in the API.

The shared feature builder creates these market channels:

- daily percentage return and one-, two-, and three-day return lags;
- three- and five-day moving-average gaps;
- five- and ten-day rolling volatility;
- three- and five-day price momentum; and
- cumulative price trend since the start of the data window.

The football event service assigns stage weights of group 1, R32 2, R16 3,
quarter-final 4, semi-final 5, and final 6, then aligns these channels to
business days:

- `exposure_score`: tournament visibility proxy based on stage and match count;
- `match_day_intensity`: number of matches multiplied by stage weight;
- `upset_score`: configured upset magnitude for that date;
- `alive_relevance`: decreasing relevance as completed knockout matches
  accumulate;
- `event_decay_3` and `event_decay_5`: recent stage-intensity summaries;
- `same_day_event`: binary match-day indicator;
- `lag1_event` and `lag2_event`: previous event intensity;
- `region_relevance`: sponsor-region context; and
- `cum_stage_intensity`: cumulative tournament stage intensity.

The sponsor catalogue supplies each company's ticker, region, tournament
category, base exposure, currency, and related teams. The live exposure helper
adds 0.05 for each related team still alive, capped at 1.0. These values are
hand-built proxies, not measured advertising impressions.

## Stock validation and forecast behavior

The shared prediction target is the next trading day's return. Models use a
chronological 80/20 train/holdout split when enough rows are available and
report:

- mean absolute error (`mae`);
- root mean squared error (`rmse`);
- directional accuracy; and
- the number of holdout/backtest rows.

Football model metadata reserves fields for log loss, Brier score, and holdout
accuracy. Those values are `null` unless a model has enough suitable local
data and an implemented validation calculation. Do not interpret a missing
metric as a zero error.

All stock models forecast recursively:

1. predict the next-day return;
2. convert it to the next predicted close;
3. reuse the forecast return as a lag for later steps; and
4. repeat for the next business day.

If the usable sample is too small, the model uses mean historical drift
instead of fitting an unreliable estimator.

The GBR and ElasticNet uncertainty bands are:

```text
predicted price +/- 1.28 * residual_std * sqrt(horizon) * last_close
```

This is an approximate 80% residual cone, not a calibrated market-risk
interval. SARIMAX instead uses its own forecast confidence interval. The stock
ensemble averages the three members' lower and upper paths.

## Training, caching, and persistence

Pressing **Predict** sends the selected model names, parameters, ticker, and
`persist` flag to `POST /api/recalculate`. The backend:

1. loads the current team, player, match, stock, and football-event frames;
2. creates a signature from the model configuration and current data;
3. reloads a compatible joblib model from `data/processed/models/`, or trains
   and saves a new model;
4. recalculates unplayed football probabilities and the Monte Carlo bracket;
5. creates the seven-day stock forecast; and
6. saves the last selection to
   `data/processed/last_model_config.json` when persistence is enabled.

This cache is local. Changing model parameters, ticker, or source data changes
the signature and causes retraining.

## End-to-end data flow

```text
FIFA match/team/player JSON
        |
        +--> team feature engineering
        |        |
        |        +--> selected football model
        |                    |
        |                    +--> 10,000-run bracket simulation
        |
        +--> date-level football signals --------+
                                                  |
Yahoo Finance adjusted closes --> market features +--> selected stock model
                                                       |
                                                       +--> 7-day forecast
```

The football and stock models are related through date-level tournament
signals only. A team's predicted match result does not directly force a stock
price up or down.

## Limitations

- Completed knockout results are real, but current team/player feature values
  are generated estimates rather than a live licensed statistics feed.
- Football features are normalized relative to the current 32-team dataset,
  so values change when the comparison population changes.
- The stock training window contains only a few weeks of observations.
- Sponsor exposure and upset magnitude are manually configured proxies.
- No broad market index, sector return, currency factor, earnings event, or
  intraday information is included.
- Yahoo Finance access is unofficial and may be unavailable or rate-limited.
- Injuries, starting lineups, red-card timing, travel, and betting-market odds
  are not modelled.
- Correlation in this demonstration must not be interpreted as causation.

## Updating FIFA results

The schema contract is defined in `data/sample/generate_sample_data.py`.

When a match finishes:

1. set its `status` to `"completed"`;
2. fill `home_score`, `away_score`, `penalties`, and `winner`;
3. resolve the teams in the next bracket match;
4. update team/player statistics if a real statistics source is available;
5. call `GET /api/recalculate` or restart the application.

The simulator will lock the completed result and predict only the remaining
fixtures.

For live data, implement `fetch_live_teams`, `fetch_live_players`, and
`fetch_live_matches` in `services/fifa_data_service.py`, then set
`USE_LIVE_DATA = True`.
