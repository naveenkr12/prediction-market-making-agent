# Prediction Market Making Agent

**Capstone project — MS in Quantitative Computational Finance, Georgia Institute of Technology**
Author: Naveen Singh

## Overview

This project builds a reinforcement-learning market-making agent for prediction
markets (Kalshi / Polymarket-style binary contracts) and benchmarks it against
the **published theoretical optimum**: Feil & Nendel's (2026) Hamilton-Jacobi-Bellman
(HJB) optimal quoting model for prediction markets. See `docs/Prediction_Market_Making_Proposal.pdf`
for the full project proposal.

The core question: how close can a model-free RL agent get to the
model-based, provably-optimal quoting strategy — and where, if anywhere, does
it do better (e.g. under real market microstructure the theoretical model
doesn't capture)?

## Repository structure

```
.
├── docs/
│   └── Prediction_Market_Making_Proposal.pdf   — full project proposal
├── data_pipeline/                               — Kalshi & Polymarket data collection
│   └── (real trade/candlestick data for 60 settled binary contracts:
│        30 Economics + 30 Sports markets, ~90-day lookback)
├── calibration/                                 — Paper 2 five-minute signed flow,
│                                                    Kyle lambda and uncertainty
├── simulator/                                   — calibrated market driver and
│                                                    complete maker environment
├── hjb_baseline/                                 — reproduction of the theoretical
│   │                                                benchmark (Feil & Nendel's HJB
│   │                                                optimal market maker)
│   └── README.md                                — implementation notes, fixes,
│                                                    and fidelity vs. the published
│                                                    Table 2 / Figures 1-5
├── rl_agent/                                     — RL quoting agents
│   ├── README.md                                — v2: first PPO agent in the
│   │                                                calibrated simulator
│   └── v3/                                      — v3: flow-aware, live-compatible
│       └── README.md                              agent (300 s and 60 s versions),
│                                                    results and limitations
└── dashboard/                                    — live paper-quoting dashboard on
                                                     Kalshi data (no orders sent)
```

## Status

| Component | Status |
|---|---|
| Data pipeline (Kalshi + Polymarket) | Initial sample plus Paper 2 Step 1: 60 eligible Kalshi contracts across 60 event clusters / 25 series, with 68,866 trades and supplementary candles |
| HJB baseline reproduction (published optimum) | Done — optimal-strategy Table 2 numbers match the paper closely (see `hjb_baseline/README.md`) |
| Empirical Kyle lambda (Paper 2) | Done — 14,318 five-minute intervals; all 60 primary contract fits, uncertainty and sensitivity checks |
| Simulator price-impact parameter | Done — train-only broad-based/sports/pooled beta, uncertainty scenarios and reference replay checks |
| Simulator market activity, sizes and background noise | Done — training event/phase counts, paired empirical marks and quote-based flow-adjusted noise; trailing quote coverage documented |
| Full quoting/fill/cash/inventory environment | Done — post-only quotes, queue/participation fill scenarios, collateral, fractional fees, binary settlement and ledger validation; historical L2 fill calibration pending |
| HJB reference in calibrated environment | Done — frozen policy adapter, explicit normalized time/partial-inventory handling; 25,200 paired synthetic episodes with PnL, VaR/ES, inventory and drawdown checks |
| RL quoting agent v2 (`rl_agent/`) | Done — first PPO agent; it learned to quote almost never (about 0.1 fills per episode), so its result is close to "no quote". Kept for the record |
| RL quoting agent v3 (`rl_agent/v3/`) | Done — control-variate reward, live-computable features, per-side sizing, domain randomization; 300 s and 60 s decision versions (`decision_seconds` in the simulator) |
| RL vs. HJB benchmark comparison | Done in the simulator, on held-out seeds with paired evaluation. With 60 s decisions, v3 is the best or tied in every cell and the only policy with positive expected PnL under randomized conditions (+0.031 [+0.006, +0.053] per episode; +0.150 vs HJB). The edge is small: realized-PnL intervals still include zero at 3,000 episodes. Details and limitations in `rl_agent/v3/README.md` |
| Live paper-quoting dashboard (`dashboard/`) | Done — runs HJB, fixed, skew, v2 PPO and v3 on live Kalshi data with simulated fills; paper only. v3 has been run live only briefly, so it is not live-validated |

Paper 2's data-readiness sample is stored separately in
`data_pipeline/output/paper2/`: 30 economics and 30 sports contracts, all with
at least 50 usable five-minute intervals, event identifiers, resolution rules,
and full-lifetime trade pagination. Information classification yields 29
broad-based, 30 sports, and one company-specific contract; this is insufficient
for reproducing the paper's main single-name/broad-based comparison. The new
collector and per-contract eligibility report are documented in
`data_pipeline/README.md`. The `calibration/` module now builds five-minute
signed-flow data and estimates per-market Kyle lambda in log-odds and cents,
with 95% intervals, event splits, and sensitivity diagnostics. Results and
reproduction commands are in `calibration/README.md`. Estimates are often
sensitive to influential buckets. Training-only price-impact calibration now
provides low/base/high scenarios and a loadable reference-price update in
`simulator/`; see `simulator/README.md` for the endpoint adjustment, parameter
values, support limits, and validation. Market activity, trade sizes and
flow-adjusted quote-noise calibration are also complete; the runnable driver
and coverage limits are in `simulator/MARKET_DYNAMICS.md`. The complete research
trading environment now connects these drivers to quotes, fills, cash,
collateral, fees and settlement; see `simulator/TRADING_ENVIRONMENT.md`.
Execution assumptions need richer orderbook data before empirical policy
claims. The frozen HJB policy now runs in that environment with paired
profit/risk comparisons; see `simulator/HJB_BENCHMARK.md`. The v3 RL agent is
trained and evaluated in that environment (`rl_agent/v3/README.md`); its gains
are simulated, not live results.

Remaining work: calibrate fills on historical order-book data (the simulator's
queue and participation settings are assumptions), recalibrate volume templates
on current-season markets (live NFL/NBA volumes are far above the summer
templates), use per-series maker fees, try event-driven re-quoting, train on
recorded live books, and validate v3 on longer live paper runs.

## Reference papers

The following papers inform this project's modeling choices. They are
**not redistributed in this repository** (copyrighted, published works) —
see `docs/Prediction_Market_Making_Proposal.pdf` for full citations:

1. Feil & Nendel (2026), *Optimal Market Making in Prediction Markets* — the
   theoretical benchmark reproduced in `hjb_baseline/`.
2. Bartlett & O'Hara — Kyle's λ, informed price impact and differences across
   market types.
3. Volatility in Prediction Markets.
4. Parlay markets (prediction-market product structure).

## Quick start

- `data_pipeline/README.md` — how the Kalshi/Polymarket data was collected,
  and how to re-run it.
- `calibration/README.md` — how to construct five-minute signed flow, estimate
  Kyle lambda, and interpret the sample's uncertainty and sensitivity.
- `simulator/README.md` — training-only price-impact parameters, uncertainty
  scenarios, replay validation, and the reference-price update API.
- `simulator/MARKET_DYNAMICS.md` — calibrated trade frequency, paired size/side
  sampling, background innovations, exposure/quote support and driver usage.
- `simulator/TRADING_ENVIRONMENT.md` — complete quoting/fill/accounting API,
  train-only spreads, fees, settlement, assumptions and reproducible episodes.
- `simulator/HJB_BENCHMARK.md` — frozen HJB strategy in the calibrated
  environment, paired PnL/VaR/ES comparisons and uncertainty limits.
- `rl_agent/README.md` — v2 PPO agent in the calibrated simulator, including a
  documented reward-shaping bug and its fix.
- `rl_agent/v3/README.md` — v3 agent: what changed and why, held-out results at
  300 s and 60 s, what the policy learned, limitations, and how to reproduce.
- `dashboard/README.md` — run the live paper-quoting dashboard
  (`python -m dashboard.server --decision-seconds 60 --close-within-hours 1`,
  then open http://127.0.0.1:8050).
- `hjb_baseline/README.md` — how the HJB benchmark was implemented, how to
  reproduce it, and an honest fidelity assessment against the published
  results (what matches closely, what doesn't, and why).
