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
├── hjb_baseline/                                 — reproduction of the theoretical
│   │                                                benchmark (Feil & Nendel's HJB
│   │                                                optimal market maker)
│   └── README.md                                — implementation notes, fixes,
│                                                    and fidelity vs. the published
│                                                    Table 2 / Figures 1-5
└── rl_agent/                                     — RL quoting agent (in progress)
```

## Status

| Component | Status |
|---|---|
| Data pipeline (Kalshi + Polymarket) | Done — 60 real settled contracts collected, with trades + candlesticks |
| HJB baseline reproduction (published optimum) | Done — optimal-strategy Table 2 numbers match the paper closely (see `hjb_baseline/README.md`) |
| Parameter calibration to collected data | Not started |
| RL quoting agent | Not started |
| RL vs. HJB benchmark comparison | Not started |

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
- `hjb_baseline/README.md` — how the HJB benchmark was implemented, how to
  reproduce it, and an honest fidelity assessment against the published
  results (what matches closely, what doesn't, and why).
