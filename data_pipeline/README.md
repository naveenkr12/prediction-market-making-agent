# Prediction-market data pipeline

Step 2 of the capstone roadmap ("Build the data pipeline"), for
*A market-making agent for prediction markets, benchmarked against the
published optimum* (Practice of QCF, Fall 2026).

Pulls, from Kalshi's public API (primary source) and optionally
Polymarket's public APIs (comparison source):
- trade tape (timestamps, prices, sizes, taker direction)
- minute bid/ask candlesticks
- market metadata and settlement outcomes

for a sample of **standalone binary contracts** across the **economics**
and **sports** categories, over a recent **90-day window** — matching the
"Data needed and feasibility" section of the proposal.

## Status

Tested live end-to-end against both Kalshi and Polymarket's production
APIs. A run of **60 contracts** (30 economics / 30 sports, within the
proposal's 50–100 target) completed in ~4 minutes with 60/60 trades and
60/60 candlestick pulls succeeding — see `output/` for the actual data and
`output/logs/` for the run log and validation report.

## Quick start

```bash
pip install -r requirements.txt

# Primary collection (Kalshi) -- ~4 min for 60 contracts
python3 collect.py --target-total 60

# Optional comparison source (Polymarket) -- a handful of high-volume markets
python3 polymarket_collect.py --n-markets 15

# Sanity-check the output
python3 validate.py
```

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunables: date window, category list, sample size, candlestick lookback, rate-limit pacing. |
| `kalshi_client.py` | Kalshi Trade API v2 client — series/events/markets listing, trades, candlesticks, with pagination + retry/backoff. |
| `polymarket_client.py` | Polymarket Gamma/CLOB/Data-API client (optional comparison source). |
| `contract_selection.py` | Picks standalone binary contracts (see "Standalone, defined" below). |
| `collect.py` | Orchestrates selection + collection, writes parquet + a JSON run log. |
| `polymarket_collect.py` | Optional: pulls top-volume closed Polymarket markets for comparison. |
| `validate.py` | Coverage + unit-sanity checks (price bounds, missing files, zero-trade share). |
| `utils.py` | Small shared helpers (timestamp parsing, safe filenames). |

## Output layout

```
output/
  markets/kalshi_selected_contracts.parquet   # one row per selected contract + metadata/result
  trades/<TICKER>.parquet                     # trade tape per contract
  candlesticks/<TICKER>.parquet                # minute OHLC per contract
  polymarket/                                  # optional comparison source
  logs/run_<timestamp>.json                    # per-contract success/failure log
  logs/validation_report.json                  # output of validate.py
```

## Standalone, defined

The proposal says "50–100 standalone contracts" without pinning down what
disqualifies a contract. This pipeline uses a concrete, documented
definition:

- `market_type == "binary"` (a plain Yes/No contract, not a scalar/range
  ladder bucket)
- not part of a multivariate-event collection (`mve_filter=exclude` on the
  query; `mve_collection_ticker` empty on the result)
- already resolved within the lookback window (`status` in `{settled,
  finalized}`, `close_time` inside the 90-day window)

This keeps every contract directly comparable to the Feil & Nendel
single-contract HJB model. Correlated / multi-leg contracts (the
ParlayMarket paper) are intentionally excluded — out of scope for the core
project per the proposal.

## Known limitations / next steps

- **Candlestick lookback is capped at 14 days before close**
  (`config.CANDLESTICK_LOOKBACK_DAYS`). Kalshi caps a single candlesticks
  request at 5000 bars; many markets are listed weeks or months before
  they close with a long, thin pre-market window. Pulling full lifetime
  minute bars is supported (`KalshiClient.get_candlesticks_full` chunks
  around the cap) but multiplies runtime — widen or remove the cap
  (`None`) once you have a compute/time budget for it, e.g. as a
  background job.
- **Unauthenticated rate limits are tighter than documented.** Kalshi's
  published rate-limit tiers are for authenticated keys; anonymous
  requests 429 quickly under bursts. `config.REQUEST_DELAY_S` /
  `BACKOFF_BASE_S` are tuned from live testing — loosen them only if you
  add an API key.
- **~35% of the sample has zero trades in the window** (illiquid
  markets, e.g. exact-score soccer outcomes) — expected, and flagged by
  `validate.py`. If the modeling step needs a minimum-liquidity floor,
  filter `output/markets/kalshi_selected_contracts.parquet` by trade count
  before building the simulator (this is the calibration step in the
  roadmap, not something this pipeline should decide on its own).
- **Polymarket collection is metadata-light by design** (per the
  proposal, it's an optional comparison, not the primary source) — it
  does not attempt the same "standalone" filtering as Kalshi.
