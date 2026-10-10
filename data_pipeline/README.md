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

## Paper 2: calibration-ready collection (Step 1)

The initial 60-contract sample contains only 814 saved trades across three
series. Only two contracts meet the paper's 50-active-five-minute-interval
threshold, and event/rule metadata and full-history routing were not retained.
Use the separate collector below for adverse-selection calibration. It keeps
the original sample and all HJB results intact.

```bash
# From the repository root; explicit date makes discovery/resume reproducible.
python data_pipeline/calibration_quality.py
python data_pipeline/collect_calibration.py --end-date 2026-10-09 --target-total 60
python data_pipeline/calibration_quality.py --input-dir data_pipeline/output/paper2 --report-dir data_pipeline/output/paper2/reports
# Test dependency: pip install -r data_pipeline/requirements-dev.txt
python -m pytest data_pipeline/tests -q
```

The new collector:

- discovers both live and archived standalone, settled binary markets closing
  in the preceding 90 days; full lifetime trade tapes can begin before that window;
- uses a 10,000-contract metadata-volume prefilter, then interleaves series and
  categories; preferred recurring series and moderate-volume ranking make this
  an explicitly documented convenience sample, not a random population sample;
- accepts up to 30 economics and 30 sports contracts, at most one per event
  cluster and ten per series; related CPI, jobs, and PCE monthly releases share
  clusters across series;
- retains actual API event identifiers, descriptions, resolution rules,
  settlement outcomes, timestamps, reported volume, and raw market metadata;
- uses the archive's trade cutoff to query both trade endpoints as needed,
  exhausts cursor pagination, and deduplicates overlapping trade IDs;
- checks at least 50 five-minute intervals with interior opening/closing trade
  prices, valid quantities/taker directions, verified pagination, metadata
  completeness, and no material shortfall against the market's reported volume;
- excludes block trades from the usable tape but retains raw trades and records
  exclusion counts; exact 0/1 prices are retained but cannot be log-odds endpoints;
- collects supplementary minute quotes for the final three days. Quote coverage
  is reported separately and is not a prerequisite for trade-price lambda;
- saves rejection reasons, checkpoints each contract, and resumes successful
  downloads. A truncated tape is never accepted as complete.

Output is under `data_pipeline/output/paper2/`:

```text
settings.json                         # frozen window, filters, seed, sampling caps
markets/candidate_metadata.json       # all screened candidates, full metadata
markets/kalshi_selected_contracts.parquet # accepted sample, flattened metadata
market_metadata/<ticker>.json         # raw API market record
trades/<ticker>.parquet                # full raw tapes, including rejected candidates
candlesticks/<ticker>.parquet          # supplementary quote/trade candles
collection_status/<ticker>.json        # history completeness/query bounds/errors
reports/contract_quality.csv           # per-contract eligibility and exclusion reasons
reports/quality_summary.json           # sample size, event diversity, category shortfall
original_sample_audit/                 # audit of the original 60 contracts
logs/discovery.json                    # discovery errors and endpoint provenance
logs/archive_cutoffs.json              # archive boundaries used by this run
```

Re-run with identical settings/output path to resume. Use `--retry-failed` to
retry failed tapes, `--max-candidates` to increase the attempt budget, and a new
`--output-dir` when changing the sample definition. Exit code 2 means the target
was not reached; inspect the summary rather than treating a partial run as done.
If a category runs out of eligible events, `--expand-discovery` searches twice
as many series in that category and logs the expansion. It reuses completed
tapes and keeps the same liquidity/coverage thresholds and event/series caps.
If there are still too few eligible events in one category, the explicit
`--allow-unbalanced` option fills the remaining slots from the other category
after trying equal quotas. `selected_by_category`, `quota_shortfall`,
`balance_policy`, and `balanced_target_reached` report the actual outcome;
the event cap and 50-interval requirement are never relaxed.

`information_class` is deliberately conservative: sports remains `sports`,
an explicit aggregate-series allowlist maps to `broad_based`, and everything
else is `needs_review` until reviewed. Economics is never automatically mapped
to broad-based. A review of saved rules maps Argentina inflation and Brazil's
central-bank decisions to broad-based, and Nvidia H200 deliveries to single-name.
The collected sample has only one single-name contract, so it cannot reproduce
the paper's main two-group contrast by itself.
Event diversity reduces duplicated outcomes but does not establish statistical
independence. Eligibility establishes data readiness, not a precise lambda
estimate; price/flow variation and estimator uncertainty belong to Step 2/3.

API routing follows the official [historical-data guide](https://docs.kalshi.com/getting_started/historical_data).
Historical market filters differ from live filters: timestamps and MVE exclusion
are applied locally when listing a historical series. Minute-candle fields also
vary by endpoint; raw quote data are preserved for later normalization.

Verified run (as of 2026-10-09): 80 contracts audited, 20 rejected, and 60 selected
across 60 event clusters and 25 series (30 economics / 30 sports). The selected
tapes contain 68,866 trades; usable five-minute interval counts range from 50 to
1,105 (median 181). All 60 have supplementary candles. Reconciliation accepts
observed volume within 95-105% of metadata volume; the run's minimum coverage
is 99.42%. Historical and live discovery for the large MLB player-stat series
`KXMLBHRR` hit the metadata pagination budget and were explicitly excluded;
these two discovery errors are retained in the log and do not truncate any
accepted trade tape. Classification was reviewed from saved resolution rules:
29 broad-based, 30 sports, and one single-name contract. This is a convenience
sample, and one single-name event cannot support a two-group replication.

## Paper 2: aggregation and estimation (Steps 2/3)

The verified saved tapes now feed [`../calibration/README.md`](../calibration/README.md).
From the repository root:

```bash
python calibration/estimate_lambda.py
python calibration/validate_results.py
```

The offline pipeline rechecks selected-contract eligibility, builds 14,318
observed five-minute intervals from 68,866 trades, and fits 60 full-life
contract-level log-odds lambdas. It also saves raw-cent fits, HC1/calendar-HAC
confidence intervals, influence and timestamp-tie diagnostics, three
sensitivity subsets, event splits, and train-only group summaries under
`output/paper2/calibration/`. The calibration README records actual results
and limitations; data eligibility does not guarantee estimator precision.

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
