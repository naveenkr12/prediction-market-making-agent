# RL training data: Kalshi + Polymarket

Dense, labelled market data for training the RL quoting agent, in four
categories on both venues: **Sports, Crypto, Financials, Economics**.
Data is written to `PM_RL_DATA_DIR` (default:
`.../Practice of QCF/Work/Data`), never into the repository.

## What exists and what does not

| Data | Kalshi | Polymarket | Source |
|---|---|---|---|
| Full trade tape (time, YES price, size, taker side) | yes | yes | historical |
| Best bid / ask, 1-minute | yes (minute candles) | **no**, midpoint only | historical |
| Full-depth order book (L2) | live only | live only | `rl_record_orderbooks.py` |

Neither venue publishes historical depth, and Kalshi's streaming book needs
an API key. Real L2 therefore has to be recorded going forward. Historical
data gives the agent real prices, spreads and order flow now; recorded L2
adds queue depth for markets recorded before they settle.

## Pipeline

Run from `data_pipeline/` with the existing requirements (`requests`,
`pandas`, `pyarrow`):

```bash
python3 rl_collect_kalshi.py              # ~40 accepted contracts per category
python3 rl_collect_polymarket.py          # ~20 accepted contracts per category
python3 rl_build_bars.py                  # unified bars, labels, splits, manifest
python3 rl_validate.py                    # integrity checks, non-zero exit on failure
python3 rl_record_orderbooks.py           # live L2 recorder; leave running
python3 -m pytest tests/test_rl_dataset.py -q
```

Collectors resume from per-contract status files, so an interrupted run can
simply be restarted. All settings are in `rl_config.py`.

## Selection and density

- Settled/resolved binary markets that closed in the lookback window:
  60 days by default, 14 days for Crypto (hourly strike ladders list
  250k–400k markets per series in 60 days), and 150 days for Economics
  (releases are monthly).
- Crypto allows 25 contracts per series instead of 12. Only hourly BTC is
  consistently dense, and each hourly contract is a separate event.
- Kalshi files WTI oil under "Commodities". It is collected and labelled as
  Financials here.
- One market per event (the most-traded one), ranked by volume, at most
  12 per series, round-robin across series. **This sample is selected for
  liquidity on purpose. It is not a random sample of each venue.**
- Kept only if the episode window has at least 200 trades and at least 25%
  of its minutes contain a trade or quote update. Every rejection is
  recorded with its reason in `<venue>/status/` and the run log.

## Time alignment with Paper 1

Feil & Nendel use a unitless horizon (T = 1, 1000 Monte Carlo steps), with
parameters "chosen to illustrate qualitative effects rather than to
calibrate the model to a specific contract". Real contracts are mapped onto
that clock:

- **Episode** = the final `episode_hours` of each contract's life:
  Sports 6 h, Crypto 2 h, Financials 8 h, Economics 24 h. Each spans the
  category's active window.
- **tau** = (t − episode start) / episode length ∈ [0, 1] on every bar and
  trade, so `hjb_policy`'s `t = T · tau` ("episode" time mapping) applies
  unchanged.
- **Bars** are 1 minute, the finest resolution at which either venue
  publishes historical quotes.
- `rl_loader.paper_grid(market_id, n_steps=1000)` resamples any episode onto
  the paper's uniform grid: quotes as of each step start, and trade counts and
  flows executed within the step. That is the same information the paper's
  Monte Carlo draws as Poisson arrivals. This makes swapping simulated paths
  for real ones a drop-in change.

## Compatibility with existing code

- Raw Kalshi trades and candles keep Kalshi's schema, the same as
  `output/paper2`. Polymarket trades are converted to that schema (YES price,
  YES/NO taker side) before saving. `calibration_quality.normalize_trades`
  reads both venues.
- Kalshi L2 rows use exactly `capture_orderbook.py`'s row format. Polymarket
  books are stored in the same `payload.orderbook_fp` shape, so
  `simulator.execution.BookSnapshot.from_kalshi` parses either venue.
- Signs follow `calibration/build_intervals.py`: +1 = YES taker, and
  `signed_flow = buy_contracts − sell_contracts`.
- Splits are chronological by close time within venue × category (70/15/15),
  and an event never straddles two splits.

## Output layout

```
Data/
  kalshi/      markets/contracts.parquet, trades/, candlesticks/, status/, logs/
  polymarket/  markets/contracts.parquet, trades/, price_history/, status/, logs/
  dataset/     contracts.parquet, bars.parquet, trades/<venue>/, manifest.json,
               validation_report.json
  live_orderbooks/  universe.jsonl, <venue>/<UTC date>/<market>.jsonl.gz
```

## Collected sample (2026-10-11)

1,234 contracts, 501,643 one-minute bars and 16.4 million trades, with all
validation checks passing. Folder size is 1.7 GB: 838 MB of raw venue
downloads plus 879 MB of cleaned, training-ready data (`dataset/`).
Medians per episode:

| Venue / category | Contracts | Train / val / test | Episode | Trades | Trades/min | Spread |
|---|---:|---|---:|---:|---:|---:|
| Kalshi Sports | 560 | 392 / 84 / 84 | 6 h | 18,650 | 51.7 | $0.01 |
| Kalshi Crypto | 120 | 84 / 18 / 18 | 2 h | 942 | 12.9 | $0.02 |
| Kalshi Financials | 107 | 75 / 16 / 16 | 8 h | 443 | 0.9 | $0.03 |
| Kalshi Economics | 68 | 48 / 10 / 10 | 24 h | 614 | 0.4 | $0.02 |
| Polymarket Sports | 250 | 175 / 38 / 37 | 6 h | 2,583 | 7.2 | n/a |
| Polymarket Crypto | 60 | 42 / 9 / 9 | 2 h | 564 | 4.7 | n/a |
| Polymarket Financials | 60 | 42 / 9 / 9 | 8 h | 740 | 1.5 | n/a |
| Polymarket Economics | 9 | 6 / 1 / 2 | 24 h | 1,055 | 0.7 | n/a |

Lookbacks and targets: Sports 120 days (560 Kalshi / 250 Polymarket),
Crypto 30 days (120 / 60), Financials 120 days (120 / 60), and Economics
365 days (120 / 40). Kalshi Financials (107) and both Economics groups fell
short because too few markets pass the density gates. Polymarket Economics
is limited to central-bank decisions and macro releases (`topical` in
`rl_collect_polymarket.py`). Its tags also carry earnings, crypto races and
Fed-chair politics, and those are excluded. Raw files of rejected contracts
are deleted; their status files keep the reasons.

Paper 1's illustrative intensity implies about 150 taker arrivals per
episode at p = 0.5. Real episodes have 400 to 19,000 (medians). Real Kalshi
spreads (1–3 cents) are also well inside the frozen HJB policy's 5 cents.

## Caveats

- `y_outcome` is the settlement label. It is retrospective and must never
  be fed to a policy as an observation.
- Kalshi candles appear only in minutes where something changed. Other
  minutes carry the last quote forward with `quote_observed=False` and
  `quote_age_s`.
- Polymarket `yes_bid`/`yes_ask` are NaN. `p_ref` is the venue's published
  midpoint. For games, YES is the first-listed outcome (team), as given in
  `yes_sub_title`.
- Polymarket sizes are shares and Kalshi sizes are contracts. Both pay $1
  at settlement, so the units match.
- Real episodes have far more trades than Paper 1's illustrative intensity
  (about 150 arrivals per episode at p = 0.5). `manifest.json` reports both,
  so they can be compared before calibrating `A0`, `A1` or the trade size.
