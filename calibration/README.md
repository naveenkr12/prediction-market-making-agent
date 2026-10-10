# Paper 2: five-minute signed flow and Kyle lambda

This module implements Bartlett & O'Hara's *Adverse Selection in Prediction
Markets: Evidence from Kalshi* (August 10, 2026), Section 4.1, equation (1),
using the verified Step 1 trade tapes. It produces empirical price-impact
estimates for later simulator development.

## Reproduce from saved data

Run from the repository root. No API requests are needed.

```bash
python -m pip install -r calibration/requirements-dev.txt
python calibration/estimate_lambda.py
python calibration/validate_results.py
python -m pytest data_pipeline/tests calibration/tests -q
```

To build the intervals without fitting regressions:

```bash
python calibration/build_intervals.py
```

Both entry points accept `--input-dir` and `--output-dir`. Defaults are
`data_pipeline/output/paper2` and its `calibration/` subdirectory. Estimation
also accepts `--min-intervals 50 --hac-lags 6 --bootstrap-draws 2000 --seed 42`.
The 50-interval threshold is enforced separately for each response and
sensitivity fit; failed subsets receive an explicit status and missing lambda.

The verified local run used `/opt/anaconda3/bin/python`: Python 3.12.2,
NumPy 1.26.4, pandas 3.0.1, SciPy 1.15.2, statsmodels 0.14.2,
Matplotlib 3.10.1, and PyArrow 23.0.1. Runtime versions, input SHA-256 hashes,
source hashes, and bootstrap settings are saved. Direct OLS imports avoid
unrelated time-series API compatibility problems in this environment.

## Dataset definition

For each contract, use half-open UTC intervals `[start, start + 5 minutes)`.
Retain intervals containing at least one usable trade. Single-trade intervals
have zero within-interval price change and remain in the estimation sample.
Empty intervals are absent; the dataset does not interpolate trades or prices.

For trade `i`, `s_i = +1` for a YES taker and `-1` for a NO taker.
`c_i` is the number of contracts, including fractional quantities. Then

```text
signed_flow_contracts = sum(s_i * c_i) = buy_contracts - sell_contracts
delta_logodds = logit(close_price) - logit(open_price)
delta_price_cents = 100 * (close_price - open_price)
```

`open_price` and `close_price` are the first and last YES transaction prices
**inside the same interval**, in probability units. Price change uses these
transaction endpoints. Supplementary candles and quote midpoints are not
inputs to this estimator. The first trade contributes to signed flow, following
the paper's stated interval definition.

Trades use the Step 1 normalizer: deduplicate IDs, remove block trades, validate
quantity/side/price, and sort by timestamp then trade ID. Collection status and
contract eligibility are rechecked before aggregation. Exact probabilities 0
and 1 remain in the raw-price dataset but have undefined log odds; affected
endpoints are excluded from log-odds fits without epsilon clipping.

Different prices at an identical endpoint timestamp make execution ordering
ambiguous. Trade-ID order makes the main result reproducible, but does not
establish the true order. Flags identify these intervals, and a separate
unambiguous-endpoint fit excludes them. Ties inside an interval that do not
affect its endpoints do not trigger this exclusion.

Each interval contains metadata (`ticker`, event/cluster, series, category,
information class, settlement result, split), timestamps, trade count,
buy/sell/total/signed contracts, YES price endpoints/extrema/VWAP, signed
imbalance, price changes, and sensitivity/quality flags. Settlement outcomes
and full-life sensitivity flags are retrospective metadata, not inputs for a
future live quoting policy.

## Estimator and uncertainty

For each market, fit an intercept and an unweighted OLS slope:

```text
delta_logodds_t = intercept + lambda * signed_flow_contracts_t + residual_t
```

The companion raw-price fit uses `delta_price_cents_t`. Thus primary lambda
units are **log odds per net contract**; raw-price lambda units are **cents per
net contract**. Signed flow uses contract quantity, not dollar turnover or
normalized imbalance. Numerical centering/scaling of the regressor is reversed
before saving coefficients. Negative slopes are retained. Constant flow or
constant price change receives a diagnostic status instead of a manufactured
zero coefficient.

Per fit, outputs include the point estimate, intercept, R-squared, HC1 robust
standard errors, and 95% normal intervals. The main diagnostic intervals use
calendar-aware Newey-West covariance with Bartlett weights and six actual
five-minute lags (30 minutes), with the `n/(n-2)` correction. Only existing
score pairs exactly `k * 5 minutes` apart contribute to lag `k`. Empty intervals
contribute zero scores to the covariance calculation and are never added as
regression observations. Six lags is an analysis setting, not a parameter
reported by the paper, and does not capture every possible long-run dependency.

This handling matters because standard [statsmodels HAC](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)
assumes consecutive, equally spaced observations. Tests compare our covariance
to that implementation on a regular grid and to an explicit zero-score grid
when intervals are missing. HC1 conventions follow the official
[robust covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).
These uncertainty estimates and influence checks extend the paper's point
estimator.

Two-sided p-values receive [Benjamini-Hochberg adjustments](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
within each response/variant family. The primary family contains 60 full-life
log-odds fits. Train-only q-values are calculated solely from training events.
Separate families make sensitivity analyses exploratory; their significance
counts do not establish joint control across every analysis tried. Standard BH
assumptions still matter when markets share common information shocks.

Leverage, Cook's distance, the largest bucket's share of absolute signed flow,
and the coefficient after removing the highest-leverage bucket are saved.
Flags use leverage above 0.2 and leave-one-out relative change above 50%.
These diagnostics do not automatically trim observations. The saved
`illustrative_p50_move_per_100_contracts_pp` converts the slope's contribution
for 100 net contracts at an initial probability of 0.5 into percentage points;
it omits the intercept and is not a causal prediction or a spread prescription.

## Sensitivities and event holdouts

Each response has four variants:

| Variant | Sample rule |
|---|---|
| `full` | All active intervals with valid response endpoints |
| `interior_endpoints_30_70` | Both opening and closing YES probabilities in [0.3, 0.7] |
| `early_80pct_life` | Last trade at or before 80% of elapsed time from the first trade to the timestamp cumulative volume first reaches 99% |
| `unambiguous_endpoints` | No endpoint timestamp containing multiple different prices |

The middle-price rule is an explicit endpoint-based operationalization of the
paper's 30–70 cent robustness discussion. It is not a claim that an unspecified
paper filtering implementation has been reproduced exactly. The early-life
rule follows footnote 20's time-based definition; it uses full-history volume
and is a retrospective sensitivity, not an observable live feature. Buckets
crossing its cutoff are excluded.

Event clusters, ordered by their latest close time, are assigned 60/20/20 to
train/validation/test. All contracts related to the same cluster stay together.
This run has 36/12/12 contracts. Histories can overlap in calendar time, so this
is an event holdout rather than a strictly causal walk-forward evaluation.
A future chronological live-policy backtest will need explicit training
cutoffs relative to the evaluation start. Use `scope=train_only` summaries for
later parameter development and preserve the validation/test events.

Group tables give equal-weight contract means, medians, quartiles, and
whole-event bootstrap 95% percentile intervals (2,000 draws, seed 42).
Bootstrap intervals describe sampling variability across estimated event
slopes; they do not propagate within-market estimator uncertainty, remove
convenience-sample bias, or establish independence across events. No group
bootstrap interval is emitted for a group with only one event.

## Verified results, 2026-10-09

The accepted sample has **68,866 trades, 14,318 active intervals, and 60
contracts** across 60 event clusters. All 60 full-life log-odds fits succeed.
The saved tapes span 2025-11-22 through 2026-10-05 UTC; the discovery window
selected market close dates, while estimation uses each saved full lifetime.

51 slopes are positive and nine negative. 19 positive slopes have unadjusted
calendar-HAC `p < 0.05`; 17 remain positive with BH `q < 0.05`. Forty of the
60 primary 95% intervals include zero. Of 14,318 intervals, 7,405 (51.7%)
contain one trade, and 1,955 (13.7%) have ambiguous endpoint prices.

The following descriptive values are **log odds per 1,000 net contracts**.
Their means are not pooled regression slopes or controlled differences.

| Scope | Class | Contracts | Mean | Median | Bootstrap 95% interval for mean |
|---|---|---:|---:|---:|---|
| All, exploratory | Broad-based | 29 | 0.072963 | 0.036733 | [0.037531, 0.111038] |
| All, exploratory | Sports | 30 | 0.053508 | 0.020939 | [0.027481, 0.082201] |
| All, exploratory | Single-name | 1 | 0.208398 | 0.208398 | Not estimated |
| Train only | Broad-based | 20 | 0.080090 | 0.043286 | [0.043889, 0.121359] |
| Train only | Sports | 15 | 0.048961 | 0.012537 | [0.014227, 0.089623] |
| Train only | Single-name | 1 | 0.208398 | 0.208398 | Not estimated |

54/60 primary fits have a bucket with leverage above 0.2. In 35/60, dropping
the highest-leverage bucket changes lambda by more than 50%. These findings
make a single fixed adverse-selection parameter premature without sensitivity
analysis. Price/flow association alone does not measure the fraction of
privately informed traders. One single-name event cannot reproduce the paper's
single-name versus broad-based comparison, and sports is a separate class.

| Log-odds variant | Successful fits | Insufficient intervals | Constant price change | Sign changes versus full among successful fits |
|---|---:|---:|---:|---:|
| Full | 60 | 0 | 0 | — |
| Interior endpoints | 22 | 38 | 0 | 5/22 |
| Early life | 35 | 25 | 0 | 6/35 |
| Unambiguous endpoints | 54 | 4 | 2 | 7/54 |

The sensitivity variants often change sample size substantially. Comparing
group means across them also changes the contributing contracts; use matched
tickers when assessing estimator sensitivity.

## Output files

All outputs are under `data_pipeline/output/paper2/calibration/`:

| File | Contents |
|---|---|
| `five_minute_intervals.parquet` / `.csv` | One observed contract/bucket row |
| `intervals/<ticker>.parquet` | Same dataset partitioned by contract |
| `aggregation_audit.csv` | Counts, exclusions, quantities and endpoint diagnostics per contract |
| `aggregation_manifest.json` | Definitions, trade span, split counts, input/source hashes |
| `event_splits.csv` | Event-preserving train/validation/test membership |
| `lambda_by_contract.csv` | 60 primary full-life log-odds estimates, CIs and diagnostics |
| `lambda_estimates.parquet` / `.csv` | All 480 attempted response/variant fits and failure statuses |
| `lambda_group_summary.csv` | All-event exploratory and train-only descriptive summaries |
| `lambda_summary.json` | Counts, settings, runtime versions, source hashes and limitations |
| `figures/lambda_logodds_by_contract.png` | Contract estimates with calendar-HAC intervals |
| `validation_report.json` | Offline reconciliation and independent numerical checks |

Validation checks all 480 attempted rows, independently verifies the 342
successful slopes/intercepts and HC1/calendar-HAC standard errors, and
reconciles interval counts, quantities, signed flow, and endpoint prices to
all selected tapes. The combined calibration/data/environment suite now passes
83 tests, including collection
checks, fractional flow/direction, time boundaries, undefined log odds,
timestamp ties, minimum samples, negative/degenerate slopes, calendar gaps,
train-only multiple-testing adjustments, and complete saved-file integration.

## Integration with the market-making project

Use these estimates as empirical targets for an adverse-selection extension
of the simulator. Kyle lambda is a price-response coefficient; the HJB
baseline's order-arrival intensity is a different quantity with different
units. A simulator adaptation must specify how fills and informational price
innovations interact, then check its simulated five-minute lambda against
training-event estimates and their uncertainty. Changing that interaction
also changes the model behind the published HJB optimum, so retain the
existing reproduction as its own benchmark.

Interval construction and empirical lambda estimation are complete.
Training-only price-impact parameter calibration is now implemented by
`calibrate_price_impact.py`; see [`../simulator/README.md`](../simulator/README.md)
for beta's endpoint adjustment, low/base/high uncertainty scenarios, usage,
and replay checks. Market activity, paired trade size/side and quote-based
background noise are now calibrated by `calibrate_market_dynamics.py`; see
[`../simulator/MARKET_DYNAMICS.md`](../simulator/MARKET_DYNAMICS.md).
`calibrate_execution.py` adds a training-only spread bank and verifies the
frozen split/upstream quote hashes. The complete quoting/accounting environment
is documented in [`../simulator/TRADING_ENVIRONMENT.md`](../simulator/TRADING_ENVIRONMENT.md).
The frozen HJB adapter and paired synthetic profit/risk evaluation are complete;
see [`../simulator/HJB_BENCHMARK.md`](../simulator/HJB_BENCHMARK.md).
Queue/capture calibration with richer orderbook data and RL evaluation remain
subsequent work.
