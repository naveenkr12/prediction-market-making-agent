# Training-calibrated price-impact extension

The simulator now has a loadable market-flow price-impact parameter with
zero/low/base/high scenarios. Calibration uses **36 training contracts,
9,424 five-minute intervals, and 44,506 trades**. Validation/test raw tapes
are not loaded, and their estimates do not determine any parameter or range.

The price-impact parameter step is complete. Training-calibrated market trade
activity, paired size/side marks and background innovations are now implemented
too; see [MARKET_DYNAMICS.md](MARKET_DYNAMICS.md). The complete research quoting
environment is implemented; see [TRADING_ENVIRONMENT.md](TRADING_ENVIRONMENT.md)
for post-only quotes, partial fills, cash/collateral, fees, settlement and
ledger validation. The frozen HJB strategy now runs in the calibrated
environment with paired profit/risk comparisons; see
[HJB_BENCHMARK.md](HJB_BENCHMARK.md). Inventory/risk calibration, richer
execution data and RL policy evaluation remain subsequent work.

## Run

From the repository root, with the calibration dependencies installed:

```bash
python calibration/calibrate_price_impact.py
python -m simulator.demo_price_impact --group broad_based --scenario base --signed-flows 100 -40 20
python -m simulator.demo_price_impact --group sports --scenario high --signed-flows 100
python -m pytest calibration/tests data_pipeline/tests -q
```

The first command operates entirely on saved data and writes to
`data_pipeline/output/paper2/calibration/simulator/`. Options include
`--raw-input`, `--calibration-dir`, `--output-dir`, `--min-events 5`,
`--bootstrap-draws 2000`, and `--seed 42`. Calibration verifies training tape
hashes and recomputes the saved training regressions before choosing parameters.

## Price rule and execution convention

For latent log odds `L` and signed aggregate market flow `F`:

```text
L_next = L + independent_innovation_logodds + beta * F
p_next = logistic(L_next)
```

YES takers contribute positive contracts and NO takers negative contracts.
`beta` has units **log odds per net market contract**, including fractional
quantities. Multiply it by contract quantity once. The jump is not a rate and
is not multiplied by elapsed time or by trade count. A separate base driver
can supply its already-generated log-odds innovation.

Reference observations are recorded **after** each trade-flow update. This is
an explicit modelling proxy for the paper's transaction-price response.
The saved data do not separately identify a causal fair-price process,
bid/ask bounce, or a quote-execution model. A full market-making environment
must account for execution prices and validate its transaction-price lambda
separately. This log-odds specification is a prediction-market adaptation;
it does not claim to reproduce the complete equilibrium in
[Kyle's original model](https://www.econometricsociety.org/publications/econometrica/browse/1985/11/01/continuous-auctions-and-insider-trading).

Maker inventory uses the opposite sign: a maker selling ten YES contracts
has `yes_inventory_change = -10`, which corresponds to `F = +10`.
`advance_maker_fill` performs that conversion for an own-fill-only experiment.
For aggregate market flow that already includes the maker fill, use `advance`
once; adding the fill again would double-count its impact. A quote/fill engine
should book cash and inventory at the actual execution quote before applying
the resulting reference-price jump.

## Why simulator beta differs from observed Kyle lambda

The empirical regressor includes every trade's signed quantity in the bucket.
For post-trade reference observations, the first trade's jump is already in
the opening reference. Thus, for a zero-innovation bucket:

```text
F = sum(signed trade quantities)
E = F - first_trade_signed_quantity
delta_L = beta * E
lambda_simulated = beta * Cov(F, E) / Var(F)
```

For each training contract and sensitivity sample, calculate the endpoint
factor `r = Cov(F,E)/Var(F)`. Choose the class parameter:

```text
beta_base = max(0, median(training contract lambdas)) / median(training r)
```

For nonnegative beta, `median(beta*r) = beta*median(r)`, so this matches the
median empirical lambda under the chosen observation convention. It does
not match every contract or the mean/pooled OLS coefficient. This adjustment
must be recalculated if prices are observed before trades: the corresponding
effective flow would omit the last trade rather than the first.

All successfully estimated training contracts contribute, including negative
and statistically insignificant lambdas. There is no positive-only or
significance screening. The simulator constrains beta to be nonnegative;
projection occurs only after calculating the group median and is recorded.
Zero remains an explicit control scenario. At least five training event
clusters are required for a class or sensitivity calibration by default.
This is a support rule, not a statistical guarantee.

## Uncertainty and sensitivities

The base parameter uses the full training sample. For each class, also
calibrate on these samples:

- opening and closing prices both in 30–70 cents;
- first 80% of the elapsed lifetime ending at the 99%-volume timestamp;
- intervals with unambiguous endpoint prices;
- full intervals after dropping each contract's highest-leverage bucket.

The last variant recomputes coefficients **and** uncertainty after the drop;
it is a diagnostic, not an automatic edit to the main sample. Each subset
still requires 50 intervals per regression. A sensitivity with fewer than
five usable event clusters remains unsupported and cannot set the range.

For each supported variant, whole event clusters are resampled 2,000 times,
recomputing the median-lambda/median-factor ratio and its nonnegative
projection. The low/high envelope spans all supported variant point values,
their 2.5/97.5 bootstrap percentiles, and the median individual calendar-HAC
lower/upper bounds converted with the variant's median endpoint factor.
Individual bounds are not group confidence intervals. The final envelope is
a **stress scenario range**, not a 95% confidence interval for a causal
structural parameter. Event resampling does not remove cross-event dependence
or selection bias in this convenience sample.

The low parameter is zero for all supported groups because the uncertainty
envelope includes zero. Consequently `low` and `zero` coincide in this run.
The sports high scenario comes from the early-life bootstrap based on only
six training events, so it is much wider than the full-sample base range.

## Calibrated values, 2026-10-09

All values below are **beta, log odds per net contract**:

| Profile | Training events | Low | Base | High | Base move for +100 contracts at p=0.5 |
|---|---:|---:|---:|---:|---:|
| Broad-based | 20 | 0 | 0.0000547379 | 0.000223431 | +0.13684 percentage points |
| Sports | 15 | 0 | 0.0000126983 | 0.000483927 | +0.03175 percentage points |
| Pooled, explicit fallback | 36 | 0 | 0.0000313551 | 0.000157369 | +0.07839 percentage points |
| Single-name | 1 | — | — | — | Unsupported |

Full-training median lambda targets are approximately 0.0000432858 for
broad-based, 0.0000125368 for sports, and 0.0000279491 pooled. Their median
endpoint factors are 0.790784, 0.987277, and 0.891375 respectively.
The full-variant event-bootstrap beta ranges are [0.0000193763, 0.000118921]
for broad-based and [0, 0.0000906806] for sports. The wider low/high table also
incorporates the sensitivity and individual-uncertainty envelopes.

There is no automatic single-name fallback: requesting that profile raises
an error. Choosing `pooled` explicitly uses the heterogeneous mixture of all
36 training markets, including the single-name event; it makes no company
versus aggregate-market claim.

## Use from code

```python
from simulator.price_impact import ImpactParameters, ImpactPriceProcess

parameters = ImpactParameters.load("broad_based", scenario="base")
market = ImpactPriceProcess(initial_probability=0.5, parameters=parameters)

# +100 net YES contracts, followed by an independently generated innovation.
first = market.advance(signed_market_contracts=100)
second = market.advance(signed_market_contracts=-40, innovation_logodds=0.01)

# Own-fill-only example: the maker sells 10 YES, so market flow is +10.
fill = market.advance_maker_fill(yes_inventory_change=-10)
```

The process retains latent log odds instead of repeatedly taking logits of
rounded probabilities. Float display is restricted to the closest
representable interior probabilities, with saturation flags; it does not
clip the latent update. Extremely large stress flows can saturate displayed
prices. No full-training **base** replay prices saturate. Full-training high
replays have 39 saturated trade displays in broad-based, 15 in sports, and
31 pooled. For those stress cases, reference log odds should be retained for
analysis; finite-precision price round-trips no longer reproduce the same
coefficient. Output checks report both calculations.

The extension changes the price dynamics assumed by the original HJB model.
Its previously solved policy can later be evaluated as a reference strategy
under impact; it is not the proven optimum for this modified process.
The existing published-parameter reproduction remains its own benchmark.

## Artifacts and verification

| File under `data_pipeline/output/paper2/calibration/simulator/` | Purpose |
|---|---|
| `price_impact_config.json` | Loadable parameters, method, support status, scenarios, provenance |
| `price_impact_scenarios.csv` | Compact numeric table and probability-scale examples |
| `training_impact_intervals.parquet` | Training-only buckets plus first-trade and effective signed flow |
| `training_impact_targets.csv` | 180 attempted training regressions and endpoint factors |
| `replay_contract_checks.csv` | 1,100 contract/variant/scenario reference checks, float precision diagnostics |
| `replay_group_checks.csv` | Mean/median replay values versus empirical training targets |
| `validation_report.json` | Training tape/estimate verification and replay audit |

139 of 180 training fits succeed; failed sensitivity subsets remain explicit.
The largest difference between latent-reference replay lambda and the
analytical endpoint mapping is below 6e-19. All three base profiles also
match their training median when log odds are reconstructed from displayed
float probabilities. Replay segments restart each bucket and use zero
independent innovations; these are calibration consistency checks rather
than a full-life simulation, a held-out test, or a policy/PnL result.

The combined suite now passes **34 tests**, including deliberately corrupting
held-out flow/price/estimate values and confirming identical parameters,
refusing any held-out raw-tape read, maker-side sign conversion, fractional
quantity units, zero control, unsupported groups, endpoint attenuation,
and preserving latent state after numerical price saturation.
