# Calibrated trade activity, sizes and background price movement

This stage uses **36 training contracts and 44,506 trades** to calibrate
exchange trade-execution frequency, paired quantity/side distributions and
flow-adjusted background innovations. Quotes are loaded for those same 36
contracts; 24 have enough observed quote pairs for the main volatility fit.
Validation/test raw tapes and quotes are excluded.

## Reproduce and use

From the repository root with the existing calibration requirements installed:

```bash
python calibration/calibrate_market_dynamics.py
python calibration/validate_market_dynamics.py
python -m simulator.demo_market_dynamics --group sports --windows 6 --seed 42
python -m pytest calibration/tests data_pipeline/tests -q
```

Outputs are under `data_pipeline/output/paper2/calibration/simulator/market_dynamics/`.
Calibration also accepts `--raw-input`, `--calibration-dir`, `--output-dir`,
`--bootstrap-draws 2000` and `--seed 42`. No API downloads are required.

```python
import numpy as np
from simulator.market_dynamics import MarketDynamics
from simulator.price_impact import ImpactPriceProcess

rng = np.random.default_rng(42)
driver = MarketDynamics("sports", rng, impact_scenario="base",
                        arrival_scenario="base", noise_scenario="base")
price = ImpactPriceProcess(0.5, driver.impact)
window = driver.advance_window(price, life_fraction=0.95, rng=rng)
print(window["n_trades"], window["signed_flow_contracts"], window["probability"])
print(window["volatility_extrapolated"])
```

The driver chooses one training event template uniformly per episode; an
explicit `ticker=` can select a supported template. All durations are calendar
**seconds**. Windows last five minutes, with the terminal window shortened
to remaining listing time. Advance lifetime fraction by duration divided by
lifetime seconds. Rates use the window's starting lifetime phase. Generated
event times, quantities and signs are available for a later execution engine.
The reference update applies aggregate flow once plus its background
innovation. It does not decide which market trades fill the maker.

## Frequency, bursts and trade size

For each contract, execution rate is `usable trade count / (close_time -
open_time)` in seconds. Full listing exposure includes quiet periods before
the first and after the last observed trade. Ten equal elapsed-time lifetime
phases have their own rates, retaining the individual event's activity shape.

Count variance includes every full UTC five-minute interval completely inside
each phase, including zeros. Partial boundary intervals are excluded from
count moments while their trades/exposure remain in the phase rate. For mean
count `m` and sample variance `v`, fit `alpha = max(0, (v-m)/m^2)` and sample:

```text
U ~ Gamma(shape=1/alpha, scale=alpha)
N | U ~ Poisson(rate * window_seconds * U)
E[N] = rate * duration
Var[N] = E[N] + alpha * E[N]^2
```

At alpha zero use Poisson. This gamma-Poisson model accommodates overdispersed
counts; see official [NumPy gamma](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html)
and [negative-binomial](https://numpy.org/doc/2.3/reference/random/generated/numpy.random.Generator.negative_binomial.html)
documentation. Median full-lifetime Fano ratios are 5.50 broad-based and
77.64 sports; fitting dispersion per phase separates part of lifecycle
nonstationarity. A new intensity multiplier is drawn per window, so the model
does not reproduce every cross-window dependency or parent-order execution batch.

Quantity and taker sign are resampled as **pairs** within the chosen event and
phase. YES is +1, NO -1. Fractional quantities and rare large trades remain
uncapped. A phase without marks explicitly falls back to that training
event's full mark bank; zero-rate phases normally produce no trades.
Low/high arrival scenarios scale phase rates using whole-event bootstrap
bounds for the class median rate; base scale is one.

These are exchange trade-execution arrivals. Multiple executions can share
one timestamp. They do not identify distinct parent orders or maker
quote-conditioned fill intensities.

## Quote-based background volatility

Saved flattened quote fields use dollar-probability names or historical
`yes_bid_close`/`yes_ask_close` aliases, which are probability values in this
dataset. Only [0,1] values are accepted; there is no guessed cents conversion.
Reject missing, crossed, invalid or conflicting timestamp quotes, undefined
midpoint log odds, and spreads greater than 20 cents.

Use midpoint quotes at exact UTC five-minute endpoints, retaining consecutive
observed endpoints only. Missing bars are never forward-filled. Signed flow
is integrated over `[left,right)`, including zero-flow intervals where quotes
were observed. This quote response differs from Paper 2's transaction endpoints.

First calculate `fixed residual = delta_midpoint_logodds - beta * flow`.
At the existing base beta, five of twelve supported broad-based contracts and
six of twelve sports contracts have absolute residual/flow correlation above
0.3. The whole fixed residual would therefore retain substantial flow-related
movement. For the background component, remove the remaining fitted linear
association, equivalently:

```text
delta_midpoint_logodds = intercept + quote_flow_slope * flow + eta
remaining_flow_slope_after_beta = quote_flow_slope - beta
sigma_seconds = sqrt(sum(eta^2) / (n - regression_rank) / interval_seconds)
```

Include an intercept; constant flow uses an intercept-only regression. Each
fit needs at least 50 quote pairs and each class/sensitivity needs five event
clusters. Fixed-beta sigma/correlation, quote slope, fitted drift/intercept,
serial correlation, kurtosis and variance decomposition remain diagnostic
outputs. The transaction-based impact beta is not silently replaced by the
quote-based slope. Primary eta/sigma is invariant to beta scenarios; equal
bootstrap draws avoid artificial differences in their uncertainty bands.

Residual linear orthogonality does not identify pure fundamental volatility
or prove independence. Sampling frequency also matters with microstructure
noise; see the primary [Aït-Sahalia and Mykland study](https://www.nber.org/papers/w9611).
Ten-minute midpoint and five-minute 30–70 cent endpoint samples are retained
as sensitivities. The interior sample has fewer than five supported events
in every class, so it cannot set the class range. Low/high sigma envelopes
span supported estimates and whole-event bootstrap percentiles (2,000 draws,
seed 42), as sampling/sensitivity scenarios rather than causal confidence bounds.

## Innovation sampling and coverage

Default innovations resample centered, standardized **empirical residuals**.
Median excess kurtosis is about 13.1 broad-based and 31.8 sports; this retains
their heavy-tailed marginal shape. `noise_distribution="gaussian"` provides
an explicit alternative. Scale innovations as `sigma * sqrt(seconds)`.
Fitted intercepts are diagnostics, not injected forecasting drift.

Supported templates use their own residual bank and sigma, scaled by the
selected class sigma scenario relative to its base median. If quote support
is insufficient, explicitly borrow a supported training event's noise bank
from the same profile. Single-name remains unsupported; pooled use is explicit.

Quotes cover the saved trailing window, usually three days, rather than the
full listing life. Earlier-life use or borrowing another event's noise template
sets `volatility_extrapolated=True`. That use is a scenario assumption, not
measured early-life volatility. Independent residual resampling retains tails
but not serial dependence: observed median lag-one correlations are about
-0.26 broad-based and +0.35 sports. Multiwindow/policy evaluation must test
that approximation. A positive stress sigma with a degenerate empirical bank
uses a flagged Gaussian fallback instead of silently generating no noise.

The driver centers innovations in log odds; it is not the probability
martingale specification behind the original HJB optimum. Normalized-horizon
units would require `rate_normalized = rate_seconds * T_seconds` and
`sigma_normalized = sigma_seconds * sqrt(T_seconds)`. Quote-distance functions
and maker fill probabilities still require a separate execution model.

## Verified values, 2026-10-09

Rates average full listing exposure, including quiet time. The runtime
retains individual event templates; these are descriptive class centers.

| Class | Activity events | Mean trades/hour | Median trades/hour | Mean size, equal-event | Median of event median sizes | Noise-supported events |
|---|---:|---:|---:|---:|---:|---:|
| Broad-based | 20 | 3.21 | 0.88 | 171.85 contracts | 10 contracts | 12 |
| Sports | 15 | 9.90 | 7.14 | 93.09 contracts | 19 contracts | 12 |
| Pooled | 36 | 5.92 | 2.22 | 137.17 contracts | 13.99 contracts | 24 |

Large trades pull means above medians. Pooled-execution mean sizes are 198.02
broad-based and 115.49 sports. Exposure-pooled rates differ too; output names
distinguish weighting schemes.

| Class | Sigma low | Sigma base | Sigma high | Approximate five-minute one-sigma move at p=0.5 |
|---|---:|---:|---:|---:|
| Broad-based | 0.004411 | 0.007567 | 0.010965 | 3.28 percentage points |
| Sports | 0.007706 | 0.011484 | 0.016561 | 4.97 percentage points |
| Pooled | 0.006426 | 0.010466 | 0.014219 | 4.53 percentage points |

Sigma units are **log odds per square-root second**. Probability-scale examples
use the local derivative at 0.5; they are approximations, not price forecasts.
Five-minute sampling supports 12+12 events; ten-minute sampling supports seven
broad-based and nine sports events.

## Artifacts and validation

In `data_pipeline/output/paper2/calibration/simulator/market_dynamics/`:

| File | Contents |
|---|---|
| `market_dynamics_config.json` | Loadable templates, scenarios, units, coverage and hashes |
| `market_dynamics_summary.csv` | Mean/median rates, sizes and sigma |
| `trade_mark_bank.npz` | Paired quantity/sign/phase marks and standardized residuals |
| `training_activity.csv` | Full-lifetime exposure and trade/size/side/count statistics |
| `training_phases.csv` | Ten phase rates and implicit-zero count moments per event |
| `training_quote_audit.csv` | Quote validity, spread and coverage |
| `training_noise.csv` | 852 attempted profile/scenario/variant fits and failure statuses |
| `training_noise_intervals.parquet` | Observed quote changes and matching flow, labelled by profile |
| `calibration_report.json` | Training isolation and fit totals |
| `validation_report.json` | Independent exposure, mark and noise checks |

The 852 rows include class/pooled profiles and four impact scenarios, rather
than 852 different events. There are 352 supported fits; other subsets remain
explicitly insufficient. Intervals are reused across profiles, not pooled
across them for estimation.

The full suite passes **34 tests**, including holdout corruption/read refusal,
quiet-time exposure, fractional marks, quote aliases/conflicts, missing grid
endpoints, half-open flow boundaries, orthogonal residuals, seconds scaling,
gamma-Poisson moments, empirical innovations and config integrity. Independent
validation verifies 360 phase/exposure/count cases and 352 noise fits. Maximum
absolute fitted residual/flow correlation is below 6e-16; this verifies linear
orthogonality rather than independence or held-out prediction quality.

Calibrated inputs, the market reference driver and a complete quoting/fill
environment are implemented. See [TRADING_ENVIRONMENT.md](TRADING_ENVIRONMENT.md)
for cash/collateral, fees, settlement, train-only spread sampling and execution
assumptions. Richer orderbook calibration and HJB/RL evaluation remain.
