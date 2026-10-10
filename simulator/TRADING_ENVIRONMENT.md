# Complete calibrated trading environment

`TradingEnvironment` connects the existing train-only activity, size/side,
background innovation and price-impact drivers to post-only quotes, partial
maker fills, cash, signed YES inventory, collateral, fees and binary settlement.
It runs complete episodes and exports a reconstructable event ledger.
This is a research simulation; fills are scenarios pending historical L2 data.

## Run

From the repository root, using the Python environment containing the
calibration dependencies (`/opt/anaconda3/bin/python` on the current machine):

```bash
python calibration/calibrate_execution.py
python -m simulator.demo_trading_env
python -m simulator.validate_trading_env data_pipeline/output/paper2/calibration/simulator/trading_environment
python -m pytest simulator/tests calibration/tests data_pipeline/tests -q
```

The default demo runs two classes, zero/base/high impact, three fill scenarios,
three reference policies, and three seeds: **162 episodes**. Each episode starts
one hour before the selected training template's close and continues through
settlement. The starting probability is an explicit 0.5 scenario, not an
estimated historical initial price. Episode duration uses actual seconds; the
last window can be shorter than five minutes.

```bash
python -m simulator.demo_trading_env --groups sports --episodes 10 --impact-scenarios base --fill-scenarios base --policies fixed skew
python -m simulator.demo_trading_env --maker-fee-multiplier 0 --output-dir /tmp/maker-fee-control
```

The fee-zero-multiplier control still rounds fractional cash changes to the
account precision. It therefore can incur rounding fees.

## Use from a policy

```python
from simulator.trading_env import EnvironmentConfig, TradingEnvironment
from simulator.execution import QuoteAction
from simulator.policies import quote_policy

env = TradingEnvironment(EnvironmentConfig(group="sports", impact_scenario="base"))
obs, info = env.reset(seed=42, options={"seconds_to_close": 3600})
while True:
    action = quote_policy(obs, size=10, inventory_skew=0.0005)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(env.render())
```

`QuoteAction(bid=.48, ask=.52, bid_size=10, ask_size=10)` replaces both quotes.
Sizes are in contracts, including fractional minimum lots. `step(None)` retains
existing orders, including queue progress and rounding accumulators.
`step(QuoteAction())` cancels both sides. Replacement loses queue priority and
starts new order fee accumulators. Quotes persist until replaced, canceled,
filled, expiry or truncation. Cancellation has no trading fee.

Actions with crossed own quotes, off-grid prices/sizes, nonfinite inputs or
immediate liquidity-taking quotes raise before advancing time. Valid sizes
which exceed inventory/collateral capacity are clipped at order admission;
requested/admitted sizes are reported in `info`. There is no taker-order API.
Default tick $0.01 and minimum lot 0.01 contracts are configurable scenario
choices; real deployments must use each market's actual price/quantity grid.

The core follows the modern `reset -> (observation, info)` and
`step -> (observation, reward, terminated, truncated, info)` interface described
in [Gymnasium's API](https://gymnasium.farama.org/api/env/), but does not depend
on or subclass Gymnasium. Gymnasium is not installed here. A Gymnasium wrapper,
action encoding and observation normalization belong to the subsequent RL step.

## Event sequence and impact

1. Validate/replace quotes at the currently observable synthetic external book.
2. Draw market executions and the empirical terminal noise for the next window.
3. At each execution time, apply the intervening background innovation and
   obtain the current external book from the training spread bank.
4. Match the opposite standing maker quote, subject to its limit, queue ahead,
   remaining size, participation cap and admitted cash/inventory capacity.
5. Book cash and inventory at the **standing quote price**, with fees.
6. Apply `beta * sign * full_market_quantity` to latent log odds once, whether
   or not the maker filled. The fill is a subset of this market volume;
   `advance_maker_fill` must not be applied again.
7. Apply the remaining background innovation and expose the next observation.
8. At close, cancel orders and settle. A step cap before close cancels orders
   and returns `truncated=True`, leaving the position marked and unsettled.

Independent random streams separate market marks, spreads, intrawindow noise
bridges and settlement from policy fills. With a fixed seed and market scenario,
different policies see the same market path and settlement draw. Quote changes
do not change aggregate market execution counts/marks in this small-maker model.

The noise bridge preserves the calibrated five-minute innovation exactly:
`Z(t) = (t/T)*terminal_innovation + sigma*(W(t)-(t/T)*W(T))`. Its conditional
Gaussian fluctuations vanish at `T`. Intrawindow diffusion is an assumption;
the training data calibrate endpoints, not tick-by-tick volatility. Independent
mark/spread/residual sampling omits observed multiwindow serial dependence.

The beta calibrates the **post-trade reference proxy**. Maker cash uses
pre-impact quote execution prices. The complete environment's transaction-price
lambda is consequently not asserted to reproduce the paper's measured lambda;
that requires a separate execution-price validation and richer orderbook data.

## Cash, inventory, collateral and rewards

Inventory `q` is signed YES exposure; a maker bid fill increases `q`, an ask
fill decreases it. The bookkeeping cash `C` changes by `-price*quantity-fee`
on a bid and `+price*quantity-fee` on an ask. A short YES position represents
the equivalent collateralized NO exposure:

```text
locked collateral = max(-q, 0)
free funds before open-order reserves = C - max(-q, 0)
marked wealth = C + q*p
settled wealth = C + q*Y, Y in {0,1}
```

`cash` includes the locked collateral; it is not the spendable exchange balance.
`available_cash` subtracts collateral and outstanding quote reserves. Both
sides reserve their worst-case purchase/short cost and fee bound without
netting unknown future opposite fills. Admission prioritizes the bid when funds
are scarce. This is conservative, including for orders which close inventory.
Cash, fees and quantities use `Decimal`; no uncollateralized short or inventory
overshoot is allowed. Settlement removes the position and releases collateral.
For a short, the signed settlement adjustment can be negative.

Reward is the change in marked wealth, including the terminal settlement jump,
minus optional `gamma * integral(q^2 dt)` and `gamma_T * terminal_q^2`. Default
penalties are **zero**. The integral uses actual seconds and the inventory held
between each fill; terminal inventory is measured before settlement. With zero
penalties, episode rewards telescope to realized PnL at settlement, or marked
PnL on truncation. Risk coefficients are configurable, not data-calibrated yet.

Synthetic settlement draws `Y ~ Bernoulli(p_close)` using an independent random
stream. No historical result/settlement value is loaded or put in observations.
Immediate settlement at close is a simplification: real resolution delays,
funding costs, early determination, disputes and position netting across
markets are outside this single-market environment. The log-odds process does
not make probability a martingale, so this settlement assumption does not
establish a calibrated fair-value or causal private-information model.

## Spreads, queues and fill support

Spread calibration reads only the **36 training quote files**, requiring at
least 50 valid observations with `0 < bid < ask < 1`. It reuses the earlier
quote checks (timestamps inside lifetime, conflicting timestamp removal,
maximum probability spread 0.20). **30 events and 32,626 observations** support
the spread bank: 15 broad-based, 15 sports, and 30 pooled. No held-out quote file
is loaded. Input hashes and the frozen train split are checked.

The sampler places half of an empirical log-odds spread on either side of the
current reference and rounds outward to the scenario tick. The arithmetic
midpoint can differ from the reference. Near floating-point saturation the
displayed book retains at least one tick; latent log odds are retained.
Unsupported events borrow a supported same-class spread bank, explicitly
labelled. Borrowing or applying trailing-window quotes/noise outside their
source coverage sets extrapolation flags.

An incoming YES taker can reach the current external ask and fill a maker ask
at that limit or better; a NO taker can fill a bid at the current external bid
or better. Better-priced maker quotes have zero initial proxy queue ahead;
joining or worse-priced quotes use the configured queue. A market execution
must consume queue ahead before the participation cap applies to remaining
volume. A fill never exceeds that volume or the order's remaining size.
This does not infer queue cancellations or credit hypothetical deep-book sweeps.

| Scenario | Proxy queue ahead | Share of post-queue market volume |
|---|---:|---:|
| Conservative | 100 contracts | 5% |
| Base | 50 contracts | 10% |
| Optimistic | 0 contracts | 100% |

These are **assumed sensitivity settings**, not estimates or confidence bounds.
Five-minute quote decisions, instantaneous replacement, independent synthetic
books and an observable proxy queue simplify real execution. Changes in the
external book alone do not create fills; the driver needs a market execution.
Policy observations include this synthetic queue state; a future live-data
experiment must mask/estimate it. The current saved minute candles do not
identify maker queue rank, depth, latency or quote-conditioned fill intensities.

`BookSnapshot.from_kalshi(payload)` parses fixed-point L2 data and converts
NO bids into YES asks (`1-NO_bid`). Its levels can initialize `queue_ahead` for
the matcher, as verified in tests. The default environment uses the spread
proxy. The L2 adapter is not a chronological exchange replay driver.

A bounded collector is ready for **active** market tickers:

```bash
python data_pipeline/capture_orderbook.py --ticker ACTIVE_MARKET_TICKER --samples 12 --interval-seconds 5
```

It uses the [public orderbook endpoint](https://docs.kalshi.com/api-reference/market/get-market-orderbook),
writes raw fixed-point snapshots and local request/availability timestamps,
and submits no orders. No live snapshots have been collected in this step;
the existing calibration markets are settled, and current snapshots cannot
recover their historical depth. Ordered trade/book-delta recording with gap
checks is still needed for causal as-of matching. Snapshots alone also do not
identify an individual maker's queue rank.

## Fees and historical limitations

The simulator implements the quadratic maker formula with an explicit
multiplier. The [July 7, 2026 fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
sets the base maker coefficient to 0.0175 and lists market-specific multipliers;
the unlisted default is zero. The demo deliberately uses multiplier **1** as a
maker-fee scenario for all templates, rather than reconstructing dated series
fees. This is not a historical fee backtest. Nonquadratic fee products, rebates
outside the rounding accumulator and incentives are not represented.

Following the [official fee-rounding specification](https://docs.kalshi.com/getting_started/fee_rounding),
each model fee rounds upward to six decimal dollars; signed revenue less that
fee rounds downward to the balance precision. The difference accumulates per
order and is rebated in precision-sized units, capped to keep each net fee
nonnegative. Precision is configurable: $0.0001 (direct member, demo default)
or $0.01 (non-direct member). Replacement resets the accumulator. Settlement
fees are zero under the cited schedule. Verify the dated product/series fee
profile and account type before an empirical performance study.

## Validation and artifacts

The full suite, including the new HJB adapter/evaluation, passes **83 tests**.
Tests cover both settlement outcomes for
long/short positions, fractional lots, cash/quote reserves, inventory bounds,
partial fills, queue consumption, sign conventions, post-only validation,
per-order fee rebates, bridge endpoints, short expiry windows, truncation,
reward telescoping, risk units, seed reproducibility and training isolation.
The disk-ledger validator independently reconstructs money/inventory and
aggregate impact, and checks code/config hashes. It detects corrupted cash and
double-counted impact. See [TEST_PLAN.md](TEST_PLAN.md) for targets and gaps.

Verified default run: **162 settled episodes**, **5,211 market executions** and
**535 maker fills** across sensitivity scenarios. Totals repeat market paths
across policies/scenarios; they are not independent historical trades. The
no-quote control has zero PnL. Three seeds per cell validate mechanics, not a
strategy ranking, VaR/ES estimate or profitability claim.

Under `data_pipeline/output/paper2/calibration/simulator/`:

| Location | Output |
|---|---|
| `execution/execution_config.json` | Frozen training spread profiles, support, hashes |
| `execution/spread_bank.npz` | Per-event empirical log-odds spread samples |
| `execution/training_spread_audit.csv` | Quote validity and support by training event |
| `trading_environment/episode_summary.csv` | PnL, fees, fills, settlement and risk per run |
| `trading_environment/scenario_summary.csv` | Descriptive scenario means for mechanics checks |
| `trading_environment/*_ledger.csv` | Exact-money quote/fill/noise/impact/cancel/settlement records |
| `trading_environment/*_steps.csv` | Observations, rewards and termination at each decision |
| `trading_environment/manifest.json` | Seeds, parameters, assumptions, source/config hashes |
| `trading_environment/validation_report.json` | Independent reconstruction results |

The original HJB reproduction is unchanged. Its policy is not proven optimal
under this impacted reference/queue model. Remaining steps are calibrating
risk settings, validating execution with richer data, and training/evaluating
RL with frozen held-out event splits and appropriate dependence-aware
uncertainty. The frozen HJB adapter and paired profit/risk evaluation are now
implemented; see [HJB_BENCHMARK.md](HJB_BENCHMARK.md).
