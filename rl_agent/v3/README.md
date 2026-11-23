# RL market maker v3: live-compatible, flow-aware, variance-reduced

v3 replaces the v2 PPO agent (`rl_agent/calibrated_env.py`). On held-out
seeds, v2 had learned to almost never quote. v3 is the only policy tested
with **positive expected PnL across randomized scenarios**. It beats the HJB
benchmark there by +$0.18 per episode (95% CI +0.14 to +0.22). It can also run
in the live dashboard as strategy `v3`.

The honest headline: the edge is small. It is about +$0.03 to +$0.09 per
market-hour, and realized PnL is still indistinguishable from zero at 3,000
episodes because settlement noise dominates. Details are below.

## What was wrong with v2, and what v3 changes

| Problem found | v3 change |
|---|---|
| The reward was dominated by noise no quote controls: per-episode std 4-8 from diffusion and Bernoulli settlement, against a spread edge of about 0.1. PPO couldn't find the signal and converged to "never quote" | **Control-variate reward.** Training reward = spread captured at fill (vs. the reference at that instant) - fees + impact moves on held inventory. The zero-mean diffusion term (q dp from background innovations) and settlement term q(Y - p) are removed, so the optimum is unchanged with far lower variance. `test_v3.py` checks that the decomposition reconstructs realized PnL exactly |
| No risk control in units that make sense | **Mean-variance penalty** lambda/2 * E[Var]. It covers diffusion over each window and settlement variance q^2 p(1-p) at close, with lambda = 0.002 |
| The policy observed the simulator's *latent* probability, which doesn't exist live | **Live-computable features only** (`features.py`): book mid and spread, inventory, clock, last-window and cumulative signed taker flow, volume, trade count, and mid changes. The same module drives the dashboard |
| Only 2 actions (fixed 10-lot, two-sided) | **4 actions**: skew, width, and a separate size per side (0-20). The agent can quote one side only or stay out |
| One scenario cell, one seed, fixed p0 = 0.5. Simulated sports flow is about 75% YES takers, so a policy could just memorise "go long" | **Domain randomization**: sports 60% / broad 40%, high impact 15%, conservative fills 30%, p0 ~ U(0.15, 0.85), and a **YES/NO mirror of the taker flow** 50% of the time. The agent has to read the direction from observed flow |

## Results (held-out test seeds 500000-502999, 3,000 episodes per cell, paired)

`exp_pnl` is the variance-reduced estimator of expected PnL. It has the same
mean as realized PnL with roughly 10-50x lower variance. `mean_pnl` is raw
realized cash PnL. Per-episode capital is $1,000, the episode is the last hour
before close, and there is one decision every 300 s.

**Randomized scenarios (the target distribution)**

| Policy | exp PnL [95% CI] | realized PnL | PnL std | ES95 loss | fills |
|---|---|---|---|---|---|
| **v3 (RL)** | **+0.027 [+0.010, +0.043]** | -0.002 | **2.86** | **7.6** | 1.1 |
| Tuned heuristic (grid-searched) | -0.087 [-0.124, -0.052] | -0.098 | 3.82 | 9.9 | 1.6 |
| HJB (Paper 1) | -0.154 [-0.197, -0.112] | -0.131 | 5.32 | 13.1 | 3.9 |
| v2 PPO | -0.049 [-0.064, -0.036] | -0.045 | 1.71 | 5.6 | 1.5 |
| Fixed spread | -0.251 | -0.191 | 6.29 | 16.6 | 6.8 |
| Inventory skew | -0.221 | -0.187 | 4.89 | 12.8 | 6.9 |
| No quote | 0 | 0 | 0 | 0 | 0 |

v3 - HJB (expected): **+0.182 [+0.141, +0.223]**.

**Standard cells (p0 = 0.5, base impact and fills, no mirror)**

| Cell | v3 exp PnL | Tuned | HJB | v3 std vs tuned std |
|---|---|---|---|---|
| Sports | +0.056 [+0.045, +0.068] | +0.068 | -0.034 | 2.77 vs 4.98 |
| Broad-based | +0.088 [+0.073, +0.102] | +0.109 | +0.079 | 3.11 vs 3.57 |

In the standard cells, v3 roughly matches a heuristic grid-searched on them,
with 25-45% less PnL dispersion. Its advantage shows when conditions vary:
flow direction, impact, fill quality and starting price.

## What the agent learned (`diagnose.py`)

- **It stays out early and quotes more toward close.** In sports it quotes 2%
  of decisions at the start of the hour and 71% at the last decision.
- **It skews with order flow.** The correlation of quote centre with cumulative
  signed taker flow is 0.59.
- **It quotes wide** (about 4.9c half-width) **and small**: about 3 lots per
  side on one-tick sports books, about 7 on wide broad-based books. On
  one-tick books your quote sits behind the queue, so you mostly fill on large
  trades, which are the most toxic because each moves the price against you
  right after you fill.

## Why the edge is small (the simulator's economics)

The control-variate split shows that, for every heuristic swept (width, size,
skew, flow-lean, one-sided), **impact losses exceed net spread capture**
(`frontier.py`). The maker fee scenario is multiplier 1, at
about 0.44c/contract at p = 0.5. That is close to the whole half-spread on
one-tick books, where you fill mostly on large impactful trades, and quotes
stay stale for 300 s. A passive maker has little to harvest here. v3 earns
its small positive edge by selecting *when*, *which side* and *how much*.

## Live (paper) check

`python -m dashboard.server` now includes `v3`. Run 2026-11-23 on markets
in their final hour (BTC hourly, WTI, S&P hourly, about 40 min): v3 placed
one-sided quotes and had no fills (PnL 0). In the same markets, fixed/skew were
marked at -3.04 and v2 at -4.80, almost all from being picked off on the BTC
market; HJB had no fills. Marks are at close, with BTC settlement pending.
That is anecdotal, not evidence.

**Known distribution shift.** Live NFL and NBA volumes are far larger than the
summer-2026 templates the simulator was calibrated on. For example, about 2,000
contracts of net flow arrived in 60 s, which pushes the volume features outside
their training range. Before the final hour the dashboard pins the episode
clock at its start, where v3 mostly waits.

## Limitations

- Simulated edge only. Fills, queues and impact are the calibrated scenarios,
  not exchange replay.
- **The expected-PnL estimator assumes zero-mean noise terms.** That holds in
  randomized and sports cells (|realized - exp| within noise). In the
  broad-based standard cell, the baselines that quote off the *latent* price
  show a -0.15 to -0.22 gap (about 2 SE), so treat broad-cell `exp_pnl` for
  those baselines with care.
- One training seed was selected on validation seeds 900000+ among 3 runs
  (lambda = 0.002, lambda = 0.0005, and the first run fine-tuned for 2M more
  steps). All had the same expected PnL within noise. The lowest-risk run was
  kept.

## Files

| File | Purpose |
|---|---|
| `features.py` | Live/sim shared observation and action decoding |
| `env_v3.py` | Gymnasium env, control-variate reward, scenario randomization |
| `train_v3.py` | PPO training (`--resume` to fine-tune) |
| `evaluate_v3.py` | Paired evaluation vs all baselines, bootstrap CIs |
| `tune_heuristic.py` | Grid search for the strongest simple baseline |
| `diagnose.py` | What the policy does (actions vs flow, time, group) |
| `models/ppo_v3.zip` | Selected model (3M steps, lambda = 0.002) |
| `results/summary_test.csv` | Table above; `episodes_test.csv` has per-episode rows |
| `tests/test_v3.py` | PnL-decomposition identity, mirror, spaces, dashboard adapter |

## Reproduce

```bash
pip install -r data_pipeline/requirements.txt -r calibration/requirements.txt -r rl_agent/requirements-rl.txt
python -m pytest rl_agent/v3/tests -q
python -m rl_agent.v3.train_v3 --timesteps 3000000 --risk-lambda 0.002 --seed 1 --out rl_agent/v3/models/ppo_v3
python rl_agent/v3/tune_heuristic.py
python -m rl_agent.v3.evaluate_v3 --model rl_agent/v3/models/ppo_v3.zip --n 3000
```

## Next steps that would move the edge most

1. Shorter decision windows (60 s) in the simulator, to cut stale-quote losses.
   The dashboard already supports `--decision-seconds 60`.
2. Recalibrate volume templates on current-season NFL/NBA data, to close the
   live feature shift.
3. Real per-series maker fees. Many Kalshi series list a zero maker multiplier,
   which would add about 0.44c/contract to every fill here.
4. Train on recorded live books (`data_pipeline/rl_record_orderbooks.py`) as a
   replay environment.
