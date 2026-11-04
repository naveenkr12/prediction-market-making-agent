# RL quoting agent -- calibrated simulator

This is the capstone's RL step against **the calibrated environment**
(`../simulator/trading_env.py`, real-Kalshi-calibrated arrivals, price
impact, queueing, fees, fractional settlement) -- not the toy Paper 1 world
`paper1_env.py` / `train_ppo.py` / `evaluate.py` use (see `PAPER1_RL.md`
for that earlier, separate check). HJB is a frozen published-model policy
applied to an environment it was never derived for; this is a first attempt
at giving a model-free agent a fair shot at the same game. One scenario
cell (`broad_based`, impact=`base`, fill=`base`), one training seed -- a
first cut, not the capstone's full robustness study.

## Files

| File | Purpose |
|---|---|
| `calibrated_env.py` | Gymnasium wrapper around `TradingEnvironment`. Same observation HJB conditions on (`[episode_time_fraction, 2p-1, q/Q]`), same quote-size convention (`min(delta, headroom)` per side), same tick/post-only projection (`simulator.hjb_policy.executable_action`, reused unmodified). See its module docstring for the full design rationale, **including a reward-shaping bug this file documents rather than hides.** |
| `calibrate_reward.py` | Reproduces the empirical measurement behind the training-reward risk-penalty coefficients (run it if the scenario cell or environment changes). |
| `check_calibrated_env.py` | Proves the wrapper is not buggy: HJB's own quotes, round-tripped through the wrapper's action encoding, reproduce the direct `TradingEnvironment` + `HJBReferencePolicy` path exactly, seed for seed (correlation 1.0000 on 60 held-out episodes). |
| `train_capstone_ppo.py` | PPO training (Stable-Baselines3), 8 parallel env copies, `n_steps=256`. |
| `evaluate_capstone.py` | Paired HJB / fixed-spread / inventory-skew / no-quote / RL evaluation on identical held-out seeds; reuses `simulator.benchmark_metrics.pnl_statistics` so the scorecard matches `simulator/evaluate_hjb.py`'s own. |
| `models/ppo_capstone_broad_v2.zip` | The trained agent behind the results below (3M timesteps, `broad_based`/`base`/`base`, seed 0). |
| `models/*_buggy_reward_DO_NOT_USE.*` | Kept for the record (see "A bug, caught and fixed" below) -- not a usable model. |
| `results/` | `episodes_broad_based.csv` (raw, 300 x 5 policies), `summary_broad_based.csv` (the table below), `rl_vs_hjb_broad_based.txt` (paired bootstrap). |

## Run

```bash
pip install -r ../data_pipeline/requirements.txt -r requirements-rl.txt
python check_calibrated_env.py
python train_capstone_ppo.py --timesteps 3000000 --n-envs 8 --out models/ppo_capstone_broad_v2
python evaluate_capstone.py --model models/ppo_capstone_broad_v2 --n-episodes 300
```

## A bug, caught and fixed (read this before reusing `TRAIN_GAMMA`/`TRAIN_GAMMA_T`)

The first version of `calibrated_env.py` reused `hjb_baseline`'s own risk
coefficients (`gamma=4e-3`, `gamma_T=1e-3`) unchanged as the RL training
reward's inventory penalty. That is wrong by construction: those
coefficients weight an integral of `inventory**2 * dt`, and the paper's
`dt` lives on a **normalized clock with a whole episode spanning `T=1`**,
while `TradingEnvironment` integrates over **real seconds** (an episode
here spans a 3,600-second hour). Reusing the coefficient inflates the
running penalty by a factor of order 3,600.

Measured empirically (100 HJB-policy episodes, this environment): the raw
running `inventory**2 * seconds` integral averages ~144,000, against a
realized-PnL scale of a few dollars. At the original coefficient, the
penalty term alone averages in the hundreds to thousands of dollars per
episode -- it swamps anything a quoting strategy can earn. A PPO agent
trained against that reward correctly learned the trivial fix: quote so
wide you almost never fill. The first full 3M-step training run converged
to **0.03 maker fills per held-out episode** (vs. HJB's ~3.7) and "won" on
mean PnL only because doing almost nothing is close to the zero-PnL
`none` baseline, not because it quoted well. The evaluation table and
model from that run are kept at `models/*_buggy_reward_DO_NOT_USE.*` and
`models/train_broad_buggy_reward_DO_NOT_USE.log` rather than deleted, so
the mistake is visible rather than quietly fixed.

`calibrate_reward.py` reproduces the measurement and suggests a coefficient
so that, at HJB's own inventory behaviour, the running and terminal penalty
each average roughly 10% of the realized-PnL scale (`|mean| + std`) rather
than multiples of it. `calibrated_env.py` now uses `TRAIN_GAMMA=3e-6`,
`TRAIN_GAMMA_T=5e-3` -- about 1,000x smaller on the running term than the
original mistake. This is a judgment call (how large a nudge is "a nudge,
not a dominant term"), not a derived constant, and it should be revisited
if the scenario cell changes.

## Results (`broad_based`, impact=`base`, fill=`base`; 300 held-out episodes, 1-hour window before settlement)

| Policy | Mean PnL | Std PnL | VaR 95% loss | ES 95% loss | Worst PnL | Mean fills | Mean max |q| |
|---|---:|---:|---:|---:|---:|---:|---:|
| HJB | -0.436 | 4.081 | 6.671 | 14.210 | -26.347 | 3.66 | 4.59 |
| Fixed spread | -0.568 | 4.424 | 6.945 | 14.861 | -27.475 | 4.76 | 5.13 |
| Inventory skew | -0.520 | 4.131 | 6.329 | 13.666 | -26.639 | 4.73 | 4.91 |
| No quotes | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| **RL (PPO, corrected reward)** | **0.004** | **0.487** | **0.000** | **0.515** | **-7.255** | **0.10** | **0.19** |

Paired bootstrap (same seeds, 5,000 draws): RL - HJB mean PnL per episode =
**+0.440**, 95% CI **[-0.010, 0.899]** -- the CI (just barely) includes
zero. RL wins on 17.7% of individual episodes (it wins by less often but
by more when it does, because it is nearly always flat).

## What this does, and does not, show

**Does not show:** that RL learned to out-quote HJB. RL's mean-PnL edge is
not statistically distinguishable from zero at 95% confidence, and even the
point estimate is explained almost entirely by participation, not skill:
RL fills **0.10** times per episode vs. HJB's **3.66**. This is a much
milder version of the same pattern as the buggy run, not a different
phenomenon -- lowering the risk-penalty coefficient 1,000x reduced, but did
not eliminate, the agent's preference for abstaining over quoting.

**Does show, tentatively:** a real risk-reduction effect. RL's PnL
variance is **~8x smaller** than HJB's (std 0.49 vs. 4.08), its expected
shortfall is **~28x smaller** (0.52 vs. 14.21), and its worst observed
episode is **~3.6x less bad** (-7.26 vs. -26.35) -- for a mean PnL that is
flat rather than HJB's mildly negative one. Put differently: in this
scenario cell, over this sample, simple reference strategies (HJB, fixed,
skew) earn a mean PnL statistically indistinguishable from (and in this
sample, somewhat below) zero while bearing substantial variance; a quick
side check across other scenario cells (`broad_based`/optimistic fill,
`sports`/base, `broad_based`/zero-impact; ~150 episodes each, not in
`results/`) found the *sign* of that mean PnL flips across independently
drawn samples for `broad_based`, i.e. the true edge for naive
market-making here is close to zero and noisy, not reliably negative --
except for `sports`, where it looked consistently negative (t ~ -2) in
that one check. RL's response -- mostly abstain, take the occasional
clearly favourable quote -- is a reasonable low-variance reaction to a
near-zero, noisy raw edge, not evidence that it has found a better way to
*provide liquidity* than HJB.

## Honest limitations

- One scenario cell, one training seed. No multi-seed variance estimate on
  the RL result itself -- unlike the paired bootstrap over held-out
  episodes, re-running training with a different seed could move the
  headline numbers more than the bootstrap CI suggests.
- The "near-zero/noisy raw edge" claim above comes from a quick,
  undocumented side check (~150 episodes per cell, printed to the
  terminal, not saved to `results/`) -- it is suggestive, not a result on
  the same footing as the main table.
- `TRAIN_GAMMA`/`TRAIN_GAMMA_T` are a judgment call tuned to ~10% of PnL
  scale; the low-fill outcome may partly reflect that choice rather than
  being environment-inherent. Worth trying: `reward_mode="raw"` (no
  shaping at all), a direct Sharpe/mean-variance objective instead of a
  fixed-coefficient penalty, and an entropy bonus or wider action-space
  bounds to discourage premature convergence to "mostly don't quote."
- `simulator/evaluate_hjb.py`'s own benchmark sweeps
  `group x impact_scenario x fill_scenario x policy` with hierarchical
  bootstrap over (ticker, replicate) pairs; this evaluation does one cell
  with a flat paired bootstrap over seeds. Extending RL training and
  evaluation across that full grid is the natural next step, not done
  here.
