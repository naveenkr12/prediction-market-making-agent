# RL agent on the Paper 1 benchmark

Goal: train a model-free RL quoting agent (PPO) in Feil & Nendel's own market
model and compare it with their HJB optimum on Table 2 metrics. HJB is
provably optimal in this world, so success means RL gets close to it. This
validates the RL machinery before it moves into `../simulator/`.

| File | Purpose |
|---|---|
| `paper1_env.py` | Paper 1 market as a vectorised game; reuses `../hjb_baseline/` formulas unchanged |
| `baselines.py` | HJB (saved production V) and myopic players, via the original `monte_carlo.py` functions |
| `check_env.py` | HJB inside the game must reproduce Table 2 (~12.35-12.39 mean PnL) |
| `train_ppo.py` | PPO training; saves `models/ppo_paper1.zip` and `models/logs/progress.csv` |
| `evaluate.py` | HJB vs myopic vs RL on the same 10,000 held-out paths; table, CSV and chart in `results/` |

## Design (so the race is fair)

- Observation: `[t/T, 2p-1, q/Q]`, the same state the HJB policy uses.
- Action: two numbers in [-1, 1] mapped to bid/ask distances from p in [-0.03, 0.09].
- Reward: the paper's objective. Training uses its reduced form
  (spread captured minus `gamma q^2 varsigma^2 dt`, terminal `-gamma_T q_T^2 p_T(1-p_T)`);
  the dropped mark-to-market term `q dp` has zero expectation for every policy.
- 1,000 decisions per episode, inventory limit 100, 10-contract fills, as in the team reproduction.
- Price path and settlement depend only on the seed, so all players face identical markets.
  The evaluation seed (777777) is never used in training.
- Myopic is simulated without an inventory bound, as in the paper's footnote 4.

## Run

```bash
pip install -r data_pipeline/requirements.txt -r rl_agent/requirements-rl.txt
python rl_agent/check_env.py
python rl_agent/train_ppo.py
python rl_agent/evaluate.py
```

## Development check (2026-10-31, short run, not the final result)

400-path environment check: HJB 12.40 mean PnL (paper 12.39), CHECK PASSED.
6M-step PPO run (3 minutes), 2,000 held-out paths:

| | PnL mean | PnL std | mean abs q_T | VaR 5% | ES 5% | objective |
|---|---:|---:|---:|---:|---:|---:|
| HJB | 12.25 | 10.37 | 15.04 | 4.15 | 9.04 | 12.24 |
| Myopic | 12.89 | 34.95 | 56.93 | 44.01 | 62.76 | 11.16 |
| RL (PPO) | 11.86 | 13.77 | 21.86 | 11.29 | 18.57 | 11.91 |

RL was still improving when stopped; the default run is 20M steps.
