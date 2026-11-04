"""Reproduce the measurement behind calibrated_env.py's TRAIN_GAMMA / TRAIN_GAMMA_T.

Runs the frozen HJB reference policy for N episodes in the calibrated
environment and reports the raw (unweighted, coefficient=1) running
inventory**2*seconds integral and terminal inventory**2 -- the two
quantities the environment's inventory_penalty_per_contract2_second and
terminal_penalty_per_contract2 multiply -- against the realized PnL scale,
so a reward-shaping coefficient can be picked by how large a fraction of
typical PnL it should contribute, instead of being inherited from a
different model's unrelated units (see calibrated_env.py's docstring).

    python rl_agent/calibrate_reward.py --group broad_based --n-episodes 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simulator.hjb_policy import HJBReferencePolicy  # noqa: E402
from simulator.trading_env import EnvironmentConfig, TradingEnvironment  # noqa: E402


def measure(cfg, pol, n_episodes, seconds_to_close, which):
    """which='running' -> raw sum(q**2 * dt); which='terminal' -> raw q_T**2."""
    probe_cfg = EnvironmentConfig(**{
        **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
        "inventory_penalty_per_contract2_second": 1.0 if which == "running" else 0.0,
        "terminal_penalty_per_contract2": 1.0 if which == "terminal" else 0.0,
    })
    vals, pnls = [], []
    for seed in range(n_episodes):
        env = TradingEnvironment(probe_cfg)
        obs, _ = env.reset(seed=seed, options={"seconds_to_close": seconds_to_close})
        while not env.done:
            action, _ = pol.action(obs, start_elapsed=env.start_elapsed, lifetime=env.lifetime,
                                    inventory_limit=probe_cfg.inventory_limit, tick=probe_cfg.price_tick,
                                    lot=probe_cfg.quantity_lot)
            obs, *_ = env.step(action)
        vals.append(env.total_risk_penalty)
        pnls.append(float(env.account.cash - env.account.initial_cash))
    return np.array(vals), np.array(pnls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="broad_based", choices=["broad_based", "sports"])
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--seconds-to-close", type=float, default=3600.0)
    ap.add_argument("--target-fraction", type=float, default=0.1,
                     help="each penalty term, at HJB's own inventory behaviour, should average "
                          "roughly this fraction of |mean PnL| + std PnL combined scale")
    a = ap.parse_args()

    cfg = EnvironmentConfig(group=a.group)
    pol = HJBReferencePolicy()
    run_integral, pnls = measure(cfg, pol, a.n_episodes, a.seconds_to_close, "running")
    term_integral, _ = measure(cfg, pol, a.n_episodes, a.seconds_to_close, "terminal")

    scale = abs(pnls.mean()) + pnls.std(ddof=1)
    target = a.target_fraction * scale
    gamma_run = target / run_integral.mean()
    gamma_term = target / term_integral.mean()

    print(f"group={a.group} n_episodes={a.n_episodes} seconds_to_close={a.seconds_to_close}")
    print(f"HJB realized PnL: mean={pnls.mean():.4f} std={pnls.std(ddof=1):.4f} scale={scale:.4f}")
    print(f"running q^2*sec integral: mean={run_integral.mean():.1f}  -> suggested gamma ~= {gamma_run:.3e}")
    print(f"terminal q_T^2 integral:  mean={term_integral.mean():.2f} -> suggested gamma_T ~= {gamma_term:.3e}")
    print("\nThese are a starting point (each term ~target_fraction of PnL scale at HJB's own "
          "inventory discipline), not a precise derivation -- retune target_fraction and retrain "
          "if the agent still degenerates to near-zero fills, or stops managing inventory at all.")


if __name__ == "__main__":
    main()
