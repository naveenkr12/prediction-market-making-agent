"""Paired RL vs. HJB vs. simple-baseline evaluation in the calibrated simulator.

Mirrors simulator/evaluate_hjb.py's scorecard (realized PnL, VaR/ES,
inventory control; simulator.benchmark_metrics.pnl_statistics) so the RL
number sits on the *same* scale as the capstone's existing HJB benchmark --
but over a plain paired bootstrap on held-out seeds, not the full
ticker x impact x fill scenario grid evaluate_hjb.py sweeps. This is a first
cut, one scenario cell at a time, not the capstone's final robustness study.

    python rl_agent/evaluate_capstone.py --model rl_agent/models/ppo_capstone_broad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stable_baselines3 import PPO  # noqa: E402

from calibrated_env import CapstoneQuotingEnv, _obs_vector  # noqa: E402
from simulator.benchmark_metrics import pnl_statistics  # noqa: E402
from simulator.execution import QuoteAction  # noqa: E402
from simulator.hjb_policy import HJBReferencePolicy  # noqa: E402
from simulator.policies import quote_policy  # noqa: E402
from simulator.trading_env import EnvironmentConfig, TradingEnvironment  # noqa: E402

SECONDS_TO_CLOSE = 3600.0


def run_episode(policy_name, cfg, seed, *, hjb=None, model=None):
    env = TradingEnvironment(cfg)
    obs, _ = env.reset(seed=seed, options={"seconds_to_close": SECONDS_TO_CLOSE})
    while not env.done:
        if policy_name == "hjb":
            action, _ = hjb.action(obs, start_elapsed=env.start_elapsed, lifetime=env.lifetime,
                                    inventory_limit=cfg.inventory_limit, tick=cfg.price_tick, lot=cfg.quantity_lot)
        elif policy_name == "fixed":
            action = quote_policy(obs, inventory_skew=0.0, tick=cfg.price_tick)
        elif policy_name == "skew":
            action = quote_policy(obs, inventory_skew=0.0005, tick=cfg.price_tick)
        elif policy_name == "none":
            action = QuoteAction()
        elif policy_name == "rl":
            vec = _obs_vector(env, obs)
            act, _ = model.predict(vec, deterministic=True)
            shim = CapstoneQuotingEnv.__new__(CapstoneQuotingEnv)
            shim.config, shim._raw_obs = cfg, obs
            action, _ = shim._action_to_quotes(act)
        else:
            raise ValueError(policy_name)
        obs, _, _, _, info = env.step(action)
    realized_pnl = float(env.account.cash - env.account.initial_cash)
    max_abs_inventory = max(abs(float(r["inventory"])) for r in env.ledger)
    return {
        "policy": policy_name, "seed": seed, "ticker": env.dynamics.ticker,
        "realized_pnl": realized_pnl, "initial_cash": cfg.initial_cash,
        "maker_fills": sum(1 for r in env.ledger if r["kind"] == "fill"),
        "max_abs_inventory": max_abs_inventory,
        "abs_terminal_inventory": abs(float(env.account.inventory)),
        "fees": float(env.account.total_fees),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models" / "ppo_capstone_broad"))
    ap.add_argument("--group", default="broad_based", choices=["broad_based", "sports"])
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=10_000)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    cfg = EnvironmentConfig(group=a.group)
    hjb = HJBReferencePolicy()
    model = PPO.load(a.model)
    seeds = list(range(a.seed_start, a.seed_start + a.n_episodes))

    rows = []
    for policy_name in ("hjb", "fixed", "skew", "none", "rl"):
        for seed in seeds:
            rows.append(run_episode(policy_name, cfg, seed, hjb=hjb, model=model))
    data = pd.DataFrame(rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / f"episodes_{a.group}.csv", index=False)

    summary = []
    for policy_name, frame in data.groupby("policy"):
        stats = pnl_statistics(frame.realized_pnl.to_numpy())
        stats.update(policy=policy_name, group=a.group,
                     mean_fills=float(frame.maker_fills.mean()),
                     mean_max_abs_inventory=float(frame.max_abs_inventory.mean()),
                     mean_abs_terminal_inventory=float(frame.abs_terminal_inventory.mean()),
                     mean_fees=float(frame.fees.mean()))
        summary.append(stats)
    summary = pd.DataFrame(summary).set_index("policy")
    cols = ["episodes", "mean_pnl", "std_pnl", "var95_loss", "es95_loss", "worst_pnl",
            "mean_max_abs_inventory", "mean_abs_terminal_inventory", "mean_fills", "mean_fees"]
    print(summary[cols].round(4).to_string())
    summary[cols].to_csv(out / f"summary_{a.group}.csv")

    # Paired bootstrap on RL - HJB mean PnL difference (seeds are the paired unit:
    # every policy sees the identical price/settlement path at a given seed).
    pivot = data.pivot(index="seed", columns="policy", values="realized_pnl")
    diff = (pivot["rl"] - pivot["hjb"]).to_numpy()
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, size=(a.bootstrap_draws, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    win_frac = float((diff > 0).mean())
    print(f"\nRL - HJB mean PnL per episode: {diff.mean():.4f}  "
          f"95% bootstrap CI [{lo:.4f}, {hi:.4f}]  RL-wins-episode fraction: {win_frac:.3f}")
    with open(out / f"rl_vs_hjb_{a.group}.txt", "w") as f:
        f.write(f"episodes={len(diff)} mean_diff={diff.mean():.6f} ci95=[{lo:.6f},{hi:.6f}] "
                f"win_fraction={win_frac:.4f}\n")


if __name__ == "__main__":
    main()
