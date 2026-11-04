"""
Step 4: the race. HJB, myopic and the trained RL agent play the SAME 10,000
fresh games (seed never used in training), and we fill in Paper 1's Table 2.

    python rl_agent/evaluate.py
    python rl_agent/evaluate.py --model rl_agent/models/ppo_paper1.zip --paths 10000

Outputs: printed table, rl_agent/results/table2_rl_vs_hjb.csv and
rl_agent/results/pnl_and_inventory.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from paper1_env import simulate, table2_stats  # noqa: E402
from baselines import hjb_policy, myopic_policy, rl_policy  # noqa: E402

COLS = ["PnL mean", "PnL std", "mean |q_T|", "VaR 5%", "ES 5%", "objective"]
PAPER = {"Optimal (paper)": [12.39, 10.34, 15.23, 4.20, 9.68, None],
         "Baseline (paper)": [12.47, 28.11, 49.37, 32.41, 40.50, None]}


def plot(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"HJB": "#1f6fb4", "Myopic": "#c0504d", "RL (PPO)": "#2e9b5f"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        for ax, key, lo, hi in [(axes[0], "pnl", -80, 100), (axes[1], "q_T", -120, 120)]:
            x = np.linspace(lo, hi, 600)
            data = r[key]
            if np.std(data) > 1e-9:
                ax.plot(x, gaussian_kde(data)(x), lw=2.2, color=colors[name], label=name)
    axes[0].set(title="Final profit and loss", xlabel="PnL", ylabel="Density")
    axes[1].set(title="Terminal inventory", xlabel="$q_T$", ylabel="Density")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Paper 1 market: HJB optimum vs myopic vs RL agent (same 10,000 paths)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "models" / "ppo_paper1.zip"))
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=777_777)  # held out: never used in training
    a = ap.parse_args()

    model = PPO.load(a.model, device="cpu")
    players = [("HJB", hjb_policy(), True), ("Myopic", myopic_policy(), False), ("RL (PPO)", rl_policy(model), True)]
    results, stats = {}, {}
    for name, pol, bound in players:
        results[name] = simulate(pol, a.paths, a.seed, enforce_bound=bound)
        stats[name] = table2_stats(results[name])
        print(f"{name}: done", flush=True)

    print(f"\n{'':<20}" + "".join(f"{c:>11}" for c in COLS))
    for k, v in PAPER.items():
        print(f"{k:<20}" + "".join(f"{x:>11.2f}" if x is not None else f"{'-':>11}" for x in v))
    for k, s in stats.items():
        print(f"{k:<20}" + "".join(f"{s[c]:>11.2f}" for c in COLS))

    diff = results["RL (PPO)"]["objective"] - results["HJB"]["objective"]
    print(f"\nRL minus HJB, mean objective on identical paths: {diff.mean():+.2f} "
          f"(standard error {diff.std() / np.sqrt(len(diff)):.2f}). "
          "HJB is optimal here, so ~0 or slightly negative is success.")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / "table2_rl_vs_hjb.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy"] + COLS)
        for k, v in PAPER.items():
            w.writerow([k] + v)
        for k, s in stats.items():
            w.writerow([k] + [round(s[c], 4) for c in COLS])
    plot(results, out / "pnl_and_inventory.png")
    print(f"Saved {out / 'table2_rl_vs_hjb.csv'} and {out / 'pnl_and_inventory.png'}")


if __name__ == "__main__":
    main()
