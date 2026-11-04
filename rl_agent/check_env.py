"""
Step 2 sanity check: do HJB and myopic, played inside paper1_env, reproduce
Paper 1's Table 2? If HJB lands near 12.35-12.39, the game is correct and
any RL result later is measured on the right yardstick.

    python rl_agent/check_env.py              # full: 10,000 paths (a few minutes)
    python rl_agent/check_env.py --paths 500  # quick look
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper1_env import simulate, table2_stats  # noqa: E402
from baselines import hjb_policy, myopic_policy  # noqa: E402

PAPER = {"Optimal (paper)": [12.39, 10.34, 15.23, 4.20, 9.68],
         "Baseline (paper)": [12.47, 28.11, 49.37, 32.41, 40.50]}
TEAM = {"Optimal (team repro)": [12.35, 10.36, 15.13, 4.40, 9.32],
        "Baseline (team repro)": [13.00, 35.78, 59.29, None, None]}
COLS = ["PnL mean", "PnL std", "mean |q_T|", "VaR 5%", "ES 5%"]


def fmt_row(name, vals):
    cells = "".join(f"{v:>11.2f}" if v is not None else f"{'-':>11}" for v in vals)
    return f"{name:<24}{cells}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    rows = {}
    for name, pol, bound in [("HJB (this env)", hjb_policy(), True),
                             ("Myopic (this env)", myopic_policy(), False)]:  # paper: myopic unbounded
        t0 = time.time()
        s = table2_stats(simulate(pol, a.paths, a.seed, enforce_bound=bound, n_steps=a.steps))
        rows[name] = s
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'':<24}" + "".join(f"{c:>11}" for c in COLS))
    for d in (PAPER, TEAM):
        for k, v in d.items():
            print(fmt_row(k, v))
    for k, s in rows.items():
        print(fmt_row(k, [s[c] for c in COLS]))
    se = rows["HJB (this env)"]["PnL std"] / a.paths ** 0.5
    print(f"\nHJB mean PnL Monte Carlo standard error: {se:.2f}")
    print(f"HJB mean objective (PnL minus penalties): {rows['HJB (this env)']['objective']:.2f}"
          "  [saved V(0, 0.5, 0) = 12.22]")
    gap = abs(rows["HJB (this env)"]["PnL mean"] - 12.35)
    print("CHECK PASSED" if gap < max(3 * se, 0.3) else "CHECK FAILED: HJB mean is off, do not train yet")


if __name__ == "__main__":
    main()
