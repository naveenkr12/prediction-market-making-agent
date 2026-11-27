"""Stress test analysis: RL v3 vs HJB in Rithvik's information worlds (M0-M3, snipers off/on).

Reads the output of
    STRESS_V3=1 python -m simulator.information.evaluate --output-dir rl_agent/stress_test/results ...
and produces:
    v3_vs_hjb.csv        paired difference v3 - HJB per world, with a 95% range
    stress_test_pnl.png  mean profit per world for HJB, RL v3 and inventory skew
    v3_minus_hjb.png     the paired difference v3 - HJB, with 95% ranges

Paired = both players on the identical episodes (same contract, same seed, same prices).
The 95% range is a hierarchical bootstrap: resample contracts, then replicates within each
contract, the same scheme Rithvik uses for his risk tables.

    python rl_agent/stress_test/analyze.py
    python rl_agent/stress_test/analyze.py --results rl_agent/stress_test/results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORLDS = ["M0", "M1", "M2", "M3"]
WORLD_LABELS = {"M0": "Base", "M1": "Informed\ntraders", "M2": "News\njumps", "M3": "Both"}
SCOPES = {"sports": "Sports", "broad_based": "Economics"}
SNIPERS = {"S0": "snipers off", "S1": "snipers on"}
# Series order is fixed; colours are validated categorical slots 1-3 (CVD-safe), plus marker shapes.
SERIES = [("hjb", "HJB (Paper 1)", "#257de0", "o"),
          ("v3", "RL v3", "#f47230", "s"),
          ("skew", "Inventory skew", "#00b894", "^")]
INK, INK_2, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def paired_bootstrap(diff: np.ndarray, draws: int, rng) -> np.ndarray:
    """diff[t, r] = v3 - HJB for contract t, replicate r. Returns bootstrap means."""
    n_t, n_r = diff.shape
    t_idx = rng.integers(0, n_t, size=(draws, n_t))
    r_idx = rng.integers(0, n_r, size=(draws, n_t, n_r))
    return diff[t_idx[:, :, None], r_idx].mean(axis=(1, 2))


def paired_table(ep: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for scope in SCOPES:
        for level in SNIPERS:
            for world in WORLDS:
                sub = ep[(ep.group == scope) & (ep.snipe_level == level) & (ep.arm == world)]
                wide = sub.pivot_table(index=["ticker", "replicate"], columns="policy",
                                       values="realized_pnl", aggfunc="first")
                wide = wide.dropna(subset=["hjb", "v3"])
                if wide.empty:
                    continue
                diff = (wide["v3"] - wide["hjb"]).unstack("replicate").dropna(axis=0).to_numpy()
                boot = paired_bootstrap(diff, draws, rng)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append({"market": SCOPES[scope], "snipers": SNIPERS[level], "world": world,
                             "contracts": diff.shape[0], "replicates": diff.shape[1],
                             "hjb_mean": wide["hjb"].mean(), "v3_mean": wide["v3"].mean(),
                             "v3_minus_hjb": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                             "clear_difference": "yes" if (lo > 0 or hi < 0) else "no"})
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.axhline(0, color=INK_2, lw=1, zorder=1)


def plot_levels(risk: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey="row")
    x = np.arange(len(WORLDS))
    label_ends = []
    for i, scope in enumerate(SCOPES):
        for j, level in enumerate(SNIPERS):
            ax = axes[i, j]
            style(ax)
            ends = []
            for k, (policy, label, color, marker) in enumerate(SERIES):
                d = risk[(risk.scope == scope) & (risk.snipe_level == level) & (risk.policy == policy)]
                d = d.set_index("arm").reindex(WORLDS)
                if d.mean_pnl.isna().all():
                    continue
                xs = x + (k - 1) * 0.06
                ax.fill_between(xs, d.mean_pnl_bootstrap95_low, d.mean_pnl_bootstrap95_high,
                                color=color, alpha=0.12, lw=0, zorder=2)
                ax.plot(xs, d.mean_pnl, color=color, lw=2, marker=marker, ms=8, zorder=3,
                        mec="white", mew=2, label=label)
                ends.append([d.mean_pnl.iloc[-1], label, xs[-1]])
            label_ends.append((ax, ends))
            ax.set_xticks(x, [WORLD_LABELS[w] for w in WORLDS])
            ax.set_xlim(-0.3, len(WORLDS) - 0.3 + 0.9)
            ax.set_title(f"{SCOPES[scope]}, {SNIPERS[level]}", fontsize=11, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("Mean profit per market-hour ($)", color=INK_2, fontsize=9.5)
    for ax, ends in label_ends:          # direct labels, spread so they never collide
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        ends.sort(key=lambda e: e[0])
        ys = [e[0] for e in ends]
        for n in range(1, len(ys)):
            ys[n] = max(ys[n], ys[n - 1] + gap)
        for yv, (_, label, xe) in zip(ys, ends):
            ax.annotate(label, (xe, yv), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=3, frameon=False,
               fontsize=9.5)
    fig.suptitle("Stress test: profit as the market gets tougher (shaded = 95% range)",
                 x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_difference(table: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(WORLDS))[::-1]
    for ax, market in zip(axes, SCOPES.values()):
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.axvline(0, color=INK_2, lw=1)
        for k, (snip, color, marker) in enumerate((("snipers off", INK_2, "o"),
                                                   ("snipers on", INK, "s"))):
            d = table[(table.market == market) & (table.snipers == snip)].set_index("world").reindex(WORLDS)
            ys = y + (0.12 if k == 0 else -0.12)
            ax.errorbar(d.v3_minus_hjb, ys, xerr=[d.v3_minus_hjb - d.ci95_low, d.ci95_high - d.v3_minus_hjb],
                        fmt=marker, color=color, ms=8, mec="white", mew=2, elinewidth=2, capsize=0, label=snip)
        ax.set_yticks(y, [WORLD_LABELS[w].replace("\n", " ") for w in WORLDS])
        ax.set_title(market, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("RL v3 minus HJB, profit per market-hour ($)", color=INK_2, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, frameon=False,
               fontsize=9.5)
    fig.suptitle("Paired comparison on identical episodes: right of zero = RL v3 does better",
                 x=0.01, y=0.99, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    res = Path(a.results)

    ep = pd.read_csv(res / "episode_results.csv")
    ep = ep[ep.arm.isin(WORLDS) & ep.roster.eq("R_star")]
    if "v3" not in set(ep.policy):
        raise SystemExit("No v3 rows found: run evaluate with STRESS_V3=1 first.")
    risk = pd.read_csv(res / "risk_summary.csv")
    risk = risk[risk.roster.eq("R_star") & risk.arm.isin(WORLDS)]

    table = paired_table(ep, a.draws, a.seed)
    table.to_csv(res / "v3_vs_hjb.csv", index=False)
    plot_levels(risk, res / "stress_test_pnl.png")
    plot_difference(table, res / "v3_minus_hjb.png")

    show = table.copy()
    for c in ("hjb_mean", "v3_mean", "v3_minus_hjb", "ci95_low", "ci95_high"):
        show[c] = show[c].round(2)
    show["world"] = show.world.map(lambda w: WORLD_LABELS[w].replace("\n", " "))
    print(show[["market", "snipers", "world", "hjb_mean", "v3_mean", "v3_minus_hjb", "ci95_low", "ci95_high",
                "clear_difference"]].to_string(index=False))
    print(f"\nSaved {res / 'v3_vs_hjb.csv'}, {res / 'stress_test_pnl.png'}, {res / 'v3_minus_hjb.png'}")


if __name__ == "__main__":
    main()
