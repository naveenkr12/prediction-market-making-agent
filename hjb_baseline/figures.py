"""
Reproduce Figures 1-5 from Feil & Nendel (Section 4.3/4.4) given a solved V.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import ModelParams
from hjb_solver import Grid, build_grid, solve_hjb
from quotes import build_interpolator, optimal_quotes
from monte_carlo import run_monte_carlo


def figure1_spread_vs_time(par: ModelParams, grid: Grid, V: np.ndarray, out_path: str) -> None:
    """Bid-ask spread at flat inventory (q=0) vs time, for several price levels."""
    p_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
    interp = build_interpolator(grid, V)
    t_values = grid.t

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for p in p_values:
        spreads = np.empty_like(t_values)
        for i, t in enumerate(t_values):
            pi_b, pi_a = optimal_quotes(t, p, 0.0, par, grid, interp)
            spreads[i] = pi_a - pi_b
        ax.plot(t_values, spreads, label=f"p = {p:.2f}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Spread")
    ax.set_title("Reproduction of Figure 1: bid-ask spread at flat inventory")
    ax.legend(title="Price", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure2_skew_vs_price(par: ModelParams, grid: Grid, V: np.ndarray, out_path: str) -> None:
    """Skew at t=0 vs price, for several inventory levels."""
    q_values = [-50, -25, 0, 25, 50]
    interp = build_interpolator(grid, V)
    p_values = np.linspace(grid.p.min(), grid.p.max(), 200)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for q in q_values:
        skews = np.empty_like(p_values)
        for i, p in enumerate(p_values):
            pi_b, pi_a = optimal_quotes(0.0, p, q, par, grid, interp)
            skews[i] = (pi_a + pi_b) / 2 - p
        ax.plot(p_values, skews, label=f"q = {q}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Price p")
    ax.set_ylabel("Skew")
    ax.set_title("Reproduction of Figure 2: skew at t=0 vs price")
    ax.legend(title="Inventory", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure5_pnl_histograms(mc_result: dict, out_path: str) -> None:
    """Distribution of terminal PnL (top) and terminal |inventory| (bottom)
    for the optimal vs myopic-baseline strategies, from a Monte Carlo run.
    Directly visualizes the Table 2 comparison."""
    pnl_b, pnl_o = mc_result["pnl_baseline"], mc_result["pnl_optimal"]
    qT_b, qT_o = mc_result["qT_baseline"], mc_result["qT_optimal"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    lo = min(pnl_b.min(), pnl_o.min())
    hi = max(pnl_b.max(), pnl_o.max())
    bins = np.linspace(lo, hi, 80)
    ax1.hist(pnl_b, bins=bins, alpha=0.55, label="Myopic baseline", color="tab:red", density=True)
    ax1.hist(pnl_o, bins=bins, alpha=0.55, label="Optimal (HJB)", color="tab:blue", density=True)
    ax1.set_xlabel("Terminal PnL")
    ax1.set_ylabel("Density")
    ax1.set_title("Figure 5a: terminal PnL distribution")
    ax1.legend(fontsize=8)

    bins_q = np.linspace(0, max(np.abs(qT_b).max(), np.abs(qT_o).max()), 60)
    ax2.hist(np.abs(qT_b), bins=bins_q, alpha=0.55, label="Myopic baseline", color="tab:red", density=True)
    ax2.hist(np.abs(qT_o), bins=bins_q, alpha=0.55, label="Optimal (HJB)", color="tab:blue", density=True)
    ax2.set_xlabel("Terminal |inventory|")
    ax2.set_title("Figure 5b: terminal inventory distribution")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure4_risk_aversion_sensitivity(
    base_par: ModelParams,
    gammas: list[float],
    out_path: str,
    n_t: int = 50,
    n_p: int = 51,
    n_pi: int = 71,
    n_paths: int = 3000,
    n_steps: int = 300,
    seed: int = 0,
) -> dict:
    """
    Sensitivity of the optimal quoting policy to the running-inventory risk
    aversion gamma (Table 1's gamma; gamma_T held fixed at base_par.gamma_T).

    For each gamma: solve the HJB PDE (at a reduced grid resolution -- this
    is a 2D sweep, so production resolution is not affordable in reasonable
    time), read off the flat-inventory spread at (t=0, p=0.5, q=0), and run
    a smaller Monte Carlo to get the optimal-strategy PnL std and mean |q_T|.
    Not literally Feil & Nendel's Figure 4 (which gamma/gamma_T grid they
    used for it is not fully specified in the paper text we extracted), but
    reproduces the qualitative claim it supports: higher risk aversion ->
    wider spreads -> lower PnL variance and tighter terminal inventory.
    """
    spreads, pnl_stds, abs_qTs = [], [], []
    for g in gammas:
        par = dataclasses.replace(base_par, gamma=g)
        grid = build_grid(par, n_t=n_t, n_p=n_p, n_pi=n_pi)
        V = solve_hjb(par, grid, damping=0.25, verbose=False)
        interp = build_interpolator(grid, V)
        pi_b, pi_a = optimal_quotes(0.0, 0.5, 0.0, par, grid, interp)
        spreads.append(pi_a - pi_b)

        mc = run_monte_carlo(par, grid, V, n_paths=n_paths, n_steps=n_steps, seed=seed)
        pnl_stds.append(mc["optimal"]["PnL_std"])
        abs_qTs.append(mc["optimal"]["mean_abs_qT"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(gammas, spreads, "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Risk aversion gamma")
    axes[0].set_ylabel("Spread at t=0, p=0.5, q=0")
    axes[0].set_title("Spread vs risk aversion")

    axes[1].plot(gammas, pnl_stds, "o-", color="tab:orange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Risk aversion gamma")
    axes[1].set_ylabel("PnL std (optimal strategy)")
    axes[1].set_title("PnL volatility vs risk aversion")

    axes[2].plot(gammas, abs_qTs, "o-", color="tab:green")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Risk aversion gamma")
    axes[2].set_ylabel("Mean |terminal inventory|")
    axes[2].set_title("Inventory control vs risk aversion")

    fig.suptitle("Reproduction of Figure 4: risk-aversion sensitivity (reduced-resolution grid)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"gammas": gammas, "spread": spreads, "pnl_std": pnl_stds, "mean_abs_qT": abs_qTs}


def figure3_skew_heatmap(par: ModelParams, grid: Grid, V: np.ndarray, out_path: str) -> None:
    """Skew at p=1/2 as a function of time and inventory."""
    interp = build_interpolator(grid, V)
    t_values = grid.t
    q_values = np.linspace(-par.Q * 0.8, par.Q * 0.8, 41)

    skew = np.empty((len(t_values), len(q_values)))
    for i, t in enumerate(t_values):
        for j, q in enumerate(q_values):
            pi_b, pi_a = optimal_quotes(t, 0.5, q, par, grid, interp)
            skew[i, j] = (pi_a + pi_b) / 2 - 0.5

    fig, ax = plt.subplots(figsize=(6, 4.5))
    vmax = np.nanmax(np.abs(skew))
    im = ax.pcolormesh(q_values, t_values, skew, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax, label="Skew")
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Time t")
    ax.set_title("Reproduction of Figure 3: skew at p=1/2 vs (t,q)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
