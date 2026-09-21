"""
Section 4.4: Monte Carlo comparison of the optimal quoting strategy against
a myopic benchmark that maximizes instantaneous expected mark-to-market
profit (i.e., the same H^o maximization but with z = 0 -- no inventory
value/continuation-value term, so it ignores inventory risk entirely, and
is therefore also independent of q -- exactly "instantaneous expected
mark-to-market profit"). This interpretation is not spelled out
symbol-by-symbol in the paper's prose, but it is the natural reading of
"myopic ... maximizes the instantaneous expected mark-to-market profit"
given the model's own H^o(t,p;z) structure, and it reproduces Table 2
closely (see output/data/table2_comparison.csv).

Only the optimal strategy is subject to the inventory bound Q (footnote 4
in the paper: "the myopic benchmark is simulated without an inventory
bound").
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from params import ModelParams
from hjb_solver import Grid
import intensities as I
import price_process as PP


def _argmax_H_batch(
    t: float, p: np.ndarray, pi_grid: np.ndarray, z: np.ndarray, par: ModelParams, side: str
) -> np.ndarray:
    if side == "b":
        lam = I.lambda_bid(t, p[:, None], pi_grid[None, :], par)
        g = I.G_bid(p[:, None], pi_grid[None, :])
    else:
        lam = I.lambda_ask(t, p[:, None], pi_grid[None, :], par)
        g = I.G_ask(p[:, None], pi_grid[None, :])
    psi = lam * (g - z[:, None])
    idx = np.argmax(psi, axis=1)
    return pi_grid[idx]


def _batch_optimal_quotes(
    t: float, p: np.ndarray, q: np.ndarray, par: ModelParams, grid: Grid, interp: RegularGridInterpolator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(p)
    t_col = np.full(n, t)
    q_clip = np.clip(q, grid.q.min(), grid.q.max())
    v0 = interp(np.column_stack([t_col, p, q_clip]))

    mask_b = q < par.Q - 1e-9
    mask_a = q > -par.Q + 1e-9

    q_up = np.clip(q_clip + par.delta, grid.q.min(), grid.q.max())
    q_dn = np.clip(q_clip - par.delta, grid.q.min(), grid.q.max())
    v_up = interp(np.column_stack([t_col, p, q_up]))
    v_dn = interp(np.column_stack([t_col, p, q_dn]))

    z_b = (v0 - v_up) / par.delta
    z_a = (v0 - v_dn) / par.delta

    pi_b = np.full(n, np.nan)
    pi_a = np.full(n, np.nan)
    if mask_b.any():
        pi_b[mask_b] = _argmax_H_batch(t, p[mask_b], grid.pi, z_b[mask_b], par, side="b")
    if mask_a.any():
        pi_a[mask_a] = _argmax_H_batch(t, p[mask_a], grid.pi, z_a[mask_a], par, side="a")
    return pi_b, pi_a


def _myopic_quotes(t: float, p: np.ndarray, pi_grid: np.ndarray, par: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros_like(p)
    pi_b = _argmax_H_batch(t, p, pi_grid, zero, par, side="b")
    pi_a = _argmax_H_batch(t, p, pi_grid, zero, par, side="a")
    return pi_b, pi_a


def run_monte_carlo(
    par: ModelParams,
    grid: Grid,
    V: np.ndarray,
    n_paths: int = 10_000,
    n_steps: int = 500,
    p0: float = 0.5,
    seed: int = 0,
) -> dict:
    interp = RegularGridInterpolator((grid.t, grid.p, grid.q), V, bounds_error=False, fill_value=None)

    rng = np.random.default_rng(seed)
    dt = par.T / n_steps
    t_grid = np.linspace(0.0, par.T, n_steps + 1)

    p = np.full(n_paths, p0)
    q_opt = np.zeros(n_paths)
    x_opt = np.zeros(n_paths)
    q_myo = np.zeros(n_paths)
    x_myo = np.zeros(n_paths)

    p_lo, p_hi = 1e-3, 1 - 1e-3

    for n in range(n_steps):
        t = t_grid[n]

        # -- optimal strategy quotes (inventory-aware, bounded) ---------
        pi_b_opt, pi_a_opt = _batch_optimal_quotes(t, p, q_opt, par, grid, interp)
        pi_b_safe = np.nan_to_num(pi_b_opt, nan=0.0)
        pi_a_safe = np.nan_to_num(pi_a_opt, nan=0.0)
        lam_b_opt = np.where(np.isnan(pi_b_opt), 0.0, I.lambda_bid(t, p, pi_b_safe, par))
        lam_a_opt = np.where(np.isnan(pi_a_opt), 0.0, I.lambda_ask(t, p, pi_a_safe, par))
        n_buy_opt = rng.poisson(np.clip(lam_b_opt, 0, None) * dt)
        n_sell_opt = rng.poisson(np.clip(lam_a_opt, 0, None) * dt)
        # respect the inventory bound: clip fills that would breach it
        max_buy = np.floor(np.clip((par.Q - q_opt) / par.delta, 0, None)).astype(int)
        max_sell = np.floor(np.clip((q_opt + par.Q) / par.delta, 0, None)).astype(int)
        n_buy_opt = np.minimum(n_buy_opt, max_buy)
        n_sell_opt = np.minimum(n_sell_opt, max_sell)
        x_opt += par.delta * pi_a_safe * n_sell_opt - par.delta * pi_b_safe * n_buy_opt
        q_opt += par.delta * (n_buy_opt - n_sell_opt)

        # -- myopic strategy quotes (no inventory awareness, no bound) --
        pi_b_myo, pi_a_myo = _myopic_quotes(t, p, grid.pi, par)
        lam_b_myo = I.lambda_bid(t, p, pi_b_myo, par)
        lam_a_myo = I.lambda_ask(t, p, pi_a_myo, par)
        n_buy_myo = rng.poisson(np.clip(lam_b_myo, 0, None) * dt)
        n_sell_myo = rng.poisson(np.clip(lam_a_myo, 0, None) * dt)
        x_myo += par.delta * pi_a_myo * n_sell_myo - par.delta * pi_b_myo * n_buy_myo
        q_myo += par.delta * (n_buy_myo - n_sell_myo)

        # -- advance the shared price path --------------------------------
        vol = p * (1 - p) * PP.sigma_bar(t, p, par)
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        p = np.clip(p + vol * dW, p_lo, p_hi)

    Y = rng.binomial(1, p)  # terminal settlement, shared across both strategies
    pnl_opt = x_opt + q_opt * Y
    pnl_myo = x_myo + q_myo * Y

    def stats(pnl: np.ndarray, q_t: np.ndarray) -> dict:
        var5 = -np.quantile(pnl, 0.05)
        es5 = -pnl[pnl <= np.quantile(pnl, 0.05)].mean()
        return {
            "PnL_mean": pnl.mean(),
            "PnL_std": pnl.std(),
            "mean_abs_qT": np.abs(q_t).mean(),
            "VaR_5pct": var5,
            "ES_5pct": es5,
        }

    return {
        "baseline": stats(pnl_myo, q_myo),
        "optimal": stats(pnl_opt, q_opt),
        "pnl_baseline": pnl_myo,
        "pnl_optimal": pnl_opt,
        "qT_baseline": q_myo,
        "qT_optimal": q_opt,
    }
