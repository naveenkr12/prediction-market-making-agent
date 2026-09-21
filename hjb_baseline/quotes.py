"""
Optimal quote extraction, Theorem 3.3 + Section 4.2 ("trilinear
interpolation... followed by the same maximization method").

For arbitrary (t,p,q):
    z_b(t,p,q) = (V(t,p,q) - V(t,p,q+Delta)) / Delta
    z_a(t,p,q) = (V(t,p,q) - V(t,p,q-Delta)) / Delta
    pi^{b,*} = argmax_{pi in [0,1]} Lambda^b(t,p,pi) (G^b(p,pi) - z_b)
    pi^{a,*} = argmax_{pi in [0,1]} Lambda^a(t,p,pi) (G^a(p,pi) - z_a)

V itself is interpolated (multilinearly) on the (t,p,q) grid; q is then
treated as continuous only insofar as V(.,.,q+/-Delta) is evaluated by the
same interpolator (q +/- Delta always lands back on a grid point for any
q that is itself a grid point, which is all we need for the figures).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from params import ModelParams
from hjb_solver import Grid
import intensities as I


def build_interpolator(grid: Grid, V: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator((grid.t, grid.p, grid.q), V, bounds_error=False, fill_value=None)


def _argmax_H(t: float, p: float, pi_grid: np.ndarray, z: float, par: ModelParams, side: str) -> tuple[float, float]:
    if side == "b":
        lam = I.lambda_bid(t, p, pi_grid, par)
        g = I.G_bid(p, pi_grid)
    else:
        lam = I.lambda_ask(t, p, pi_grid, par)
        g = I.G_ask(p, pi_grid)
    psi = lam * (g - z)
    idx = np.argmax(psi)
    return pi_grid[idx], par.delta * psi[idx]


def optimal_quotes(
    t: float, p: float, q: float, par: ModelParams, grid: Grid, interp: RegularGridInterpolator
) -> tuple[float, float]:
    """Returns (pi_b, pi_a); NaN on a side that is switched off at an inventory boundary."""
    Q = par.Q
    v0 = interp([[t, p, q]])[0]

    if q < Q - 1e-9:
        v_up = interp([[t, p, q + par.delta]])[0]
        z_b = (v0 - v_up) / par.delta
        pi_b, _ = _argmax_H(t, p, grid.pi, z_b, par, side="b")
    else:
        pi_b = np.nan

    if q > -Q + 1e-9:
        v_dn = interp([[t, p, q - par.delta]])[0]
        z_a = (v0 - v_dn) / par.delta
        pi_a, _ = _argmax_H(t, p, grid.pi, z_a, par, side="a")
    else:
        pi_a = np.nan

    return pi_b, pi_a


def spread_skew_grid(
    t: float, p_values: np.ndarray, q: float, par: ModelParams, grid: Grid, interp: RegularGridInterpolator
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized-by-loop spread/skew at fixed (t,q) over an array of prices."""
    spread = np.empty_like(p_values, dtype=float)
    skew = np.empty_like(p_values, dtype=float)
    for i, p in enumerate(p_values):
        pi_b, pi_a = optimal_quotes(t, p, q, par, grid, interp)
        spread[i] = pi_a - pi_b
        skew[i] = (pi_a + pi_b) / 2 - p
    return spread, skew
