"""
Price process, Section 2.1.1 + Section 4.1 of Feil & Nendel.

Price is a logistic transform of a latent belief diffusion, engineered so
that (p_t) is a martingale on (0,1):

    dp_t = p_t (1 - p_t) sigma_bar(t, p_t) dW_t

with

    sigma_bar(t, p) := sigma(t, f^{-1}(p)) = sigma(t, ln(p/(1-p)))

and the volatility-of-latent-belief specification (Section 4.1, eq. for
sigma(t,x)):

    sigma(t, x) = sigma0 + sigma1*(t/T)^eta + sigma2/(1 + x^2)
"""

from __future__ import annotations

import numpy as np

from params import ModelParams

EPS = 1e-9  # numerical floor so log-odds / divisions never blow up at p=0,1


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigma_latent(t: np.ndarray | float, x: np.ndarray, par: ModelParams) -> np.ndarray:
    """sigma(t, x): volatility of the latent belief process L_t."""
    return par.sigma0 + par.sigma1 * (t / par.T) ** par.eta + par.sigma2 / (1 + x ** 2)


def sigma_bar(t: np.ndarray | float, p: np.ndarray, par: ModelParams) -> np.ndarray:
    """sigma_bar(t,p) = sigma(t, f^{-1}(p)): volatility of the price process (p_t)."""
    x = logit(p)
    return sigma_latent(t, x, par)


def simulate_price_paths(
    par: ModelParams,
    n_paths: int,
    n_steps: int,
    p0: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama simulation of dp_t = p(1-p) sigma_bar(t,p) dW_t,
    projected onto [EPS, 1-EPS] (Section 4.4: "using the Euler-Maruyama
    method and project prices onto [1e-3, 1-1e-3]").

    Returns (t_grid, paths) where paths has shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = par.T / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, par.T, n_steps + 1)

    p = np.full(n_paths, p0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = p

    p_lo, p_hi = 1e-3, 1 - 1e-3
    for n in range(n_steps):
        t = t_grid[n]
        vol = p * (1 - p) * sigma_bar(t, p, par)
        dW = rng.normal(0.0, sqrt_dt, size=n_paths)
        p = p + vol * dW
        p = np.clip(p, p_lo, p_hi)
        paths[:, n + 1] = p

    return t_grid, paths
