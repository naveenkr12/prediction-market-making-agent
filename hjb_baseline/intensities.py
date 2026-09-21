"""
Order arrival intensities, Section 2.1.3 (general form) + Section 4.1
(concrete specification used for the numerical study) of Feil & Nendel.

    Lambda_o(t,p,pi) = A(t,p) * B_o(t,p,pi),   o in {b,a}

    A(t,p) := (A0 + A1*(exp(xi*t/T)-1)/(exp(xi)-1)) * sqrt(p(1-p))

    B^b(t,p,pi) := (2*pi/(pi+p))^nu * exp(-k(t)*(p-pi))
    B^a(t,p,pi) := (2*(1-pi)/(2-pi-p))^nu * exp(-k(t)*(pi-p))

    k(t) := k0 + (k1-k0)*(exp(kappa*t/T)-1)/(exp(kappa)-1)

NOTE ON NOTATION: the paper reuses the symbol "pi" for the market maker's
quote (pi^b, pi^a), *not* the constant 3.14159..., inside B^b/B^a -- this
is confirmed by the surrounding text ("exponential execution intensities
in the bid and ask spreads, delta^b = p - pi and delta^a = pi - p").
G^b(p,pi) = p - pi and G^a(p,pi) = pi - p, matching Section 2.3's
Hamiltonian G^o.
"""

from __future__ import annotations

import numpy as np

from params import ModelParams

EPS = 1e-12


def k_of_t(t: np.ndarray | float, par: ModelParams) -> np.ndarray | float:
    return par.k0 + (par.k1 - par.k0) * (np.exp(par.kappa * t / par.T) - 1) / (np.exp(par.kappa) - 1)


def activity(t: np.ndarray | float, p: np.ndarray, par: ModelParams) -> np.ndarray:
    """A(t,p): overall market activity level."""
    time_term = par.A0 + par.A1 * (np.exp(par.xi * t / par.T) - 1) / (np.exp(par.xi) - 1)
    return time_term * np.sqrt(np.clip(p * (1 - p), 0.0, None))


def G_bid(p: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return p - pi


def G_ask(p: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return pi - p


def shape_bid(t: np.ndarray | float, p: np.ndarray, pi: np.ndarray, par: ModelParams) -> np.ndarray:
    k = k_of_t(t, par)
    base = np.clip(2 * pi / np.clip(pi + p, EPS, None), 0.0, None)
    return base ** par.nu * np.exp(-k * (p - pi))


def shape_ask(t: np.ndarray | float, p: np.ndarray, pi: np.ndarray, par: ModelParams) -> np.ndarray:
    k = k_of_t(t, par)
    base = np.clip(2 * (1 - pi) / np.clip(2 - pi - p, EPS, None), 0.0, None)
    return base ** par.nu * np.exp(-k * (pi - p))


def lambda_bid(t: np.ndarray | float, p: np.ndarray, pi: np.ndarray, par: ModelParams) -> np.ndarray:
    return activity(t, p, par) * shape_bid(t, p, pi, par)


def lambda_ask(t: np.ndarray | float, p: np.ndarray, pi: np.ndarray, par: ModelParams) -> np.ndarray:
    return activity(t, p, par) * shape_ask(t, p, pi, par)
