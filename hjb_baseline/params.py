"""
Model parameters, exactly Table 1 of Feil & Nendel (2026),
"Optimal Market Making in Prediction Markets".

Used as the default parameterization for reproducing their published
Figures 1-4 and Table 2. Swap these for data-calibrated values (see the
Kalshi data pipeline) once the paper's own numbers have been reproduced --
that is a *separate*, later step, not part of baseline verification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    T: float = 1.0  # horizon

    # Volatility: sigma(t,x) = sigma0 + sigma1*(t/T)^eta + sigma2/(1+x^2)
    sigma0: float = 0.6
    sigma1: float = 0.3
    sigma2: float = 0.1
    eta: float = 3.0

    # Trading activity: A(t,p) = (A0 + A1*(exp(xi*t/T)-1)/(exp(xi)-1)) * sqrt(p(1-p))
    A0: float = 100.0
    A1: float = 150.0
    xi: float = 2.0

    # Intensity shape: B(t,p,pi) with boundary decay nu and spread sensitivity k(t)
    nu: float = 1.0
    k0: float = 35.0
    k1: float = 50.0
    kappa: float = 1.5

    # Risk and inventory
    gamma: float = 4e-3       # running inventory penalty
    gamma_T: float = 1e-3     # terminal inventory (settlement) penalty
    Q: float = 100.0          # inventory limit
    delta: float = 10.0       # trade size (paper's "Delta")


DEFAULT = ModelParams()
