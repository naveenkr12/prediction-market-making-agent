"""Independent sequential posterior calculator for validation (V2).

Deliberately written from first principles with the standard library only; it
does not import `kernels.py` or NumPy. Given the initial probability and the
logged public signals of an M-family episode, it rebuilds Bayes' rule

    log P(Theta=1 | signals) - log P(Theta=0 | signals)
        = logit(p0) + sum_i [log P(signal_i | Theta=1) - log P(signal_i | Theta=0)]

from each signal's two class-conditional likelihoods (densities for Gaussian
background observations, probabilities for discrete signs). Signal laws:

- background: X | Theta ~ Normal((2Theta-1) v/2, v), v = s*sigma_c^2*h.
- flow, P(buy | Theta=1) = alpha + (1-alpha)u, P(buy | Theta=0) = (1-alpha)u:
    law "fg":       1 - alpha = max(exp(-eta c), 1e-12), u logged per trade;
    law "fg_half":  alpha = tanh(beta c/2), u = 1/2, so P(buy|1) = 1/(1+e^{-beta c});
    law "constant": alpha = alpha0, u logged;
    law "uninformed" (F0): identical likelihoods, no update.
- jump: P(+ | Theta=1) = 1/(1+e^{-a}), P(+ | Theta=0) = 1/(1+e^{a}).
"""
from __future__ import annotations

import math

FG_ONE_MINUS_ALPHA_FLOOR = 1e-12


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0/(1.0+math.exp(-x))
    e = math.exp(x)
    return e/(1.0+e)


def _log_logistic(x: float) -> float:
    """log(1/(1+e^{-x})) without overflow."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x-math.log1p(math.exp(x))


def _gaussian_log_density(x: float, mean: float, variance: float) -> float:
    return -0.5*math.log(2*math.pi*variance)-(x-mean)**2/(2*variance)


def _sign(signal: dict) -> int:
    s = int(signal["sign"])
    if s not in (-1, 1):
        raise AssertionError("Logged signs must be -1 or +1")
    return s


def class_log_likelihoods(signal: dict) -> tuple[float, float]:
    """(log P(signal | Theta=1), log P(signal | Theta=0)) for one logged signal."""
    kind = signal["kind"]
    if kind == "background":
        x, v = float(signal["x"]), float(signal["variance"])
        if v < 0:
            raise ValueError("Negative background variance")
        if v == 0:
            if x != 0:
                raise ValueError("Nonzero observation from a zero-variance experiment")
            return 0.0, 0.0
        return _gaussian_log_density(x, v/2, v), _gaussian_log_density(x, -v/2, v)
    if kind == "flow":
        law, s = signal["law"], _sign(signal)
        if law == "uninformed":
            return 0.0, 0.0
        if law == "fg_half":
            x = float(signal["beta"])*float(signal["size"])
            # P(buy|1) = (1+alpha)/2 = 1/(1+e^{-x}); P(buy|0) = (1-alpha)/2 = 1/(1+e^{x}).
            return (_log_logistic(x), _log_logistic(-x)) if s == 1 else (_log_logistic(-x), _log_logistic(x))
        u = float(signal["u"])
        if not 0 < u < 1:
            raise ValueError("Uninformed YES rate must be in (0,1)")
        if law == "fg":
            one_minus = max(math.exp(-float(signal["eta"])*float(signal["size"])), FG_ONE_MINUS_ALPHA_FLOOR)
        elif law == "constant":
            one_minus = 1.0-float(signal["alpha"])
        else:
            raise ValueError(f"Unknown flow law {law}")
        if s == 1:
            return math.log(1.0-one_minus*(1.0-u)), math.log(one_minus*u)
        return math.log(one_minus*(1.0-u)), math.log(1.0-one_minus*u)
    if kind == "jump":
        a, s = float(signal["magnitude"]), _sign(signal)
        return (_log_logistic(a), _log_logistic(-a)) if s == 1 else (_log_logistic(-a), _log_logistic(a))
    raise ValueError(f"Unknown signal kind {kind}")


def posterior_log_odds_path(initial_probability: float, signals: list[dict]) -> list[float]:
    """Log odds after each logged signal, accumulated sequentially in log space."""
    p0 = float(initial_probability)
    if not 0 < p0 < 1:
        raise ValueError("Initial probability must be in (0,1)")
    log_odds = math.log(p0)-math.log1p(-p0)
    path = []
    for signal in signals:
        one, zero = class_log_likelihoods(signal)
        log_odds += one-zero
        path.append(log_odds)
    return path


def posterior_probability_path(initial_probability: float, signals: list[dict]) -> list[float]:
    return [_logistic(x) for x in posterior_log_odds_path(initial_probability, signals)]
