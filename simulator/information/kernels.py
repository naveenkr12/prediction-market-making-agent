"""Pure latent log-odds update kernels and samplers (spec Section 4); no I/O.

Latent L is float64 log odds of the hidden outcome Theta in {0,1}; p = expit(L)
is never clipped here (use `display_probability` only for observations,
quotes and HJB inputs). Every function accepts NumPy scalars or arrays and
consumes only the uniforms/increments passed in, so callers control streams.

Global sign convention: an execution sign is +1 for a YES taker and -1 for a
NO taker; uninformed signs are S_u = 2*1{V_u < u} - 1 everywhere.

Families (Section 4):
- M ("Theta-known" generators): signals are drawn conditional on Theta;
  updates are exact Bayesian log-likelihood ratios (LLR), p-independent.
- Theta-free (L family): signals are drawn from the exact marginal law given
  the current p, using the same LLR update, so p is a martingale.
- Legacy-law: driftless Gaussian background, fixed impact (FL), fair-coin jumps.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..price_impact import displayed_probability

ALPHA_FLOOR_ONE_MINUS = 1e-12       # alpha <= 1 - 1e-12  <=>  1 - alpha >= 1e-12
TTR_BUCKETS = ("B0", "B1", "B2", "B3")
TTR_EDGES_SECONDS = (0.0, 3600.0, 10800.0, 86400.0, np.inf)   # [lo, hi) per bucket


# ----------------------------------------------------------------- probability
def probability(log_odds):
    """Unclipped expit(L): use for filtering, LLRs, sniping crossing and validation."""
    return expit(np.asarray(log_odds, dtype=float))


def display_probability(log_odds):
    """Clipped interior float probability (legacy `displayed_probability`)."""
    return displayed_probability(np.asarray(log_odds, dtype=float))


def saturated(log_odds):
    """True where float expit(L) has reached exactly 0 or 1 (counted, never altered)."""
    p = probability(log_odds)
    return (p == 0.0) | (p == 1.0)


def check_signs(signs):
    """Assert every realized execution/jump sign is in {-1, +1}; return int8 array."""
    s = np.asarray(signs)
    if s.size and not np.isin(s, (-1, 1)).all():
        raise AssertionError("Realized signs must be in {-1, +1}")
    return s.astype(np.int8)


def ttr_bucket_index(ttr_seconds):
    """B0 [0,3600), B1 [3600,10800), B2 [10800,86400), B3 [86400,inf); negative TTR -> -1."""
    t = np.asarray(ttr_seconds, dtype=float)
    if np.isnan(t).any():
        raise ValueError("TTR must not be NaN")
    idx = np.searchsorted(np.asarray(TTR_EDGES_SECONDS[1:-1]), t, side="right")
    return np.where(t < 0, -1, idx)


# ------------------------------------------------------------------ background
def background_m(log_odds, theta, sigma2, h, dW):
    """M background: X = (sigma2*h/2)(2Theta-1) + sqrt(sigma2)*dW, dW ~ N(0,h); L += X.

    `sigma2` = s*sigma_c^2 (log-odds^2 per second). Returns (L_new, X); the
    Gaussian experiment's LLR equals X exactly.
    """
    sigma2, h = np.asarray(sigma2, dtype=float), np.asarray(h, dtype=float)
    if np.any(sigma2 < 0) or np.any(h < 0):
        raise ValueError("Variance and interval must be nonnegative")
    x = (sigma2*h/2)*(2*np.asarray(theta)-1) + np.sqrt(sigma2)*np.asarray(dW, dtype=float)
    return np.asarray(log_odds, dtype=float)+x, x


def background_theta_free(log_odds, v_aux, sigma2, h, dW):
    """L background: Theta~ = 1{V_aux < expit(L)} (L-aux stream), then the M step with Theta~.

    Exact marginal transition of the Gaussian experiment; returns (L_new, X, Theta~).
    """
    theta_tilde = (np.asarray(v_aux, dtype=float) < probability(log_odds)).astype(np.int8)
    updated, x = background_m(log_odds, theta_tilde, sigma2, h, dW)
    return updated, x, theta_tilde


def background_legacy(log_odds, sigma2, dW):
    """Legacy-law (L0c) background: L += sigma*dW (driftless in log odds)."""
    x = np.sqrt(np.asarray(sigma2, dtype=float))*np.asarray(dW, dtype=float)
    return np.asarray(log_odds, dtype=float)+x, x


def gaussian_llr(x, sigma2, h):
    """LLR of X under N(+-sigma2*h/2, sigma2*h); identically X (V1 check helper)."""
    v = np.asarray(sigma2, dtype=float)*np.asarray(h, dtype=float)
    x = np.asarray(x, dtype=float)
    return ((x+v/2)**2-(x-v/2)**2)/(2*v)


# ------------------------------------------------------------------------ flow
def uninformed_sign(v_u, u):
    """S_u = 2*1{V_u < u} - 1 in {-1,+1} (the only uninformed-sign rule)."""
    return (2*(np.asarray(v_u, dtype=float) < np.asarray(u, dtype=float)).astype(np.int8)-1).astype(np.int8)


def fg_alpha_terms(size, eta):
    """FG: alpha(c; eta) = min(1 - exp(-eta c), 1 - 1e-12), returned as (alpha, 1 - alpha).

    1 - alpha = max(exp(-eta c), 1e-12) is carried separately so LLRs stay accurate
    for both tiny and near-one alpha (the cap is exactly the spec's min()).
    """
    c, eta = np.asarray(size, dtype=float), float(eta)
    if not np.isfinite(eta) or eta < 0 or np.any(~np.isfinite(c)) or np.any(c < 0):
        raise ValueError("FG requires finite nonnegative eta and sizes")
    one_minus = np.exp(-eta*c)
    alpha = -np.expm1(-eta*c)
    capped = one_minus < ALPHA_FLOOR_ONE_MINUS
    one_minus = np.where(capped, ALPHA_FLOOR_ONE_MINUS, one_minus)
    alpha = np.where(capped, 1.0-ALPHA_FLOOR_ONE_MINUS, alpha)
    return alpha, one_minus


def fg_half_alpha_terms(size, beta):
    """FG-half: alpha = tanh(beta c / 2), 1 - alpha = 2 expit(-beta c) (u = 1/2)."""
    x = float(beta)*np.asarray(size, dtype=float)
    if not np.isfinite(beta) or beta < 0 or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("FG-half requires finite nonnegative beta and sizes")
    return np.tanh(x/2), 2*expit(-x)


def constant_alpha_terms(alpha0, shape=()):
    """Constant-alpha (Tier 2) variant: alpha = alpha0 for every size."""
    a = float(alpha0)
    if not 0 <= a <= 1-ALPHA_FLOOR_ONE_MINUS:
        raise ValueError("Constant alpha must lie in [0, 1 - 1e-12]")
    return np.full(shape, a), np.full(shape, 1.0-a)


def gm_llr(sign, alpha, one_minus_alpha, u):
    """Glosten-Milgrom LLR log P(sign|Theta=1)/P(sign|Theta=0), p-independent.

    buy (+1): log[(alpha + (1-alpha)u) / ((1-alpha)u)]
    sell (-1): log[((1-alpha)(1-u)) / (alpha + (1-alpha)(1-u))]
    """
    s = check_signs(sign)
    a, om, u = (np.asarray(v, dtype=float) for v in (alpha, one_minus_alpha, u))
    if np.any(u <= 0) or np.any(u >= 1) or np.any(om <= 0) or np.any(a < 0):
        raise ValueError("GM LLR requires u in (0,1), alpha >= 0 and 1 - alpha > 0")
    buy = np.log1p(a/(om*u))
    sell = -np.log1p(a/(om*(1-u)))
    return np.where(s == 1, buy, sell)


def fg_half_llr(sign, size, beta):
    """FG-half LLR in closed form: +-beta*c (exactly the GM LLR at u=1/2, alpha=tanh(beta c/2))."""
    return check_signs(sign)*(float(beta)*np.asarray(size, dtype=float))


def flow_sign_m(theta, v_i, v_u, alpha, u):
    """M sampler: I = 1{V_I < alpha}; sign = 2Theta-1 if I else S_u. Returns (sign, I)."""
    informed = np.asarray(v_i, dtype=float) < np.asarray(alpha, dtype=float)
    sign = np.where(informed, 2*np.asarray(theta)-1, uninformed_sign(v_u, u))
    return check_signs(sign), informed


def buy_probability_theta_free(log_odds, alpha, one_minus_alpha, u):
    """Marginal P(buy | p) = (1-alpha)u + alpha*p with unclipped p = expit(L)."""
    return np.asarray(one_minus_alpha, dtype=float)*np.asarray(u, dtype=float) + \
        np.asarray(alpha, dtype=float)*probability(log_odds)


def flow_sign_theta_free(log_odds, v_f, alpha, one_minus_alpha, u):
    """Theta-free sampler: buy iff V_F < (1-alpha)u + alpha*expit(L) (single uniform)."""
    q = buy_probability_theta_free(log_odds, alpha, one_minus_alpha, u)
    return check_signs(np.where(np.asarray(v_f, dtype=float) < q, 1, -1))


def flow_update(log_odds, sign, alpha, one_minus_alpha, u):
    """Bayesian flow update L += GM LLR; returns (L_new, LLR)."""
    llr = gm_llr(sign, alpha, one_minus_alpha, u)
    return np.asarray(log_odds, dtype=float)+llr, llr


def fl_update(log_odds, sign, size, beta, innovation=0.0):
    """Legacy fixed impact FL with `ImpactPriceProcess.advance` arithmetic order.

    updated = before + innovation + beta * (sign * c); with innovation 0.0 this is
    bit-identical to `price.advance(float(sign*quantity))`.
    """
    impact = float(beta)*float(int(check_signs(sign))*float(size))
    return float(log_odds)+float(innovation)+impact, impact


# ----------------------------------------------------------------------- jumps
def jump_up_probability_m(theta, magnitude):
    """M: P(+ | Theta) = expit(a) if Theta=1 else expit(-a)."""
    a = np.asarray(magnitude, dtype=float)
    return np.where(np.asarray(theta) == 1, expit(a), expit(-a))


def jump_up_probability_theta_free(log_odds, magnitude):
    """Theta-free: q(p, a) = p expit(a) + (1-p) expit(-a), unclipped p."""
    p, a = probability(log_odds), np.asarray(magnitude, dtype=float)
    return p*expit(a)+(1-p)*expit(-a)


def jump_sign_m(theta, v_j, magnitude):
    return check_signs(np.where(np.asarray(v_j, dtype=float) < jump_up_probability_m(theta, magnitude), 1, -1))


def jump_sign_theta_free(log_odds, v_j, magnitude):
    return check_signs(np.where(np.asarray(v_j, dtype=float) < jump_up_probability_theta_free(log_odds, magnitude), 1, -1))


def jump_sign_legacy(v_j):
    """Legacy-law jumps: + iff V_J < 1/2 (fair coin, not a likelihood experiment)."""
    return check_signs(np.where(np.asarray(v_j, dtype=float) < 0.5, 1, -1))


def jump_update(log_odds, sign, magnitude):
    """L += sign * a (the sign-tilt experiment's LLR is exactly +-a)."""
    step = check_signs(sign)*np.asarray(magnitude, dtype=float)
    return np.asarray(log_odds, dtype=float)+step, step


def poisson_jump_offsets(ttr_start, duration, rates_per_second, rng):
    """Piecewise-constant Poisson jump times on [0, duration) seconds of an episode.

    TTR at offset t is ttr_start - t; rate is `rates_per_second[bucket]` for the
    TTR bucket (B0..B3). Segments at TTR <= 0 (after release) get rate zero.
    Consumes, per positive-length segment in time order, one Poisson count
    then that many uniforms. Returns sorted offsets.
    """
    if not np.isfinite(ttr_start) or not np.isfinite(duration) or duration < 0:
        raise ValueError("Invalid TTR start or duration")
    rates = [float(rates_per_second[b]) for b in TTR_BUCKETS]
    if any(not np.isfinite(r) or r < 0 for r in rates):
        raise ValueError("Jump rates must be finite and nonnegative")
    # TTR breakpoints, descending through time, mapped to elapsed offsets.
    cuts = sorted({0.0, float(duration), *[ttr_start-e for e in TTR_EDGES_SECONDS[1:-1] if 0 < ttr_start-e < duration],
                   *([ttr_start] if 0 < ttr_start < duration else [])})
    offsets = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid_ttr = ttr_start-(lo+hi)/2
        if hi <= lo or mid_ttr <= 0:
            continue
        rate = rates[int(ttr_bucket_index(mid_ttr))]
        n = int(rng.poisson(rate*(hi-lo)))
        offsets.extend((lo+(hi-lo)*rng.random(n)).tolist())
    return np.sort(np.asarray(offsets, dtype=float))


# ------------------------------------------------------------------ settlement
def settle_m(u_m, p0):
    """M family: Theta = 1{U_M < p0}; the same Theta drives all M generators and pays out."""
    return int(float(u_m) < float(p0))


def settle_l(u_l, p_close):
    """L family: pays 1{U_L < p_close}, p_close = unclipped expit(L) at t_c-; U_L is used nowhere else."""
    return int(float(u_l) < float(p_close))
