"""Terminal-LLR variance matching for the M family (spec Section 8; V9).

M-arm experiments are fixed and p-independent, so given Theta the closing log
odds are L_c = L0 + Lambda_B + Lambda_N with Lambda_N the flow and jump LLRs and
Lambda_B ~ N((2Theta-1)V/2, V), V = s * sigma_c(donor)^2 * T_ep, independent of
Lambda_N given Theta. Draws of (donor, Theta, Lambda_N) come from a vectorized
sampler that uses the skeleton's generators (legacy `sample_counts` window
counts, phase-eligible mark-bank draws, the kernels' flow and jump sign rules,
the calibrated u, eta, jump rates and magnitude bank). Per draw the
Rao-Blackwellized expected terminal uncertainty is

    h_i(V) = q_i * E_Z[expit(-(l_i + V/2 + sqrt(V) Z))],  l_i = L0 + Lambda_N,i,  q_i = expit(l_i)

(= E[p_c(1 - p_c) | N-signals]), decreasing in V. The logistic-normal integral
uses probabilists' Gauss-Hermite in Z when sqrt(V) <= 1 and Gauss-Legendre in
the logistic variable (E = E_Lambda[Phi((Lambda - m)/sqrt(V))], Lambda logistic
= logit(u), u uniform, integrated over Lambda in [-40, 40]) when sqrt(V) > 1;
every evaluation is accepted only if two orders agree within 1e-9 absolute.

Procedure per (template, arm, p0): N = 200,000 draws; target U* = U_M3(1) on an
independent draw set; s_hi doubled from 4 until U_A(s_hi) < U* (cap 1,024 ->
unmatched); bisection; independent-draw validation with a delta-method 95% CI
that includes the target's uncertainty; accept if |gap| < 2% and the CI covers
0, else double N (at most 3 times) and then mark unmatched.

CLI: python -m simulator.information.matching --output-dir DIR [--workers K]
writes roster.json and matching_report.csv (frozen before any policy run).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.special import expit, ndtr

from ..market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, MarketDynamics, file_hash, sample_counts
from ..price_impact import ImpactPriceProcess
from . import kernels as K
from .paths import Arm, make_arm
from .skeleton import SEED, event_roster

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT/"data_pipeline"/"output"/"paper3"/"experiment"
MATCH_STREAM_TAG = 31_008           # distinguishes matching draws from episode seeds
DRAWS = 200_000
MAX_DOUBLINGS = 3
S_HI_START, S_HI_CAP = 4.0, 1024.0
GAP_TOLERANCE = 0.02
Z95 = 1.959963984540054
CHUNK = 20_000                      # episodes per sampler chunk (fixed: draws depend on it)
MATCHED_ARMS = ("M0", "M1", "M2")
TARGET_ARM = "M3"
ARM_CODES = {"M0": 0, "M1": 1, "M2": 2, "M3": 3}
ROLE_CODES = {"target": 0, "fit": 1, "val": 2}
P0_LABELS = ("0.5", "emp")

# Quadrature (error-controlled).
QUAD_TOL = 1e-9
GH_ORDERS = (64, 128)
GL_ORDERS = (256, 512)
GL_HALF_WIDTH = 40.0                # logistic mass outside [-40, 40] is < 1e-17
MAX_REFINEMENTS = 3
EVAL_CHUNK = 16_384


class QuadratureError(RuntimeError):
    pass


# ----------------------------------------------------------------- quadrature
@lru_cache(maxsize=None)
def _hermite(n: int):
    z, w = hermegauss(n)
    return z, w/np.sqrt(2*np.pi)


@lru_cache(maxsize=None)
def _legendre(n: int):
    x, w = leggauss(n)
    lam = GL_HALF_WIDTH*x
    return lam, GL_HALF_WIDTH*w*expit(lam)*expit(-lam)


def _evaluate(method: str, order: int, m: np.ndarray, sd: float) -> np.ndarray:
    out = np.empty(len(m))
    if method == "hermite":
        z, w = _hermite(order)
        for a in range(0, len(m), EVAL_CHUNK):
            b = min(len(m), a+EVAL_CHUNK)
            out[a:b] = expit(-(m[a:b, None]+sd*z[None, :])) @ w
    else:
        lam, w = _legendre(order)
        for a in range(0, len(m), EVAL_CHUNK):
            b = min(len(m), a+EVAL_CHUNK)
            out[a:b] = ndtr((lam[None, :]-m[a:b, None])/sd) @ w
    return out


def complement_expectation(ell, V: float, tol: float = QUAD_TOL):
    """E_Z[expit(-(l + V/2 + sqrt(V) Z))] for an array of l at one V >= 0.

    Returns (values, max |order1 - order2|, (method, orders)). Raises
    QuadratureError if two orders do not agree within `tol` after refinement.
    """
    ell = np.asarray(ell, dtype=float)
    V = float(V)
    if not np.isfinite(V) or V < 0 or not np.isfinite(ell).all():
        raise ValueError("Quadrature needs finite l and V >= 0")
    if V == 0.0:
        return expit(-ell), 0.0, ("exact", (0, 0))
    m, sd = ell+V/2, np.sqrt(V)
    method = "hermite" if sd <= 1.0 else "legendre"
    orders = GH_ORDERS if method == "hermite" else GL_ORDERS
    for _ in range(MAX_REFINEMENTS+1):
        low, high = _evaluate(method, orders[0], m, sd), _evaluate(method, orders[1], m, sd)
        diff = float(np.max(np.abs(high-low))) if len(m) else 0.0
        if diff <= tol:
            return high, diff, (method, orders)
        orders = (2*orders[0], 2*orders[1])
    raise QuadratureError(f"Quadrature orders disagree by {diff:.3e} at V={V}")


def h_values(ell, V: float, tol: float = QUAD_TOL):
    """Per-draw h(V) = expit(l) * E_Z[expit(-(l + V/2 + sqrt(V) Z))]; returns (h, max order diff)."""
    values, diff, _ = complement_expectation(ell, V, tol)
    return expit(np.asarray(ell, dtype=float))*values, diff


def quad_reference(ell: float, V: float) -> float:
    """Independent scipy.integrate.quad value of E_Z[expit(-(l + V/2 + sqrt(V) Z))]."""
    ell, V = float(ell), float(V)
    if V == 0:
        return float(expit(-ell))
    m, sd = ell+V/2, np.sqrt(V)
    f = lambda z: expit(-(m+sd*z))*np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    z0 = -m/sd
    cuts = sorted({-38.0, 38.0, *[c for c in (z0-30/sd, z0-1/sd, z0, z0+1/sd, z0+30/sd) if -38 < c < 38]})
    return float(sum(integrate.quad(f, a, b, epsabs=1e-15, epsrel=1e-13, limit=400)[0] for a, b in zip(cuts[:-1], cuts[1:])))


def quadrature_check(cases: int = 1000, seed: int = 20261031) -> dict:
    """Compare the error-controlled quadrature with scipy.integrate.quad on random (l, V)."""
    rng = np.random.default_rng(seed)
    ell = rng.uniform(-30, 30, cases)
    sd = np.exp(rng.uniform(np.log(1e-3), np.log(40.0), cases))
    sd[:cases//20] = rng.uniform(0.9, 1.1, cases//20)            # around the method switch
    V = sd**2
    V[0] = 0.0
    errors = np.empty(cases)
    for i in range(cases):
        value, _, _ = complement_expectation(np.array([ell[i]]), V[i])
        errors[i] = abs(value[0]-quad_reference(ell[i], V[i]))
    return {"cases": cases, "seed": seed, "max_abs_error": float(errors.max()),
            "mean_abs_error": float(errors.mean()), "tolerance": QUAD_TOL, "passed": bool(errors.max() <= QUAD_TOL)}


# --------------------------------------------------------------- template law
class _SegmentRecorder:
    """Stand-in generator: records the Poisson means `kernels.poisson_jump_offsets` would draw."""
    def __init__(self):
        self.means = []

    def poisson(self, lam):
        self.means.append(float(lam))
        return 0

    def random(self, n):
        return np.zeros(int(n))


@dataclass
class TemplateDesign:
    """Deterministic, policy-independent law of one template's M-arm signals (Section 5 generators)."""
    ticker: str
    group: str
    information_class: str
    event_index: int
    lifetime: float
    seconds: float
    windows: list                     # dicts: duration, phase, rate, alpha, eligible
    quantities: np.ndarray            # mark bank sizes
    phases: np.ndarray
    donors: list
    donor_sigma_c: np.ndarray
    donor_fixed: bool
    jump_means: list                  # Poisson means per TTR segment (skeleton order)
    jump_bank: np.ndarray
    impact: object
    calibration: object = field(repr=False)
    jump_rate_scale: float = 1.0
    _tables: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, ticker: str, calibration, *, hours_to_close: float = 1.0, jump_rate_scale: float = 1.0,
              dynamics_path: Path = DYNAMICS_CONFIG) -> "TemplateDesign":
        roster = {t: (g, i) for g, t, i in event_roster(dynamics_path)}
        if ticker not in roster:
            raise KeyError(f"{ticker} is not a training event template")
        group, index = roster[ticker]
        raw = json.loads(Path(dynamics_path).read_text())
        # Throwaway generator: the explicit ticker consumes no selection draw; a donor draw (if any) is ignored.
        dynamics = MarketDynamics(group, np.random.default_rng(0), ticker, "base", "base", "base",
                                  config_path=dynamics_path)
        lifetime = float(dynamics.template["lifetime_seconds"])
        seconds = float(min(hours_to_close*3600, raw["event_templates"][ticker]["lifetime_seconds"]))
        start = lifetime-seconds
        windows, elapsed, probe = [], start, np.random.default_rng(0)
        while True:
            window = dynamics.sample_window(elapsed/lifetime, probe)   # deterministic duration/phase
            duration, phase = float(window["duration_seconds"]), int(window["phase"])
            record = dynamics.template["phases"][phase]
            eligible = np.flatnonzero(dynamics.phases == phase)
            if len(eligible) == 0:
                eligible = np.arange(len(dynamics.quantities))
            windows.append({"duration": duration, "phase": phase,
                            "rate": float(record["arrival_rate_per_second"])*dynamics.rate_scale,
                            "alpha": float(record["gamma_poisson_alpha"]), "eligible": eligible})
            elapsed = min(lifetime, elapsed+duration)
            if lifetime-elapsed < 1e-7:
                break
        if "noise" in dynamics.template:
            donors, fixed = [ticker], True
        else:
            donors, fixed = list(raw["groups"][group]["volatility"]["base"]["supported_tickers"]), False
        sigma_c = np.array([calibration.sigma_c(d) for d in donors], dtype=float)
        info_class = calibration.information_class(ticker)
        recorder = _SegmentRecorder()
        rates = calibration.jump_rates(info_class, jump_rate_scale)
        K.poisson_jump_offsets(calibration.ttr_at_close(ticker)+seconds, seconds, rates, recorder)
        return cls(ticker=ticker, group=group, information_class=info_class, event_index=index, lifetime=lifetime,
                   seconds=seconds, windows=windows, quantities=dynamics.quantities.astype(float),
                   phases=dynamics.phases, donors=donors, donor_sigma_c=sigma_c, donor_fixed=fixed,
                   jump_means=recorder.means, jump_bank=calibration.magnitude_bank(info_class),
                   impact=dynamics.impact, calibration=calibration, jump_rate_scale=float(jump_rate_scale))

    # ------------------------------------------------------------------ helpers
    def initial_log_odds(self, p0: float) -> float:
        """Exactly the paths' L0 = ImpactPriceProcess(p0).log_odds."""
        return float(ImpactPriceProcess(float(p0), self.impact).log_odds)

    @property
    def sigma_c_identically_zero(self) -> bool:
        return bool(np.all(self.donor_sigma_c == 0.0))

    def flow_tables(self, flow: str, phase: int) -> dict:
        """Per-mark (alpha, 1 - alpha, u, buy LLR, sell LLR) for the window phase (kernels' operations)."""
        key = (flow, phase)
        if key not in self._tables:
            c = self.quantities
            u = np.asarray(self.calibration.uninformed_rate(self.ticker, c, np.full(len(c), phase)), dtype=float)
            if flow == "fg":
                alpha, one_minus = K.fg_alpha_terms(c, self.calibration.eta(self.information_class))
            elif flow == "fg_constant":
                alpha, one_minus = K.constant_alpha_terms(self.calibration.constant_alpha(self.information_class), (len(c),))
            else:
                raise ValueError(f"No flow LLR table for {flow}")
            buy = K.gm_llr(np.ones(len(c), np.int8), alpha, one_minus, u)
            sell = K.gm_llr(-np.ones(len(c), np.int8), alpha, one_minus, u)
            self._tables[key] = {"alpha": alpha, "one_minus": one_minus, "u": u, "buy": buy, "sell": sell}
        return self._tables[key]


@dataclass
class Draws:
    """Vectorized draws of one (template, arm, p0, role, size)."""
    ell: np.ndarray                    # L0 + Lambda_N
    sigma_c2: np.ndarray               # sigma_c(donor)^2 per draw
    theta: np.ndarray
    lam_flow: np.ndarray
    lam_jump: np.ndarray
    n_trades: np.ndarray
    n_jumps: np.ndarray
    donor: np.ndarray
    seconds: float
    _groups: list | None = None

    def __len__(self):
        return len(self.ell)

    def groups(self):
        """[(sigma_c^2, unique l, inverse index, draw indices)] — prepared once per draw set."""
        if self._groups is None:
            self._groups = []
            for value in np.unique(self.sigma_c2):
                idx = np.flatnonzero(self.sigma_c2 == value)
                unique, inverse = np.unique(self.ell[idx], return_inverse=True)
                self._groups.append((float(value), unique, inverse, idx))
        return self._groups


def sample_draws(design: TemplateDesign, arm: Arm, n: int, rng: np.random.Generator) -> Draws:
    """Vectorized sampler of (donor, Theta, Lambda_N) with the skeleton/path law of `arm`."""
    if arm.family != "M":
        raise ValueError("Matching draws are defined for M arms")
    if arm.flow not in {"f0", "fg", "fg_constant"} or arm.jumps not in {"none", "m"}:
        raise ValueError("Unsupported M arm components")
    L0 = design.initial_log_odds(arm.p0)
    parts = []
    for a in range(0, n, CHUNK):
        m = min(n, a+CHUNK)-a
        if design.donor_fixed:
            donor = np.zeros(m, dtype=np.int64)
        else:
            donor = rng.choice(len(design.donors), size=m)            # legacy rng.choice(supported) law
        theta = (rng.random(m) < arm.p0).astype(np.int8)             # Theta = 1{U_M < p0}
        lam_flow, n_trades = np.zeros(m), np.zeros(m, dtype=np.int64)
        for w in design.windows:
            counts = sample_counts(w["rate"], w["alpha"], rng, draws=m, duration_seconds=w["duration"])
            n_trades += counts
            total = int(counts.sum())
            if total == 0:
                continue
            ids = rng.choice(w["eligible"], size=total, replace=True)
            if arm.flow == "f0":
                continue                                             # uninformed: signs carry no LLR
            v_i, v_u = rng.random(total), rng.random(total)
            episode = np.repeat(np.arange(m), counts)
            tab = design.flow_tables(arm.flow, w["phase"])
            sign, _ = K.flow_sign_m(theta[episode], v_i, v_u, tab["alpha"][ids], tab["u"][ids])
            llr = np.where(sign == 1, tab["buy"][ids], tab["sell"][ids])
            lam_flow += np.bincount(episode, weights=llr, minlength=m)
        lam_jump, n_jumps = np.zeros(m), np.zeros(m, dtype=np.int64)
        for mean in design.jump_means:
            n_jumps += rng.poisson(mean, size=m)
        if arm.jumps == "m":
            total = int(n_jumps.sum())
            if total:
                if not len(design.jump_bank):
                    raise ValueError("Empty jump magnitude bank")
                episode = np.repeat(np.arange(m), n_jumps)
                a_mag = rng.choice(design.jump_bank, size=total)
                v_j = rng.random(total)
                sign = K.jump_sign_m(theta[episode], v_j, a_mag)
                lam_jump = np.bincount(episode, weights=sign*a_mag, minlength=m)
        parts.append((donor, theta, lam_flow, lam_jump, n_trades, n_jumps))
    donor, theta, lam_flow, lam_jump, n_trades, n_jumps = (np.concatenate(x) for x in zip(*parts))
    return Draws(ell=L0+lam_flow+lam_jump, sigma_c2=design.donor_sigma_c[donor]**2, theta=theta, lam_flow=lam_flow,
                 lam_jump=lam_jump, n_trades=n_trades, n_jumps=n_jumps, donor=donor, seconds=design.seconds)


def aligned_llr(design: TemplateDesign, skeleton, arm: Arm) -> dict:
    """Lambda_N of one skeleton episode from the sampler's per-mark tables and sign rules (exact alignment check)."""
    theta = K.settle_m(skeleton.u_m, arm.p0)
    per_trade = np.zeros(skeleton.n_executions)
    if arm.flow in {"fg", "fg_constant"} and skeleton.n_executions:
        for k, w in enumerate(skeleton.windows):
            sl = w.exec_slice
            if sl.stop == sl.start:
                continue
            tab = design.flow_tables(arm.flow, design.windows[k]["phase"])
            q = skeleton.exec_quantity[sl]
            ids = np.array([np.flatnonzero(design.quantities == c)[0] for c in q])
            sign, _ = K.flow_sign_m(theta, skeleton.v_i[sl], skeleton.v_u[sl], tab["alpha"][ids], tab["u"][ids])
            per_trade[sl] = np.where(sign == 1, tab["buy"][ids], tab["sell"][ids])
    per_jump = np.zeros(len(skeleton.jump_elapsed))
    if arm.jumps == "m" and len(per_jump):
        sign = K.jump_sign_m(theta, skeleton.v_j, skeleton.jump_magnitude)
        per_jump = sign*skeleton.jump_magnitude
    return {"theta": theta, "per_trade": per_trade, "per_jump": per_jump,
            "lam_flow": float(per_trade.sum()), "lam_jump": float(per_jump.sum()),
            "ell": design.initial_log_odds(arm.p0)+float(per_trade.sum())+float(per_jump.sum())}


# -------------------------------------------------------------- RB estimator
@dataclass
class Estimate:
    mean: float
    var: float          # sample variance of h_i (ddof = 1)
    n: int
    quad_diff: float

    @property
    def se(self) -> float:
        return float(np.sqrt(self.var/self.n)) if self.n > 1 else 0.0


def rb_uncertainty(draws: Draws, s: float) -> Estimate:
    """U_A(s) = mean_i h_i(s * sigma_c(donor_i)^2 * T_ep) with error-controlled quadrature."""
    if not np.isfinite(s) or s < 0:
        raise ValueError("Background scale must be finite and >= 0")
    h = np.empty(len(draws))
    worst = 0.0
    for sigma_c2, unique, inverse, idx in draws.groups():
        values, diff = h_values(unique, s*sigma_c2*draws.seconds)
        worst = max(worst, diff)
        h[idx] = values[inverse]
    return Estimate(float(h.mean()), float(h.var(ddof=1)) if len(h) > 1 else 0.0, len(h), worst)


def validation_gap(val: Estimate, target: Estimate) -> dict:
    """gap = U_val/U* - 1 with a delta-method 95% CI including validation and target uncertainty."""
    gap = val.mean/target.mean-1
    se = float(np.sqrt(val.var/val.n/target.mean**2+(val.mean**2/target.mean**4)*target.var/target.n))
    low, high = gap-Z95*se, gap+Z95*se
    return {"gap": gap, "gap_se": se, "ci_low": low, "ci_high": high,
            "accepted": bool(abs(gap) < GAP_TOLERANCE and low <= 0 <= high)}


def solve_scale(draws: Draws, target: float, *, s_hi_start: float = S_HI_START, s_hi_cap: float = S_HI_CAP,
                rel_tol: float = 1e-9, max_iterations: int = 200) -> dict:
    """Bracket (s_hi doubled from 4, cap 1,024) and bisect U_A(s) = target on the monotone RB curve."""
    worst = 0.0
    at_zero = rb_uncertainty(draws, 0.0)
    worst = max(worst, at_zero.quad_diff)
    if not at_zero.mean > target:
        return {"status": "target_not_above_arm_at_s0", "s": None, "s_hi": None, "iterations": 0, "quad_diff": worst,
                "fit": at_zero.mean}
    lo, hi = 0.0, s_hi_start
    while True:
        est = rb_uncertainty(draws, hi)
        worst = max(worst, est.quad_diff)
        if est.mean < target:
            break
        if hi >= s_hi_cap:
            return {"status": "unmatched_cap", "s": None, "s_hi": hi, "iterations": 0, "quad_diff": worst,
                    "fit": est.mean}
        lo, hi = hi, 2*hi
    s_hi_final = hi
    iterations = 0
    mid, est = (lo+hi)/2, None
    while iterations < max_iterations:
        mid = (lo+hi)/2
        est = rb_uncertainty(draws, mid)
        worst = max(worst, est.quad_diff)
        iterations += 1
        if abs(est.mean-target) <= rel_tol*target or hi-lo <= rel_tol*max(hi, 1.0):
            break
        lo, hi = (mid, hi) if est.mean > target else (lo, mid)
    return {"status": "solved", "s": float(mid), "s_hi": s_hi_final, "iterations": iterations, "quad_diff": worst,
            "fit": est.mean}


def _rng(event_index: int, p0_label: str, arm: str, role: str, level: int, seed: int = SEED):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, MATCH_STREAM_TAG, event_index, P0_LABELS.index(p0_label), ARM_CODES[arm], ROLE_CODES[role], level]))


def match_template(ticker: str, calibration, *, p0_label: str = "0.5", draws: int = DRAWS, arms=MATCHED_ARMS,
                   seed: int = SEED, max_doublings: int = MAX_DOUBLINGS) -> list[dict]:
    """Run Section 8 for one (template, p0); returns one report row per M arm (M3 = target row)."""
    t_start = time.perf_counter()
    design = TemplateDesign.build(ticker, calibration)
    p0 = 0.5 if p0_label == "0.5" else calibration.p0_emp(ticker)
    base = {"ticker": ticker, "information_class": design.information_class, "event_index": design.event_index,
            "p0_label": p0_label, "p0": p0, "episode_seconds": design.seconds,
            "donor_fixed": design.donor_fixed, "donors": len(design.donors),
            "donor_sigma_c_min": float(design.donor_sigma_c.min()), "donor_sigma_c_max": float(design.donor_sigma_c.max())}
    rows = []
    if calibration.early_revelation(ticker) or ticker not in calibration.candidate_roster():
        for arm in (*arms, TARGET_ARM):
            rows.append({**base, "arm": arm, "status": "excluded_early_revelation"
                         if calibration.early_revelation(ticker) else "excluded_not_candidate"})
        return rows
    targets = {}

    def target(level):
        if level not in targets:
            n = draws*2**level
            d = sample_draws(design, make_arm(TARGET_ARM, p0=p0), n, _rng(design.event_index, p0_label, TARGET_ARM,
                                                                         "target", level, seed))
            targets[level] = rb_uncertainty(d, 1.0)
        return targets[level]

    for arm_name in arms:
        a_start = time.perf_counter()
        row = {**base, "arm": arm_name}
        if design.sigma_c_identically_zero:
            t0 = target(0)
            rows.append({**row, "status": "unmatched_sigma_c_zero", "U_star": t0.mean, "U_star_se": t0.se,
                         "draws": draws, "level": 0, "runtime_seconds": time.perf_counter()-a_start})
            continue
        for level in range(max_doublings+1):
            n = draws*2**level
            tgt = target(level)
            fit_draws = sample_draws(design, make_arm(arm_name, s=1.0, p0=p0),
                                     n, _rng(design.event_index, p0_label, arm_name, "fit", level, seed))
            solved = solve_scale(fit_draws, tgt.mean)
            row.update(draws=n, level=level, U_star=tgt.mean, U_star_se=tgt.se, fit_status=solved["status"],
                       s_hi=solved["s_hi"], bisection_iterations=solved["iterations"], U_fit=solved["fit"],
                       quad_max_order_diff=max(solved["quad_diff"], tgt.quad_diff))
            if solved["status"] == "unmatched_cap":
                row.update(status="unmatched_cap", s=None)
                break
            if solved["status"] != "solved":
                row.update(status="unmatched_"+solved["status"], s=None)
                continue                                              # retry at doubled N
            del fit_draws
            val_draws = sample_draws(design, make_arm(arm_name, s=1.0, p0=p0),
                                     n, _rng(design.event_index, p0_label, arm_name, "val", level, seed))
            val = rb_uncertainty(val_draws, solved["s"])
            del val_draws
            gap = validation_gap(val, tgt)
            row.update(s=solved["s"], U_val=val.mean, U_val_se=val.se, **gap,
                       quad_max_order_diff=max(row["quad_max_order_diff"], val.quad_diff),
                       status="matched" if gap["accepted"] else "unmatched_validation")
            if gap["accepted"]:
                break
        row["runtime_seconds"] = time.perf_counter()-a_start
        rows.append(row)
    t0 = target(0)
    rows.append({**base, "arm": TARGET_ARM, "status": "target", "s": 1.0, "U_star": t0.mean, "U_star_se": t0.se,
                 "draws": draws, "level": 0, "quad_max_order_diff": t0.quad_diff,
                 "runtime_seconds": time.perf_counter()-t_start})
    return rows


# ------------------------------------------------------------------- roster
def build_roster(report: pd.DataFrame, calibration) -> dict:
    """R* (p0 = 0.5) and R*_emp (p0_emp): candidates (no early revelation) matched for every M arm."""
    out = {}
    candidates = calibration.candidate_roster()
    for label, key in (("0.5", "R_star"), ("emp", "R_star_emp")):
        frame = report[report.p0_label.eq(label)]
        members, scales, excluded = [], {}, {}
        for ticker in [t for _, t, _ in event_roster()]:
            rows = frame[frame.ticker.eq(ticker)].set_index("arm")
            if ticker not in candidates:
                excluded[ticker] = "early_revelation" if calibration.early_revelation(ticker) else "not_candidate"
                continue
            statuses = {a: rows.loc[a, "status"] for a in MATCHED_ARMS if a in rows.index}
            if len(statuses) == len(MATCHED_ARMS) and all(v == "matched" for v in statuses.values()):
                members.append(ticker)
                scales[ticker] = {a: float(rows.loc[a, "s"]) for a in MATCHED_ARMS}
                scales[ticker][TARGET_ARM] = 1.0
            else:
                excluded[ticker] = ";".join(f"{a}:{v}" for a, v in statuses.items() if v != "matched") or "missing"
        out[key] = {"p0_label": label, "templates": members, "matched_s": scales, "excluded": excluded}
    return out


def roster_hash(roster: dict) -> str:
    core = {k: roster[k] for k in ("R_star", "R_star_emp")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def load_roster(path) -> dict:
    roster = json.loads(Path(path).read_text())
    if roster_hash(roster) != roster["roster_sha256"]:
        raise ValueError("roster.json content does not match its recorded hash")
    return roster


def _task(args):
    ticker, p0_label, draws, seed, calibration_dir = args
    from .config import InformationCalibration
    calibration = InformationCalibration.load(calibration_dir) if calibration_dir else InformationCalibration.load()
    return match_template(ticker, calibration, p0_label=p0_label, draws=draws, seed=seed)


def run(output_dir=DEFAULT_OUTPUT_DIR, *, draws: int = DRAWS, workers: int | None = None, tickers=None,
        seed: int = SEED, calibration_dir=None, quadrature_cases: int = 1000) -> dict:
    from .config import DEFAULT_DIR, InformationCalibration
    calibration_dir = Path(calibration_dir) if calibration_dir else DEFAULT_DIR
    calibration = InformationCalibration.load(calibration_dir)
    report_json = json.loads((calibration_dir/"calibration_report.json").read_text())
    if not report_json.get("passed", False):
        raise RuntimeError("Information calibration has not passed; matching refuses to run")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    quad = quadrature_check(quadrature_cases)
    if not quad["passed"]:
        raise QuadratureError(f"Quadrature check failed: {quad}")
    selected = [t for _, t, _ in event_roster() if tickers is None or t in tickers]
    tasks = [(t, label, draws, seed, str(calibration_dir)) for t in selected for label in P0_LABELS]
    workers = workers or max(1, (os.cpu_count() or 2)-1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_task, tasks):
            rows.extend(result)
    report = pd.DataFrame(rows)
    report.to_csv(output_dir/"matching_report.csv", index=False)
    roster = build_roster(report, calibration)
    source = Path(__file__)
    roster.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_before_policy_runs": True,
        "rule": "candidate (train, not early revelation) templates matched (|gap| < 2%, CI covers 0) for M0, M1, M2; "
                "M3 is the s = 1 target",
        "design": {"draws": draws, "max_doublings": MAX_DOUBLINGS, "s_hi_start": S_HI_START, "s_hi_cap": S_HI_CAP,
                   "gap_tolerance": GAP_TOLERANCE, "seed": seed, "match_stream_tag": MATCH_STREAM_TAG,
                   "quadrature": {"tolerance": QUAD_TOL, "hermite_orders": GH_ORDERS, "legendre_orders": GL_ORDERS,
                                  "legendre_half_width": GL_HALF_WIDTH, "switch": "sqrt(V) <= 1 -> Hermite"},
                   "tickers": selected, "workers": workers},
        "p0_emp": {t: calibration.p0_emp(t) for t in selected},
        "quadrature_check": quad,
        "max_quadrature_order_difference": float(report.get("quad_max_order_diff", pd.Series([0.0])).max()),
        "calibration_sha256": calibration.config["file_sha256"],
        "calibration_config_sha256": file_hash(calibration_dir/"information_config.json"),
        "code_sha256": {str(source.relative_to(ROOT)): file_hash(source)},
        "runtime_seconds": time.perf_counter()-t0})
    roster["roster_sha256"] = roster_hash(roster)
    (output_dir/"roster.json").write_text(json.dumps(roster, indent=2, default=float)+"\n")
    return {"R_star": roster["R_star"]["templates"], "R_star_emp": roster["R_star_emp"]["templates"],
            "runtime_seconds": roster["runtime_seconds"], "output_dir": str(output_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--quadrature-cases", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, draws=args.draws, workers=args.workers, tickers=args.tickers,
                         calibration_dir=args.calibration_dir, quadrature_cases=args.quadrature_cases), indent=2))
