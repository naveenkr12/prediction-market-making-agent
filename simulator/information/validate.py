"""Offline validation CLI for Paper 3 (spec Section 10: V1-V13).

    python -m simulator.information.validate [--scale full|reduced] [--experiment-dir DIR]
                                             [--output REPORT.json] [--workers K] [--only V4 V9 ...]

Full scale follows the spec (V2 >= 500 episodes per M arm; V4 >= 20,000
price-path episodes per M arm per class, no policy loop; V9 cross-check
3 templates x M0-M2 x 2,000 no-quote episodes; full golden sweep; L0 anchor on
all 35 templates x 20 replicates x 4 policies). `--scale reduced` runs every
item on small samples for smoke checks (results are not acceptance evidence).
Validation episodes use replicate indices offset far from the experiment's
(0..19), so they are independent draws of the same law.

Statistical items: V4 joint chi-square on QR-reduced instrument columns (rank
tolerance 1e-10), alpha = 0.01, Holm across M arms within class; V5 the same
for outcome calibration; V6 the budget identity is exact per episode and its
single weighted-drift aggregate sum (2p_- - 1) dp is tested for mean zero.
L arms are reported as diagnostics only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.special import expit

from ..demo_trading_env import FILL_SCENARIOS
from ..execution import QuoteAction
from ..market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, file_hash
from . import kernels as K
from . import metrics as Mx
from . import reference as R
from .environment import InformationTradingEnvironment, golden_equivalence_sweep, l0_anchor_check, run_policy
from .matching import load_roster
from .paths import build_path, make_arm, with_p0
from .skeleton import EXECUTION, JUMP, PROPOSAL_ASK, PROPOSAL_BID, WINDOW_END, build_skeleton, event_roster
from .taker_fees import TakerFeeSchedule, snipe_profitable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT/"data_pipeline"/"output"/"paper3"/"experiment"
SAVED_BENCHMARK = ROOT/"data_pipeline/output/paper2/calibration/simulator/hjb_benchmark/episode_results.csv"
ALPHA = 0.01
RANK_TOL = 1e-10
M_ARMS = ("M0", "M1", "M2", "M3", "M1-native")
L_DIAGNOSTIC_ARMS = ("L0b", "L0c", "L1J", "L1", "FGH-L", "FGH-LJ")
CHANNELS = ("background", "flow", "jump")
INSTRUMENTS = ("1", "p", "p2", "ttr_B0", "ttr_B1", "ttr_B2", "ttr_B3", "size_bin", "phase", "last_flow_sign",
               "time_since_last_jump")
V5_TIMES = (900.0, 1800.0, 2700.0, "close")
OFFSETS = {"v4": 1_000_000, "v2": 2_000_000, "v9": 3_000_000, "v12": 4_000_000, "v8": 5_000_000}
SCALES = {
    "full": {"v2_episodes": 500, "v3_episodes": 100, "v4_episodes": 20_000, "v9_templates": 3, "v9_episodes": 2_000,
             "v12_replicates": 20, "v8_templates": 2, "v8_replicates": 2, "golden_tickers": None, "golden_seeds": (0, 1),
             "anchor_replicates": None, "anchor_tickers": None},
    "reduced": {"v2_episodes": 20, "v3_episodes": 10, "v4_episodes": 400, "v9_templates": 3, "v9_episodes": 100,
                "v12_replicates": 2, "v8_templates": 1, "v8_replicates": 1, "golden_tickers": 3, "golden_seeds": (0,),
                "anchor_replicates": [0], "anchor_tickers": 3}}

_STATE: dict = {}


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def _single_threaded_children():
    """Spawned workers read these at import: one BLAS thread per process (no oversubscription)."""
    for key in THREAD_VARIABLES:
        os.environ.setdefault(key, "1")


def _init(calibration_dir):
    from .config import InformationCalibration
    from ..hjb_policy import HJBReferencePolicy
    _STATE["calibration"] = InformationCalibration.load(calibration_dir) if calibration_dir else InformationCalibration.load()
    _STATE["hjb"] = HJBReferencePolicy(time_mapping="episode")


def _cal():
    if "calibration" not in _STATE:
        _init(None)
    return _STATE["calibration"]


# ------------------------------------------------------------------ helpers
def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    """Holm step-down adjusted p-values and rejections."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m-i)*pvalues[k]))
        out[k] = {"p": pvalues[k], "p_holm": running, "reject": bool(running < alpha)}
    return out


def wald_qr(X: np.ndarray, names: list[str], tol: float = RANK_TOL) -> dict:
    """Joint chi-square that E[X] = 0 on QR-reduced (pivoted, centered, scaled) columns."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    centered = X-X.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    usable = np.flatnonzero(norms > 0)
    dropped = [names[j] for j in range(X.shape[1]) if j not in set(usable)]
    if len(usable) == 0 or n < 3:
        return {"n": n, "rank": 0, "statistic": 0.0, "df": 0, "p": 1.0, "dropped": dropped,
                "nonzero_constant_columns": [names[j] for j in range(X.shape[1]) if norms[j] == 0 and X[0, j] != 0]}
    scaled = centered[:, usable]/norms[usable]
    _, r, piv = linalg.qr(scaled, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int(np.sum(diag > tol*diag[0]))
    keep = usable[piv[:rank]]
    dropped += [names[j] for j in usable if j not in set(keep)]
    Y = X[:, keep]
    mean = Y.mean(axis=0)
    cov = np.cov(Y, rowvar=False, ddof=1).reshape(rank, rank)
    statistic = float(n*mean@np.linalg.solve(cov, mean))
    return {"n": n, "rank": rank, "statistic": statistic, "df": rank, "p": float(stats.chi2.sf(statistic, rank)),
            "kept": [names[j] for j in keep], "dropped": dropped,
            "nonzero_constant_columns": [names[j] for j in range(X.shape[1]) if norms[j] == 0 and X[0, j] != 0]}


def _arm(name, ticker, roster, p0_label="0.5"):
    cal = _cal()
    key = "R_star" if p0_label == "0.5" else "R_star_emp"
    if name in ("M0", "M1", "M2"):
        arm = make_arm(name, s=roster[key]["matched_s"][ticker][name])
    else:
        arm = make_arm(name)
    return arm if p0_label == "0.5" else with_p0(arm, cal.p0_emp(ticker))


def _node_windows(path) -> np.ndarray:
    out = np.empty(path.node_count, dtype=np.int64)
    for k, sl in enumerate(path.window_node_slices):
        out[sl] = k
    return out


# ----------------------------------------------------- V4/V5/V6/V10 stats
def episode_statistics(path, calibration) -> dict:
    """Martingale moments (V4), outcome calibration (V5), budget identity (V6), fidelity (V10) of one path."""
    sk, arm = path.skeleton, path.arm
    kinds = np.asarray(path.kinds)
    refs = np.asarray(path.refs)
    elapsed = np.asarray(path.elapsed, dtype=float)
    l_bg, l_after = np.asarray(path.l_bg), np.asarray(path.l_after)
    l_prev = np.r_[path.initial_log_odds, l_after[:-1]]
    p_prev, p_bg, p_after = expit(l_prev), expit(l_bg), expit(l_after)
    n = len(kinds)
    h = np.diff(np.r_[sk.start, elapsed])
    is_exec = kinds == EXECUTION
    is_jump = (kinds == JUMP) & path.jump_applied
    # Pre-event public state.
    node_exec = np.where(is_exec, refs, -1)
    last_node = np.maximum.accumulate(np.where(is_exec, np.arange(n), -1))          # node of last exec <= i
    last_j_incl = np.where(last_node >= 0, node_exec[np.maximum(last_node, 0)], -1)  # exec index (-1: none)
    last_j_before = np.r_[-1, last_j_incl[:-1]]
    sign_before = np.r_[0, np.asarray(path.exec_sign, dtype=float)][last_j_before+1]
    sizes = sk.exec_quantity
    size_bins = calibration.size_bin(sk.information_class, sizes) if len(sizes) else np.zeros(0, int)
    bin_incl = np.r_[-1, size_bins][last_j_incl+1]
    bin_before = np.r_[-1, bin_incl[:-1]]
    jump_time = np.where(is_jump, elapsed, -np.inf)
    last_jump_incl = np.maximum.accumulate(jump_time)
    last_jump_before = np.r_[-np.inf, last_jump_incl[:-1]]
    phase = np.array([int(w.window["phase"]) for w in sk.windows])[_node_windows(path)]

    def instruments(p, t, size_bin, last_sign, last_jump):
        ttr = sk.ttr_at_close+sk.lifetime-t
        bucket = K.ttr_bucket_index(ttr)
        since = t-np.where(np.isfinite(last_jump), last_jump, sk.start)
        cols = [np.ones_like(p), p, p*p] + [(bucket == b).astype(float) for b in range(4)] + [
            size_bin.astype(float), phase.astype(float), last_sign.astype(float), since]
        return np.vstack(cols)

    t_start = elapsed-h
    phi_bg = instruments(p_prev, t_start, bin_before, sign_before, last_jump_before)
    d_bg = p_bg-p_prev
    phi_ev = instruments(p_bg, elapsed, bin_incl, sign_before, last_jump_before)
    d_ev = p_after-p_bg
    flow_mask = is_exec & (arm.flow not in ("f0",))
    moments = np.concatenate([phi_bg@d_bg, phi_ev@(d_ev*flow_mask), phi_ev@(d_ev*is_jump)])
    # V6: budget identity and its single weighted-drift aggregate.
    p0 = float(expit(path.initial_log_odds))
    sq = {"background": float(np.sum(d_bg**2)), "flow": float(np.sum((d_ev*flow_mask)**2)),
          "jump": float(np.sum((d_ev*is_jump)**2))}
    other = d_ev*~(flow_mask | is_jump)
    if np.any(other != 0):
        raise AssertionError("Price moved on a node outside background/flow/jump")
    drift = float(np.sum((2*p_prev-1)*d_bg)+np.sum((2*p_bg-1)*d_ev))
    p_c = float(p_after[-1])
    identity_error = p0*(1-p0)-p_c*(1-p_c)-sum(sq.values())-drift
    # V5: p_t and the outcome.
    outcome = path.theta if path.theta is not None else path.settlement["outcome"]
    p_t = []
    for t in V5_TIMES:
        if t == "close":
            p_t.append(p_c)
        else:
            k = int(np.searchsorted(elapsed, sk.start+t, side="right"))-1
            p_t.append(float(p_after[k]) if k >= 0 else p0)
    p_t = np.asarray(p_t)
    v5 = np.concatenate([(outcome-p_t)*f for f in (np.ones(4), p_t, p_t**2)])
    # V10: window log-odds changes, Kyle sums, jump counts.
    ends = [sl.stop-1 for sl in path.window_node_slices]
    window_l = np.r_[path.initial_log_odds, l_after[ends]]
    dL = np.diff(window_l)
    absd = np.sort(np.abs(dL))[::-1]
    F, kdL = Mx.kyle_windows(path)
    return {"moments": moments, "v5": v5, "p_t": p_t, "outcome": int(outcome), "budget": sq, "drift": drift,
            "identity_error": identity_error, "pc_uncertainty": p_c*(1-p_c), "p0_uncertainty": p0*(1-p0),
            "window_dl": dL, "top3_share": float(absd[:3].sum()/absd.sum()) if absd.sum() > 0 else np.nan,
            "kyle": (len(F), float(F.sum()), float(kdL.sum()), float(F@F), float(F@kdL), Mx.ols_slope(F, kdL)),
            "jumps": int(is_jump.sum()), "jump_abs": float(np.sum(np.abs(path.jump_step))) if path.jump_applied else 0.0}


def _v4_task(args):
    cls, templates, start, stop, roster, arms = args
    cal = _cal()
    out = {a: [] for a in arms}
    for e in range(start, stop):
        ticker = templates[e % len(templates)]
        sk = build_skeleton(ticker, OFFSETS["v4"]+e//len(templates), calibration=cal)
        for name in arms:
            st = episode_statistics(build_path(sk, _arm(name, ticker, roster), calibration=cal), cal)
            st["ticker"] = ticker
            out[name].append(st)
    return cls, out


def run_v4_family(roster, templates_by_class, episodes, workers, arms) -> dict:
    tasks = []
    chunk = 100
    for cls, templates in templates_by_class.items():
        for a in range(0, episodes, chunk):
            tasks.append((cls, templates, a, min(episodes, a+chunk), roster, arms))
    collected = {cls: {a: [] for a in arms} for cls in templates_by_class}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(_STATE.get("calibration_dir"),)) as pool:
        for cls, out in pool.map(_v4_task, tasks):
            for a, rows in out.items():
                collected[cls][a].extend(rows)
    return collected


def v4_v5_v6_v10(collected, calibration) -> dict:
    names = [f"{c}:{i}" for c in CHANNELS for i in INSTRUMENTS]
    v5_names = [f"{f}@{t}" for f in ("1", "p", "p2") for t in V5_TIMES]
    v4, v5, v6, v10 = {}, {}, {}, {"simulated": {}, "train": train_fidelity(calibration)}
    for cls, by_arm in collected.items():
        p4, p5, p6 = {}, {}, {}
        for arm, rows in by_arm.items():
            X = np.vstack([r["moments"] for r in rows])
            test = wald_qr(X, names)
            test["mean_abs_moment_max"] = float(np.max(np.abs(X.mean(axis=0))))
            Y = np.vstack([r["v5"] for r in rows])
            test5 = wald_qr(Y, v5_names)
            p_t = np.vstack([r["p_t"] for r in rows])
            outcome = np.array([r["outcome"] for r in rows])
            reliability = {}
            for j, t in enumerate(V5_TIMES):
                deciles = np.minimum((p_t[:, j]*10).astype(int), 9)
                reliability[str(t)] = [{"decile": d, "n": int(np.sum(deciles == d)),
                                        "mean_p": float(p_t[deciles == d, j].mean()),
                                        "mean_outcome": float(outcome[deciles == d].mean())}
                                       for d in range(10) if np.any(deciles == d)]
            drift = np.array([r["drift"] for r in rows])
            ident = np.array([r["identity_error"] for r in rows])
            se = drift.std(ddof=1)/np.sqrt(len(drift)) if len(drift) > 1 else np.inf
            p_drift = float(2*stats.norm.sf(abs(drift.mean())/se)) if se > 0 else (1.0 if drift.mean() == 0 else 0.0)
            budget = {c: float(np.mean([r["budget"][c] for r in rows])) for c in CHANNELS}
            v6_row = {"episodes": len(rows), "identity_max_abs_error": float(np.max(np.abs(ident))),
                      "lhs_mean_p0_uncertainty_minus_close_uncertainty": float(np.mean(
                          [r["p0_uncertainty"]-r["pc_uncertainty"] for r in rows])),
                      "budget_mean_sum_dp2_by_channel": budget, "weighted_drift_aggregate_mean": float(drift.mean()),
                      "weighted_drift_aggregate_se": float(se), "p": p_drift,
                      "label": "one weighted-drift aggregate: sum (2 p_- - 1) dp"}
            kyle = pd.DataFrame([dict(zip(("n", "sf", "sl", "sff", "sfl", "slope"), r["kyle"]), ticker=r["ticker"])
                                 for r in rows])
            per_t = []
            for t, d in kyle.groupby("ticker"):
                nn = d.n.sum()
                sxx = d.sff.sum()-d.sf.sum()**2/nn if nn > 1 else 0
                sxy = d.sfl.sum()-d.sf.sum()*d.sl.sum()/nn if nn > 1 else 0
                per_t.append({"pooled": sxy/sxx if sxx > 1e-24 else 0.0, "replicate_averaged": d.slope.mean()})
            per_t = pd.DataFrame(per_t)
            dl = np.concatenate([r["window_dl"] for r in rows])
            by_ticker = {}
            for r in rows:
                by_ticker.setdefault(r["ticker"], []).append(r["window_dl"])
            kurt = [float(stats.kurtosis(np.concatenate(v), fisher=True)) for v in by_ticker.values()
                    if np.concatenate(v).std() > 0]
            v10["simulated"][f"{arm}|{cls}"] = {
                "kyle_lambda_median_pooled": float(per_t.pooled.median()),
                "kyle_lambda_median_replicate_averaged": float(per_t.replicate_averaged.median()),
                "kyle_lambda_target": calibration.config["classes"][cls]["flow"]["target_lambda_median"],
                "mean_jumps_per_episode": float(np.mean([r["jumps"] for r in rows])),
                "mean_abs_jump_size": float(np.sum([r["jump_abs"] for r in rows])/max(1, np.sum([r["jumps"] for r in rows]))),
                "bank_mean_magnitude": float(np.mean(calibration.magnitude_bank(cls))),
                "median_top3_share_abs_dL": float(np.nanmedian([r["top3_share"] for r in rows])),
                "median_template_excess_kurtosis_300s_dL": float(np.median(kurt)) if kurt else None,
                "pooled_excess_kurtosis_300s_dL": float(stats.kurtosis(dl)) if dl.std() > 0 else None}
            if arm in M_ARMS:
                p4[arm], p5[arm], p6[arm] = test["p"], test5["p"], p_drift
            v4[f"{arm}|{cls}"] = {**test, "role": "test" if arm in M_ARMS else "diagnostic"}
            v5[f"{arm}|{cls}"] = {**test5, "role": "test" if arm in M_ARMS else "diagnostic", "reliability": reliability}
            v6[f"{arm}|{cls}"] = {**v6_row, "role": "test" if arm in M_ARMS else "diagnostic"}
        for store, pv in ((v4, p4), (v5, p5), (v6, p6)):
            for arm, res in holm(pv).items():
                store[f"{arm}|{cls}"]["holm"] = res
    ok4 = all(not v["holm"]["reject"] and not v["nonzero_constant_columns"] for v in v4.values() if v["role"] == "test")
    ok5 = all(not v["holm"]["reject"] for v in v5.values() if v["role"] == "test")
    ok6 = all(not v["holm"]["reject"] and v["identity_max_abs_error"] < 1e-12 for v in v6.values() if v["role"] == "test")
    return {"V4": {"passed": ok4, "alpha": ALPHA, "rank_tolerance": RANK_TOL, "instruments": list(INSTRUMENTS),
                   "channels": list(CHANNELS), "results": v4},
            "V5": {"passed": ok5, "alpha": ALPHA, "times_seconds": [str(t) for t in V5_TIMES], "results": v5},
            "V6": {"passed": ok6, "alpha": ALPHA, "results": v6},
            "V10": {"passed": None, "status": "report", **v10}}


def train_fidelity(calibration) -> dict:
    """Top-3 share of |dL| (per 12-pair hour block) and excess kurtosis of 300-s dL on train data (recomputed)."""
    d = pd.read_parquet(DYNAMICS_CONFIG.parent/"training_noise_intervals.parquet")
    d = d[d.information_class.ne("pooled") & d.split.eq("train")]
    own = d.ticker.map(lambda t: calibration.config["templates"].get(t, {}).get("information_class"))
    d = d[own.eq(d.information_class) & d.duration_seconds.eq(300)]
    out = {}
    for cls, g in d.groupby("information_class"):
        kurt, top3 = [], []
        for _, t in g.groupby("ticker"):
            t = t.sort_values("left_time")
            y = t.delta_logodds_midpoint.to_numpy(dtype=float)
            if len(y) > 3 and y.std() > 0:
                kurt.append(float(stats.kurtosis(y, fisher=True)))
            for a in range(0, len(y)-11, 12):
                block = np.sort(np.abs(y[a:a+12]))[::-1]
                if block.sum() > 0:
                    top3.append(float(block[:3].sum()/block.sum()))
        out[cls] = {"contracts": int(g.ticker.nunique()), "pairs": int(len(g)),
                    "median_contract_excess_kurtosis_300s_dL": float(np.median(kurt)) if kurt else None,
                    "median_top3_share_abs_dL_per_12_pair_block": float(np.median(top3)) if top3 else None,
                    "pooled_excess_kurtosis_300s_dL": float(stats.kurtosis(g.delta_logodds_midpoint))}
    return out


# --------------------------------------------------------------- V1-V3
def v1_kernels() -> dict:
    checks = {}
    p = np.linspace(0.01, 0.99, 41)[:, None, None]
    a = np.array([0.0, 1e-6, 0.01, 0.3, 0.9, 1-1e-12])[None, :, None]
    u = np.array([0.02, 0.3, 0.5, 0.98])[None, None, :]
    a, u, p = np.broadcast_arrays(a, u, p)
    om = 1-a
    L = np.log(p)-np.log1p(-p)
    q = om*u+a*p
    up = expit(L+K.gm_llr(np.ones_like(p, dtype=np.int8), a, om, u))-p
    dn = expit(L+K.gm_llr(-np.ones_like(p, dtype=np.int8), a, om, u))-p
    checks["gm_buy_closed_form"] = float(np.max(np.abs(up-a*p*(1-p)/q)))
    checks["gm_sell_closed_form"] = float(np.max(np.abs(dn+a*p*(1-p)/(1-q))))
    checks["gm_zero_mean"] = float(np.max(np.abs(q*up+(1-q)*dn)))
    checks["gm_variance"] = float(np.max(np.abs(q*up**2+(1-q)*dn**2-a**2*p**2*(1-p)**2/(q*(1-q)))))
    c = np.array([0.01, 1.0, 37.5, 1e4])
    beta = 1.3e-5
    ah, omh = K.fg_half_alpha_terms(c, beta)
    checks["fg_half_llr_is_beta_c"] = float(max(np.max(np.abs(K.gm_llr(np.ones(4, np.int8), ah, omh, .5)-beta*c)/(beta*c)),
                                                np.max(np.abs(K.fg_half_llr(-np.ones(4, np.int8), c, beta)+beta*c))))
    mag = np.array([0.1, 1.0, 3.0, 12.0])
    checks["sign_tilt_llr"] = float(np.max(np.abs(np.log(K.jump_up_probability_m(1, mag)/K.jump_up_probability_m(0, mag))-mag)))
    pl = np.linspace(-6, 6, 25)[:, None]
    qa = K.jump_up_probability_theta_free(pl, mag[None, :])
    checks["jump_martingale"] = float(np.max(np.abs(qa*expit(pl+mag)+(1-qa)*expit(pl-mag)-expit(pl))))
    x = np.linspace(-3, 3, 13)
    checks["gaussian_llr_equals_x"] = float(np.max(np.abs(K.gaussian_llr(x, 2e-5, 300.0)-x)))
    rng = np.random.default_rng(7)
    n = 400_000
    vf = rng.random(n)
    s = K.flow_sign_theta_free(0.4, vf, 0.2, 0.8, 0.3)
    target = 0.8*0.3+0.2*expit(0.4)
    checks["theta_free_flow_law_z"] = float(abs((s == 1).mean()-target)/np.sqrt(target*(1-target)/n))
    js = K.jump_sign_theta_free(-0.7, rng.random(n), 1.5)
    tq = float(K.jump_up_probability_theta_free(-0.7, 1.5))
    checks["theta_free_jump_law_z"] = float(abs((js == 1).mean()-tq)/np.sqrt(tq*(1-tq)/n))
    new, _, _ = K.background_theta_free(np.full(n, 0.3), rng.random(n), 1e-3, 300.0, rng.normal(0, np.sqrt(300.0), n))
    pp = expit(new)
    checks["theta_free_background_martingale_z"] = float(abs(pp.mean()-expit(0.3))/(pp.std()/np.sqrt(n)))
    big = K.gm_llr(np.array([1, -1], np.int8), np.array([1-1e-12]*2), np.array([1e-12]*2), np.array([0.02, 0.98]))
    extreme = np.array([800.0, -800.0])
    checks["extreme_llr_finite"] = float(np.isfinite(big).all() and np.isfinite(extreme+big).all() and
                                         K.saturated(extreme).all())
    tol = {"gm_buy_closed_form": 1e-12, "gm_sell_closed_form": 1e-12, "gm_zero_mean": 1e-12, "gm_variance": 1e-12,
           "fg_half_llr_is_beta_c": 1e-14, "sign_tilt_llr": 1e-12, "jump_martingale": 1e-14,
           "gaussian_llr_equals_x": 1e-9}
    ok = all(checks[k] <= v for k, v in tol.items()) and all(
        checks[k] < 4.5 for k in ("theta_free_flow_law_z", "theta_free_jump_law_z", "theta_free_background_martingale_z")) \
        and checks["extreme_llr_finite"] == 1.0
    return {"passed": bool(ok), "checks": checks, "tolerances": tol}


def _v2_task(args):
    ticker, replicate, roster, arms, permute = args
    cal = _cal()
    sk = build_skeleton(ticker, replicate, calibration=cal)
    out = []
    rng = np.random.default_rng([replicate, 17])
    for name, label in arms:
        path = build_path(sk, _arm(name, ticker, roster, label), calibration=cal)
        signals, values = path.signals()
        ref = R.posterior_log_odds_path(path.arm.p0, signals)
        worst = float(np.max(np.abs(np.asarray(ref)-np.asarray(values))))
        perm = None
        if permute:
            order = rng.permutation(len(signals))
            perm = abs(R.posterior_log_odds_path(path.arm.p0, [signals[i] for i in order])[-1]-ref[-1])
        out.append({"arm": name+("" if label == "0.5" else "-emp"), "ticker": ticker, "max_abs": worst,
                    "nodes": len(values), "permutation_terminal_abs": perm})
    return out


def v2_v3(roster, rstar, rstar_emp, episodes, permuted, workers) -> dict:
    arms = [(a, "0.5") for a in M_ARMS]
    tasks = [(rstar[e % len(rstar)], OFFSETS["v2"]+e//len(rstar), roster, arms, e < permuted) for e in range(episodes)]
    if rstar_emp:
        tasks += [(rstar_emp[e % len(rstar_emp)], OFFSETS["v2"]+e//len(rstar_emp), roster,
                   [(a, "emp") for a in ("M0", "M1", "M2", "M3")], e < permuted) for e in range(episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(_STATE.get("calibration_dir"),)) as pool:
        for out in pool.map(_v2_task, tasks, chunksize=4):
            rows.extend(out)
    d = pd.DataFrame(rows)
    per_arm = d.groupby("arm").agg(episodes=("max_abs", "size"), max_abs=("max_abs", "max"),
                                   nodes=("nodes", "sum")).reset_index().to_dict("records")
    perm = d.dropna(subset=["permutation_terminal_abs"])
    return {"V2": {"passed": bool(d.max_abs.max() < 1e-12), "tolerance": 1e-12, "per_arm": per_arm},
            "V3": {"passed": bool(perm.permutation_terminal_abs.max() < 1e-9) if len(perm) else None,
                   "tolerance": 1e-9, "episodes": len(perm),
                   "max_terminal_abs_difference": float(perm.permutation_terminal_abs.max()) if len(perm) else None,
                   "note": "random permutations of all logged (fixed, p-independent) M signals; terminal posterior "
                           "is order-invariant up to float summation"}}


# ------------------------------------------------------------------ V8
def v8_hashes(roster, rstar, templates, replicates) -> dict:
    cal, hjb = _cal(), _STATE["hjb"]
    rows = []
    for ticker in rstar[:templates]:
        for r in range(replicates):
            sk = build_skeleton(ticker, OFFSETS["v8"]+r, calibration=cal)
            base = sk.base_input_hash()
            signal = {}
            for name in ("golden",)+L_DIAGNOSTIC_ARMS+M_ARMS:
                path = build_path(sk, _arm(name, ticker, roster) if name != "golden" else make_arm("golden"),
                                  calibration=cal)
                levels = ("S0", "S1") if path.arm.family == "M" else ("S0",)
                walks = set()
                for level in levels:
                    for policy in ("hjb", "fixed", "skew", "none"):
                        env = InformationTradingEnvironment(path, snipe_level=level)
                        run_policy(env, policy, hjb, seed=sk.path_seed)
                        walks.add(tuple(sorted(env.walk_hashes().items())))
                signal[name] = path.signal_path_hash()
                rows.append({"ticker": ticker, "replicate": r, "arm": name, "walk_hash_sets": len(walks),
                             "matches_path_trace": dict(next(iter(walks))) == path.walk_trace_hashes(),
                             "base_input_unchanged": sk.base_input_hash() == base})
            rows.append({"ticker": ticker, "replicate": r, "arm": "*", "distinct_signal_hashes":
                         len(set(signal.values())) == len(signal)})
    d = pd.DataFrame(rows)
    arms = d[d.arm.ne("*")]
    ok = bool((arms.walk_hash_sets == 1).all() and arms.matches_path_trace.all() and arms.base_input_unchanged.all()
              and d[d.arm.eq("*")].distinct_signal_hashes.all())
    return {"passed": ok, "episodes": int(d[d.arm.eq("*")].shape[0]), "arm_checks": len(arms)}


# ------------------------------------------------------------------ V9
def _v9_task(args):
    ticker, arm_name, start, stop, roster, sval = args
    cal = _cal()
    out = []
    for r in range(start, stop):
        sk = build_skeleton(ticker, OFFSETS["v9"]+r, calibration=cal)
        path = build_path(sk, _arm(arm_name, ticker, roster), calibration=cal)
        env = InformationTradingEnvironment(path)
        run_policy(env, "none", _STATE["hjb"], seed=sk.path_seed)
        pc = path.settlement["p_close"]
        elapsed = np.asarray(path.elapsed)
        prof = []
        for t in (900.0, 1800.0, 2700.0):
            k = int(np.searchsorted(elapsed, sk.start+t, side="right"))-1
            p = float(expit(path.l_after[k])) if k >= 0 else float(expit(path.initial_log_odds))
            prof.append(p*(1-p))
        out.append([pc*(1-pc), *prof, float(env.account.cash-env.account.initial_cash)])
    return ticker, arm_name, out


def v9_matching(roster, report: pd.DataFrame, rstar, templates, episodes, workers) -> dict:
    report = report.copy()
    in_r = report[report.ticker.isin(rstar) & report.arm.isin(["M0", "M1", "M2"]) & report.p0_label.eq("0.5")]
    gaps_ok = bool(in_r.status.eq("matched").all() and (in_r.gap.abs() < 0.02).all() and
                   ((in_r.ci_low <= 0) & (in_r.ci_high >= 0)).all())
    emp = report[report.ticker.isin(roster["R_star_emp"]["templates"]) & report.arm.isin(["M0", "M1", "M2"]) &
                 report.p0_label.eq("emp")]
    gaps_emp_ok = bool(emp.status.eq("matched").all() and (emp.gap.abs() < 0.02).all())
    # Cross-check templates: first broad and first sports template of R*, then the next R* template.
    groups = {t: g for g, t, _ in event_roster()}
    chosen = []
    for cls in ("broad_based", "sports"):
        chosen += [t for t in rstar if groups[t] == cls][:1]
    chosen += [t for t in rstar if t not in chosen][:max(0, templates-len(chosen))]
    chosen = chosen[:templates]
    tasks, chunk = [], 100
    for t in chosen:
        for arm in ("M0", "M1", "M2"):
            for a in range(0, episodes, chunk):
                tasks.append((t, arm, a, min(episodes, a+chunk), roster, None))
    collected = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(_STATE.get("calibration_dir"),)) as pool:
        for t, arm, out in pool.map(_v9_task, tasks):
            collected.setdefault((t, arm), []).extend(out)
    rows, pvals = [], {}
    for (t, arm), values in sorted(collected.items()):
        v = np.asarray(values)
        mean, se = float(v[:, 0].mean()), float(v[:, 0].std(ddof=1)/np.sqrt(len(v)))
        rep = in_r[(in_r.ticker == t) & (in_r.arm == arm)].iloc[0]
        z = (mean-rep.U_val)/np.hypot(se, rep.U_val_se)
        p = float(2*stats.norm.sf(abs(z)))
        pvals[f"{t}|{arm}"] = p
        rows.append({"ticker": t, "arm": arm, "s": float(rep.s), "episodes": len(v), "env_mean_pc_uncertainty": mean,
                     "env_se": se, "rb_U_val": float(rep.U_val), "rb_U_val_se": float(rep.U_val_se),
                     "U_star": float(rep.U_star), "z": float(z), "p": p,
                     "time_profile_mean_p_uncertainty": {"900": float(v[:, 1].mean()), "1800": float(v[:, 2].mean()),
                                                          "2700": float(v[:, 3].mean()), "close": mean},
                     "none_policy_max_abs_pnl": float(np.max(np.abs(v[:, 4])))})
    adjusted = holm(pvals)
    for row in rows:
        row["holm"] = adjusted[f"{row['ticker']}|{row['arm']}"]
    cross_ok = all(not r["holm"]["reject"] and r["none_policy_max_abs_pnl"] == 0 for r in rows)
    return {"passed": bool(gaps_ok and cross_ok), "r_star_gaps_ok": gaps_ok, "r_star_emp_gaps_ok": gaps_emp_ok,
            "r_star_rows": len(in_r), "max_abs_gap": float(in_r.gap.abs().max()) if len(in_r) else None,
            "cross_check": rows, "alpha": ALPHA}


# ----------------------------------------------------------------- V12
def crossing_exposure(path, ledger, schedule=TakerFeeSchedule(), tick="0.01") -> dict:
    """Snipe count and an unbiased crossed-exposure estimate (PASTA probes).

    Bid and ask proposals are independent Poisson(1/l) processes, independent of
    prices. Each side-Y proposal node therefore probes the other side X's state
    (alive order crossed at that node's exact unclipped p, same taker-fee rule)
    at a Poisson time, so E[# side-Y probes finding X crossed] = (1/l) E[time X is
    alive and crossed] = E[# side-X snipes] under the claimed hazard.
    """
    kinds, elapsed, p_bg = path.kinds, path.elapsed, path.p_bg
    rows = [r for r in ledger if r["kind"] in ("quote", "cancel", "fill", "snipe")]
    orders, k = {}, 0
    snipes = {"bid": 0, "ask": 0}
    probes_crossed = {"bid": 0, "ask": 0}      # keyed by the probed (crossed) side
    proposals = {"bid": 0, "ask": 0}
    for i in range(len(kinds)):
        if kinds[i] not in (PROPOSAL_BID, PROPOSAL_ASK):
            continue
        t = elapsed[i]
        # Order state strictly before this node (rows at earlier nodes with the same time are included).
        while k < len(rows) and rows[k]["elapsed_seconds"] < t-1e-12:
            r = rows[k]
            if r["kind"] == "quote":
                orders[r["side"]] = [r["price"], float(r["quantity"])]
            elif r["kind"] == "cancel":
                orders.pop(r["side"], None)
            else:
                o = orders.get(r["side"])
                if o is not None:
                    o[1] = round(o[1]-float(r["quantity"]), 10)
                    if o[1] <= 1e-12:
                        orders.pop(r["side"])
                if r["kind"] == "snipe":
                    snipes[r["side"]] += 1
            k += 1
        side = "bid" if kinds[i] == PROPOSAL_BID else "ask"
        other = "ask" if side == "bid" else "bid"
        proposals[side] += 1
        o = orders.get(other)
        if o is None:
            continue
        p, m = float(p_bg[i]), float(o[0])
        gap = p-m if other == "ask" else m-p
        if gap > 0.005 and snipe_profitable(other, str(o[1]), o[0], p, tick, schedule):
            probes_crossed[other] += 1
    for r in rows[k:]:
        if r["kind"] == "snipe":
            snipes[r["side"]] += 1
    S, P = sum(snipes.values()), sum(probes_crossed.values())
    return {"snipes": S, "probes_crossed": P, "snipes_by_side": snipes, "probes_crossed_by_side": probes_crossed,
            "proposals": proposals, "exposure_seconds": P*path.skeleton.mean_latency}


def hazard_test(snipes: int, probes: int, mean_latency: float) -> dict:
    """Rate = snipes / (probes * l) with an exact conditional binomial CI (S | S + P ~ Bin(n, 1/2) under H0)."""
    n = snipes+probes
    if n == 0:
        return {"rate_per_second": None, "ci95": [None, None], "p": None}
    test = stats.binomtest(snipes, n, 0.5)
    ci = test.proportion_ci(0.95, method="exact")
    ratio = lambda q: q/(1-q) if q < 1 else np.inf
    return {"rate_per_second": snipes/(probes*mean_latency) if probes else np.inf,
            "ci95": [ratio(ci.low)/mean_latency, ratio(ci.high)/mean_latency], "p": float(test.pvalue)}


def _v12_task(args):
    ticker, replicate, roster, arms = args
    cal, hjb = _cal(), _STATE["hjb"]
    sk = build_skeleton(ticker, OFFSETS["v12"]+replicate, calibration=cal)
    out = []
    for name in arms:
        path = build_path(sk, _arm(name, ticker, roster), calibration=cal)
        for policy in ("fixed", "hjb"):
            env = InformationTradingEnvironment(path, snipe_level="S1")
            run_policy(env, policy, hjb, seed=sk.path_seed)
            res = crossing_exposure(path, env.ledger)
            if res["snipes"] != env.snipe_count:
                raise AssertionError("Snipe count reconstruction failed")
            out.append({"ticker": ticker, "replicate": replicate, "arm": name, "policy": policy, "snipes": res["snipes"],
                        "probes_crossed": res["probes_crossed"]})
    return out


def v12_hazard(roster, rstar, replicates, workers, mean_latency=2.0) -> dict:
    tasks = [(t, r, roster, ("M0", "M1", "M2", "M3")) for t in rstar for r in range(replicates)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(_STATE.get("calibration_dir"),)) as pool:
        for out in pool.map(_v12_task, tasks, chunksize=2):
            rows.extend(out)
    d = pd.DataFrame(rows)
    S, P = int(d.snipes.sum()), int(d.probes_crossed.sum())
    target = 1.0/mean_latency
    clustered = cluster_ratio_test(d, mean_latency)
    return {"passed": bool(clustered["ci95"][0] <= target <= clustered["ci95"][1]) if P > 0 else None,
            "snipes": S, "probes_crossed": P, "exposure_seconds_estimate": P*mean_latency, **clustered,
            "pooled_binomial_secondary": hazard_test(S, P, mean_latency), "target_rate": target,
            "episodes": int(len(d)), "by_arm": d.groupby("arm")[["snipes", "probes_crossed"]].sum().to_dict("index"),
            "note": "PASTA probe: opposite-side proposals (independent Poisson(1/l)) sample the crossed state; "
                    "exposure estimate = probes finding the order crossed x l. Primary CI: ratio estimator with "
                    "cluster-robust (template, replicate) delta-method SE, because arms/policies share skeletons; "
                    "the pooled binomial CI (secondary) treats them as independent and is anti-conservative.",
            "grid_convergence": "not run: zero-latency grid (Tier 2) is not implemented"}


def cluster_ratio_test(d: pd.DataFrame, mean_latency: float) -> dict:
    """rate = sum S / (sum P * l); delta-method SE clustered on (ticker, replicate)."""
    c = d.groupby(["ticker", "replicate"])[["snipes", "probes_crossed"]].sum()
    S, P = c.snipes.to_numpy(float), c.probes_crossed.to_numpy(float)
    n = len(c)
    if P.sum() == 0 or n < 2:
        return {"rate_per_second": None, "ci95": [None, None], "p": None, "clusters": n}
    ratio = S.sum()/P.sum()
    resid = S-ratio*P
    se = float(np.sqrt(n/(n-1)*np.sum(resid**2))/P.sum())
    z = (ratio-1.0)/se if se > 0 else 0.0
    return {"rate_per_second": ratio/mean_latency, "ci95": [(ratio-1.959963984540054*se)/mean_latency,
            (ratio+1.959963984540054*se)/mean_latency], "p": float(2*stats.norm.sf(abs(z))), "clusters": n,
            "snipe_to_probe_ratio": float(ratio), "ratio_se": se}


# ----------------------------------------------------------- V7, V11, V13
def v7_golden_anchor(scale) -> dict:
    cal = _cal()
    tickers = None
    if scale["golden_tickers"]:
        roster = event_roster()
        tickers = [roster[0][1], roster[len(roster)//2][1], roster[-1][1]][:scale["golden_tickers"]]
    t0 = time.perf_counter()
    golden = golden_equivalence_sweep(cal, seeds=scale["golden_seeds"], tickers=tickers, hjb=_STATE["hjb"])
    t1 = time.perf_counter()
    anchor_tickers = None
    if scale["anchor_tickers"]:
        anchor_tickers = [t for _, t, _ in event_roster()][::12][:scale["anchor_tickers"]]
    anchor = l0_anchor_check(SAVED_BENCHMARK, hjb=_STATE["hjb"], tickers=anchor_tickers,
                             replicates=scale["anchor_replicates"])
    return {"passed": bool(anchor["max_abs_difference"] <= 1e-9), "golden_episodes": len(golden),
            "golden_templates": len({r["ticker"] for r in golden}), "golden_seconds": t1-t0,
            "anchor": anchor, "anchor_seconds": time.perf_counter()-t1}


def v11_from_experiment(experiment_dir: Path) -> dict:
    manifest_path = experiment_dir/"manifest.json"
    if not manifest_path.exists():
        return {"passed": None, "status": "not_run: no experiment manifest (per-episode audit runs inside evaluate.py)"}
    manifest = json.loads(manifest_path.read_text())
    data = pd.read_csv(experiment_dir/"episode_results.csv", usecols=["attribution_error", "policy", "realized_pnl",
                                                                       "arm", "ticker", "replicate", "signal_sha256",
                                                                       "book_sha256", "settlement_sha256"])
    hashes_ok = bool((data.groupby(["arm", "ticker", "replicate"])[["signal_sha256", "book_sha256", "settlement_sha256"]]
                      .nunique() == 1).all().all())
    return {"passed": bool(manifest["checks"]["all_passed"] and data.attribution_error.abs().max() < 1e-8 and
                           data[data.policy.eq("none")].realized_pnl.eq(0).all() and hashes_ok),
            "episodes": len(data), "attribution_max_abs_error": float(data.attribution_error.abs().max()),
            "hashes_shared_within_arm": hashes_ok, "manifest_checks": manifest["checks"]}


def v13_isolation(run_c2: bool) -> dict:
    cal = _cal()
    report = json.loads((cal.directory/"calibration_report.json").read_text())
    splits = pd.read_csv(ROOT/"data_pipeline/output/paper2/calibration/event_splits.csv")
    train = set(splits[splits.split.eq("train")].ticker) if "split" in splits else set()
    roster = {t for _, t, _ in event_roster()}
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    lines = [l for l in status.splitlines() if l.strip()]
    tracked_changes = [l for l in lines if not l.startswith("??")]
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-status", "HEAD"], capture_output=True, text=True).stdout
    preserved = ["hjb_baseline", "simulator", "calibration", "data_pipeline/output/paper2/calibration/simulator"]
    c4 = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *preserved], capture_output=True,
                        text=True).stdout
    c4_changes = [l for l in c4.splitlines() if l.strip() and not l.startswith("??")]
    c2 = {"status": "not_run (use --c2 to validate a temporary copy of the Paper 2 hjb_benchmark directory)"}
    if run_c2:
        from ..validate_hjb_benchmark import validate
        src = ROOT/"data_pipeline/output/paper2/calibration/simulator/hjb_benchmark"
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)/"hjb_benchmark"
            shutil.copytree(src, copy)
            result = validate(copy)
        c2 = {"status": "run_on_temporary_copy", "passed": bool(result.get("passed"))}
    ok = (report.get("heldout_files_loaded") == 0 and roster <= train and not tracked_changes and not diff.strip()
          and not c4_changes and c2.get("passed", True))
    return {"passed": bool(ok), "calibration_heldout_files_loaded": report.get("heldout_files_loaded"),
            "event_templates_all_train": roster <= train, "C1_tracked_changes": tracked_changes,
            "C1_git_diff_head": diff.strip(), "C4_preserved_paths_changed": c4_changes, "C2": c2,
            "untracked_entries": len(lines)-len(tracked_changes)}


# --------------------------------------------------------------------- run
def run(output=None, *, scale_name="full", experiment_dir=DEFAULT_EXPERIMENT, workers=None, only=None,
        calibration_dir=None, run_c2=False) -> dict:
    t0 = time.perf_counter()
    scale = SCALES[scale_name]
    experiment_dir = Path(experiment_dir)
    _STATE["calibration_dir"] = str(calibration_dir) if calibration_dir else None
    _init(_STATE["calibration_dir"])
    cal = _STATE["calibration"]
    roster = load_roster(experiment_dir/"roster.json")
    report = pd.read_csv(experiment_dir/"matching_report.csv")
    rstar, rstar_emp = roster["R_star"]["templates"], roster["R_star_emp"]["templates"]
    groups = {t: g for g, t, _ in event_roster()}
    by_class = {c: [t for t in rstar if groups[t] == c] for c in ("broad_based", "sports")}
    workers = workers or max(1, (os.cpu_count() or 2)-1)
    _single_threaded_children()
    want = lambda key: only is None or key in only
    result, timings = {}, {}

    def timed(key, fn):
        s = time.perf_counter()
        out = fn()
        timings[key] = time.perf_counter()-s
        return out

    if want("V1"):
        result["V1"] = timed("V1", v1_kernels)
    if want("V2") or want("V3"):
        result.update(timed("V2_V3", lambda: v2_v3(roster, rstar, rstar_emp, scale["v2_episodes"],
                                                   scale["v3_episodes"], workers)))
    if any(want(k) for k in ("V4", "V5", "V6", "V10")):
        collected = timed("V4_generation", lambda: run_v4_family(roster, by_class, scale["v4_episodes"], workers,
                                                                 M_ARMS+L_DIAGNOSTIC_ARMS))
        result.update(timed("V4_tests", lambda: v4_v5_v6_v10(collected, cal)))
        result["V10"]["flow_replay"] = {c: {k: cal.config["classes"][c]["flow"].get(k) for k in (
            "target_lambda_median", "calibration_statistic", "calibration_relative_gap", "validation")}
            for c in ("broad_based", "sports")}
        result["V10"]["calibration_jump_counts"] = {c: cal.config["classes"][c]["jumps"].get("buckets") for c in
                                                    ("broad_based", "sports")}
        from .matching import TemplateDesign
        result["V10"]["expected_jumps_per_episode_by_class"] = {
            c: float(np.mean([sum(TemplateDesign.build(t, cal).jump_means) for t in ts])) for c, ts in by_class.items()
            if ts}
        del collected
    if want("V7"):
        result["V7"] = timed("V7", lambda: v7_golden_anchor(scale))
    if want("V8"):
        result["V8"] = timed("V8", lambda: v8_hashes(roster, rstar, scale["v8_templates"], scale["v8_replicates"]))
    if want("V9"):
        result["V9"] = timed("V9", lambda: v9_matching(roster, report, rstar, scale["v9_templates"],
                                                       scale["v9_episodes"], workers))
    if want("V11"):
        result["V11"] = v11_from_experiment(experiment_dir)
    if want("V12"):
        result["V12"] = timed("V12", lambda: v12_hazard(roster, rstar, scale["v12_replicates"], workers))
    if want("V13"):
        result["V13"] = timed("V13", lambda: v13_isolation(run_c2))
    tested = {k: v.get("passed") for k, v in result.items()}
    out = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "scale": scale_name, "scale_parameters": scale,
           "experiment_dir": str(experiment_dir), "roster_sha256": roster["roster_sha256"], "workers": workers,
           "cpu_count": os.cpu_count(), "summary": tested,
           "passed": bool(all(v is not False for v in tested.values())),
           "timings_seconds": {**timings, "total": time.perf_counter()-t0}, "results": result}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        from .evaluate import _finite
        Path(output).write_text(json.dumps(_finite(json.loads(json.dumps(
            out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))), indent=2, allow_nan=False)+"\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", choices=list(SCALES), default="full")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, help="JSON report (default: EXPERIMENT_DIR/validation_report.json)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--c2", action="store_true", help="validate a temporary copy of the Paper 2 benchmark (C-2)")
    args = parser.parse_args()
    output = args.output or args.experiment_dir/"validation_report.json"
    res = run(output, scale_name=args.scale, experiment_dir=args.experiment_dir, workers=args.workers, only=args.only,
              calibration_dir=args.calibration_dir, run_c2=args.c2)
    print(json.dumps({"passed": res["passed"], "summary": res["summary"], "timings": res["timings_seconds"],
                      "output": str(output)}, indent=2))
