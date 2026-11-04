"""Training-only Paper 3 information-arrival calibration (implementation spec Section 3).

Decomposes the legacy quote-midpoint residual variance into a continuous
background (sigma_c) and predeclared reduced-form jump proxies (intensity by
time-to-release bucket and an empirical magnitude bank), estimates uninformed
YES-taker rates u, calibrates the informed-flow intensity eta of
alpha(c; eta) = min(1 - exp(-eta c), 1 - 1e-12) by a frozen-uniform flow-only
replay to the train-median Kyle lambda, and records empirical starting
probabilities and release-time verification. Validation/test tickers are
refused at every raw-file access. Units: seconds; log odds; contracts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, normalize_trades, safe_filename
from calibrate_market_dynamics import residual_noise_fit
from estimate_lambda import interval_subset

sys.path.insert(0, str(PROJECT_ROOT))
from simulator.information import kernels as K

CLASSES = ("broad_based", "sports")
BUCKETS = K.TTR_BUCKETS
PAIR_SECONDS = 300
JUMP_THRESHOLD = 4.0
BIPOWER_HALF_WINDOW = 12
MIN_LOCAL_PRODUCTS = 12
MIN_BUCKET_JUMPS = 5
POOL_ORDER = {"B0": ("B0", "B1", "B2", "B3"), "B1": ("B1", "B0", "B2", "B3"),
              "B2": ("B2", "B1", "B3", "B0"), "B3": ("B3", "B2", "B1", "B0")}
N_SIZE_BINS, U_PRIOR_WEIGHT, U_CLIP = 5, 20.0, (0.02, 0.98)
EPISODE_SECONDS = 3600.0
P0_CLIP = (0.01, 0.99)
REPLAY = {"initial_replications": 200, "max_replications": 51_200, "batches": 20, "bisection_tolerance": 0.001,
          "mc_se_tolerance": 0.0025, "validation_tolerance": 0.02, "bracket_start": 1e-6, "max_doublings": 60,
          "max_bisection_steps": 200, "seed": 20261031}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT/"data_pipeline"/"output"/"paper3"/"calibration"


class HeldOutAccessError(ValueError):
    """Raised before any validation/test raw file is opened."""


class TrainingGuard:
    """Single gate for raw-file access; only event-split training tickers pass."""
    def __init__(self, splits: pd.DataFrame):
        if splits.groupby("event_cluster").split.nunique().gt(1).any():
            raise ValueError("An event cluster crosses data splits")
        self.train = frozenset(splits.loc[splits.split.eq("train"), "ticker"])
        self.heldout = frozenset(splits.ticker)-self.train
        self.loaded: list[tuple[str, str]] = []

    def require(self, ticker: str, kind: str):
        if ticker not in self.train:
            raise HeldOutAccessError(f"Refusing {kind} access for non-training ticker {ticker}")
        self.loaded.append((kind, ticker))

    @property
    def heldout_files_loaded(self) -> int:
        return sum(t in self.heldout for _, t in self.loaded)


# ------------------------------------------------------------- Section 2 times
def release_record(ticker: str, template: dict, metadata: dict) -> dict:
    """t_c = open + lifetime; t_r per class rule; consistency flags (no data repair)."""
    cls = template["information_class"]
    t_c = pd.Timestamp(template["open_time"])+pd.Timedelta(seconds=float(template["lifetime_seconds"]))
    occurrence, flags = metadata.get("occurrence_datetime"), []
    if cls == "sports":
        t_r, source = t_c, "template_close"
        flags.append("provisional_sports_close")
    elif occurrence:
        t_r, source = pd.Timestamp(occurrence), "occurrence_datetime"
    else:
        t_r, source = t_c, "template_close"
        flags.append("close_fallback")
    expected, settled = metadata.get("expected_expiration_time"), metadata.get("settlement_ts")
    if pd.Timestamp(metadata["close_time"]) != t_c:
        flags.append("metadata_close_differs_from_template_close")
    if occurrence and expected and pd.Timestamp(occurrence) > pd.Timestamp(expected):
        flags.append("occurrence_after_expected_expiration")
    if occurrence and settled and pd.Timestamp(occurrence) > pd.Timestamp(settled):
        flags.append("occurrence_after_settlement_ts")
    early = bool(t_r < t_c)
    if early:
        flags.append("early_revelation")
    rules = " ".join(str(metadata.get("rules_primary") or "").split())
    return {"ticker": ticker, "information_class": cls, "series_ticker": metadata.get("series_ticker", ""),
            "t_c_utc": t_c.isoformat(), "t_r_utc": t_r.isoformat(), "t_r_source": source,
            "release_minus_close_seconds": float((t_r-t_c).total_seconds()), "early_revelation": early,
            "occurrence_datetime": occurrence, "expected_expiration_time": expected, "settlement_ts": settled,
            "metadata_close_time": metadata.get("close_time"), "flags": ";".join(flags),
            "rules_excerpt": rules[:240]}


# -------------------------------------------------------- 3.1 normalization
def effective_innovations(pairs: pd.DataFrame, beta: float, stored_sigma: float, min_pairs: int = 50) -> dict:
    """Legacy residual regression -> eta, d; eta_tilde = eta*sqrt(n/(n-d)).

    Asserts the recomputed legacy sigma^2 = sum eta^2/((n-d)*300) equals the stored
    template sigma^2 (relative 1e-10) and that sum eta_tilde^2/(n*300) equals it too.
    """
    d = pairs.sort_values("left_time", kind="stable").reset_index(drop=True)
    if not d.duration_seconds.eq(PAIR_SECONDS).all() or not (d.right_time-d.left_time).eq(pd.Timedelta(seconds=PAIR_SECONDS)).all():
        raise ValueError("Residual normalization requires full 300-s observed quote pairs")
    fit = residual_noise_fit(d, beta, min_pairs)
    if fit["fit_status"] != "ok":
        raise ValueError(f"Unsupported noise template: {fit['fit_status']}")
    flow = d.signed_flow_contracts.to_numpy(dtype=float)
    y = d.delta_logodds_midpoint.to_numpy(dtype=float)
    fc, yc = flow-flow.mean(), y-y.mean()
    slope = fit["quote_flow_slope"]
    eta = yc-(slope*fc if slope is not None else 0.0)
    n, dof = len(d), (2 if slope is not None else 1)
    sigma2 = float(eta@eta)/((n-dof)*PAIR_SECONDS)
    for reference, label in ((fit["sigma_logodds_per_sqrt_second"]**2, "fit"), (float(stored_sigma)**2, "stored template")):
        if not np.isclose(sigma2, reference, rtol=1e-10, atol=0):
            raise AssertionError(f"Recomputed legacy sigma^2 differs from {label} sigma^2")
    eta_tilde = eta*np.sqrt(n/(n-dof))
    if not np.isclose(float(eta_tilde@eta_tilde)/(n*PAIR_SECONDS), sigma2, rtol=1e-10, atol=0):
        raise AssertionError("Effective innovations do not reproduce legacy sigma^2")
    consecutive = np.r_[False, d.left_time.iloc[1:].to_numpy() == d.right_time.iloc[:-1].to_numpy()]
    return {"pairs": d, "eta": eta, "eta_tilde": eta_tilde, "n": n, "d": dof, "sigma2_legacy": sigma2,
            "consecutive": consecutive, "fit": fit}


def own_class_intervals(intervals: pd.DataFrame, templates: dict) -> pd.DataFrame:
    """Keep each contract's own-class rows; `pooled` rows are duplicates and are dropped."""
    d = intervals[intervals.information_class.ne("pooled")]
    own = d.ticker.map(lambda t: templates.get(t, {}).get("information_class"))
    d = d[own.eq(d.information_class)]
    if d.duplicated(["ticker", "left_time"]).any():
        raise AssertionError("Duplicate (ticker, left_time) after dropping pooled-profile rows")
    return d


def pair_ttr_buckets(pairs: pd.DataFrame, release_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """TTR at each pair's LEFT endpoint (t_r - left_time, seconds) and its bucket index (-1 after release)."""
    ttr = (pd.Timestamp(release_time)-pairs.left_time).dt.total_seconds().to_numpy(dtype=float)
    return ttr, K.ttr_bucket_index(ttr)


# -------------------------------------------------------- 3.2 jump detection
def adjacent_products(eta_tilde: np.ndarray, consecutive: np.ndarray) -> np.ndarray:
    """products[i] = |eta_i||eta_{i-1}| when pairs i-1, i are consecutive; else NaN."""
    e = np.abs(np.asarray(eta_tilde, dtype=float))
    out = np.full(len(e), np.nan)
    if len(e) > 1:
        out[1:] = e[1:]*e[:-1]
    out[~np.asarray(consecutive, dtype=bool)] = np.nan
    return out


def local_bipower_variance(eta_tilde: np.ndarray, consecutive: np.ndarray, half_window: int = BIPOWER_HALF_WINDOW,
                           min_products: int = MIN_LOCAL_PRODUCTS) -> tuple[np.ndarray, np.ndarray]:
    """Per-300-s local variance sigma_hat_k^2 = (pi/2) mean(products) near k.

    Product i (= |eta_i||eta_{i-1}|, indexed by its later pair) is within the
    window when |i - k| <= half_window observed pairs; any product involving
    pairs k-1, k or k+1 (i in {k-1, k, k+1, k+2}) is excluded. Fewer than
    `min_products` -> contract-level (pi/2) mean of all products. Returns
    (sigma_hat^2, used_local).
    """
    products = adjacent_products(eta_tilde, consecutive)
    valid = ~np.isnan(products)
    if not valid.any():
        raise ValueError("No consecutive observed pairs: bipower variation undefined")
    contract_level = np.pi/2*float(products[valid].mean())
    n = len(products)
    sigma2, local = np.empty(n), np.zeros(n, dtype=bool)
    for k in range(n):
        idx = np.arange(max(1, k-half_window), min(n-1, k+half_window)+1)
        idx = idx[(idx < k-1) | (idx > k+2)]
        values = products[idx][valid[idx]]
        if len(values) >= min_products:
            sigma2[k], local[k] = np.pi/2*float(values.mean()), True
        else:
            sigma2[k] = contract_level
    return sigma2, local


def detect_jumps(eta_tilde: np.ndarray, consecutive: np.ndarray, threshold: float = JUMP_THRESHOLD) -> dict:
    sigma2, local = local_bipower_variance(eta_tilde, consecutive)
    jump = np.abs(eta_tilde) > threshold*np.sqrt(sigma2)
    return {"jump": jump, "sigma_hat2": sigma2, "local": local}


# --------------------------------------------------- 3.3 continuous background
def decompose(eta_tilde: np.ndarray, jump: np.ndarray, sigma2_legacy: float) -> dict:
    n = len(eta_tilde)
    sigma_c2 = float(eta_tilde[~jump]@eta_tilde[~jump])/(n*PAIR_SECONDS)
    jump_var = float(eta_tilde[jump]@eta_tilde[jump])/(n*PAIR_SECONDS)
    error = abs(sigma_c2+jump_var-sigma2_legacy)
    if not np.isclose(sigma_c2+jump_var, sigma2_legacy, rtol=1e-10, atol=0):
        raise AssertionError("Continuous + jump variance does not reproduce legacy sigma^2")
    return {"sigma_c2": sigma_c2, "sigma_c": float(np.sqrt(sigma_c2)), "jump_variance_rate": jump_var,
            "identity_abs_error": float(error)}


# ------------------------------------------------------ 3.4 jump intensities
def pooled_intensity(counts: dict, exposure: dict, min_jumps: int = MIN_BUCKET_JUMPS) -> dict:
    """lambda_J per bucket with deterministic neighbour pooling of counts and exposure."""
    out = {}
    for b in BUCKETS:
        used, c, e = [], 0, 0.0
        for nb in POOL_ORDER[b]:
            used.append(nb)
            c, e = c+int(counts[nb]), e+float(exposure[nb])
            if c >= min_jumps:
                break
        if e <= 0:
            raise ValueError("Jump intensity has no exposure")
        out[b] = {"lambda_per_second": c/e, "pooled_count": c, "pooled_exposure_seconds": e,
                  "contributing_buckets": used, "pooled": len(used) > 1, "reached_min_jumps": c >= min_jumps}
    return out


def bucket_tables(rows: pd.DataFrame) -> tuple[dict, dict]:
    counts = {b: int(rows[rows.bucket.eq(b)].is_jump.sum()) for b in BUCKETS}
    exposure = {b: float(rows.bucket.eq(b).sum()*PAIR_SECONDS) for b in BUCKETS}
    return counts, exposure


def intensity_bootstrap(rows: pd.DataFrame, draws: int, rng) -> dict:
    """Contract-resampling bootstrap of lambda_J(b) (same pooling rule) and E[a^2]."""
    tickers = sorted(rows.ticker.unique())
    per = {t: (bucket_tables(g), (g.eta_tilde[g.is_jump]**2).to_numpy()) for t, g in rows.groupby("ticker")}
    lam, a2, short = {b: [] for b in BUCKETS}, [], 0
    for _ in range(draws):
        chosen = [tickers[i] for i in rng.integers(0, len(tickers), len(tickers))]
        counts = {b: sum(per[t][0][0][b] for t in chosen) for b in BUCKETS}
        exposure = {b: sum(per[t][0][1][b] for t in chosen) for b in BUCKETS}
        short += sum(counts.values()) < MIN_BUCKET_JUMPS
        for b, rec in pooled_intensity(counts, exposure).items():
            lam[b].append(rec["lambda_per_second"])
        sq = np.concatenate([per[t][1] for t in chosen])
        a2.append(float(sq.mean()) if len(sq) else np.nan)
    a2 = np.asarray(a2)
    return {"lambda_bootstrap95": {b: [float(x) for x in np.quantile(lam[b], [.025, .975])] for b in BUCKETS},
            "mean_a2_bootstrap95": [float(x) for x in np.nanquantile(a2, [.025, .975])],
            "draws": draws, "draws_without_jumps": int(np.isnan(a2).sum()), "draws_below_min_total_jumps": int(short)}


# --------------------------------------------------------------- 3.5 u rates
def size_bin_edges(quantities: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(quantities, dtype=float), np.arange(1, N_SIZE_BINS)/N_SIZE_BINS)


def size_bins(quantities, edges) -> np.ndarray:
    """Bin b covers [edge_{b-1}, edge_b): number of interior edges <= c."""
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(quantities, dtype=float), side="right")


def uninformed_rates(marks: dict, n_phases: int) -> dict:
    """marks: {ticker: (quantity, sign, phase)} for one class (pooled class marks)."""
    q = np.concatenate([m[0] for m in marks.values()])
    s = np.concatenate([m[1] for m in marks.values()])
    K.check_signs(s)
    edges = size_bin_edges(q)
    b = size_bins(q, edges)
    overall = float((s == 1).mean())
    u_class, empty_class_bins = np.empty(N_SIZE_BINS), []
    for j in range(N_SIZE_BINS):
        mask = b == j
        if mask.any():
            u_class[j] = float((s[mask] == 1).mean())
        else:
            u_class[j] = overall
            empty_class_bins.append(j)
    templates = {}
    for ticker, (tq, ts, tp) in marks.items():
        tb = size_bins(tq, edges)
        n = np.zeros((N_SIZE_BINS, n_phases)); yes = np.zeros((N_SIZE_BINS, n_phases))
        np.add.at(n, (tb, tp), 1); np.add.at(yes, (tb, tp), (ts == 1).astype(float))
        rate = np.clip((yes+U_PRIOR_WEIGHT*u_class[:, None])/(n+U_PRIOR_WEIGHT), *U_CLIP)
        nb, yb = n.sum(axis=1), yes.sum(axis=1)
        templates[ticker] = {"u_bin_phase": rate, "u_bin": np.clip((yb+U_PRIOR_WEIGHT*u_class)/(nb+U_PRIOR_WEIGHT), *U_CLIP),
                             "n_bin_phase": n, "yes_bin_phase": yes}
    return {"edges": edges, "u_class": np.clip(u_class, *U_CLIP), "u_class_raw": u_class, "overall_yes_rate": overall,
            "empty_class_bins": empty_class_bins, "templates": templates, "pooled_marks": len(q)}


# -------------------------------------------------------- 3.6 flow replay
class FlowReplay:
    """Flow-only post-trade replay with frozen per-(replication, bucket, trade) uniforms.

    Trades are contiguous by bucket, buckets contiguous by contract. Per bucket:
    Theta = 1{V_Theta < p_start}; I = 1{V_I < alpha}; S_u = 2*1{V_u < u} - 1;
    sign = 2Theta-1 if I else S_u; observed open = L after the first trade, so
    Delta L = sum of LLRs after the first trade; F = sum sign*c over all trades.
    """
    def __init__(self, sizes, u, trade_bucket, bucket_contract, p_start, seed: int, class_index: int,
                 contracts: list[str] | None = None, chunk: int = 50, cache_floats: float = 6e7):
        self.c = np.asarray(sizes, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.trade_bucket = np.asarray(trade_bucket)
        self.bucket_contract = np.asarray(bucket_contract)
        self.p_start = np.asarray(p_start, dtype=float)
        if (np.any(np.diff(self.trade_bucket) < 0) or np.any(np.diff(self.bucket_contract) < 0) or
                set(np.unique(self.trade_bucket)) != set(range(len(self.p_start)))):
            raise ValueError("Replay trades/buckets must be contiguous and sorted")
        if np.any(self.c <= 0) or np.any((self.u <= 0) | (self.u >= 1)) or np.any((self.p_start <= 0) | (self.p_start >= 1)):
            raise ValueError("Invalid replay sizes, u rates or opening prices")
        self.bucket_start = np.flatnonzero(np.r_[True, np.diff(self.trade_bucket) != 0])
        self.contract_start = np.flatnonzero(np.r_[True, np.diff(self.bucket_contract) != 0])
        self.first = np.zeros(len(self.c), dtype=bool); self.first[self.bucket_start] = True
        self.n_contracts = len(self.contract_start)
        self.contracts = contracts or [str(i) for i in range(self.n_contracts)]
        self.seed, self.class_index, self.chunk = int(seed), int(class_index), int(chunk)
        self.cache, self.cache_budget, self.cached_floats = {}, cache_floats, 0
        self.degenerate_slopes = 0

    def _uniforms(self, tag: int, chunk_index: int, r1: int):
        key = (tag, chunk_index)
        if key in self.cache:
            return self.cache[key]
        r0 = chunk_index*self.chunk
        rows = []
        for r in range(r0, r1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.class_index, tag, r]))
            rows.append((rng.random(len(self.p_start)), rng.random(len(self.c)), rng.random(len(self.c))))
        value = tuple(np.vstack(x) for x in zip(*rows))
        size = sum(v.size for v in value)
        if r1-r0 == self.chunk and self.cached_floats+size <= self.cache_budget:
            self.cache[key] = value
            self.cached_floats += size
        return value

    def slopes(self, law: str, parameter: float, replications: int, tag: int = 0) -> tuple[np.ndarray, dict]:
        """Per-replication contract slopes (R x contracts) and pooled sign/size moments."""
        if law == "fg":
            alpha, one_minus = K.fg_alpha_terms(self.c, parameter)
            u = self.u
            buy, sell = (K.gm_llr(np.full(len(self.c), s), alpha, one_minus, u) for s in (1, -1))
        elif law == "constant":
            alpha, one_minus = K.constant_alpha_terms(parameter, len(self.c))
            u = self.u
            buy, sell = (K.gm_llr(np.full(len(self.c), s), alpha, one_minus, u) for s in (1, -1))
        elif law == "fg_half":
            alpha, _ = K.fg_half_alpha_terms(self.c, parameter)
            u = np.full(len(self.c), .5)
            buy, sell = (K.fg_half_llr(np.full(len(self.c), s), self.c, parameter) for s in (1, -1))
        else:
            raise ValueError(f"Unknown replay flow law {law}")
        out, moments = [], {"trades": 0, "buys": 0.0, "signed_contracts": 0.0, "informed": 0.0,
                            "bucket_abs_flow": 0.0, "buckets": 0}
        for ci, r0 in enumerate(range(0, replications, self.chunk)):
            r1 = min(replications, r0+self.chunk)
            vt, vi, vu = self._uniforms(tag, ci, r1)
            vt, vi, vu = vt[:r1-r0], vi[:r1-r0], vu[:r1-r0]
            theta = (vt < self.p_start[None, :])[:, self.trade_bucket]
            informed = vi < alpha[None, :]
            sign = np.where(informed, 2*theta.astype(np.int8)-1, K.uninformed_sign(vu, u[None, :]))
            K.check_signs(sign)
            llr = np.where(sign == 1, buy[None, :], sell[None, :])
            llr[:, self.first] = 0.0
            dL = np.add.reduceat(llr, self.bucket_start, axis=1)
            F = np.add.reduceat(sign*self.c[None, :], self.bucket_start, axis=1)
            out.append(self._contract_slopes(F, dL))
            moments["trades"] += sign.size; moments["buys"] += float((sign == 1).sum())
            moments["signed_contracts"] += float((sign*self.c[None, :]).sum()); moments["informed"] += float(informed.sum())
            moments["bucket_abs_flow"] += float(np.abs(F).sum()); moments["buckets"] += F.size
        return np.vstack(out), moments

    def _contract_slopes(self, F, dL):
        counts = np.diff(np.r_[self.contract_start, F.shape[1]])
        mean_f = np.add.reduceat(F, self.contract_start, axis=1)/counts
        mean_l = np.add.reduceat(dL, self.contract_start, axis=1)/counts
        fc = F-np.repeat(mean_f, counts, axis=1)
        lc = dL-np.repeat(mean_l, counts, axis=1)
        sxx = np.add.reduceat(fc*fc, self.contract_start, axis=1)
        sxy = np.add.reduceat(fc*lc, self.contract_start, axis=1)
        degenerate = sxx <= 1e-24
        self.degenerate_slopes += int(degenerate.sum())
        return np.where(degenerate, 0.0, sxy/np.where(degenerate, 1.0, sxx))

    def statistic(self, law: str, parameter: float, replications: int, tag: int = 0, batches: int = REPLAY["batches"]) -> dict:
        """Class statistic = median over contracts of replication-averaged slopes; batch-means MC SE."""
        slopes, moments = self.slopes(law, parameter, replications, tag)
        per_contract = slopes.mean(axis=0)
        groups = np.array_split(np.arange(replications), batches)
        batch_stats = np.array([np.median(slopes[g].mean(axis=0)) for g in groups])
        return {"statistic": float(np.median(per_contract)), "per_contract": per_contract,
                "mc_se": float(batch_stats.std(ddof=1)/np.sqrt(batches)), "moments": moments}


def solve_parameter(replay: FlowReplay, law: str, target: float, replications: int, upper_limit: float = np.inf,
                    trace: list | None = None, label: str = "") -> tuple[float, dict]:
    """Bracket from 0 (g<0), double from 1e-6 until g>0, then bisect to |g|/target < 0.1% (amendment A-1)."""
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Replay target must be positive")
    log = trace if trace is not None else []
    def g(x):
        res = replay.statistic(law, x, replications)
        log.append({"variant": label, "stage": "search", "replications": replications, "parameter": x,
                    "statistic": res["statistic"], "relative_gap": res["statistic"]/target-1, "mc_se": res["mc_se"]})
        return res["statistic"]-target, res
    g0, _ = g(0.0)
    if not g0 < 0:
        raise AssertionError("Replay objective at zero informativeness must be negative")
    lo, hi = 0.0, REPLAY["bracket_start"]
    for _ in range(REPLAY["max_doublings"]+1):
        hi = min(hi, upper_limit)
        g_hi, res = g(hi)
        if g_hi > 0:
            break
        if hi >= upper_limit:
            raise RuntimeError(f"{label}: objective still negative at the parameter upper limit")
        lo, hi = hi, 2*hi
    else:
        raise RuntimeError(f"{label}: no positive bracket after {REPLAY['max_doublings']} doublings")
    if abs(g_hi)/target < REPLAY["bisection_tolerance"]:
        return hi, res
    for _ in range(REPLAY["max_bisection_steps"]):
        mid = (lo+hi)/2
        g_mid, res = g(mid)
        if abs(g_mid)/target < REPLAY["bisection_tolerance"]:
            return mid, res
        lo, hi = (mid, hi) if g_mid < 0 else (lo, mid)
    raise RuntimeError(f"{label}: bisection did not reach tolerance")


def calibrate_replay_law(replay: FlowReplay, law: str, target: float, trace: list, label: str,
                         upper_limit: float = np.inf) -> dict:
    """Replication doubling until MC SE < 0.25% of target, then independent validation (R_val = R)."""
    R = REPLAY["initial_replications"]
    while True:
        parameter, res = solve_parameter(replay, law, target, R, upper_limit, trace, label)
        trace.append({"variant": label, "stage": "replication_check", "replications": R, "parameter": parameter,
                      "statistic": res["statistic"], "relative_gap": res["statistic"]/target-1, "mc_se": res["mc_se"]})
        if res["mc_se"] < REPLAY["mc_se_tolerance"]*target:
            break
        if 2*R > REPLAY["max_replications"]:
            raise RuntimeError(f"{label}: MC SE tolerance not reached by {REPLAY['max_replications']} replications")
        R *= 2
    val = replay.statistic(law, parameter, R, tag=1)
    rel = val["statistic"]/target-1
    # Amendment A-2: the fitted parameter carries calibration-sample MC error, so the validation CI
    # combines the calibration and validation standard errors in quadrature.
    half = 1.959963984540054*np.hypot(res["mc_se"], val["mc_se"])/target
    ci = [rel-half, rel+half]
    passed = bool(abs(rel) < REPLAY["validation_tolerance"] and ci[0] <= 0 <= ci[1])
    trace.append({"variant": label, "stage": "validation", "replications": R, "parameter": parameter,
                  "statistic": val["statistic"], "relative_gap": rel, "mc_se": val["mc_se"]})
    return {"parameter": parameter, "replications": R, "statistic": res["statistic"], "mc_se": res["mc_se"],
            "relative_gap": res["statistic"]/target-1, "per_contract": res["per_contract"], "moments": res["moments"],
            "validation": {"statistic": val["statistic"], "mc_se": val["mc_se"], "relative_error": rel,
                           "ci95_relative_error": ci, "passed": passed,
                           "status": "validated" if passed else "calibration_limitation_reported_not_retuned"}}


def information_variance_p50(alpha, u) -> np.ndarray:
    """Per-trade E[(Delta p)^2] at p = 1/2: alpha^2 p^2(1-p)^2 / (q(1-q)), q = (1-alpha)u + alpha/2."""
    a, u = np.asarray(alpha, dtype=float), np.asarray(u, dtype=float)
    q = (1-a)*u+a/2
    return a**2/16/(q*(1-q))


# ---------------------------------------------------------------- 3.7 p0_emp
def empirical_start_probability(tape: pd.DataFrame, t_c: pd.Timestamp) -> dict:
    start = t_c-pd.Timedelta(seconds=EPISODE_SECONDS)
    d = tape.sort_values(["timestamp", "trade_id"], kind="stable")
    before = d[d.timestamp < start]
    if len(before):
        row, source = before.iloc[-1], "last_trade_before_episode_start"
    else:
        after = d[d.timestamp >= start]
        if after.empty:
            raise ValueError("Empty training tape")
        row, source = after.iloc[0], "first_trade_at_or_after_episode_start"
    p = float(row.price)
    sim = float(np.clip(p, *P0_CLIP))
    return {"p0_emp": p, "p0_source": source, "p0_trade_timestamp_utc": row.timestamp.isoformat(),
            "p0_sim": sim, "p0_clipped": bool(sim != p)}


# ------------------------------------------------------------------- driver
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def run(raw_input: Path = DEFAULT_INPUT, calibration_dir: Path = DEFAULT_OUTPUT, output_dir: Path = DEFAULT_OUTPUT_DIR,
        bootstrap_draws: int = 2000, seed: int = 42) -> dict:
    if bootstrap_draws < 20 or seed < 0:
        raise ValueError("Require at least twenty bootstrap draws and a nonnegative seed")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = calibration_dir/"simulator"
    dyn_dir = sim_dir/"market_dynamics"
    paths = {"dynamics": dyn_dir/"market_dynamics_config.json", "impact": sim_dir/"price_impact_config.json",
             "splits": calibration_dir/"event_splits.csv", "noise_intervals": dyn_dir/"training_noise_intervals.parquet",
             "mark_bank": dyn_dir/"trade_mark_bank.npz", "buckets": calibration_dir/"five_minute_intervals.parquet",
             "estimates": calibration_dir/"lambda_estimates.parquet",
             "markets": raw_input/"markets"/"kalshi_selected_contracts.parquet"}
    dynamics, impact = json.loads(paths["dynamics"].read_text()), json.loads(paths["impact"].read_text())
    hashes = {k: file_hash(p) for k, p in paths.items()}
    expected = {"impact": dynamics["upstream_sha256"]["price_impact_config.json"],
                "splits": dynamics["upstream_sha256"]["event_splits.csv"], "mark_bank": dynamics["mark_bank_sha256"],
                "buckets": impact["upstream_file_sha256"]["five_minute_intervals.parquet"],
                "estimates": impact["upstream_file_sha256"]["lambda_estimates.parquet"],
                "markets": dynamics["raw_input_sha256"]["markets/kalshi_selected_contracts.parquet"]}
    for key, value in expected.items():
        if hashes[key] != value:
            raise ValueError(f"Upstream input changed since Paper 2 calibration: {key}")
    splits = pd.read_csv(paths["splits"])
    guard = TrainingGuard(splits)
    if sorted(guard.train) != sorted(dynamics["training"]["tickers"]) or sorted(guard.train) != sorted(impact["training"]["tickers"]):
        raise ValueError("Training tickers differ across Paper 2 calibrations")
    markets = pd.read_parquet(paths["markets"])
    markets = markets[markets.ticker.isin(guard.train)].set_index("ticker")
    templates = dynamics["event_templates"]
    class_tickers = {c: sorted(t for t, tp in templates.items() if tp["information_class"] == c) for c in CLASSES}
    if not set().union(*class_tickers.values()) <= guard.train:
        raise ValueError("A template is not a training contract")

    # ---- Section 2: release times (training metadata only)
    meta_hashes, release_rows = {}, []
    for cls in CLASSES:
        for ticker in class_tickers[cls]:
            guard.require(ticker, "market_metadata")
            path = raw_input/"market_metadata"/f"{safe_filename(ticker)}.json"
            meta_hashes[str(path.relative_to(raw_input))] = file_hash(path)
            md = json.loads(path.read_text())
            if md["ticker"] != ticker or pd.Timestamp(md["open_time"]) != pd.Timestamp(markets.loc[ticker, "open_time"]) or \
                    pd.Timestamp(md["close_time"]) != pd.Timestamp(markets.loc[ticker, "close_time"]):
                raise ValueError(f"Metadata JSON disagrees with verified contract table: {ticker}")
            release_rows.append(release_record(ticker, templates[ticker], md))
    release = pd.DataFrame(release_rows)
    t_r = {r.ticker: pd.Timestamp(r.t_r_utc) for r in release.itertuples()}
    t_c = {r.ticker: pd.Timestamp(r.t_c_utc) for r in release.itertuples()}

    # ---- 3.1-3.3: residual normalization, detection, decomposition
    intervals = own_class_intervals(pd.read_parquet(paths["noise_intervals"]), templates)
    if not intervals.split.eq("train").all() or not set(intervals.ticker) <= guard.train:
        raise ValueError("Non-training quote pairs present")
    detection_rows, decomposition_rows, contract_info = [], [], {}
    for cls in CLASSES:
        vol = dynamics["groups"][cls]["volatility"]["base"]
        supported = sorted(t for t in class_tickers[cls] if "noise" in templates[t])
        if supported != sorted(vol["supported_tickers"]):
            raise AssertionError("Noise-supported templates differ from the legacy donor set")
        beta = float(impact["groups"][cls]["scenarios"]["base"])
        for ticker in supported:
            res = effective_innovations(intervals[intervals.ticker.eq(ticker)], beta,
                                        templates[ticker]["noise"]["sigma_logodds_per_sqrt_second"])
            det = detect_jumps(res["eta_tilde"], res["consecutive"])
            dec = decompose(res["eta_tilde"], det["jump"], res["sigma2_legacy"])
            pairs = res["pairs"]
            ttr, bidx = pair_ttr_buckets(pairs, t_r[ticker])
            for k in range(res["n"]):
                detection_rows.append({"ticker": ticker, "information_class": cls, "pair_index": k,
                    "left_time": pairs.left_time.iloc[k].isoformat(), "right_time": pairs.right_time.iloc[k].isoformat(),
                    "consecutive_with_previous": bool(res["consecutive"][k]), "eta": float(res["eta"][k]),
                    "eta_tilde": float(res["eta_tilde"][k]), "sigma_hat_300s": float(np.sqrt(det["sigma_hat2"][k])),
                    "local_bipower": bool(det["local"][k]), "is_jump": bool(det["jump"][k]),
                    "ttr_left_seconds": float(ttr[k]), "bucket": BUCKETS[bidx[k]] if bidx[k] >= 0 else "post_release"})
            contract_info[ticker] = {"n_pairs": res["n"], "d": res["d"], "sigma_legacy": float(np.sqrt(res["sigma2_legacy"])),
                                     "sigma_c": dec["sigma_c"], "n_jumps": int(det["jump"].sum())}
            decomposition_rows.append({"ticker": ticker, "information_class": cls, "n_pairs": res["n"], "d": res["d"],
                "consecutive_pairs": int(res["consecutive"].sum()), "local_bipower_pairs": int(det["local"].sum()),
                "sigma_stored": templates[ticker]["noise"]["sigma_logodds_per_sqrt_second"],
                "sigma2_legacy_recomputed": res["sigma2_legacy"], "sigma2_stored": templates[ticker]["noise"]["sigma_logodds_per_sqrt_second"]**2,
                "sigma_c": dec["sigma_c"], "sigma_c2": dec["sigma_c2"], "jump_variance_rate": dec["jump_variance_rate"],
                "identity_abs_error": dec["identity_abs_error"], "n_jumps": int(det["jump"].sum()),
                "jump_variance_share": dec["jump_variance_rate"]/res["sigma2_legacy"] if res["sigma2_legacy"] > 0 else None})
    detection, decomposition = pd.DataFrame(detection_rows), pd.DataFrame(decomposition_rows)

    # ---- 3.4: intensities, magnitudes, bootstrap
    class_reports, bank = {}, {}
    for index, cls in enumerate(CLASSES):
        rows = detection[detection.information_class.eq(cls)]
        post = rows[rows.bucket.eq("post_release")]
        rows_in = rows[rows.bucket.ne("post_release")]
        counts, exposure = bucket_tables(rows_in)
        pooled_class = sum(counts.values()) < MIN_BUCKET_JUMPS
        source_rows = detection[detection.bucket.ne("post_release")] if pooled_class else rows_in
        if pooled_class:
            counts, exposure = bucket_tables(source_rows)
        lam = pooled_intensity(counts, exposure)
        magnitudes = np.abs(source_rows.eta_tilde[source_rows.is_jump].to_numpy())
        bank[f"{cls}_abs_eta_tilde"] = magnitudes
        boot = intensity_bootstrap(source_rows, bootstrap_draws, np.random.default_rng(np.random.SeedSequence([seed, index])))
        mean_a2 = float((magnitudes**2).mean()) if len(magnitudes) else None
        dp2 = float(((expit(magnitudes)-.5)**2).mean()) if len(magnitudes) else 0.0
        sc2 = decomposition[decomposition.information_class.eq(cls)].sigma_c2
        by_bucket = {}
        for b in BUCKETS:
            rec = lam[b]
            by_bucket[b] = {**rec, "detected_jumps": counts[b], "exposure_seconds": exposure[b],
                            "implied_jumps": rec["lambda_per_second"]*exposure[b],
                            "bootstrap95_lambda": boot["lambda_bootstrap95"][b],
                            "probability_variance_p50_jump_per_hour": rec["lambda_per_second"]*dp2*3600,
                            "probability_variance_p50_continuous_median_per_hour": float(sc2.median())/16*3600}
        class_reports[cls] = {"pooled_class_profile": bool(pooled_class), "post_release_pairs_excluded": len(post),
            "post_release_jumps_excluded": int(post.is_jump.sum()), "total_detected_jumps": int(rows.is_jump.sum()),
            "buckets": by_bucket, "magnitude_bank_key": f"{cls}_abs_eta_tilde", "magnitude_count": len(magnitudes),
            "magnitude_quantiles": {str(q): float(np.quantile(magnitudes, q)) for q in (0, .1, .25, .5, .75, .9, 1)} if len(magnitudes) else None,
            "mean_a2": mean_a2, "mean_a2_bootstrap95": boot["mean_a2_bootstrap95"],
            "mean_probability_jump2_p50": dp2, "bootstrap": {k: v for k, v in boot.items() if k not in ("lambda_bootstrap95", "mean_a2_bootstrap95")},
            "supported_tickers": sorted(rows.ticker.unique()),
            "sigma_c": {t: contract_info[t]["sigma_c"] for t in sorted(rows.ticker.unique())},
            "label": "reduced-form jump proxies from 300-s quote-midpoint residuals"}

    # ---- 3.5: uninformed sign rates from the verified training mark bank
    rates_npz, u_reports = {}, {}
    with np.load(paths["mark_bank"], allow_pickle=False) as mb:
        marks = {c: {t: (mb[f"{templates[t]['mark_bank_key']}_quantity"].astype(float), mb[f"{templates[t]['mark_bank_key']}_sign"].astype(np.int8),
                         mb[f"{templates[t]['mark_bank_key']}_phase"].astype(np.int64)) for t in class_tickers[c]} for c in CLASSES}
    n_phases = int(dynamics["n_lifetime_phases"])
    u_rates = {c: uninformed_rates(marks[c], n_phases) for c in CLASSES}
    for cls in CLASSES:
        ur = u_rates[cls]
        rates_npz[f"{cls}_size_bin_edges"] = ur["edges"]; rates_npz[f"{cls}_u_class"] = ur["u_class"]
        for t, rec in ur["templates"].items():
            key = templates[t]["mark_bank_key"]
            rates_npz[f"{key}_u_bin_phase"] = rec["u_bin_phase"]; rates_npz[f"{key}_u_bin"] = rec["u_bin"]
            rates_npz[f"{key}_n_bin_phase"] = rec["n_bin_phase"]
        u_reports[cls] = {"size_bin_edges": ur["edges"], "u_class": ur["u_class"], "u_class_raw": ur["u_class_raw"],
                          "overall_yes_rate": ur["overall_yes_rate"], "empty_class_bins": ur["empty_class_bins"],
                          "pooled_marks": ur["pooled_marks"], "prior_weight": U_PRIOR_WEIGHT, "clip": U_CLIP}

    # ---- 3.6: flow replay (training tapes only, hash verified) and 3.7 p0_emp
    buckets_all = pd.read_parquet(paths["buckets"])
    estimates = pd.read_parquet(paths["estimates"])
    targets = estimates[estimates.split.eq("train") & estimates.response.eq("logodds") & estimates.variant.eq("full") &
                        estimates.fit_status.eq("ok")]
    tape_hashes, tapes = {}, {}
    def load_tape(ticker):
        guard.require(ticker, "trades")
        path = raw_input/"trades"/f"{safe_filename(ticker)}.parquet"
        rel = str(path.relative_to(raw_input)); checksum = file_hash(path)
        if checksum != impact["training_tape_sha256"].get(rel):
            raise ValueError(f"Training tape changed: {ticker}")
        tape_hashes[rel] = checksum
        tape, stats = normalize_trades(pd.read_parquet(path), ticker)
        if stats["invalid_trade_rows"]:
            raise ValueError(f"Invalid training trades: {ticker}")
        return tape.sort_values(["timestamp", "trade_id"], kind="stable").reset_index(drop=True)
    flow_reports, replay_trace, contract_rows = {}, [], []
    for index, cls in enumerate(CLASSES):
        contracts = sorted(set(targets[targets.information_class.eq(cls)].ticker))
        if contracts != class_tickers[cls] or len(contracts) != impact["groups"][cls]["training_contracts"]:
            raise AssertionError("Replay contracts differ from the class training targets")
        sizes, us, tb, bc, p_start, empirical_signs = [], [], [], [], [], []
        ur = u_rates[cls]
        n_bucket = 0
        for ci, ticker in enumerate(contracts):
            tape = tapes[ticker] = load_tape(ticker)
            tq, ts, tp = marks[cls][ticker]
            tmpl = templates[ticker]
            opened = pd.Timestamp(tmpl["open_time"])
            phase = np.minimum(((tape.timestamp-opened).dt.total_seconds().to_numpy()/float(tmpl["lifetime_seconds"])*n_phases).astype(int), n_phases-1)
            if not (np.array_equal(tq, tape.quantity.to_numpy(float)) and np.array_equal(ts, tape.sign.to_numpy()) and np.array_equal(tp, phase)):
                raise AssertionError(f"Mark bank and training tape disagree: {ticker}")
            d = interval_subset(buckets_all[buckets_all.ticker.eq(ticker) & buckets_all.split.eq("train")], "full", "logodds")
            bstart = tape.timestamp.dt.floor("5min")
            keep = bstart.isin(set(d.bucket_start))
            t = tape[keep]
            grouped = t.assign(bucket_start=bstart[keep], signed=t.sign*t.quantity).groupby("bucket_start", sort=True)
            if (grouped.size().index.tolist() != d.bucket_start.tolist() or not np.array_equal(grouped.size().to_numpy(), d.n_trades.to_numpy())
                    or not np.allclose(grouped.signed.sum().to_numpy(), d.signed_flow_contracts.to_numpy(), rtol=1e-10, atol=1e-8)
                    or not np.array_equal(grouped.price.first().to_numpy(), d.open_price.to_numpy())):
                raise AssertionError(f"Training tape/bucket mismatch: {ticker}")
            starts_ns = pd.DatetimeIndex(d.bucket_start).as_unit("ns").asi8
            trade_ns = pd.DatetimeIndex(bstart[keep]).as_unit("ns").asi8
            local_bucket = np.searchsorted(starts_ns, trade_ns)
            if not np.array_equal(starts_ns[local_bucket], trade_ns):
                raise AssertionError(f"Trade-to-bucket mapping failed: {ticker}")
            sizes.append(t.quantity.to_numpy(float)); empirical_signs.append(t.sign.to_numpy())
            us.append(ur["templates"][ticker]["u_bin_phase"][size_bins(t.quantity.to_numpy(float), ur["edges"]), phase[keep.to_numpy()]])
            tb.append(n_bucket+local_bucket); bc.append(np.full(len(d), ci)); p_start.append(d.open_price.to_numpy(float))
            n_bucket += len(d)
        replay = FlowReplay(np.concatenate(sizes), np.concatenate(us), np.concatenate(tb), np.concatenate(bc),
                            np.concatenate(p_start), REPLAY["seed"], index, contracts)
        target = float(impact["groups"][cls]["target_lambda_median"])
        beta = float(impact["groups"][cls]["scenarios"]["base"])
        trace = []
        fg = calibrate_replay_law(replay, "fg", target, trace, "fg")
        eta = fg["parameter"]
        half = replay.statistic("fg_half", beta, fg["replications"])
        half_val = replay.statistic("fg_half", beta, fg["replications"], tag=1)
        trace.append({"variant": "fg_half", "stage": "report", "replications": fg["replications"], "parameter": beta,
                      "statistic": half["statistic"], "relative_gap": half["statistic"]/target-1, "mc_se": half["mc_se"]})
        try:
            const = calibrate_replay_law(replay, "constant", target, trace, "constant_alpha", upper_limit=1-K.ALPHA_FLOOR_ONE_MINUS)
            const_report = {"status": "calibrated", "alpha0": const["parameter"], "replications": const["replications"],
                            "statistic": const["statistic"], "mc_se": const["mc_se"], "validation": const["validation"]}
        except RuntimeError as exc:
            const_report = {"status": "failed", "reason": str(exc)}
        for row in trace:
            replay_trace.append({"information_class": cls, "target_lambda": target, **row})
        for ticker, slope in zip(contracts, fg["per_contract"]):
            contract_rows.append({"information_class": cls, "target_lambda": target, "variant": "fg", "stage": "contract_slope",
                                  "ticker": ticker, "replications": fg["replications"], "parameter": eta, "statistic": float(slope)})
        all_sizes = np.concatenate([m[0] for m in marks[cls].values()])
        size_q = {str(q): float(np.quantile(all_sizes, q)) for q in (.1, .5, .9, .99)}
        alpha_q = {k: float(K.fg_alpha_terms(v, eta)[0]) for k, v in size_q.items()}
        alpha_trades, _ = K.fg_alpha_terms(replay.c, eta)
        info_var = information_variance_p50(alpha_trades, replay.u)
        mom = fg["moments"]
        emp_signs = np.concatenate(empirical_signs)
        flow_reports[cls] = {"law": "alpha(c; eta) = min(1 - exp(-eta c), 1 - 1e-12)", "eta_per_contract": eta,
            "target_lambda_median": target, "target_source": "price_impact_config.json groups[class].target_lambda_median",
            "replications": fg["replications"], "calibration_statistic": fg["statistic"], "calibration_mc_se": fg["mc_se"],
            "calibration_relative_gap": fg["relative_gap"], "validation": fg["validation"],
            "contracts": len(contracts), "buckets": n_bucket, "trades": len(replay.c),
            "size_quantiles": size_q, "alpha_at_size_quantiles": alpha_q,
            "simulated_moments": {"buy_fraction": mom["buys"]/mom["trades"], "mean_signed_contracts_per_trade": mom["signed_contracts"]/mom["trades"],
                                  "informed_fraction": mom["informed"]/mom["trades"], "mean_abs_bucket_flow": mom["bucket_abs_flow"]/mom["buckets"]},
            "empirical_moments": {"buy_fraction": float((emp_signs == 1).mean()),
                                  "mean_signed_contracts_per_trade": float((emp_signs*replay.c).mean()), "mean_size": float(replay.c.mean())},
            "information_variance_p50": {"mean_per_trade": float(info_var.mean()), "mean_per_bucket": float(info_var.sum()/n_bucket),
                                         "units": "probability^2 at p = 1/2"},
            "fg_half": {"beta": beta, "u": .5, "alpha": "tanh(beta c / 2)", "statistic": half["statistic"], "mc_se": half["mc_se"],
                        "relative_to_target": half["statistic"]/target, "validation_uniform_statistic": half_val["statistic"],
                        "validation_uniform_mc_se": half_val["mc_se"]},
            "constant_alpha": const_report, "degenerate_contract_slopes": replay.degenerate_slopes,
            "assumption": "alpha small => YES rate approximately u; u from training marks; observation after each trade"}

    p0_rows = {}
    for cls in CLASSES:
        for ticker in class_tickers[cls]:
            p0_rows[ticker] = empirical_start_probability(tapes[ticker], t_c[ticker])

    # ---- 3.8 outputs
    np.savez_compressed(output_dir/"jump_magnitude_bank.npz", **bank)
    np.savez_compressed(output_dir/"uninformed_sign_rates.npz", **rates_npz)
    detection.to_csv(output_dir/"jump_detection.csv", index=False)
    decomposition.to_csv(output_dir/"decomposition_check.csv", index=False)
    release.to_csv(output_dir/"release_time_verification.csv", index=False)
    pd.DataFrame(replay_trace+contract_rows).to_csv(output_dir/"flow_replay_calibration.csv", index=False)
    files = ["jump_magnitude_bank.npz", "uninformed_sign_rates.npz", "jump_detection.csv", "decomposition_check.csv",
             "release_time_verification.csv", "flow_replay_calibration.csv"]
    template_records = {}
    for r in release.itertuples():
        ticker = r.ticker
        jumps = detection[detection.ticker.eq(ticker) & detection.is_jump]
        template_records[ticker] = {"information_class": r.information_class, "mark_bank_key": templates[ticker]["mark_bank_key"],
            "t_c_utc": r.t_c_utc, "t_r_utc": r.t_r_utc, "t_r_source": r.t_r_source, "flags": r.flags.split(";") if r.flags else [],
            "early_revelation": bool(r.early_revelation), "release_minus_close_seconds": r.release_minus_close_seconds,
            "noise_supported": ticker in contract_info, **contract_info.get(ticker, {}),
            "jumps": [{"left_time_utc": j.left_time, "ttr_left_seconds": j.ttr_left_seconds, "bucket": j.bucket,
                       "eta_tilde": j.eta_tilde} for j in jumps.itertuples()],
            **p0_rows[ticker]}
    early = sorted(release.ticker[release.early_revelation])
    config = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "units": {"time": "seconds", "sigma_c": "log_odds_per_sqrt_second", "lambda_J": "jumps_per_second",
                  "magnitude": "log_odds", "eta": "per_contract", "u": "probability_of_YES_taker"},
        "conventions": {"ttr": "TTR = t_r - t; buckets B0 [0,3600), B1 [3600,10800), B2 [10800,86400), B3 [86400,inf) seconds",
                        "exposure": "300 s per observed quote pair, bucket of the pair's left endpoint TTR",
                        "pooling_order": POOL_ORDER, "min_bucket_jumps": MIN_BUCKET_JUMPS,
                        "detection": "|eta_tilde_k| > 4 sigma_hat_k; sigma_hat_k^2 = (pi/2) mean of adjacent products |eta_i||eta_{i-1}| "
                                     "(consecutive pairs only) with |i-k| <= 12 observed pairs, excluding products involving k-1,k,k+1 "
                                     "(i in {k-1..k+2}); >=12 products else contract level",
                        "effective_innovations": "eta_tilde = eta*sqrt(n/(n-d)) so sum eta_tilde^2/(n*300) = legacy sigma^2",
                        "sign": "+1 YES taker, -1 NO taker; S_u = 2*1{V_u < u} - 1",
                        "release_time": "broad_based: metadata occurrence_datetime (else template close, flagged); sports: template close (provisional)",
                        "noise_donors": "templates without own quote support use the legacy MarketDynamics donor draw; sigma_c is the donor's",
                        "u_rates": "u[tpl,b,k] = clip((n_yes + 20 u_class[b])/(n + 20), .02, .98); b = class quintile size bin "
                                   "(side='right' of interior edges); k = legacy lifetime phase",
                        "p0_emp": "last training trade strictly before t_c - 3600 s, else first at/after; simulation start clipped to [.01,.99]"},
        "parameters": {"jump_threshold": JUMP_THRESHOLD, "bipower_half_window": BIPOWER_HALF_WINDOW,
                       "min_local_products": MIN_LOCAL_PRODUCTS, "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
                       "replay": REPLAY, "size_bins": N_SIZE_BINS, "u_prior_weight": U_PRIOR_WEIGHT},
        "training": {"tickers": sorted(guard.train), "class_tickers": class_tickers},
        "classes": {cls: {"jumps": class_reports[cls], "flow": flow_reports[cls], "uninformed": u_reports[cls],
                          "impact_beta_base": float(impact["groups"][cls]["scenarios"]["base"]),
                          "noise_donor_tickers": dynamics["groups"][cls]["volatility"]["base"]["supported_tickers"]} for cls in CLASSES},
        "templates": template_records, "early_revelation_templates": early,
        "candidate_roster": [t for cls in CLASSES for t in class_tickers[cls] if t not in early],
        "roster_note": "Candidate roster before matching; R* (Section 8) additionally drops templates unmatched for any M arm.",
        "file_sha256": {name: file_hash(output_dir/name) for name in files},
        "input_sha256": {**{f"paper2/{k}": v for k, v in hashes.items()}, **tape_hashes, **meta_hashes},
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
                          [Path(__file__).resolve(), PROJECT_ROOT/"simulator"/"information"/"kernels.py",
                           PROJECT_ROOT/"calibration"/"calibrate_market_dynamics.py", PROJECT_ROOT/"calibration"/"estimate_lambda.py",
                           PROJECT_ROOT/"calibration"/"build_intervals.py", PROJECT_ROOT/"data_pipeline"/"calibration_quality.py"]},
        "limitations": ["Jumps are reduced-form proxies: large 300-s quote-midpoint residuals, not identified news events.",
                        "Quote support covers the trailing window (usually three days); intensities use that exposure only.",
                        "Sports release time is the provisional template close (official game-end timestamps deferred).",
                        "eta matches the train-median Kyle lambda in a flow-only post-trade replay; it does not identify informed traders.",
                        "u rates treat observed YES-taker frequencies as uninformed rates (alpha small).",
                        "Bootstrap ranges resample training contracts; they are sampling scenarios, not causal bounds."]}
    (output_dir/"information_config.json").write_text(json.dumps(_jsonable(config), indent=2, allow_nan=False)+"\n")
    if guard.heldout_files_loaded:
        raise AssertionError("Held-out raw file loaded")
    report = {"passed": True, "heldout_files_loaded": guard.heldout_files_loaded,
              "training_files_loaded": {k: sorted({t for kind, t in guard.loaded if kind == k}) for k in ("market_metadata", "trades")},
              "supported_noise_templates": len(decomposition), "max_decomposition_identity_error": float(decomposition.identity_abs_error.max()),
              "early_revelation_templates": early,
              "summary": {cls: {"jumps_by_bucket": {b: class_reports[cls]["buckets"][b]["detected_jumps"] for b in BUCKETS},
                                "lambda_J_per_second": {b: class_reports[cls]["buckets"][b]["lambda_per_second"] for b in BUCKETS},
                                "contributing_buckets": {b: class_reports[cls]["buckets"][b]["contributing_buckets"] for b in BUCKETS},
                                "magnitude_bank_size": class_reports[cls]["magnitude_count"], "eta": flow_reports[cls]["eta_per_contract"],
                                "replications": flow_reports[cls]["replications"], "validation": flow_reports[cls]["validation"],
                                "fg_half_relative_to_target": flow_reports[cls]["fg_half"]["relative_to_target"],
                                "constant_alpha": flow_reports[cls]["constant_alpha"]} for cls in CLASSES},
              "checks": ["Upstream Paper 2 inputs match recorded hashes", "Training tapes match recorded hashes",
                         "Recomputed legacy sigma^2 equals stored template sigma^2 (rel 1e-10)",
                         "sigma_c^2 + jump variance = legacy sigma^2 for every supported template",
                         "Mark bank equals training tapes (quantity, sign, phase)",
                         "Replay buckets reconcile to Paper 2 five-minute intervals",
                         "No validation/test ticker file was opened"],
              "output_sha256": {**{name: file_hash(output_dir/name) for name in files},
                                "information_config.json": file_hash(output_dir/"information_config.json")}}
    (output_dir/"calibration_report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_input, args.calibration_dir, args.output_dir, args.bootstrap_draws, args.seed), indent=2))
