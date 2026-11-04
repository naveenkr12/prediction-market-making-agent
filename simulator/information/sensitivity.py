"""Paper 3 sensitivity experiments (docs/paper3/SENSITIVITY_SPEC.md): quote refresh and jump intensity.

A. quote_refresh: decision interval 300 s against 60 s. Arms M0, M1, M2 (matched s), M3 (s = 1), M0-s1,
   M2-s1, M1-native, each x S0/S1 x {300, 60}. Only `decision_interval` changes: the path (node set,
   W, executions, books, jumps, proposals, settlement) is the same precomputed object, and the policies
   (frozen HJB with time_mapping "episode", fixed, skew 0.0005, none) are called exactly as in Tier 1.
   The 300-s cells are the Tier 1 cells and must reproduce the Tier 1 rows exactly (asserted).

B. jump_intensity: doses 0x, 0.5x, 1x, 2x at s = 1, coupled by common random numbers.
   F0 family: 0x = M0-s1, then M2-s1 at 0.5x / 1x / 2x. FG family: 0x = M1-native, then M3.
   - 0x and 1x are the Tier 1 arms on the Tier 1 skeleton (asserted equal to the Tier 1 rows).
   - 0.5x: the 1x skeleton with its jumps independently thinned (keep iff U < 0.5, U from episode child
     12); dropped jumps stay in the node set as inert nodes, so W, proposals and all other inputs equal 1x.
   - 2x: the 1x jumps plus an independent Poisson(lambda_J) process (times, bank magnitudes, V_J from
     child 13), inserted into the node set by Brownian-bridge refinement of W from child 13; W at every
     native node is unchanged. See `skeleton.thinning_draws` / `skeleton.superpose_jumps`.
   Tier 1 children 0-11 are untouched (positional spawn keys).
   Information revealed per class and dose: U = E[p_c(1 - p_c)] by the Rao-Blackwellized estimator of
   `matching.py` (s = 1, N = 200,000 per template, family and dose) with standard errors, I = p0(1 - p0) - U,
   the jump share (I_d - I_0) / I_d, and the empirical mean p_c(1 - p_c) of the simulated paths.

Per episode (as Tier 1, via `evaluate.run_arm_replicate`): ledger audit, additive attribution, reward =
realized PnL, genuine settlement, walk hashes equal to the path trace, hash sharing across policies and S
levels, none-policy PnL = 0, S0 has no snipes. Runs are checkpointed per (template, cell).

Usage: python -m simulator.information.sensitivity --experiment quote_refresh|jump_intensity
       [--output-dir DIR] [--workers 10] [--replicates 20] [--tickers ...] [--fresh]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from ..market_dynamics import file_hash
from . import bootstrap as B
from . import evaluate as E
from .matching import TemplateDesign, load_roster, rb_uncertainty, sample_draws
from .paths import build_path, make_arm
from .skeleton import (DOSE_STREAMS, SEED, STREAMS, THINNING_KEEP_PROBABILITY, build_skeleton, event_roster,
                       superpose_jumps, thinning_draws)

ROOT = E.ROOT
TIER1_DIR = ROOT/"data_pipeline"/"output"/"paper3"/"experiment"
OUTPUT_ROOT = ROOT/"data_pipeline"/"output"/"paper3"/"sensitivity"
EXPERIMENTS = ("quote_refresh", "jump_intensity")
LEVELS = ("S0", "S1")
POLICIES_COMPARED = B.POLICIES_COMPARED
INFO_DRAWS = 200_000
INFO_STREAM_TAG = 61_017           # distinguishes information-revealed draws from matching and episode seeds
DOSES = (0.0, 0.5, 1.0, 2.0)
# Tier 1 columns that are labels/metadata, not outcomes (everything else shared must be equal).
TIER1_EXCLUDED = {"tier", "arm", "episode_id", "jump_rate_scale", "anchor_max_abs_difference", "price_path_sha256",
                  "constant_alpha"}


@dataclass(frozen=True)
class Cell:
    label: str                       # arm label in this experiment's episode_results.csv
    base: str                        # make_arm name
    s_source: str                    # matched | one
    decision_interval: int = 300
    dose: float = 1.0                # jump intensity relative to the native lambda_J (B)
    coupling: str = "native"         # native | none (0x, jump-free arm) | thinning | superposition
    tier1_label: str | None = None   # Tier 1 arm whose rows this cell must reproduce exactly
    group: str = ""                  # A: the 300-s arm label; B: dose family F0 | FG

    def plan(self) -> E.ArmPlan:
        return E.ArmPlan(self.label, self.base, ("R_star",), LEVELS, self.s_source, tier=3,
                         decision_interval=self.decision_interval)


QUOTE_REFRESH_ARMS = (("M0", "M0", "matched"), ("M1", "M1", "matched"), ("M2", "M2", "matched"), ("M3", "M3", "one"),
                      ("M0-s1", "M0", "one"), ("M2-s1", "M2", "one"), ("M1-native", "M1-native", "one"))
JUMP_FAMILIES = {"F0": {"zero": ("M0-s1", "M0"), "jump": ("M2-s1", "M2")},
                 "FG": {"zero": ("M1-native", "M1-native"), "jump": ("M3", "M3")}}


def quote_refresh_cells() -> list[Cell]:
    cells = []
    for label, base, s_source in QUOTE_REFRESH_ARMS:
        cells.append(Cell(label, base, s_source, 300, tier1_label=label, group=label))
        cells.append(Cell(f"{label}-dec60", base, s_source, 60, group=label))
    return cells


def dose_label(family: str, dose: float) -> str:
    spec = JUMP_FAMILIES[family]
    if dose == 0:
        return spec["zero"][0]
    if dose == 1:
        return spec["jump"][0]
    return f"{spec['jump'][0]}-x{dose:g}"


def jump_intensity_cells() -> list[Cell]:
    cells = []
    for family, spec in JUMP_FAMILIES.items():
        zero_label, zero_base = spec["zero"]
        jump_label, jump_base = spec["jump"]
        cells.append(Cell(zero_label, zero_base, "one", dose=0.0, coupling="none", tier1_label=zero_label, group=family))
        cells.append(Cell(dose_label(family, .5), jump_base, "one", dose=.5, coupling="thinning", group=family))
        cells.append(Cell(jump_label, jump_base, "one", dose=1.0, coupling="native", tier1_label=jump_label,
                          group=family))
        cells.append(Cell(dose_label(family, 2.), jump_base, "one", dose=2.0, coupling="superposition", group=family))
    return cells


def experiment_cells(experiment: str) -> list[Cell]:
    if experiment == "quote_refresh":
        return quote_refresh_cells()
    if experiment == "jump_intensity":
        return jump_intensity_cells()
    raise ValueError(f"Unknown experiment {experiment}")


# ------------------------------------------------------------------ per-episode extras
def quote_statistics(ledger) -> dict:
    """Decision stops and quote updates of one episode from its ledger.

    Every step calls the legacy `_replace`: all resting orders are cancelled and new orders are posted with a
    fresh queue position from the current book (`BookSnapshot.queue_ahead`: 0 if the price improves the
    same-side top of book, else the proxy depth) and the full requested size. A `same_price_resubmissions`
    count is a quote posted at the same price as the same-side order cancelled in that same replace (its
    queue progress and any partial fill are discarded).
    """
    decisions = max((int(r["step"]) for r in ledger), default=0)
    quotes = same = 0
    cancelled: dict = {}
    for row in ledger:
        kind = row["kind"]
        if kind == "cancel":
            cancelled[row["side"]] = row.get("_price")
            continue
        if kind == "quote":
            quotes += 1
            if cancelled.get(row["side"]) is not None and cancelled[row["side"]] == str(row["price"]):
                same += 1
            continue
        cancelled = {}
    return {"decisions": decisions, "quote_rows": quotes, "same_price_resubmissions": same,
            "changed_quotes": quotes-same}


def _annotate_cancel_prices(ledger) -> list:
    """Copy of the ledger where each cancel row carries the cancelled order's price (from its quote row)."""
    prices, out = {}, []
    for row in ledger:
        if row["kind"] == "quote":
            prices[str(row["order_id"])] = str(row["price"])
            out.append(row)
        elif row["kind"] == "cancel":
            out.append({**row, "_price": prices.get(str(row["order_id"]))})
        else:
            out.append(row)
    return out


def _sha_array(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def run_cell_replicate(ticker: str, replicate: int, cell: Cell, roster: dict, seed: int = SEED):
    """All S levels x policies of one (episode, cell) with the Tier 1 per-episode checks."""
    calibration = E._STATE["calibration"]
    plan = cell.plan()
    arm, s, p0 = E.resolve_arm(plan, ticker, roster, calibration)
    native = build_skeleton(ticker, replicate, calibration=calibration, seed=seed)
    sk, path, dose_hash, extra = native, None, "", {}
    if cell.coupling == "thinning":
        uniforms, keep = thinning_draws(native, THINNING_KEEP_PROBABILITY)
        path = build_path(native, arm, calibration=calibration, jump_keep=keep,
                          jump_keep_probability=THINNING_KEEP_PROBABILITY)
        dose_hash = _sha_array(uniforms)
        extra = {"jumps_native": len(native.jump_elapsed), "jumps_kept": int(keep.sum())}
    elif cell.coupling == "superposition":
        sk = superpose_jumps(native, calibration)
        path = build_path(sk, arm, calibration=calibration)
        dose_hash = sk.dose_input_hash
        extra = {"jumps_native": len(native.jump_elapsed), "jumps_inserted": int(sk.jump_extra.sum())}
    elif cell.coupling in ("native", "none"):          # the Tier 1 path itself (0x: a jump-free arm)
        if cell.coupling == "none" and arm.jumps != "none":
            raise ValueError("0x cells are jump-free arms")
        extra = {"jumps_native": len(native.jump_elapsed)}
    else:
        raise ValueError(f"Unknown coupling {cell.coupling}")
    rows, ledgers, diag = E.run_arm_replicate(sk, arm, plan, path)
    if len({(r["signal_sha256"], r["book_sha256"], r["settlement_sha256"]) for r in rows}) != 1:
        raise AssertionError("Hashes differ across policies/S levels within an arm")
    for r in rows:
        r.update(quote_statistics(_annotate_cancel_prices(ledgers[(r["snipe_level"], r["policy"])])))
        r["dose_input_sha256"] = dose_hash
    return rows, {**diag, **extra}, (arm, s, p0)


def _checkpoint_key(experiment, cell, ticker, replicates, seed, roster_sha, code_sha) -> str:
    return E._sha({"experiment": experiment, "cell": asdict(cell), "ticker": ticker, "replicates": replicates,
                   "seed": seed, "roster": roster_sha, "code": code_sha})


def run_task(task) -> dict:
    experiment, ticker, cell, out_dir, replicates, seed, roster, code_sha = task
    group, index = next((g, i) for g, t, i in event_roster() if t == ticker)
    ckpt = Path(out_dir)/"checkpoints"
    stem = f"{ticker}__{cell.label}"
    key = _checkpoint_key(experiment, cell, ticker, replicates, seed, roster["roster_sha256"], code_sha)
    meta_path = ckpt/(stem+".json")
    if meta_path.exists() and json.loads(meta_path.read_text()).get("key") == key and (ckpt/(stem+".csv")).exists():
        return {"ticker": ticker, "label": cell.label, "resumed": True}
    t0 = time.perf_counter()
    rows, diags = [], []
    for replicate in range(replicates):
        episode_rows, diag, (arm, s, p0) = run_cell_replicate(ticker, replicate, cell, roster, seed)
        common = {"experiment": experiment, "arm": cell.label, "base_arm": cell.base, "family": arm.family,
                  "cell_group": cell.group, "tier1_arm": cell.tier1_label or "", "roster": "R_star",
                  "ticker": ticker, "group": group, "event_index": index, "replicate": replicate, "p0": p0, "s": s,
                  "fill_scenario": "base", "decision_interval": cell.decision_interval, "jump_dose": cell.dose,
                  "jump_coupling": cell.coupling, "mean_latency": E.MEAN_LATENCY_SECONDS}
        rows += [{**common, **r} for r in episode_rows]
        diags.append({**common, **diag})
    runtime = time.perf_counter()-t0
    ckpt.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for name, data in ((stem+".csv", frame), (stem+"__paths.csv", pd.DataFrame(diags))):
        tmp = ckpt/(name+".tmp")
        data.to_csv(tmp, index=False)
        os.replace(tmp, ckpt/name)
    meta = {"key": key, "cell": asdict(cell), "ticker": ticker, "episodes": len(frame), "runtime_seconds": runtime,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {"ledger_audit": True, "attribution_max_abs_error": float(frame.attribution_error.abs().max()),
                       "hashes_shared": True, "none_policy_zero_pnl": True}}
    tmp = ckpt/(stem+".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)
    return {"ticker": ticker, "label": cell.label, "resumed": False, "runtime_seconds": runtime}


# ------------------------------------------------------------------ information revealed (B)
FAMILY_CODES = {"F0": 0, "FG": 1}


def info_arm(family: str, dose: float):
    spec = JUMP_FAMILIES[family]
    base = spec["zero"][1] if dose == 0 else spec["jump"][1]
    return make_arm(base, s=1.0) if base in ("M0", "M1", "M2") else make_arm(base)


def information_task(args) -> dict:
    """RB estimate of U = E[p_c(1 - p_c)] at s = 1 for one (template, family, dose); p0 = 0.5."""
    ticker, family, dose, draws, seed = args
    calibration = E._STATE["calibration"]
    design = TemplateDesign.build(ticker, calibration, jump_rate_scale=dose if dose > 0 else 1.0)
    arm = info_arm(family, dose)
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, INFO_STREAM_TAG, design.event_index, FAMILY_CODES[family], DOSES.index(dose)]))
    t0 = time.perf_counter()
    d = sample_draws(design, arm, draws, rng)
    est = rb_uncertainty(d, 1.0)
    return {"ticker": ticker, "group": design.group, "family": family, "dose": dose, "arm": dose_label(family, dose),
            "draws": draws, "U": est.mean, "U_se": est.se, "quad_max_order_diff": est.quad_diff,
            "expected_jumps": float(sum(design.jump_means)) if dose > 0 else 0.0,
            "mean_sampled_jumps": float(d.n_jumps.mean()) if dose > 0 else 0.0,
            "runtime_seconds": time.perf_counter()-t0}


def information_summary(per_template: pd.DataFrame, paths: pd.DataFrame, risk: pd.DataFrame, p0: float = 0.5):
    """Per (family, scope, dose): U +- SE, I, jump share, empirical mean p_c(1 - p_c), snipe PnL per unit I."""
    rows = []
    for family, scope in [(f, s) for f in JUMP_FAMILIES for s in ("all", "broad_based", "sports")]:
        zero = None
        for dose in DOSES:
            sub = per_template[(per_template.family == family) & (per_template.dose == dose)]
            if scope != "all":
                sub = sub[sub.group == scope]
            if sub.empty:
                continue
            k = len(sub)
            U, se = float(sub.U.mean()), float(np.sqrt((sub.U_se**2).sum())/k)
            info = p0*(1-p0)-U
            label = dose_label(family, dose)
            pth = paths[paths.arm == label]
            if scope != "all":
                pth = pth[pth.group == scope]
            emp = (pth.p_close_latent*(1-pth.p_close_latent)).to_numpy()
            row = {"family": family, "scope": scope, "dose": dose, "arm": label, "templates": k, "U_rb": U,
                   "U_rb_se": se, "information_revealed": info, "information_revealed_se": se,
                   "expected_jumps_per_episode": float(sub.expected_jumps.mean()),
                   "simulated_jumps_per_episode": float(pth.jumps_applied.mean()) if len(pth) else np.nan,
                   "empirical_mean_pc_1mpc": float(emp.mean()) if len(emp) else np.nan,
                   "empirical_mean_pc_1mpc_se_iid": float(emp.std(ddof=1)/np.sqrt(len(emp))) if len(emp) > 1 else np.nan,
                   "empirical_episodes": len(emp)}
            if dose == 0:
                zero = row
            if zero is not None:
                row["information_increment_vs_0x"] = info-zero["information_revealed"]
                row["information_increment_vs_0x_se"] = float(np.sqrt(se**2+zero["U_rb_se"]**2)) if dose else 0.0
                row["jump_share_of_information"] = (info-zero["information_revealed"])/info if info > 0 else np.nan
            for policy in POLICIES_COMPARED:
                r = risk[(risk.arm == label) & (risk.scope == scope) & (risk.policy == policy) &
                         (risk.snipe_level == "S1")]
                if len(r):
                    snipe = float(r.mean_snipe_pnl.iloc[0])
                    row[f"{policy}_S1_mean_snipe_pnl"] = snipe
                    row[f"{policy}_S1_snipe_pnl_per_unit_information"] = snipe/info if info > 0 else np.nan
                    if zero is not None and dose > 0:
                        z = risk[(risk.arm == zero["arm"]) & (risk.scope == scope) & (risk.policy == policy) &
                                 (risk.snipe_level == "S1")]
                        inc = info-zero["information_revealed"]
                        row[f"{policy}_S1_incremental_snipe_pnl_per_unit_incremental_information"] = (
                            (snipe-float(z.mean_snipe_pnl.iloc[0]))/inc if len(z) and inc > 0 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ contrasts
def _c(name, family, cells, policy, level, **labels):
    return B.contrast(name, family, cells, "R_star", policy=policy, snipe_level=level, **labels)


def quote_refresh_contrasts() -> list[dict]:
    out = []
    labels = [a for a, _, _ in QUOTE_REFRESH_ARMS]
    d60 = lambda a: f"{a}-dec60"
    jump_pairs = (("M2", "M0"), ("M3", "M1"), ("M2-s1", "M0-s1"), ("M3", "M1-native"))
    for policy in POLICIES_COMPARED:
        k = lambda arm, level, w: ((arm, level, policy), w)
        for a in labels:
            for level in LEVELS:
                out.append(_c(f"refresh:{a}@60-{a}@300", "refresh", [k(d60(a), level, 1), k(a, level, -1)], policy,
                              level, arm=a))
            out.append(_c(f"sniping@60:({a},S1)-({a},S0)", "sniping_at_60",
                          [k(d60(a), "S1", 1), k(d60(a), "S0", -1)], policy, "S1-S0", arm=a))
            out.append(_c(f"refresh_x_sniping:{a}", "refresh_x_sniping",
                          [k(d60(a), "S1", 1), k(d60(a), "S0", -1), k(a, "S1", -1), k(a, "S0", 1)], policy, "S1-S0",
                          arm=a))
        for x, y in jump_pairs:
            for level in LEVELS:
                out.append(_c(f"jump@300:{x}-{y}", "jump_at_300", [k(x, level, 1), k(y, level, -1)], policy, level))
                out.append(_c(f"jump@60:{x}-{y}", "jump_at_60", [k(d60(x), level, 1), k(d60(y), level, -1)], policy,
                              level))
                out.append(_c(f"refresh_x_jump:{x}-{y}", "refresh_x_jump",
                              [k(d60(x), level, 1), k(d60(y), level, -1), k(x, level, -1), k(y, level, 1)], policy,
                              level))
    return out


def slope_weights(doses=DOSES) -> list[float]:
    """OLS slope on dose levels as a linear contrast: w_d = (d - mean) / sum (d - mean)^2."""
    d = np.asarray(doses, dtype=float)
    c = d-d.mean()
    return (c/np.dot(c, c)).tolist()


def jump_intensity_contrasts() -> list[dict]:
    out = []
    weights = slope_weights()
    for family in JUMP_FAMILIES:
        lab = {d: dose_label(family, d) for d in DOSES}
        for policy in POLICIES_COMPARED:
            k = lambda arm, level, w: ((arm, level, policy), w)
            for level in LEVELS:
                for d in (.5, 1., 2.):
                    out.append(_c(f"dose_vs_0x:{lab[d]}-{lab[0.]}", "dose_vs_0x", [k(lab[d], level, 1),
                                  k(lab[0.], level, -1)], policy, level, dose_family=family, dose=d))
                for d in (0., .5, 2.):
                    out.append(_c(f"dose_vs_1x:{lab[d]}-{lab[1.]}", "dose_vs_1x", [k(lab[d], level, 1),
                                  k(lab[1.], level, -1)], policy, level, dose_family=family, dose=d))
                out.append(_c(f"dose_slope:{family}", "dose_slope",
                              [k(lab[d], level, w) for d, w in zip(DOSES, weights)], policy, level, dose_family=family))
            for d in (.5, 1., 2.):
                out.append(_c(f"sniping_x_dose:{family}@{d:g}", "sniping_x_dose",
                              [k(lab[d], "S1", 1), k(lab[d], "S0", -1), k(lab[0.], "S1", -1), k(lab[0.], "S0", 1)],
                              policy, "S1-S0", dose_family=family, dose=d))
            out.append(_c(f"dose_slope_sniping:{family}", "dose_slope_sniping",
                          [k(lab[d], lv, w*sgn) for d, w in zip(DOSES, weights) for lv, sgn in (("S1", 1), ("S0", -1))],
                          policy, "S1-S0", dose_family=family))
    return out


def evaluate_all_contrasts(data: pd.DataFrame, contrasts, rosters, draws, seed) -> pd.DataFrame:
    """Bootstrap contrasts (B.evaluate_contrasts) plus the mean snipe count on the same shared indices."""
    paired = B.evaluate_contrasts(data, contrasts, rosters, draws=draws, seed=seed)
    counts = B.evaluate_contrasts(data.assign(snipe_pnl=data.snipe_count.astype(float)), contrasts, rosters,
                                  draws=draws, seed=seed)
    counts = counts[counts.statistic.eq("mean_snipe_pnl")].assign(statistic="mean_snipe_count")
    return pd.concat([paired, counts], ignore_index=True)


def risk_with_snipe_count(data, rosters, roster_of, draws, seed) -> pd.DataFrame:
    risk = B.risk_summary(data, rosters, roster_of, draws=draws, seed=seed)
    counts = B.risk_summary(data.assign(snipe_pnl=data.snipe_count.astype(float)), rosters, roster_of, draws=draws,
                            seed=seed)
    key = ["roster", "scope", "arm", "snipe_level", "policy"]
    counts = counts[key+["mean_snipe_pnl", "mean_snipe_pnl_bootstrap95_low", "mean_snipe_pnl_bootstrap95_high"]]
    counts.columns = key+["mean_snipe_count", "mean_snipe_count_bootstrap95_low", "mean_snipe_count_bootstrap95_high"]
    risk = risk.drop(columns=["mean_snipe_count"], errors="ignore").merge(counts, on=key, how="left", validate="1:1")
    return risk


# ------------------------------------------------------------------ assertions
def tier1_equality(data: pd.DataFrame, tier1: pd.DataFrame, cells: list[Cell]) -> dict:
    """Rows of cells with a Tier 1 label must equal the Tier 1 rows exactly (outcomes and hashes)."""
    report, worst_rows = {}, 0
    key = ["ticker", "replicate", "snipe_level", "policy"]
    for cell in cells:
        if not cell.tier1_label:
            continue
        mine = data[data.arm.eq(cell.label)]
        ref = tier1[tier1.arm.eq(cell.tier1_label) & tier1.ticker.isin(mine.ticker.unique()) &
                    tier1.replicate.isin(mine.replicate.unique())]
        columns = sorted((set(mine.columns) & set(ref.columns))-TIER1_EXCLUDED-set(key))
        if len(mine) != len(ref) or len(mine) == 0:
            raise AssertionError(f"{cell.label}: {len(mine)} rows against {len(ref)} Tier 1 rows")
        merged = mine.merge(ref, on=key, suffixes=("", "_tier1"), validate="1:1")
        mismatched = []
        for c in columns:
            a, b = merged[c], merged[c+"_tier1"]
            equal = (a == b) | (a.isna() & b.isna())
            if not equal.all():
                mismatched.append(c)
        if mismatched:
            raise AssertionError(f"{cell.label} differs from Tier 1 {cell.tier1_label} in {mismatched}")
        report[cell.label] = {"tier1_arm": cell.tier1_label, "rows": len(merged), "columns_compared": len(columns),
                              "exact": True}
        worst_rows += len(merged)
    return {"cells": report, "rows_compared": worst_rows,
            "columns_compared_include": ["realized_pnl", "signal_sha256", "book_sha256", "settlement_sha256",
                                         "base_input_sha256"], "all_exact": True}


def decision_interval_equality(data: pd.DataFrame) -> dict:
    """A: within (arm, ticker, replicate) the 60-s and 300-s runs share signal/book/settlement/base hashes."""
    cols = ["signal_sha256", "book_sha256", "settlement_sha256", "base_input_sha256", "market_trades", "jumps_applied",
            "p_close_latent", "settlement_outcome"]
    a = data[data.decision_interval.eq(300)].groupby(["cell_group", "ticker", "replicate"])[cols].first()
    b = data[data.decision_interval.eq(60)].groupby(["cell_group", "ticker", "replicate"])[cols].first()
    if not a.index.equals(b.index):
        raise AssertionError("60-s and 300-s cells cover different episodes")
    bad = [c for c in cols if not (a[c] == b[c]).all()]
    if bad:
        raise AssertionError(f"60-s and 300-s paths differ in {bad}")
    return {"episodes_compared": len(a), "columns": cols, "identical": True}


def dose_coupling_checks(data: pd.DataFrame, paths: pd.DataFrame) -> dict:
    """B: 0x/0.5x/1x share the base-input hash; 0.5x applies a subset of the 1x jumps; 2x adds jumps."""
    out = {}
    for family in JUMP_FAMILIES:
        lab = {d: dose_label(family, d) for d in DOSES}
        h = {d: data[data.arm.eq(lab[d])].groupby(["ticker", "replicate"]).base_input_sha256.first() for d in DOSES}
        if not (h[0.] == h[1.]).all() or not (h[.5] == h[1.]).all():
            raise AssertionError(f"{family}: 0x, 0.5x and 1x must share the base inputs")
        p = {d: paths[paths.arm.eq(lab[d])].set_index(["ticker", "replicate"]) for d in DOSES}
        if not (p[.5].jumps_applied <= p[1.].jumps_applied).all() or not (p[.5].jumps_kept == p[.5].jumps_applied).all():
            raise AssertionError(f"{family}: thinned jumps must be a subset of the 1x jumps")
        if not (p[2.].jumps_applied == p[1.].jumps_applied+p[2.].jumps_inserted).all():
            raise AssertionError(f"{family}: 2x jumps must be the 1x jumps plus the inserted process")
        out[family] = {"base_inputs_shared_0x_0.5x_1x": True, "thinned_subset": True, "superposed_superset": True,
                       "mean_jumps": {f"{d:g}x": float(p[d].jumps_applied.mean()) for d in DOSES}}
    return out


# ------------------------------------------------------------------ runner
def run(experiment: str, output_dir=None, *, roster_path=None, tier1_dir=TIER1_DIR, replicates: int = E.REPLICATES,
        workers: int | None = None, tickers=None, bootstrap_draws: int = B.DRAWS, info_draws: int = INFO_DRAWS,
        seed: int = SEED, calibration_dir=None, fresh: bool = False) -> dict:
    from .config import DEFAULT_DIR, InformationCalibration
    import shutil
    t_start = time.perf_counter()
    cells = experiment_cells(experiment)
    calibration_dir = Path(calibration_dir) if calibration_dir else DEFAULT_DIR
    if not json.loads((calibration_dir/"calibration_report.json").read_text()).get("passed", False):
        raise RuntimeError("Information calibration has not passed")
    calibration = InformationCalibration.load(calibration_dir)
    output_dir = Path(output_dir) if output_dir else OUTPUT_ROOT/experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    tier1_dir = Path(tier1_dir)
    roster_path = Path(roster_path) if roster_path else tier1_dir/"roster.json"
    roster = load_roster(roster_path)
    if roster["calibration_config_sha256"] != file_hash(calibration_dir/"information_config.json"):
        raise ValueError("roster.json was matched on a different information calibration")
    if fresh and (output_dir/"checkpoints").exists():
        shutil.rmtree(output_dir/"checkpoints")
    templates = [t for t in roster["R_star"]["templates"] if tickers is None or t in tickers]
    if not templates:
        raise ValueError("No R* templates selected")
    code = E.code_hashes()
    code_sha = E._sha(code)
    weight = {300: 1, 60: 4}
    tasks = [(experiment, t, c, str(output_dir), replicates, seed, roster, code_sha) for t in templates for c in cells]
    tasks.sort(key=lambda task: -weight[task[2].decision_interval])
    workers = workers or max(1, (os.cpu_count() or 2)-1)
    E._single_threaded_children()
    summaries, info_rows = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=E._init_worker,
                             initargs=(str(calibration_dir), str(E.SAVED_BENCHMARK))) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        info_futures = []
        if experiment == "jump_intensity":
            info_futures = [pool.submit(information_task, (t, f, d, info_draws, seed))
                            for t in templates for f in JUMP_FAMILIES for d in DOSES]
        for k, fut in enumerate(as_completed(futures), 1):
            summaries.append(fut.result())
            if k % 25 == 0 or k == len(futures):
                print(f"{experiment}: {k}/{len(futures)} (template, cell) tasks", flush=True)
        info_rows = [f.result() for f in info_futures]
    t_sim = time.perf_counter()-t_start

    frames, path_frames, runtimes = [], [], {}
    for _, ticker, cell, *_ in tasks:
        stem = f"{ticker}__{cell.label}"
        frames.append(pd.read_csv(output_dir/"checkpoints"/(stem+".csv")))
        path_frames.append(pd.read_csv(output_dir/"checkpoints"/(stem+"__paths.csv")))
        runtimes[stem] = json.loads((output_dir/"checkpoints"/(stem+".json")).read_text())["runtime_seconds"]
    order = {c.label: k for k, c in enumerate(cells)}
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["arm", "ticker", "replicate", "snipe_level", "policy"],
                            key=lambda s: s.map(order) if s.name == "arm" else s, kind="stable").reset_index(drop=True)
    data["episode_id"] = (data.arm+"_"+data.ticker+"_"+data.replicate.astype(str)+"_"+data.snipe_level+"_"+data.policy)
    if data.episode_id.duplicated().any():
        raise AssertionError("Duplicate episode ids")
    data.to_csv(output_dir/"episode_results.csv", index=False)
    data = pd.read_csv(output_dir/"episode_results.csv")        # compare on the CSV round trip (exact floats)
    paths = pd.concat(path_frames, ignore_index=True)
    paths.to_csv(output_dir/"path_diagnostics.csv", index=False)

    tier1 = pd.read_csv(tier1_dir/"episode_results.csv")
    checks = {"episodes": len(data), "ledger_audited": len(data),
              "attribution_max_abs_error": float(data.attribution_error.abs().max()),
              "none_policy_max_abs_pnl": float(data[data.policy.eq("none")].realized_pnl.abs().max()),
              "s0_snipes": int(data[data.snipe_level.eq("S0")].snipe_count.sum()),
              "hash_groups_checked": int(data.groupby(["arm", "ticker", "replicate"]).ngroups),
              "tier1_equality": tier1_equality(data, tier1, cells)}
    if checks["none_policy_max_abs_pnl"] != 0 or checks["s0_snipes"] != 0 or checks["attribution_max_abs_error"] > 1e-8:
        raise AssertionError(f"Per-episode checks failed: {checks}")
    if experiment == "quote_refresh":
        checks["decision_interval_path_equality"] = decision_interval_equality(data)
    else:
        checks["dose_coupling"] = dose_coupling_checks(data, paths)
    checks["all_passed"] = True

    rosters = {"R_star": templates}
    roster_of = {c.label: ("R_star",) for c in cells}
    risk = risk_with_snipe_count(data, rosters, roster_of, bootstrap_draws, seed)
    risk.to_csv(output_dir/"risk_summary.csv", index=False)
    present = set(map(tuple, data[["arm", "snipe_level", "policy"]].drop_duplicates().to_numpy()))
    contrasts = (quote_refresh_contrasts() if experiment == "quote_refresh" else jump_intensity_contrasts())
    contrasts += B.policy_contrasts(sorted(present), roster_of)
    paired = evaluate_all_contrasts(data, contrasts, rosters, bootstrap_draws, seed)
    paired.to_csv(output_dir/"paired_contrasts.csv", index=False)
    extra_cols = ["decisions", "quote_rows", "same_price_resubmissions", "changed_quotes", "snipe_count", "maker_fills",
                  "fills_total", "hjb_decisions", "jumps_applied"]
    cell_summary = data.groupby(["arm", "group", "snipe_level", "policy"])[extra_cols].mean().reset_index()
    pooled = data.groupby(["arm", "snipe_level", "policy"])[extra_cols].mean().reset_index().assign(group="all")
    cell_summary = pd.concat([pooled, cell_summary], ignore_index=True)
    cell_summary.to_csv(output_dir/"cell_summary.csv", index=False)
    artifacts = ["episode_results.csv", "path_diagnostics.csv", "risk_summary.csv", "paired_contrasts.csv",
                 "cell_summary.csv"]
    info_meta = None
    if experiment == "jump_intensity":
        per_template = pd.DataFrame(info_rows)
        info_runtime = float(per_template.pop("runtime_seconds").sum())    # wall-clock: manifest only
        per_template.to_csv(output_dir/"information_revealed_by_template.csv", index=False)
        summary = information_summary(per_template, paths, risk)
        summary.to_csv(output_dir/"information_revealed.csv", index=False)
        artifacts += ["information_revealed_by_template.csv", "information_revealed.csv"]
        info_meta = {"estimator": "matching.rb_uncertainty at s = 1 on matching.sample_draws with "
                                  "TemplateDesign(jump_rate_scale = dose); p0 = 0.5",
                     "draws_per_template_family_dose": info_draws, "stream_tag": INFO_STREAM_TAG,
                     "class_aggregate": "equal-weight mean over R* templates; SE = sqrt(sum SE_t^2)/K",
                     "jump_share": "(I_d - I_0x) / I_d within the dose family",
                     "max_quadrature_order_difference": float(per_template.quad_max_order_diff.max()),
                     "worker_seconds": info_runtime}
    (output_dir/"tier1_equality.json").write_text(json.dumps(checks["tier1_equality"], indent=2)+"\n")
    artifacts.append("tier1_equality.json")
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    from ..hjb_policy import HJBReferencePolicy
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": head, "experiment": experiment,
        "specification": "docs/paper3/SENSITIVITY_SPEC.md",
        "design": {"seed": seed, "replicates": replicates, "policies": list(E.POLICIES), "levels": list(LEVELS),
                   "bootstrap_draws": bootstrap_draws, "workers": workers, "cpu_count": os.cpu_count(),
                   "templates": templates, "cells": [asdict(c) for c in cells],
                   "seeding": "path_seed = SeedSequence([seed, event_index, replicate]); episode "
                              f"SeedSequence(path_seed).spawn({len(STREAMS)+len(DOSE_STREAMS)}): children 0-11 = "
                              f"{list(STREAMS)} (Tier 1), 12 = {DOSE_STREAMS[0]}, 13 = {DOSE_STREAMS[1]}",
                   "thinning_keep_probability": THINNING_KEEP_PROBABILITY,
                   "dose_slope_weights": dict(zip([f"{d:g}" for d in DOSES], slope_weights())),
                   "hjb": HJBReferencePolicy(time_mapping="episode").provenance(),
                   "information_revealed": info_meta},
        "roster_file": str(roster_path.resolve().relative_to(ROOT)) if roster_path.resolve().is_relative_to(ROOT)
        else str(roster_path), "roster_sha256": roster["roster_sha256"], "roster_file_sha256": file_hash(roster_path),
        "tier1_episode_results_sha256": file_hash(tier1_dir/"episode_results.csv"),
        "calibration_sha256": calibration.config["file_sha256"],
        "calibration_config_sha256": file_hash(calibration_dir/"information_config.json"),
        "code_sha256": code, "preservation_manifest": E.preservation_manifest(),
        "runtimes": {"total_seconds": time.perf_counter()-t_start, "simulation_seconds": t_sim,
                     "per_template_cell_seconds": runtimes},
        "checks": checks,
        "not_committed": ["checkpoints/", "episode_results.csv (regenerable to artifact_sha256)"],
        "artifact_sha256": {name: file_hash(output_dir/name) for name in artifacts}}
    (output_dir/"manifest.json").write_text(json.dumps(E._finite(manifest), indent=2, default=str)+"\n")
    return {"output_dir": str(output_dir), "episodes": len(data), "runtime_seconds": manifest["runtimes"]["total_seconds"],
            "simulation_seconds": t_sim, "episode_results_sha256": manifest["artifact_sha256"]["episode_results.csv"],
            "tier1_rows_compared": checks["tier1_equality"]["rows_compared"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output-dir", type=Path, help="default data_pipeline/output/paper3/sensitivity/EXPERIMENT")
    parser.add_argument("--roster", type=Path, help="frozen roster.json (default: the Tier 1 experiment's)")
    parser.add_argument("--tier1-dir", type=Path, default=TIER1_DIR)
    parser.add_argument("--replicates", type=int, default=E.REPLICATES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--bootstrap-draws", type=int, default=B.DRAWS)
    parser.add_argument("--info-draws", type=int, default=INFO_DRAWS)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints")
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.output_dir, roster_path=args.roster, tier1_dir=args.tier1_dir,
                         replicates=args.replicates, workers=args.workers, tickers=args.tickers,
                         bootstrap_draws=args.bootstrap_draws, info_draws=args.info_draws,
                         calibration_dir=args.calibration_dir, fresh=args.fresh), indent=2, default=str))
