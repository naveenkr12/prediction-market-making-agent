"""Paper 3 experiment runner (spec Section 9): Tier 1, and Tier 2 where hooks exist (--tier2).

Seeding and template order follow Section 5 (`skeleton.path_seed`, global
event index over broad_based then sports, tickers sorted). Every episode is
run for the four policies exactly as `simulator.evaluate_hjb` calls them
(frozen HJB with time_mapping "episode", fixed, skew 0.0005, none).
Multiprocessing is over templates; results are checkpointed per
(template, arm) and resumed when the checkpoint key (plan, s, p0, seed,
replicates, roster hash, code hash) is unchanged.

Per-episode checks (any failure aborts the run): ledger audit, additive
attribution reconciles, reward equals realized PnL, genuine settlement,
signal/book/settlement hashes equal across policies and S levels within
(template, replicate, arm) and equal to the precomputed path's trace,
none-policy PnL = 0. L0 rows reproduce the saved Paper 2
`hjb_benchmark/episode_results.csv` (impact base, fill base) to 1e-9 and its
price-path hash.

Usage: python -m simulator.information.evaluate [--output-dir DIR] [--roster PATH]
       [--replicates 20] [--workers K] [--tickers ...] [--tier2]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from ..benchmark_metrics import episode_diagnostics
from ..demo_trading_env import FILL_SCENARIOS
from ..evaluate_hjb import path_hashes
from ..market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, file_hash
from ..spreads import DEFAULT_CONFIG as SPREAD_CONFIG
from ..trading_env import EnvironmentConfig, TradingEnvironment
from ..validate_trading_env import audit_ledger
from . import bootstrap as B
from . import metrics as Mx
from .environment import ANCHOR_COLUMNS, InformationTradingEnvironment, run_policy
from .matching import load_roster
from .paths import build_path, make_arm, with_p0
from .skeleton import MEAN_LATENCY_SECONDS, SEED, build_skeleton, event_roster, path_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT/"data_pipeline"/"output"/"paper3"/"experiment"
SAVED_BENCHMARK = ROOT/"data_pipeline/output/paper2/calibration/simulator/hjb_benchmark/episode_results.csv"
POLICIES = ("hjb", "fixed", "skew", "none")
REPLICATES = 20
ANCHOR_TOLERANCE = 1e-9
TIER2_NOT_IMPLEMENTED = {
    "zero_latency_grid": "Zero-latency (0.25 s / 0.0625 s Brownian-bridge grid) sniping is not implemented in P3 "
                         "(build_skeleton rejects mean_latency <= 0).",
    "s1_thinned": "S1-thinned exogenous executions (pilot-estimated pi_thin per class and policy) not implemented.",
    "ttr_clock_arrivals": "TTR-clock arrival sensitivity not implemented.",
    "early_revelation": "Early-revelation templates (absorbing p = Theta at t_r, M family only) not implemented."}


@dataclass(frozen=True)
class ArmPlan:
    label: str                       # unique cell label
    base: str                        # make_arm name, or "L0" (TradingEnvironment itself)
    rosters: tuple                   # rosters the arm is run on (primary first): all35 | R_star | R_star_emp
    levels: tuple = ("S0",)
    s_source: str = "one"            # one | matched | matched_emp
    p0_source: str = "0.5"           # 0.5 | emp
    tier: int = 1
    constant_alpha: bool = False
    jump_rate_scale: float = 1.0
    mean_latency: float = MEAN_LATENCY_SECONDS
    decision_interval: int = 300
    fill: str = "base"


def tier1_plan() -> list[ArmPlan]:
    both = ("S0", "S1")
    plan = [ArmPlan("L0", "L0", ("all35", "R_star"))]
    plan += [ArmPlan(m, m, ("R_star",), both, "matched") for m in ("M0", "M1", "M2")]
    plan += [ArmPlan("M3", "M3", ("R_star",), both)]
    plan += [ArmPlan(n, n, ("R_star",)) for n in ("L0b", "L0c", "L1J", "L1")]
    plan += [ArmPlan("M1-native", "M1-native", ("R_star",), both)]
    plan += [ArmPlan(n, n, ("R_star",)) for n in ("FGH-L", "FGH-LJ")]
    # Secondary native (s = 1): M1 at s = 1 is the M1-native arm and M3 at s = 1 is M3 (identical arms; not re-run).
    plan += [ArmPlan(f"{m}-s1", m, ("R_star",), both) for m in ("M0", "M2")]
    plan += [ArmPlan(f"{m}-emp", m, ("R_star_emp",), both, "matched_emp", "emp") for m in ("M0", "M1", "M2")]
    plan += [ArmPlan("M3-emp", "M3", ("R_star_emp",), both, "one", "emp")]
    return plan


def tier2_plan(calibration) -> list[ArmPlan]:
    both, plan = ("S0", "S1"), []
    matched = lambda m: "matched" if m in ("M0", "M1", "M2") else "one"
    for scale in (0.5, 2.0):
        plan += [ArmPlan(f"{m}-jx{scale:g}", m, ("R_star",), both, matched(m), tier=2, jump_rate_scale=scale)
                 for m in ("M2", "M3")]
        plan += [ArmPlan(f"{n}-jx{scale:g}", n, ("R_star",), tier=2, jump_rate_scale=scale)
                 for n in ("L0c", "L1J", "FGH-LJ")]
    try:
        for cls in ("broad_based", "sports"):
            calibration.constant_alpha(cls)
        plan += [ArmPlan(f"{m}-const", m, ("R_star",), both, matched(m), tier=2, constant_alpha=True)
                 for m in ("M1", "M3", "M1-native")]
    except (ValueError, KeyError):
        pass
    for fill in ("conservative", "optimistic"):
        plan += [ArmPlan(f"{m}-fill-{fill}", m, ("R_star",), both, matched(m), tier=2, fill=fill)
                 for m in ("M0", "M1", "M2", "M3")]
    plan += [ArmPlan(f"{m}-dec60", m, ("R_star",), both, matched(m), tier=2, decision_interval=60)
             for m in ("M0", "M1", "M2", "M3")]
    plan += [ArmPlan(f"{m}-lat30", m, ("R_star",), ("S1",), matched(m), tier=2, mean_latency=30.0)
             for m in ("M0", "M1", "M2", "M3")]
    return plan


# ------------------------------------------------------------- worker state
_STATE: dict = {}


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def _single_threaded_children():
    """Spawned workers read these at import: one BLAS thread per process (no oversubscription)."""
    for key in THREAD_VARIABLES:
        os.environ.setdefault(key, "1")


def _init_worker(calibration_dir, saved_path):
    from ..hjb_policy import HJBReferencePolicy
    from .config import InformationCalibration
    _STATE["calibration"] = InformationCalibration.load(calibration_dir) if calibration_dir else InformationCalibration.load()
    _STATE["hjb"] = HJBReferencePolicy(time_mapping="episode")
    saved = pd.read_csv(saved_path)
    saved = saved[saved.impact_scenario.eq("base") & saved.fill_scenario.eq("base")]
    _STATE["saved"] = {(r["ticker"], int(r["replicate"]), r["policy"]): r for r in saved.to_dict("records")}


def code_hashes() -> dict:
    files = sorted((ROOT/"simulator"/"information").glob("*.py"))
    files += [ROOT/"simulator"/n for n in ("trading_env.py", "market_dynamics.py", "price_impact.py", "spreads.py",
                                           "execution.py", "accounting.py", "fees.py", "policies.py", "hjb_policy.py",
                                           "evaluate_hjb.py", "benchmark_metrics.py", "validate_trading_env.py")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def resolve_arm(plan: ArmPlan, ticker: str, roster: dict, calibration):
    """(Arm, s, p0) for one plan cell on one template (None for L0)."""
    if plan.base == "L0":
        return None, None, 0.5
    p0 = 0.5 if plan.p0_source == "0.5" else calibration.p0_emp(ticker)
    if plan.s_source == "one":
        s = 1.0
    else:
        key = "R_star" if plan.s_source == "matched" else "R_star_emp"
        s = float(roster[key]["matched_s"][ticker][plan.base])
    if plan.base in ("M0", "M1", "M2"):
        arm = make_arm(plan.base, s=s, constant_alpha=plan.constant_alpha)
    elif plan.base in ("M3", "M1-native"):
        arm = make_arm(plan.base, constant_alpha=plan.constant_alpha)
    else:
        arm = make_arm(plan.base)
    arm = replace(with_p0(arm, p0), name=plan.label)
    return arm, s, p0


# ------------------------------------------------------------ episode rows
def _drawdown(ledger, initial_cash):
    peak, drawdown, max_q = float(initial_cash), 0., 0.
    for row in ledger:
        q = float(row["inventory"])
        equity = float(row["cash"])+q*float(row["probability"])
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
        max_q = max(max_q, abs(q))
    return drawdown, max_q


def _common_row(env, records, ledger, initial_cash, initial_p, lifetime, ttr_close):
    attribution = Mx.attribution(ledger, initial_cash, initial_p)
    pnl = float(env.account.cash-env.account.initial_cash)
    if abs(attribution["pnl"]-pnl) > 1e-8 or abs(env.total_reward-pnl) > 1e-8:
        raise AssertionError("Realized PnL, reward and attribution disagree")
    if not env.account.settled:
        raise AssertionError("Episode did not reach genuine settlement")
    diag = Mx.diagnostics(ledger, initial_probability=initial_p, lifetime=lifetime, release_minus_close_seconds=ttr_close)
    drawdown, max_q = _drawdown(ledger, initial_cash)
    info = records[-1]["info"]
    q_c = float(info["terminal_inventory"])
    fills = sum(r["kind"] == "fill" for r in ledger)
    return {"realized_pnl": pnl, "cash": float(env.account.cash), "terminal_inventory": q_c,
            "abs_terminal_inventory": abs(q_c), "settlement_outcome": int(info["settlement_outcome"]),
            "closing_probability": float(env.price.probability), "maker_fills": fills,
            "maker_contracts": float(sum(float(r["quantity"]) for r in ledger if r["kind"] == "fill")),
            "snipe_count": diag["snipe_count"], "fills_total": fills+diag["snipe_count"],
            "snipe_volume": diag["snipe_volume"], "snipe_pnl": diag["snipe_pnl"],
            "fees": attribution["fees"], "spread_capture_fills": attribution["spread_capture_fills"],
            "spread_capture_snipes": attribution["spread_capture_snipes"],
            "background_price_pnl": attribution["background_price_pnl"], "flow_price_pnl": attribution["flow_price_pnl"],
            "jump_price_pnl": attribution["jump_price_pnl"], "settlement_surprise_pnl": attribution["settlement_surprise_pnl"],
            "attribution_error": attribution["attribution_error"], "max_drawdown": drawdown, "max_abs_inventory": max_q,
            "flow_markout": diag["flow_markout"], "jump_markout": diag["jump_markout"], "markout_5m": diag["markout_5m"],
            "market_trades": sum(r["kind"] == "market" for r in ledger), "ledger_rows": len(ledger)}


def run_l0_replicate(group, ticker, index, replicate, *, seed=SEED, hours_to_close=1.0):
    """L0 = TradingEnvironment itself with Section 5 seeding; checked against the saved benchmark rows."""
    hjb, saved, calibration = _STATE["hjb"], _STATE["saved"], _STATE["calibration"]
    queue, participation = FILL_SCENARIOS["base"]
    pseed = path_seed(index, replicate, seed)
    lifetime = json.loads(DYNAMICS_CONFIG.read_text())["event_templates"][ticker]["lifetime_seconds"]
    seconds = min(hours_to_close*3600, lifetime)
    cfg = EnvironmentConfig(group=group, ticker=ticker, impact_scenario="base", queue_ahead_contracts=queue,
                            participation=participation)
    rows, price_hash, ledgers = [], None, {}
    for policy in POLICIES:
        env = TradingEnvironment(cfg)
        records, actions = run_policy(env, policy, hjb, seed=pseed, options={"seconds_to_close": seconds})
        obs = records[0]["observation"]
        audit = audit_ledger(env.ledger, initial_cash=cfg.initial_cash, initial_log_odds=obs["log_odds"],
                             beta=env.dynamics.impact.beta, inventory_limit=cfg.inventory_limit,
                             maker_fee_multiplier=cfg.maker_fee_multiplier, balance_precision=cfg.balance_precision,
                             quantity_lot=cfg.quantity_lot, participation=cfg.participation)
        legacy = episode_diagnostics(env.ledger, cfg.initial_cash, obs["probability"], env.start_elapsed, env.elapsed)
        row = _common_row(env, records, env.ledger, cfg.initial_cash, obs["probability"], env.lifetime,
                          calibration.ttr_at_close(ticker))
        _, phash = path_hashes(env)
        if price_hash is None:
            price_hash = phash
        if phash != price_hash:
            raise AssertionError("L0 policies did not see identical price paths")
        if policy == "none" and (row["realized_pnl"] != 0 or audit["maker_fills"]):
            raise AssertionError("No-quote control must have zero PnL")
        got = {"seed": pseed, "realized_pnl": row["realized_pnl"], "cash": row["cash"],
               "terminal_inventory": row["terminal_inventory"], "settlement_outcome": row["settlement_outcome"],
               "closing_probability": env.price.probability, "market_trades": audit["market_trades"],
               "hjb_decisions": len(actions), **{k: legacy[k] for k in ("maker_fills", "maker_contracts", "fees",
               "spread_capture", "background_price_pnl", "flow_price_pnl", "settlement_surprise_pnl", "max_drawdown")}}
        ref = saved[(ticker, replicate, policy)]
        worst = max(abs(float(got[c])-float(ref[c])) for c in ANCHOR_COLUMNS)
        if worst > ANCHOR_TOLERANCE or phash != ref["price_path_sha256"]:
            raise AssertionError(f"L0 anchor mismatch {ticker} r{replicate} {policy}: {worst}")
        row.update(anchor_max_abs_difference=worst, price_path_sha256=phash, hjb_decisions=len(actions),
                   signal_sha256=phash, book_sha256=phash, settlement_sha256=_sha(row["settlement_outcome"]),
                   donor=env.dynamics.noise_template_ticker, sigma=env.dynamics.sigma, seed=pseed, policy=policy,
                   jumps_applied=0)
        rows.append(row)
        ledgers[policy] = env.ledger
    return rows, ledgers


def path_diagnostics(path, calibration) -> dict:
    F, dL = Mx.kyle_windows(path)
    out = {**Mx.flow_moments(path), **Mx.information_variance(path, calibration),
           "kyle_windows": len(F), "kyle_slope_episode": Mx.ols_slope(F, dL), "kyle_sum_f": float(F.sum()),
           "kyle_sum_dl": float(dL.sum()), "kyle_sum_ff": float(np.dot(F, F)), "kyle_sum_fdl": float(np.dot(F, dL)),
           "saturated_nodes": path.saturated_nodes, "p_close_latent": path.settlement["p_close"],
           "outcome": path.settlement["outcome"], "theta": path.theta if path.theta is not None else -1,
           "sigma2": path.sigma2, "sigma_c": path.sigma_c}
    return out


def run_arm_replicate(sk, arm, plan: ArmPlan, path=None):
    """All S levels x policies of one (episode, arm), with the per-episode checks.

    `path` (optional) is a prebuilt path of `arm` on `sk` (the sensitivity runner passes coupled-dose paths).
    """
    calibration, hjb = _STATE["calibration"], _STATE["hjb"]
    if path is None:
        path = build_path(sk, arm, calibration=calibration)
    if path.skeleton is not sk or path.arm != arm:
        raise ValueError("Prebuilt path does not belong to this skeleton and arm")
    queue, participation = FILL_SCENARIOS[plan.fill]
    cfg = replace(sk.config, initial_probability=arm.p0, queue_ahead_contracts=queue, participation=participation)
    expected = path.walk_trace_hashes()
    rows, ledgers = [], {}
    base_hash = sk.base_input_hash()
    for level in plan.levels:
        for policy in POLICIES:
            env = InformationTradingEnvironment(path, cfg, snipe_level=level, decision_interval=plan.decision_interval)
            records, actions = run_policy(env, policy, hjb, seed=sk.path_seed)
            audit = Mx.audit_information_ledger(
                env.ledger, initial_cash=cfg.initial_cash, initial_log_odds=path.initial_log_odds,
                inventory_limit=cfg.inventory_limit, maker_fee_multiplier=cfg.maker_fee_multiplier,
                balance_precision=cfg.balance_precision, quantity_lot=cfg.quantity_lot, participation=cfg.participation,
                tick=cfg.price_tick, beta=sk.dynamics.impact.beta, expected_executions=sk.n_executions,
                expected_jumps=path.active_jump_count)
            row = _common_row(env, records, env.ledger, cfg.initial_cash, records[0]["observation"]["probability"],
                              sk.lifetime, sk.ttr_at_close)
            hashes = env.walk_hashes()
            if hashes != expected:
                raise AssertionError(f"{arm.name} {level} {policy}: walk hashes differ from the path trace")
            if policy == "none" and (row["realized_pnl"] != 0 or audit["maker_fills"] or audit["snipes"]):
                raise AssertionError("No-quote control must have zero PnL")
            if level == "S0" and audit["snipes"]:
                raise AssertionError("S0 must not snipe")
            row.update(policy=policy, snipe_level=level, hjb_decisions=len(actions), signal_sha256=hashes["signal"],
                       book_sha256=hashes["book"], settlement_sha256=hashes["settlement"], base_input_sha256=base_hash,
                       jumps_applied=env.jump_count, donor=sk.noise_ticker, sigma=path.sigma_c, seed=sk.path_seed,
                       p_close_latent=path.settlement["p_close"])
            rows.append(row)
            ledgers[(level, policy)] = env.ledger
    if sk.base_input_hash() != base_hash:
        raise AssertionError("Skeleton base inputs were mutated")
    return rows, ledgers, path_diagnostics(path, calibration)


def _checkpoint_key(plan, ticker, replicates, seed, roster_sha, code_sha) -> str:
    return _sha({"plan": asdict(plan), "ticker": ticker, "replicates": replicates, "seed": seed, "roster": roster_sha,
                 "code": code_sha})


def run_template(task) -> dict:
    ticker, plans, out_dir, replicates, seed, roster, code_sha, representative = task
    calibration = _STATE["calibration"]
    group, index = next((g, i) for g, t, i in event_roster() if t == ticker)
    out_dir = Path(out_dir)
    ckpt = out_dir/"checkpoints"
    ledger_dir = out_dir/"ledgers"
    skeletons: dict = {}
    summary = {"ticker": ticker, "cells": []}
    for plan in plans:
        key = _checkpoint_key(plan, ticker, replicates, seed, roster["roster_sha256"], code_sha)
        stem = f"{ticker}__{plan.label}"
        meta_path = ckpt/(stem+".json")
        if meta_path.exists() and json.loads(meta_path.read_text()).get("key") == key and \
                (ckpt/(stem+".csv")).exists():
            summary["cells"].append({"label": plan.label, "resumed": True})
            continue
        t0 = time.perf_counter()
        rows, diags = [], []
        arm, s, p0 = resolve_arm(plan, ticker, roster, calibration)
        for replicate in range(replicates):
            if plan.base == "L0":
                episode_rows, ledgers = run_l0_replicate(group, ticker, index, replicate, seed=seed)
                diag = None
            else:
                skey = (replicate, plan.jump_rate_scale, plan.mean_latency)
                if skey not in skeletons:
                    skeletons[skey] = build_skeleton(ticker, replicate, calibration=calibration, seed=seed,
                                                     jump_rate_scale=plan.jump_rate_scale, mean_latency=plan.mean_latency)
                sk = skeletons[skey]
                episode_rows, ledgers, diag = run_arm_replicate(sk, arm, plan)
                by_level = {}
                for r in episode_rows:
                    by_level.setdefault("all", set()).add((r["signal_sha256"], r["book_sha256"], r["settlement_sha256"]))
                if len(by_level["all"]) != 1:
                    raise AssertionError("Hashes differ across policies/S levels within an arm")
            common = {"tier": plan.tier, "arm": plan.label, "base_arm": plan.base,
                      "family": "legacy" if plan.base == "L0" else arm.family, "roster": plan.rosters[0],
                      "ticker": ticker, "group": group, "event_index": index, "replicate": replicate, "p0": p0,
                      "s": s, "fill_scenario": plan.fill, "decision_interval": plan.decision_interval,
                      "jump_rate_scale": plan.jump_rate_scale, "mean_latency": plan.mean_latency,
                      "constant_alpha": plan.constant_alpha}
            for r in episode_rows:
                r.setdefault("snipe_level", "S0")
                rows.append({**common, **r})
            if diag is not None:
                diags.append({**common, **diag})
            if representative and replicate == 0:
                ledger_dir.mkdir(parents=True, exist_ok=True)
                for k, ledger in ledgers.items():
                    level, policy = k if isinstance(k, tuple) else ("S0", k)
                    if policy == "hjb":
                        pd.DataFrame(ledger).to_csv(ledger_dir/f"{ticker}__{plan.label}__{level}__hjb_ledger.csv",
                                                    index=False)
        runtime = time.perf_counter()-t0
        ckpt.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        for name, data in ((stem+".csv", frame), (stem+"__paths.csv", pd.DataFrame(diags))):
            tmp = ckpt/(name+".tmp")
            data.to_csv(tmp, index=False)
            os.replace(tmp, ckpt/name)
        meta = {"key": key, "plan": asdict(plan), "ticker": ticker, "episodes": len(frame), "runtime_seconds": runtime,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "checks": {"ledger_audit": True, "attribution_max_abs_error": float(frame.attribution_error.abs().max()),
                           "hashes_shared": True, "none_policy_zero_pnl": True,
                           "anchor_max_abs_difference": float(frame.anchor_max_abs_difference.max())
                           if "anchor_max_abs_difference" in frame else None}}
        tmp = ckpt/(stem+".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
        summary["cells"].append({"label": plan.label, "resumed": False, "runtime_seconds": runtime})
    return summary


# ------------------------------------------------------------ aggregation
def arm_diagnostics(paths: pd.DataFrame, calibration) -> tuple[pd.DataFrame, dict]:
    """Per (arm, class) sign/size moments, information variance, simulated Kyle lambda; fidelity summary."""
    rows = []
    fidelity = {"kyle_lambda": {}, "jumps": {}}
    for (arm, cls), d in paths.groupby(["arm", "group"], sort=True):
        per_template = []
        for ticker, t in d.groupby("ticker"):
            n = t.kyle_windows.sum()
            sxx = t.kyle_sum_ff.sum()-t.kyle_sum_f.sum()**2/n if n > 1 else 0.0
            sxy = t.kyle_sum_fdl.sum()-t.kyle_sum_f.sum()*t.kyle_sum_dl.sum()/n if n > 1 else 0.0
            per_template.append({"pooled": sxy/sxx if sxx > 1e-24 else 0.0,
                                 "replicate_averaged": float(t.kyle_slope_episode.mean())})
        pt = pd.DataFrame(per_template)
        target = calibration.config["classes"][cls]["flow"].get("target_lambda_median")
        expected_jumps = None
        row = {"arm": arm, "group": cls, "episodes": len(d), "templates": d.ticker.nunique(),
               **{f"mean_{c}": float(d[c].mean()) for c in (
                   "trades", "buy_fraction", "mean_size", "mean_signed_contracts_per_trade", "informed_fraction",
                   "mean_abs_window_flow", "jumps_applied", "mean_jump_magnitude", "up_jump_fraction",
                   "info_variance_flow", "info_variance_jump", "info_variance_background", "saturated_nodes")},
               "var_flow_llr_sum": float(d.flow_llr_sum.var(ddof=1)) if len(d) > 1 else np.nan,
               "kyle_lambda_median_replicate_averaged": float(pt.replicate_averaged.median()),
               "kyle_lambda_median_pooled": float(pt.pooled.median()), "kyle_lambda_target": target,
               "mean_p_close_uncertainty": float((d.p_close_latent*(1-d.p_close_latent)).mean())}
        rows.append(row)
        fidelity["kyle_lambda"][f"{arm}|{cls}"] = {k: row[k] for k in (
            "kyle_lambda_median_replicate_averaged", "kyle_lambda_median_pooled", "kyle_lambda_target")}
        base = d.base_arm.iloc[0]
        if make_arm(base, s=1.0).jumps != "none":
            fidelity["jumps"][f"{arm}|{cls}"] = {"mean_jumps_per_episode": row["mean_jumps_applied"],
                                                  "mean_jump_magnitude": row["mean_mean_jump_magnitude"],
                                                  "bank_mean_magnitude": float(np.mean(calibration.magnitude_bank(cls))),
                                                  "expected_jumps_per_episode": expected_jumps}
    return pd.DataFrame(rows), fidelity


def expected_jump_counts(tickers, calibration, scale=1.0) -> dict:
    from .matching import TemplateDesign
    return {t: float(sum(TemplateDesign.build(t, calibration, jump_rate_scale=scale).jump_means)) for t in tickers}


def _finite(value):
    """JSON-safe copy: NaN/inf -> None."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def training_isolation(tickers, calibration_dir) -> dict:
    """Measured training-only checks for the experiment (replaces a hard-coded held-out count).

    Every simulated template must be a train-split ticker of `event_splits.csv`. The runner reads no raw
    tape or quote file: its inputs are the Paper 2 training calibrations (market dynamics, mark bank,
    spreads, impact), the Paper 3 information calibration, the saved Paper 2 benchmark rows and the
    frozen HJB grid. The calibration's own held-out count (measured by its TrainingGuard) is recorded.
    """
    splits = pd.read_csv(ROOT/"data_pipeline/output/paper2/calibration/event_splits.csv")
    train = set(splits.loc[splits.split.eq("train"), "ticker"])
    outside = sorted(set(tickers)-train)
    if outside:
        raise ValueError(f"Simulated templates outside the train split: {outside}")
    report = json.loads((Path(calibration_dir)/"calibration_report.json").read_text())
    dynamics = json.loads(DYNAMICS_CONFIG.read_text())
    spread = json.loads(SPREAD_CONFIG.read_text())
    inputs = [DYNAMICS_CONFIG, DYNAMICS_CONFIG.parent/dynamics["mark_bank_file"],
              (DYNAMICS_CONFIG.parent/dynamics["impact_config_file"]).resolve(), SPREAD_CONFIG,
              (SPREAD_CONFIG.parent/spread["bank_file"]).resolve(), SAVED_BENCHMARK,
              *sorted(Path(calibration_dir).glob("*"))]
    return {"simulated_templates": len(tickers), "simulated_templates_in_train_split": len(set(tickers) & train),
            "simulated_templates_outside_train_split": outside,
            "information_calibration_heldout_files_loaded": report.get("heldout_files_loaded"),
            "data_files_read": sorted(str(p.resolve().relative_to(ROOT)) for p in inputs if p.is_file()),
            "raw_tape_or_quote_files_read": []}


def preservation_manifest() -> dict:
    files = sorted((ROOT/"hjb_baseline").glob("*.py"))
    files += [ROOT/"hjb_baseline/output/data/V_production.npy", ROOT/"hjb_baseline/output/data/grid_production.npz"]
    files += sorted((ROOT/"simulator").glob("*.py")) + sorted((ROOT/"calibration").glob("*.py"))
    files += sorted(p for p in (ROOT/"data_pipeline/output/paper2/calibration/simulator").rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def plot_summary(risk: pd.DataFrame, output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"capstone-matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = risk[risk.scope.eq("all") & risk.policy.isin(["hjb", "fixed", "skew"]) & risk.roster.ne("all35")]
    if sel.empty:
        return None
    cells = sel[["arm", "snipe_level"]].drop_duplicates().sort_values(["arm", "snipe_level"])
    labels = [f"{a}|{s}" for a, s in cells.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(max(8, .45*len(labels)), 5))
    x = np.arange(len(labels))
    for k, (policy, color) in enumerate((("hjb", "#245879"), ("fixed", "#9b6338"), ("skew", "#4c7765"))):
        d = sel[sel.policy.eq(policy)].set_index(["arm", "snipe_level"])
        y, lo, hi = [], [], []
        for a, s in cells.itertuples(index=False):
            r = d.loc[(a, s)] if (a, s) in d.index else None
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            y.append(np.nan if r is None else r["mean_pnl"])
            lo.append(np.nan if r is None else r["mean_pnl"]-r["mean_pnl_bootstrap95_low"])
            hi.append(np.nan if r is None else r["mean_pnl_bootstrap95_high"]-r["mean_pnl"])
        ax.errorbar(x+(k-1)*.22, y, yerr=[lo, hi], fmt="o", color=color, label=policy, ms=3, capsize=2)
    ax.axhline(0, color=".6", lw=.6)
    ax.set_xticks(x, labels, rotation=75, fontsize=7)
    ax.set_ylabel("Mean PnL per one-hour episode ($)")
    ax.set_title("Paper 3 experiment: mean PnL by arm (hierarchical bootstrap 95% ranges)")
    ax.legend(frameon=False)
    fig.tight_layout()
    name = "mean_pnl_by_arm.png"
    fig.savefig(output_dir/name, dpi=150)
    plt.close(fig)
    return name


def run(output_dir=DEFAULT_OUTPUT, *, roster_path=None, replicates: int = REPLICATES, workers: int | None = None,
        tickers=None, tier2: bool = False, bootstrap_draws: int = B.DRAWS, seed: int = SEED, calibration_dir=None,
        fresh: bool = False) -> dict:
    from .config import DEFAULT_DIR, InformationCalibration
    t_start = time.perf_counter()
    calibration_dir = Path(calibration_dir) if calibration_dir else DEFAULT_DIR
    report = json.loads((calibration_dir/"calibration_report.json").read_text())
    if not report.get("passed", False):
        raise RuntimeError("Information calibration has not passed")
    calibration = InformationCalibration.load(calibration_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = Path(roster_path) if roster_path else output_dir/"roster.json"
    roster = load_roster(roster_path)
    if roster["calibration_config_sha256"] != file_hash(calibration_dir/"information_config.json"):
        raise ValueError("roster.json was matched on a different information calibration")
    if roster_path.parent.resolve() != output_dir.resolve():
        shutil.copy2(roster_path, output_dir/"roster.json")
        shutil.copy2(roster_path.parent/"matching_report.csv", output_dir/"matching_report.csv")
    if fresh and (output_dir/"checkpoints").exists():
        shutil.rmtree(output_dir/"checkpoints")
    all_templates = [t for _, t, _ in event_roster() if tickers is None or t in tickers]
    rosters = {"all35": all_templates,
               "R_star": [t for t in roster["R_star"]["templates"] if t in all_templates],
               "R_star_emp": [t for t in roster["R_star_emp"]["templates"] if t in all_templates]}
    plan = tier1_plan()+(tier2_plan(calibration) if tier2 else [])
    code = code_hashes()
    code_sha = _sha(code)
    groups = {t: g for g, t, _ in event_roster()}
    representative = {next((t for t in rosters["R_star"] if groups[t] == cls), None) for cls in ("broad_based", "sports")}
    tasks = []
    for ticker in all_templates:
        cells = [p for p in plan if any(ticker in rosters[r] for r in p.rosters)]
        if cells:
            tasks.append((ticker, cells, str(output_dir), replicates, seed, roster, code_sha, ticker in representative))
    workers = workers or max(1, (os.cpu_count() or 2)-1)
    _single_threaded_children()
    summaries = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(calibration_dir), str(SAVED_BENCHMARK))) as pool:
        for summary in pool.map(run_template, tasks):
            summaries.append(summary)
            print(f"{summary['ticker']}: {len(summary['cells'])} cells", flush=True)
    t_sim = time.perf_counter()-t_start

    frames, paths, runtimes = [], [], {}
    for ticker, cells, *_ in tasks:
        for p in cells:
            stem = f"{ticker}__{p.label}"
            frames.append(pd.read_csv(output_dir/"checkpoints"/(stem+".csv")))
            pth = output_dir/"checkpoints"/(stem+"__paths.csv")
            if pth.stat().st_size > 2:
                try:
                    paths.append(pd.read_csv(pth))
                except pd.errors.EmptyDataError:
                    pass
            runtimes[stem] = json.loads((output_dir/"checkpoints"/(stem+".json")).read_text())["runtime_seconds"]
    data = pd.concat(frames, ignore_index=True)
    data["episode_id"] = (data.arm+"_"+data.ticker+"_"+data.replicate.astype(str)+"_"+data.snipe_level+"_"+data.policy)
    if data.episode_id.duplicated().any():
        raise AssertionError("Duplicate episode ids")
    data.to_csv(output_dir/"episode_results.csv", index=False)
    path_frame = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    path_frame.to_csv(output_dir/"path_diagnostics.csv", index=False)

    roster_of = {p.label: p.rosters for p in plan}
    risk = B.risk_summary(data, rosters, roster_of, draws=bootstrap_draws, seed=seed)
    risk.to_csv(output_dir/"risk_summary.csv", index=False)
    cells = set(map(tuple, data[["arm", "snipe_level", "policy"]].drop_duplicates().to_numpy()))
    contrasts = B.tier1_contrasts()+B.policy_contrasts(sorted(cells), roster_of)
    contrast_rosters = {k: v for k, v in rosters.items() if k in ("R_star", "R_star_emp", "all35")}
    missing = [c["contrast"] for c in contrasts if c["roster"] not in contrast_rosters]
    if missing:
        raise AssertionError(f"Contrasts on rosters that are not evaluated: {missing[:5]}")
    paired = B.evaluate_contrasts(data, contrasts, contrast_rosters, draws=bootstrap_draws, seed=seed)
    paired.to_csv(output_dir/"paired_contrasts.csv", index=False)
    if not path_frame.empty:
        diag, fidelity = arm_diagnostics(path_frame, calibration)
        expected = expected_jump_counts(sorted(set(path_frame.ticker)), calibration)
        for key, value in fidelity["jumps"].items():
            arm, cls = key.split("|")
            sub = path_frame[(path_frame.arm == arm) & (path_frame.group == cls)]
            value["expected_jumps_per_episode"] = float(np.mean([expected[t] for t in sub.ticker]))*(
                float(sub.jump_rate_scale.iloc[0]) if len(sub) else 1.0)
    else:
        diag, fidelity = pd.DataFrame(), {}
    diag.to_csv(output_dir/"arm_diagnostics.csv", index=False)
    fidelity.update(scope="Experiment-path fidelity (V10 report); full V10 is in validate.py",
                    flow_replay={c: {k: calibration.config["classes"][c]["flow"].get(k) for k in (
                        "target_lambda_median", "calibration_statistic", "calibration_relative_gap")}
                        for c in ("broad_based", "sports")})
    (output_dir/"fidelity_report.json").write_text(json.dumps(_finite(fidelity), indent=2, allow_nan=False)+"\n")
    figure = plot_summary(risk, output_dir)

    checks = {"episodes": len(data), "ledger_audited": len(data), "attribution_max_abs_error":
              float(data.attribution_error.abs().max()), "none_policy_max_abs_pnl":
              float(data[data.policy.eq("none")].realized_pnl.abs().max()),
              "l0_anchor_episodes": int(data.arm.eq("L0").sum()),
              "l0_anchor_max_abs_difference": float(data.anchor_max_abs_difference.max())
              if "anchor_max_abs_difference" in data else None, "hash_groups_checked":
              int(data.groupby(["arm", "ticker", "replicate"]).ngroups), "all_passed": True}
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = None
    from ..hjb_policy import HJBReferencePolicy
    hjb_provenance = HJBReferencePolicy(time_mapping="episode").provenance()
    artifacts = ["episode_results.csv", "path_diagnostics.csv", "risk_summary.csv", "paired_contrasts.csv",
                 "arm_diagnostics.csv", "fidelity_report.json", "roster.json", "matching_report.csv"]+(
                     [figure] if figure else [])
    artifacts += sorted(str(p.relative_to(output_dir)) for p in (output_dir/"ledgers").glob("*.csv")) \
        if (output_dir/"ledgers").exists() else []
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
        "design": {"seed": seed, "replicates": replicates, "policies": list(POLICIES), "hours_to_close": 1.0,
                   "tier2": tier2, "bootstrap_draws": bootstrap_draws, "workers": workers, "cpu_count": os.cpu_count(),
                   "tickers": all_templates, "rosters": rosters, "plan": [asdict(p) for p in plan],
                   "seeding": "path_seed = SeedSequence([seed, event_index, replicate]); episode SeedSequence(path_seed)"
                              ".spawn(12) per skeleton.STREAMS", "hjb": hjb_provenance,
                   "aliases": {"M1-s1": "M1-native", "M3-s1": "M3"}},
        "tier2_not_implemented": TIER2_NOT_IMPLEMENTED if tier2 else "tier2 not requested",
        "roster_sha256": roster["roster_sha256"], "roster_file_sha256": file_hash(output_dir/"roster.json"),
        "calibration_sha256": calibration.config["file_sha256"],
        "calibration_config_sha256": file_hash(calibration_dir/"information_config.json"),
        "config_sha256": {str(DYNAMICS_CONFIG.relative_to(ROOT)): file_hash(DYNAMICS_CONFIG),
                          str(SPREAD_CONFIG.relative_to(ROOT)): file_hash(SPREAD_CONFIG),
                          str(SAVED_BENCHMARK.relative_to(ROOT)): file_hash(SAVED_BENCHMARK)},
        "code_sha256": code, "preservation_manifest": preservation_manifest(),
        "runtimes": {"total_seconds": time.perf_counter()-t_start, "simulation_seconds": t_sim,
                     "per_template_arm_seconds": runtimes},
        "checks": checks, "training_isolation": training_isolation(all_templates, calibration_dir),
        "artifact_sha256": {name: file_hash(output_dir/name) for name in artifacts}}
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+"\n")
    return {"output_dir": str(output_dir), "episodes": len(data), "runtime_seconds": manifest["runtimes"]["total_seconds"],
            "simulation_seconds": t_sim, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roster", type=Path, help="roster.json (default: OUTPUT_DIR/roster.json)")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--tier2", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=B.DRAWS)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--fresh", action="store_true", help="discard existing checkpoints")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, roster_path=args.roster, replicates=args.replicates, workers=args.workers,
                         tickers=args.tickers, tier2=args.tier2, bootstrap_draws=args.bootstrap_draws,
                         calibration_dir=args.calibration_dir, fresh=args.fresh), indent=2, default=str))
