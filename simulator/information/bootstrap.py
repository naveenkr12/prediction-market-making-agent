"""Common-index paired hierarchical bootstrap and the Section 9.4 contrasts.

One set of resampling indices (templates, then replicates within each
template) is drawn per (roster, scope) and shared by every arm, policy and S
level of every contrast on that roster/scope. On each resample each marginal
statistic is recomputed per cell (arm, S, policy) and the contrast is the
weighted sum of the cell statistics (a difference of marginal statistics).
Primary statistics: mean PnL, VaR95 loss, ES95 loss, mean |q| before
settlement, mean drawdown, mean fills, mean snipe PnL (negative = loss).
Secondary: ES95 of the paired episode-level PnL contrast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..benchmark_metrics import expected_shortfall, pnl_statistics

DRAWS = 1000
POLICIES_COMPARED = ("hjb", "fixed", "skew")
METRICS = {"pnl": "realized_pnl", "abs_q": "abs_terminal_inventory", "drawdown": "max_drawdown",
           "fills": "fills_total", "snipe": "snipe_pnl"}
STATISTICS = ("mean_pnl", "var95_loss", "es95_loss", "mean_abs_q", "mean_drawdown", "mean_fills", "mean_snipe_pnl")
PAIRED = "es95_paired_difference_loss"


# ------------------------------------------------------------------ indices
def hierarchical_indices(templates: int, replicates: int, draws: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Template indices (draws, K) and within-template replicate indices (draws, K, R)."""
    if templates < 1 or replicates < 1 or draws < 1:
        raise ValueError("Need at least one template, replicate and draw")
    events = rng.integers(0, templates, size=(draws, templates))
    reps = rng.integers(0, replicates, size=(draws, templates, replicates))
    return events, reps


def resample(cube: np.ndarray, indices) -> np.ndarray:
    """(K, R) panel -> (draws, K*R) resampled episodes with the shared indices."""
    events, reps = indices
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 2 or not np.isfinite(cube).all():
        raise ValueError("Bootstrap needs a finite balanced (template, replicate) panel")
    return cube[events[:, :, None], reps].reshape(events.shape[0], -1)


def cell_panels(frame: pd.DataFrame, templates: list[str], replicates: list[int]) -> dict:
    """Balanced (template x replicate) arrays for every metric of one cell."""
    frame = frame.set_index(["ticker", "replicate"])
    if frame.index.duplicated().any():
        raise ValueError("Duplicate (ticker, replicate) rows in a cell")
    index = pd.MultiIndex.from_product([templates, replicates])
    out = {}
    for key, column in METRICS.items():
        values = frame[column].reindex(index)
        if values.isna().any():
            raise ValueError(f"Unbalanced panel for {column}")
        out[key] = values.to_numpy(dtype=float).reshape(len(templates), len(replicates))
    return out


def statistics(panels: dict, axis=-1) -> dict:
    """Marginal statistics of a panel set (works on (K,R) panels flattened or resampled (draws, n))."""
    pnl = panels["pnl"]
    return {"mean_pnl": pnl.mean(axis=axis), "var95_loss": np.quantile(-pnl, .95, axis=axis, method="linear"),
            "es95_loss": expected_shortfall(-pnl, axis=axis), "mean_abs_q": panels["abs_q"].mean(axis=axis),
            "mean_drawdown": panels["drawdown"].mean(axis=axis), "mean_fills": panels["fills"].mean(axis=axis),
            "mean_snipe_pnl": panels["snipe"].mean(axis=axis)}


def cell_statistics(panels: dict, indices) -> tuple[dict, dict]:
    """(point estimates on the sample, per-draw bootstrap statistics) for one cell."""
    flat = {k: v.reshape(-1) for k, v in panels.items()}
    boot = {k: resample(v, indices) for k, v in panels.items()}
    return {k: float(v) for k, v in statistics(flat).items()}, statistics(boot)


# ---------------------------------------------------------------- contrasts
def contrast(name: str, family: str, cells: list[tuple[tuple, float]], roster: str = "R_star", **labels) -> dict:
    return {"contrast": name, "family": family, "cells": cells, "roster": roster, **labels}


def _m_family(prefix: dict, family: str, roster: str, levels=("S0", "S1")) -> list[dict]:
    """flow, jump, interaction, sniping-jump contrasts for one M 2x2 (labels mapped through `prefix`)."""
    m0, m1, m2, m3 = (prefix[k] for k in ("M0", "M1", "M2", "M3"))
    out = []
    for policy in POLICIES_COMPARED:
        for s in levels:
            c = lambda arm, w, s=s: ((arm, s, policy), w)
            out += [contrast(f"flow:{m1}-{m0}", family, [c(m1, 1), c(m0, -1)], roster, policy=policy, snipe_level=s),
                    contrast(f"jump:{m2}-{m0}", family, [c(m2, 1), c(m0, -1)], roster, policy=policy, snipe_level=s),
                    contrast(f"jump:{m3}-{m1}", family, [c(m3, 1), c(m1, -1)], roster, policy=policy, snipe_level=s),
                    contrast(f"interaction:{m3}-{m2}-{m1}+{m0}", family,
                             [c(m3, 1), c(m2, -1), c(m1, -1), c(m0, 1)], roster, policy=policy, snipe_level=s)]
        if set(levels) >= {"S0", "S1"}:
            k = lambda arm, s, w: ((arm, s, policy), w)
            out += [contrast(f"sniping_jump:({m2},S1)-({m2},S0)-({m0},S1)+({m0},S0)", family,
                             [k(m2, "S1", 1), k(m2, "S0", -1), k(m0, "S1", -1), k(m0, "S0", 1)], roster,
                             policy=policy, snipe_level="S1-S0"),
                    contrast(f"sniping_jump:({m3},S1)-({m3},S0)-({m1},S1)+({m1},S0)", family,
                             [k(m3, "S1", 1), k(m3, "S0", -1), k(m1, "S1", -1), k(m1, "S0", 1)], roster,
                             policy=policy, snipe_level="S1-S0")]
            for arm in (m0, m1, m2, m3):
                out.append(contrast(f"sniping:({arm},S1)-({arm},S0)", family, [k(arm, "S1", 1), k(arm, "S0", -1)],
                                    roster, policy=policy, snipe_level="S1-S0"))
    return out


def tier1_contrasts() -> list[dict]:
    """The Section 9.4 contrast list (plus S1 - S0 main effects per M arm)."""
    out = []
    out += _m_family({"M0": "M0", "M1": "M1", "M2": "M2", "M3": "M3"}, "primary_matched", "R_star")
    # Secondary native arms at s = 1: M1 at s = 1 is M1-native and M3 at s = 1 is M3 (identical arms).
    out += _m_family({"M0": "M0-s1", "M1": "M1-native", "M2": "M2-s1", "M3": "M3"}, "secondary_native", "R_star")
    out += _m_family({"M0": "M0-emp", "M1": "M1-emp", "M2": "M2-emp", "M3": "M3-emp"}, "p0_emp", "R_star_emp")
    for policy in POLICIES_COMPARED:
        c = lambda arm, w, s="S0": ((arm, s, policy), w)
        chain = [("L0b", "L0"), ("L0c", "L0b"), ("L1J", "L0c"), ("L1J", "L1")]
        out += [contrast(f"legacy_chain:{a}-{b}", "legacy_chain", [c(a, 1), c(b, -1)], policy=policy, snipe_level="S0")
                for a, b in chain]
        out += [contrast("contextual:M1-native-L1", "contextual", [c("M1-native", 1), c("L1", -1)], policy=policy,
                         snipe_level="S0"),
                contrast("contextual:M3-L1J", "contextual", [c("M3", 1), c("L1J", -1)], policy=policy, snipe_level="S0"),
                contrast("fg_half:FGH-L-L1", "fg_half", [c("FGH-L", 1), c("L1", -1)], policy=policy, snipe_level="S0"),
                contrast("fg_half:FGH-LJ-L1J", "fg_half", [c("FGH-LJ", 1), c("L1J", -1)], policy=policy,
                         snipe_level="S0")]
    # Secondary M3 (s = 1) is the primary M3: drop exact duplicates (same name, roster, policy, S level).
    seen, unique = set(), []
    for c in out:
        key = (c["contrast"], c["roster"], c["policy"], c["snipe_level"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def policy_contrasts(cells: list[tuple[str, str, str]], roster_of: dict) -> list[dict]:
    """HJB minus fixed / skew within every (arm, S) cell present, on every roster the arm is run on.

    `roster_of` maps arm -> tuple of roster names (primary first). L0 therefore gets its
    policy contrasts on R_star (the contrast roster) and on all35 (the anchor roster).
    """
    out = []
    for arm, level in sorted({(a, s) for a, s, _ in cells}):
        for roster in tuple(roster_of.get(arm, ("R_star",))):
            for other in ("fixed", "skew"):
                if (arm, level, "hjb") in cells and (arm, level, other) in cells:
                    out.append(contrast(f"policy:hjb-{other}@{arm},{level}", "policy",
                                        [((arm, level, "hjb"), 1), ((arm, level, other), -1)], roster,
                                        policy=f"hjb-{other}", snipe_level=level))
    return out


def _quantiles(values):
    lo, hi = np.quantile(values, [.025, .975], method="linear")
    return float(lo), float(hi), float(np.std(values, ddof=1))


def evaluate_contrasts(frame: pd.DataFrame, contrasts: list[dict], rosters: dict, *, draws: int = DRAWS,
                       seed: int = 20261009, scopes=("all", "broad_based", "sports")) -> pd.DataFrame:
    """Bootstrap every contrast whose cells are all present; one shared index set per (roster, scope)."""
    rows = []
    cells_present = set(map(tuple, frame[["arm", "snipe_level", "policy"]].drop_duplicates().to_numpy()))
    replicates = sorted(frame.replicate.unique().tolist())
    groups = frame.drop_duplicates("ticker").set_index("ticker").group.to_dict()
    for r_index, (roster_name, templates_all) in enumerate(sorted(rosters.items())):
        for s_index, scope in enumerate(scopes):
            templates = [t for t in templates_all if scope == "all" or groups.get(t) == scope]
            if not templates:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, 94, r_index, s_index]))
            indices = hierarchical_indices(len(templates), len(replicates), draws, rng)
            cache = {}

            def cell(key):
                if key not in cache:
                    sub = frame[(frame.arm == key[0]) & (frame.snipe_level == key[1]) & (frame.policy == key[2]) &
                                frame.ticker.isin(templates)]
                    panels = cell_panels(sub, templates, replicates)
                    cache[key] = (panels, *cell_statistics(panels, indices))
                return cache[key]

            for spec in contrasts:
                if spec["roster"] != roster_name or not all(k in cells_present for k, _ in spec["cells"]):
                    continue
                point = {s: 0.0 for s in STATISTICS}
                boot = {s: 0.0 for s in STATISTICS}
                paired_sample = 0.0
                paired_boot = 0.0
                for key, weight in spec["cells"]:
                    panels, p, b = cell(key)
                    for s in STATISTICS:
                        point[s] = point[s]+weight*p[s]
                        boot[s] = boot[s]+weight*b[s]
                    paired_sample = paired_sample+weight*panels["pnl"].reshape(-1)
                    paired_boot = paired_boot+weight*resample(panels["pnl"], indices)
                base = {k: v for k, v in spec.items() if k != "cells"}
                base.update(scope=scope, templates=len(templates), replicates=len(replicates), draws=draws,
                            cells=";".join(f"{w:+g}*{a}|{s}|{p}" for (a, s, p), w in spec["cells"]))
                for s in STATISTICS:
                    lo, hi, se = _quantiles(boot[s])
                    rows.append({**base, "statistic": s, "kind": "difference_of_marginal_statistics",
                                 "estimate": point[s], "bootstrap95_low": lo, "bootstrap95_high": hi, "bootstrap_se": se})
                lo, hi, se = _quantiles(expected_shortfall(-paired_boot, axis=1))
                rows.append({**base, "statistic": PAIRED, "kind": "secondary_paired_difference",
                             "estimate": float(expected_shortfall(-paired_sample)), "bootstrap95_low": lo,
                             "bootstrap95_high": hi, "bootstrap_se": se})
    return pd.DataFrame(rows)


def risk_summary(frame: pd.DataFrame, rosters: dict, roster_of: dict, *, draws: int = DRAWS, seed: int = 20261009,
                 scopes=("all", "broad_based", "sports")) -> pd.DataFrame:
    """Per (roster, scope, arm, S, policy) marginal statistics with shared-index bootstrap ranges."""
    rows = []
    replicates = sorted(frame.replicate.unique().tolist())
    groups = frame.drop_duplicates("ticker").set_index("ticker").group.to_dict()
    for r_index, (roster_name, templates_all) in enumerate(sorted(rosters.items())):
        for s_index, scope in enumerate(scopes):
            templates = [t for t in templates_all if scope == "all" or groups.get(t) == scope]
            if not templates:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, 95, r_index, s_index]))
            indices = hierarchical_indices(len(templates), len(replicates), draws, rng)
            arms = [a for a in frame.arm.unique() if roster_name in roster_of.get(a, ())]
            for (arm, level, policy), sub in frame[frame.arm.isin(arms) & frame.ticker.isin(templates)].groupby(
                    ["arm", "snipe_level", "policy"], sort=True):
                if set(sub.ticker) != set(templates):
                    continue
                panels = cell_panels(sub, templates, replicates)
                point, boot = cell_statistics(panels, indices)
                row = {"roster": roster_name, "scope": scope, "arm": arm, "snipe_level": level, "policy": policy,
                       "templates": len(templates), "replicates": len(replicates),
                       **pnl_statistics(panels["pnl"].reshape(-1)), **point}
                for s in STATISTICS:
                    row[s+"_bootstrap95_low"], row[s+"_bootstrap95_high"], _ = _quantiles(boot[s])
                for column in ("maker_fills", "snipe_count", "snipe_volume", "fees", "spread_capture_fills",
                               "spread_capture_snipes", "background_price_pnl", "flow_price_pnl", "jump_price_pnl",
                               "settlement_surprise_pnl", "markout_5m", "flow_markout", "jump_markout",
                               "max_abs_inventory"):
                    if column in sub:
                        row["mean_"+column] = float(sub[column].mean())
                rows.append(row)
    return pd.DataFrame(rows)
