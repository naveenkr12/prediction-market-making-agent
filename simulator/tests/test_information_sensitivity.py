"""Bounded tests for the Paper 3 sensitivity experiments (docs/paper3/SENSITIVITY_SPEC.md).

A (quote refresh): 60-s and 300-s decisions walk the same path (executions, prices, jumps, books,
settlement); re-submitted quotes lose their queue position under the legacy `_replace` semantics.
B (jump intensity): coupled doses — nesting of the jump sets, background identical at native nodes,
thinned/superposed rate laws, Brownian-bridge insertion law, V2 posterior equality at 0.5x and 2x,
and Tier 1 rows reproduced by the sensitivity runner. Contrast lists are checked for presence.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from simulator.hjb_policy import HJBReferencePolicy
from simulator.information import bootstrap as B
from simulator.information import evaluate as E
from simulator.information import kernels as K
from simulator.information import reference as R
from simulator.information import sensitivity as S
from simulator.information.config import InformationCalibration
from simulator.information.environment import InformationTradingEnvironment, run_policy
from simulator.information.matching import TemplateDesign, load_roster
from simulator.information.paths import build_path, make_arm
from simulator.information.skeleton import (JUMP, STREAMS, brownian_bridge_insert, build_skeleton, dose_rng,
                                            superpose_jumps, superposition_draws, thinning_draws)

ROOT = Path(__file__).resolve().parents[2]
TIER1 = ROOT/"data_pipeline/output/paper3/experiment"
SPORTS, BROAD = "KXNFLGAME-26AUG28NYGNYJ-NYJ", "KXCPICORE-26JUN-T0.2"
JUMP_ARMS = {"F0": lambda: make_arm("M2", s=1.0), "FG": lambda: make_arm("M3")}


@pytest.fixture(scope="module")
def calibration():
    for attempt in range(10):
        try:
            return InformationCalibration.load()
        except (ValueError, json.JSONDecodeError, FileNotFoundError, KeyError):
            if attempt == 9:
                raise
            time.sleep(30)


@pytest.fixture(scope="module")
def hjb():
    return HJBReferencePolicy(time_mapping="episode")


@pytest.fixture(scope="module")
def dosed(calibration):
    """Native, thinned and superposed versions of sports episodes that carry native jumps."""
    out = []
    for replicate in range(40):
        sk = build_skeleton(SPORTS, replicate, calibration=calibration)
        if len(sk.jump_elapsed) >= 2:
            uniforms, keep = thinning_draws(sk)
            out.append((sk, uniforms, keep, superpose_jumps(sk, calibration)))
        if len(out) == 4:
            break
    assert len(out) == 4
    return out


def _jump_tuples(sk, path, mask=None):
    mask = path.jump_active if mask is None else mask
    return {(float(sk.jump_elapsed[j]), float(sk.jump_magnitude[j]), float(sk.v_j[j]), int(path.jump_sign[j]))
            for j in np.flatnonzero(mask)}


# --------------------------------------------------------------- streams
def test_dose_streams_are_positional_and_leave_tier1_children_unchanged(calibration):
    pseed = build_skeleton(BROAD, 3, calibration=calibration).path_seed
    twelve = np.random.SeedSequence(pseed).spawn(12)
    fourteen = np.random.SeedSequence(pseed).spawn(14)
    for a, b in zip(twelve, fourteen[:12]):
        assert a.spawn_key == b.spawn_key and np.array_equal(a.generate_state(8), b.generate_state(8))
    assert len(STREAMS) == 12
    assert np.array_equal(dose_rng(pseed, "jump_thinning").random(5), np.random.default_rng(fourteen[12]).random(5))
    assert np.array_equal(dose_rng(pseed, "jump_superposition").random(5),
                          np.random.default_rng(fourteen[13]).random(5))
    sk = build_skeleton(SPORTS, 1, calibration=calibration)
    base = sk.base_input_hash()
    thinning_draws(sk)
    sk2 = superpose_jumps(sk, calibration)
    assert sk.base_input_hash() == base == build_skeleton(SPORTS, 1, calibration=calibration).base_input_hash()
    assert sk2.base_input_hash() != base and sk2.jump_rate_scale == 2.0 and sk.jump_extra is None
    with pytest.raises(ValueError):
        superpose_jumps(sk2, calibration)
    with pytest.raises(ValueError):
        build_path(sk2, make_arm("L1J"), calibration=calibration)      # no L-aux uniforms at inserted nodes


# --------------------------------------------------------------- nesting
def test_jump_sets_are_nested_across_doses(calibration, dosed):
    for sk, _, keep, sk2 in dosed:
        for family, arm_fn in JUMP_ARMS.items():
            arm = arm_fn()
            p1 = build_path(sk, arm, calibration=calibration)
            ph = build_path(sk, arm, calibration=calibration, jump_keep=keep, jump_keep_probability=.5)
            p2 = build_path(sk2, arm, calibration=calibration)
            half, one, two = _jump_tuples(sk, ph), _jump_tuples(sk, p1), _jump_tuples(sk2, p2)
            assert half <= one <= two and len(one) == len(sk.jump_elapsed)
            assert len(two) == len(sk2.jump_elapsed) == len(one)+int(sk2.jump_extra.sum())
            assert {t[:3] for t in two if t not in one} == {
                (float(sk2.jump_elapsed[j]), float(sk2.jump_magnitude[j]), float(sk2.v_j[j]))
                for j in np.flatnonzero(sk2.jump_extra)}
            assert ph.active_jump_count == int(keep.sum()) and ph.jumps_thinned == (not keep.all())
            # Thinned jump nodes stay in the node set (inert): identical node arrays to 1x.
            assert ph.kinds == p1.kinds and ph.elapsed == p1.elapsed
            assert sum(1 for k in ph.kinds if k == JUMP) == len(sk.jump_elapsed)
            # 2x keeps every native node, in order, with the same kind and time.
            native = np.flatnonzero(sk2.node_original)
            assert np.array_equal(sk2.node_kind[native], sk.node_kind)
            assert np.array_equal(sk2.node_elapsed[native], sk.node_elapsed)
            assert np.array_equal(sk2.node_kind[~sk2.node_original], np.full((~sk2.node_original).sum(), JUMP))


def test_background_at_native_nodes_identical_across_doses(calibration, dosed):
    for sk, _, keep, sk2 in dosed:
        native = np.flatnonzero(sk2.node_original)
        assert np.max(np.abs(np.cumsum(sk2.node_dw)[native]-np.cumsum(sk.node_dw))) < 1e-12
        assert np.max(np.abs(np.cumsum(sk2.node_h)[native]-np.cumsum(sk.node_h))) < 1e-9
        untouched = np.r_[True, sk2.node_original[:-1]][native]      # native node not preceded by an insertion
        assert np.array_equal(sk2.node_dw[native][untouched], sk.node_dw[untouched])
        for zero, jump in (("M0", "M2"), ("M1-native", "M3")):
            arms = [make_arm(zero, s=1.0) if zero == "M0" else make_arm(zero), make_arm(jump, s=1.0)
                    if jump == "M2" else make_arm(jump)]
            p0 = build_path(sk, arms[0], calibration=calibration)
            p1 = build_path(sk, arms[1], calibration=calibration)
            ph = build_path(sk, arms[1], calibration=calibration, jump_keep=keep, jump_keep_probability=.5)
            p2 = build_path(sk2, arms[1], calibration=calibration)
            assert p0.x == p1.x == ph.x                                   # 0x, 0.5x, 1x: bitwise identical
            assert np.array_equal(p0.exec_sign, p1.exec_sign) and np.array_equal(p1.exec_sign, p2.exec_sign)
            assert p0.theta == p1.theta == p2.theta and p1.settlement["uniform"] == p2.settlement["uniform"]
            assert np.max(np.abs(np.cumsum(p2.x)[native]-np.cumsum(p1.x))) < 1e-12
            # Jump-free L differs from 1x only by the applied jumps.
            jumps1 = np.cumsum([p1.jump_step[r] if k == JUMP else 0.0 for k, r in zip(p1.kinds, p1.refs)])
            assert np.max(np.abs(np.asarray(p1.l_after)-jumps1-np.asarray(p0.l_after))) < 1e-10


# --------------------------------------------------------------- bridge law
def test_brownian_bridge_insertion_preserves_w_and_has_the_conditional_variance():
    rng = np.random.default_rng(7)
    H, u = 10.0, np.array([2.0, 2.0, 7.0, 10.0])
    n = 20_000
    cond = np.empty((n, len(u)))
    free = np.empty((n, len(u)+1))
    for i in range(n):
        pieces, hs = brownian_bridge_insert(H, 1.3, u, rng.standard_normal(len(u)))
        assert abs(pieces.sum()-1.3) < 1e-14 and abs(hs.sum()-H) < 1e-14
        cond[i] = np.cumsum(pieces)[:-1]
        d = np.sqrt(H)*rng.standard_normal()
        free[i] = brownian_bridge_insert(H, d, u, rng.standard_normal(len(u)))[0]
    # Conditional law given W(H) = 1.3: mean u/H * d, var u(H-u)/H, cov(W(2), W(7)) = 2*3/10.
    mean, var = u/H*1.3, u*(H-u)/H
    se = np.sqrt(np.maximum(var, 1e-12)/n)
    assert np.all(np.abs(cond.mean(axis=0)-mean) < 4.5*se+1e-12)
    assert np.all(np.abs(cond.var(axis=0)[:3]-var[:3]) < 4.5*var[:3]*np.sqrt(2/n))
    assert np.allclose(cond[:, 3], 1.3) and np.all(cond[:, 0] == cond[:, 1])      # endpoint and repeated point
    assert abs(np.cov(cond[:, 0], cond[:, 2])[0, 1]-0.6) < 4.5*np.sqrt((1.6*2.1+0.36)/n)
    # Unconditionally (W(H) ~ N(0, H)) the refined increments are independent N(0, h).
    h = np.diff(np.r_[0.0, u, H])
    assert np.all(np.abs(free.var(axis=0)[h > 0]-h[h > 0]) < 4.5*h[h > 0]*np.sqrt(2/n))
    assert np.all(free[:, h == 0] == 0)
    c = np.corrcoef(free[:, h > 0].T)
    assert np.max(np.abs(c-np.eye(len(c)))) < 4.5/np.sqrt(n)
    with pytest.raises(ValueError):
        brownian_bridge_insert(1.0, 0.0, [0.5, 0.2], [0.0, 0.0])


# --------------------------------------------------------------- rate laws
@pytest.mark.parametrize("ticker", [SPORTS, "KXGDP-26JUL30-T1.0"])
def test_thinned_and_superposed_jump_processes_have_the_scaled_poisson_law(calibration, ticker):
    cls = calibration.information_class(ticker)
    rates, bank = calibration.jump_rates(cls), calibration.magnitude_bank(cls)
    design = TemplateDesign.build(ticker, calibration)
    lam = float(sum(design.jump_means))
    seconds, ttr = design.seconds, calibration.ttr_at_close(ticker)+design.seconds
    rng_native, rng_thin, rng_sup = (np.random.default_rng(s) for s in (11, 12, 13))
    n = 20_000
    thinned, total, mags, extra_t = np.empty(n), np.empty(n), [], []
    for i in range(n):
        native = K.poisson_jump_offsets(ttr, seconds, rates, rng_native)
        thinned[i] = np.count_nonzero(rng_thin.random(len(native)) < S.THINNING_KEEP_PROBABILITY)
        offsets, magnitude, v_j = superposition_draws(ttr, seconds, rates, bank, rng_sup)
        total[i] = len(native)+len(offsets)
        mags.extend(magnitude)
        extra_t.extend(offsets)
    for counts, mean in ((thinned, .5*lam), (total, 2*lam)):
        assert abs(counts.mean()-mean) < 4.5*np.sqrt(mean/n)                       # Poisson mean
        assert abs(counts.var()/mean-1) < 4.5*np.sqrt(2/n)+4.5/np.sqrt(n*mean)     # dispersion 1
    assert abs(np.mean(mags)-bank.mean()) < 4.5*bank.std()/np.sqrt(len(mags))
    # Time-varying intensity: inserted times follow the native piecewise rate (bucket shares).
    expected = np.asarray(design.jump_means)/lam
    recorder_cuts = sorted({0.0, seconds, *[ttr-e for e in K.TTR_EDGES_SECONDS[1:-1] if 0 < ttr-e < seconds]})
    observed = np.histogram(extra_t, bins=recorder_cuts)[0]
    if len(expected) == len(observed) and len(expected) > 1:
        from scipy import stats
        assert stats.chisquare(observed, expected*observed.sum()).pvalue > 1e-4
    else:
        from scipy import stats
        assert stats.kstest(np.asarray(extra_t)/seconds, "uniform").pvalue > 1e-4


def test_skeleton_doses_thin_half_and_insert_lambda_on_real_episodes(calibration):
    lam = float(sum(TemplateDesign.build(SPORTS, calibration).jump_means))
    native = kept = inserted = 0
    reps = 80
    for replicate in range(reps):
        sk = build_skeleton(SPORTS, replicate, calibration=calibration)
        _, keep = thinning_draws(sk)
        native += len(sk.jump_elapsed)
        kept += int(keep.sum())
        inserted += int(superpose_jumps(sk, calibration).jump_extra.sum())
    assert abs(kept-native/2) < 4.5*np.sqrt(native/4)+1
    assert abs(inserted/reps-lam) < 4.5*np.sqrt(lam/reps)
    assert abs(native/reps-lam) < 4.5*np.sqrt(lam/reps)


# --------------------------------------------------------------- V2
def test_v2_posterior_equals_path_at_half_and_double_dose(calibration, dosed):
    worst, count = 0.0, 0
    for sk, _, keep, sk2 in dosed:
        for family, arm_fn in JUMP_ARMS.items():
            for path in (build_path(sk, arm_fn(), calibration=calibration, jump_keep=keep, jump_keep_probability=.5),
                         build_path(sk2, arm_fn(), calibration=calibration)):
                signals, values = path.signals()
                ref = R.posterior_log_odds_path(path.arm.p0, signals)
                assert len(ref) == len(values)
                assert sum(s["kind"] == "jump" for s in signals) == path.active_jump_count
                assert sum(s["kind"] == "background" for s in signals) == len(path.skeleton.node_kind)
                worst = max(worst, float(np.max(np.abs(np.asarray(ref)-np.asarray(values)))))
                count += 1
    assert count == 16 and worst < 1e-12


def test_dose_paths_pass_the_per_episode_checks(calibration, hjb, dosed):
    sk, _, keep, sk2 = dosed[0]
    for path in (build_path(sk, make_arm("M3"), calibration=calibration, jump_keep=keep, jump_keep_probability=.5),
                 build_path(sk2, make_arm("M3"), calibration=calibration)):
        expected = path.walk_trace_hashes()
        for level in ("S0", "S1"):
            for policy in ("hjb", "none"):
                env = InformationTradingEnvironment(path, snipe_level=level)
                records, _ = run_policy(env, policy, hjb, seed=path.skeleton.path_seed)
                from simulator.information import metrics as Mx
                cfg = env.config
                audit = Mx.audit_information_ledger(
                    env.ledger, initial_cash=cfg.initial_cash, initial_log_odds=path.initial_log_odds,
                    inventory_limit=cfg.inventory_limit, maker_fee_multiplier=cfg.maker_fee_multiplier,
                    balance_precision=cfg.balance_precision, quantity_lot=cfg.quantity_lot,
                    participation=cfg.participation, tick=cfg.price_tick, beta=path.skeleton.dynamics.impact.beta,
                    expected_executions=path.skeleton.n_executions, expected_jumps=path.active_jump_count)
                Mx.attribution(env.ledger, cfg.initial_cash, records[0]["observation"]["probability"])
                assert env.walk_hashes() == expected and env.jump_count == path.active_jump_count
                assert sum(r["kind"] == "jump" for r in env.ledger) == audit["jumps"] == path.active_jump_count
                if policy == "none":
                    assert float(env.account.cash-env.account.initial_cash) == 0


# --------------------------------------------------------------- A: 60 s vs 300 s
def test_sixty_and_three_hundred_second_decisions_share_executions_prices_and_jumps(calibration, hjb):
    roster = load_roster(TIER1/"roster.json")
    for ticker, replicate in ((SPORTS, 2), (BROAD, 0)):
        sk = build_skeleton(ticker, replicate, calibration=calibration)
        for label, base, s_source in S.QUOTE_REFRESH_ARMS:
            arm, _, _ = E.resolve_arm(E.ArmPlan(label, base, ("R_star",), ("S0", "S1"), s_source), ticker, roster,
                                      calibration)
            path = build_path(sk, arm, calibration=calibration)
            for policy in ("hjb", "fixed"):
                runs = {}
                for interval in (300, 60):
                    env = InformationTradingEnvironment(path, snipe_level="S1", decision_interval=interval)
                    records, actions = run_policy(env, policy, hjb, seed=sk.path_seed)
                    runs[interval] = (env, records, actions)
                (e3, r3, a3), (e6, r6, a6) = runs[300], runs[60]
                assert e3.walk_hashes() == e6.walk_hashes() == path.walk_trace_hashes()
                for kind, keys in (("market", ("elapsed_seconds", "market_sign", "market_quantity", "log_odds",
                                               "impact_logodds")),
                                   ("jump", ("elapsed_seconds", "jump_sign", "jump_magnitude", "log_odds"))):
                    assert [tuple(r[k] for k in keys) for r in e3.ledger if r["kind"] == kind] == \
                        [tuple(r[k] for k in keys) for r in e6.ledger if r["kind"] == kind]
                assert e3.price.log_odds == e6.price.log_odds and e3.ledger[-1]["settlement_outcome"] == \
                    e6.ledger[-1]["settlement_outcome"]
                assert len(r6)-len(r3) == len(sk.decision_elapsed)
                # Same frozen rule: the first decision (identical state) gives the identical action.
                assert repr(r3[1]["action"]) == repr(r6[1]["action"])
                if policy == "hjb":
                    assert a3[0] == a6[0] and len(a6) == len(r6)-1
                    times = [a["hjb_time"] for a in a6]
                    assert np.all(np.diff(times) > 0)


def test_resubmitted_quote_gets_a_fresh_queue_position(calibration, hjb):
    """Legacy `_replace`: cancel then re-post; the queue position is recomputed from the book (reset)."""
    from decimal import Decimal
    lost = 0
    for replicate in range(3):
        sk = build_skeleton(SPORTS, replicate, calibration=calibration)
        path = build_path(sk, make_arm("M3"), calibration=calibration)
        for policy in ("hjb", "fixed"):
            env = InformationTradingEnvironment(path, decision_interval=60)
            run_policy(env, policy, hjb, seed=sk.path_seed)
            proxy = Decimal(str(env.config.queue_ahead_contracts))
            quotes = [r for r in env.ledger if r["kind"] == "quote"]
            # Proxy books carry no depth levels: a new order's queue is 0 (improves the top) or the proxy depth.
            assert quotes and all(Decimal(r["queue_ahead"]) in (0, proxy) for r in quotes)
            assert len({r["order_id"] for r in quotes}) == len(quotes)         # every re-post is a new order
            stats = S.quote_statistics(S._annotate_cancel_prices(env.ledger))
            assert stats["quote_rows"] == len(quotes)
            assert stats["decisions"] == len(sk.windows)+len(sk.decision_elapsed)
            queue, price, cancelled = {}, {}, {}
            for row in env.ledger:
                if row["kind"] == "fill":
                    queue[str(row["order_id"])] = Decimal(row["queue_after"])
                elif row["kind"] == "cancel":
                    cancelled[row["side"]] = str(row["order_id"])
                elif row["kind"] == "quote":
                    old = cancelled.pop(row["side"], None)
                    if old is not None and price[old] == row["price"] and queue.get(old, proxy+1) < Decimal(
                            row["queue_ahead"]):
                        lost += 1                     # same price, but the earned queue progress is discarded
                    price[str(row["order_id"])], queue[str(row["order_id"])] = row["price"], Decimal(row["queue_ahead"])
                else:
                    cancelled = {}
    assert lost > 0


# --------------------------------------------------------------- runner + Tier 1 equality
def test_runner_reproduces_tier1_rows_for_the_shared_cells(calibration, tmp_path):
    roster = load_roster(TIER1/"roster.json")
    E._init_worker(None, str(E.SAVED_BENCHMARK))
    tier1 = pd.read_csv(TIER1/"episode_results.csv")
    tier1 = tier1[tier1.ticker.eq(SPORTS) & tier1.replicate.eq(0)]
    frames = {}
    for experiment in S.EXPERIMENTS:
        cells = [c for c in S.experiment_cells(experiment)
                 if c.label in {"M2-s1", "M2-s1-x0.5", "M2-s1-x2", "M0-s1", "M3-dec60", "M3"}]
        for cell in cells:
            S.run_task((experiment, SPORTS, cell, str(tmp_path/experiment), 1, E.SEED, roster, "code"))
        data = pd.concat([pd.read_csv(tmp_path/experiment/"checkpoints"/f"{SPORTS}__{c.label}.csv") for c in cells])
        # Same CSV chain as the runner and Tier 1 (checkpoint -> concat -> episode_results.csv -> read).
        data.to_csv(tmp_path/f"{experiment}.csv", index=False)
        data = pd.read_csv(tmp_path/f"{experiment}.csv")
        frames[experiment] = data
        report = S.tier1_equality(data, tier1, cells)
        assert report["all_exact"] and report["rows_compared"] == 8*sum(bool(c.tier1_label) for c in cells)
        assert data[data.policy.eq("none")].realized_pnl.eq(0).all()
    a = frames["quote_refresh"]
    assert set(a.arm) == {"M0-s1", "M2-s1", "M3", "M3-dec60"}
    d = S.decision_interval_equality(a[a.cell_group.eq("M3")])
    assert d["identical"]
    b = frames["jump_intensity"]
    assert set(b.arm) == {"M0-s1", "M2-s1", "M2-s1-x0.5", "M2-s1-x2", "M3"}
    assert b.groupby("arm").base_input_sha256.first().loc[["M0-s1", "M2-s1", "M2-s1-x0.5"]].nunique() == 1
    with pytest.raises(AssertionError):
        broken = b.copy()
        broken.loc[broken.arm.eq("M2-s1"), "realized_pnl"] += 1e-12
        S.tier1_equality(broken, tier1, [c for c in S.jump_intensity_cells() if c.label == "M2-s1"])


# --------------------------------------------------------------- contrasts
def test_sensitivity_cells_and_contrasts_are_present():
    tier1_labels = {p.label for p in E.tier1_plan()}
    for experiment in S.EXPERIMENTS:
        cells = S.experiment_cells(experiment)
        assert len({c.label for c in cells}) == len(cells)
        assert {c.tier1_label for c in cells if c.tier1_label} <= tier1_labels
    a = S.quote_refresh_contrasts()
    names = {c["contrast"] for c in a}
    for arm, _, _ in S.QUOTE_REFRESH_ARMS:
        assert {f"refresh:{arm}@60-{arm}@300", f"refresh_x_sniping:{arm}", f"sniping@60:({arm},S1)-({arm},S0)"} <= names
    for x, y in (("M2", "M0"), ("M3", "M1"), ("M2-s1", "M0-s1"), ("M3", "M1-native")):
        assert {f"jump@60:{x}-{y}", f"refresh_x_jump:{x}-{y}", f"jump@300:{x}-{y}"} <= names
    rx = next(c for c in a if c["contrast"] == "refresh_x_sniping:M2" and c["policy"] == "hjb")
    assert sorted(rx["cells"]) == sorted([(("M2-dec60", "S1", "hjb"), 1), (("M2-dec60", "S0", "hjb"), -1),
                                          (("M2", "S1", "hjb"), -1), (("M2", "S0", "hjb"), 1)])
    b = S.jump_intensity_contrasts()
    families = {c["family"] for c in b}
    assert families == {"dose_vs_0x", "dose_vs_1x", "dose_slope", "sniping_x_dose", "dose_slope_sniping"}
    for fam in S.JUMP_FAMILIES:
        assert {c["contrast"] for c in b if c.get("dose_family") == fam} >= {
            f"dose_vs_0x:{S.dose_label(fam, d)}-{S.dose_label(fam, 0.)}" for d in (.5, 1., 2.)}
    # The slope contrast is the OLS slope on dose levels 0, 0.5, 1, 2.
    w = np.asarray(S.slope_weights())
    d = np.asarray(S.DOSES)
    assert abs(w.sum()) < 1e-15 and abs(np.dot(w, 3.0+1.7*d)-1.7) < 1e-12
    assert abs(np.dot(w, d**2)-np.polyfit(d, d**2, 1)[0]) < 1e-12
    # Bootstrap on a synthetic frame: every B contrast evaluates, with the snipe-count statistic.
    labels = [S.dose_label(f, x) for f in S.JUMP_FAMILIES for x in S.DOSES]
    rng = np.random.default_rng(0)
    rows = [{"arm": a, "snipe_level": s, "policy": p, "ticker": t, "group": g, "replicate": r,
             "realized_pnl": rng.normal(), "abs_terminal_inventory": 1.0, "max_drawdown": 1.0, "fills_total": 3,
             "snipe_pnl": -0.1, "snipe_count": 2}
            for a in labels for s in ("S0", "S1") for p in ("hjb", "fixed", "skew") for t, g in (("A", "sports"),
            ("B", "broad_based")) for r in range(3)]
    frame = pd.DataFrame(rows)
    out = S.evaluate_all_contrasts(frame, b, {"R_star": ["A", "B"]}, draws=20, seed=1)
    assert set(out.contrast) == {c["contrast"] for c in b}
    count = out[out.statistic.eq("mean_snipe_count") & out.contrast.str.startswith("dose_vs_0x") & out.scope.eq("all")]
    assert np.allclose(count.estimate, 0.0) and set(B.STATISTICS) <= set(out.statistic)


# --------------------------------------------------------------- review additions
class _ContrastRates:
    """Real calibration with exaggerated, distinct per-bucket jump rates (power for the time-profile test)."""
    RATES_PER_HOUR = {"B0": 24.0, "B1": 2.0, "B2": 9.0, "B3": 0.5}

    def __init__(self, calibration):
        self._c = calibration

    def __getattr__(self, name):
        return getattr(self._c, name)

    def jump_rates(self, information_class, scale=1.0):
        return {b: scale*r/3600.0 for b, r in self.RATES_PER_HOUR.items()}


def _bucket_exposure(ttr_start, seconds):
    """Seconds of the episode [0, seconds) spent in each TTR bucket (TTR = ttr_start - t), computed directly."""
    edges = K.TTR_EDGES_SECONDS
    out = {}
    for k, b in enumerate(K.TTR_BUCKETS):
        lo, hi = edges[k], edges[k+1]
        out[b] = max(0.0, min(seconds, ttr_start-lo)-max(0.0, ttr_start-hi))
    return out


def _check_refined_geometry(sk, sk2):
    """2x skeleton: node_h equals the elapsed differences; inserted nodes strictly inside their windows; order kept."""
    expected_h = np.maximum(np.diff(np.r_[sk2.start, sk2.node_elapsed]), 0.0)
    assert np.max(np.abs(sk2.node_h-expected_h)) < 1e-9
    assert np.all(np.diff(sk2.node_window) >= 0)
    same = np.diff(sk2.node_window) == 0
    assert np.all(np.diff(sk2.node_elapsed)[same] >= 0)                          # time order within windows
    inserted = np.flatnonzero(~sk2.node_original)
    assert len(inserted) == int(sk2.jump_extra.sum())
    for i in inserted:
        w = sk2.windows[sk2.node_window[i]]
        assert w.start < sk2.node_elapsed[i] < w.end and 0 < sk2.node_offset[i] < w.duration
        assert w.node_slice.start <= i < w.node_slice.stop and sk2.node_kind[i] == JUMP
        assert sk2.jump_extra[sk2.node_ref[i]] and sk2.jump_elapsed[sk2.node_ref[i]] == sk2.node_elapsed[i]
    native = np.flatnonzero(sk2.node_original)
    assert np.array_equal(sk2.node_elapsed[native], sk.node_elapsed)
    assert np.array_equal(sk2.node_window[native], sk.node_window)


def test_superposed_skeleton_node_geometry(calibration, dosed):
    for sk, _, _, sk2 in dosed:
        _check_refined_geometry(sk, sk2)
    stub = _ContrastRates(calibration)
    ticker = "KXGDP-26JUL30-T1.0"
    for replicate in range(6):
        sk = build_skeleton(ticker, replicate, calibration=stub)
        sk2 = superpose_jumps(sk, stub)
        assert sk2.jump_extra.sum() > 0
        _check_refined_geometry(sk, sk2)


def _inserted_bucket_counts(stub, ticker, replicates, ttr_shift=0.0):
    from dataclasses import replace
    counts = dict.fromkeys(K.TTR_BUCKETS, 0)
    for replicate in range(replicates):
        sk = build_skeleton(ticker, replicate, calibration=stub)
        if ttr_shift:
            sk = replace(sk, ttr_at_close=sk.ttr_at_close+ttr_shift)          # mutation: wrong release time
        sk2 = superpose_jumps(sk, stub)
        elapsed = sk2.jump_elapsed[sk2.jump_extra]
        ttr = sk.ttr_at_close-ttr_shift+sk.lifetime-elapsed                   # true TTR of each inserted jump
        for k in K.ttr_bucket_index(ttr):
            counts[K.TTR_BUCKETS[int(k)]] += 1
    return counts, sk


def test_inserted_jumps_follow_the_ttr_bucket_intensity_through_superpose_jumps(calibration):
    """Per-bucket counts of the real superpose_jumps path against lambda_J(bucket) x exposure; a wrong
    ttr_start (release time shifted by 1,800 s) must be rejected by the same test."""
    from scipy import stats
    stub = _ContrastRates(calibration)
    ticker, reps = "KXGDP-26JUL30-T1.0", 150
    counts, sk = _inserted_bucket_counts(stub, ticker, reps)
    exposure = _bucket_exposure(sk.ttr_at_close+sk.seconds, sk.seconds)
    rates = stub.jump_rates(sk.information_class)
    live = [b for b in K.TTR_BUCKETS if exposure[b] > 0]
    assert len(live) == 2                                                     # this episode spans B1 then B0
    expected = np.array([reps*rates[b]*exposure[b] for b in live])
    observed = np.array([counts[b] for b in live])
    assert sum(counts.values()) == observed.sum()
    assert np.all(np.abs(observed-expected) < 4.5*np.sqrt(expected))          # Poisson per bucket
    assert stats.chisquare(observed, expected*observed.sum()/expected.sum()).pvalue > 1e-4
    shifted, _ = _inserted_bucket_counts(stub, ticker, reps, ttr_shift=1800.0)
    wrong = np.array([shifted[b] for b in live])
    assert stats.chisquare(wrong, expected*wrong.sum()/expected.sum()).pvalue < 1e-6 or \
        np.any(np.abs(wrong-expected) > 4.5*np.sqrt(expected))


def test_thinning_uniforms_align_one_to_one_with_native_jumps(calibration, dosed):
    for sk, uniforms, keep, _ in dosed:
        direct = dose_rng(sk.path_seed, "jump_thinning").random(len(sk.jump_elapsed))
        assert np.array_equal(direct, uniforms) and np.array_equal(keep, direct < 0.5)
        assert np.all(np.diff(sk.jump_elapsed) >= 0)                          # native jumps are in time order
        path = build_path(sk, make_arm("M3"), calibration=calibration, jump_keep=keep, jump_keep_probability=.5)
        assert np.array_equal(path.jump_active, keep)
        applied = [path.refs[i] for i, k in enumerate(path.kinds) if k == JUMP and path.jump_active[path.refs[i]]]
        assert applied == np.flatnonzero(keep).tolist()                     # node order = jump order = uniform order
        signals, _ = path.signals()
        jumps = [s for s in signals if s["kind"] == "jump"]
        assert [s["magnitude"] for s in jumps] == sk.jump_magnitude[keep].tolist()
