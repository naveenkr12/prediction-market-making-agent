"""Paper 3 skeleton / paths / environment (spec Sections 5-7; V2, V7, V8, V11, V12 bounded).

Calibration values and hashes are never pinned: the information calibration is
loaded through `config.py` at runtime, and jump-bearing episodes are produced
with a jump-rate scale rather than by relying on calibrated counts.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from simulator.benchmark_metrics import episode_diagnostics
from simulator.hjb_policy import HJBReferencePolicy
from simulator.information import kernels as K
from simulator.information import metrics as Mx
from simulator.information import reference as R
from simulator.information.config import InformationCalibration
from simulator.information.environment import (EXTRA_OBSERVATION_KEYS, InformationTradingEnvironment,
                                               golden_equivalence_sweep, l0_anchor_check, run_policy)
from simulator.information.paths import build_path, make_arm, with_p0
from simulator.information.skeleton import (DECISION, EXECUTION, JUMP, PROPOSAL_ASK, PROPOSAL_BID, SEED, STREAMS,
                                            WINDOW_END, build_skeleton, event_roster, path_seed)
from simulator.market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG
from simulator.price_impact import displayed_probability
from simulator.trading_env import EnvironmentConfig, TradingEnvironment

ROOT = Path(__file__).resolve().parents[2]
EPISODE_RESULTS = ROOT/"data_pipeline/output/paper2/calibration/simulator/hjb_benchmark/episode_results.csv"
POLICIES = ("hjb", "fixed", "skew", "none")
M_ARMS = ("M0", "M1", "M2", "M3", "M1-native")
L_ARMS = ("L0b", "L0c", "L1J", "L1", "FGH-L", "FGH-LJ")
JUMP_SCALE = 25.0      # forces jump-bearing episodes independently of calibrated counts


@pytest.fixture(scope="module")
def calibration():
    # The calibration may be regenerated concurrently; retry a transient hash mismatch.
    for attempt in range(10):
        try:
            return InformationCalibration.load()
        except (ValueError, json.JSONDecodeError, FileNotFoundError, KeyError):
            if attempt == 9:
                raise
            time.sleep(30)


@pytest.fixture(scope="module")
def hjb():
    return HJBReferencePolicy()


def arm(name, s=1.5, **kwargs):
    return make_arm(name, s=s, **kwargs) if name in {"M0", "M1", "M2"} else make_arm(name, **kwargs)


def audit(env, path):
    sk, cfg = path.skeleton, env.config
    return Mx.audit_information_ledger(env.ledger, initial_cash=cfg.initial_cash, initial_log_odds=path.initial_log_odds,
        inventory_limit=cfg.inventory_limit, maker_fee_multiplier=cfg.maker_fee_multiplier,
        balance_precision=cfg.balance_precision, quantity_lot=cfg.quantity_lot, participation=cfg.participation,
        tick=cfg.price_tick, beta=sk.dynamics.impact.beta, expected_executions=sk.n_executions,
        expected_jumps=len(sk.jump_elapsed) if path.jump_applied else 0)


def check_episode(env, path, records, policy):
    """V11 + identity + signs for one finished episode; returns the attribution."""
    report = audit(env, path)
    result = Mx.attribution(env.ledger, env.config.initial_cash, records[0]["observation"]["probability"])
    assert abs(result["pnl"]-float(env.account.cash-env.account.initial_cash)) < 1e-12
    assert abs(env.total_reward-result["pnl"]) < 1e-8
    if policy == "none":
        assert result["pnl"] == 0 and report["maker_fills"] == 0 and report["snipes"] == 0
    market = [r for r in env.ledger if r["kind"] == "market"]
    assert all(r["market_sign"] in (-1, 1) for r in market) and len(market) == path.skeleton.n_executions
    assert np.isin(path.exec_sign, (-1, 1)).all()
    if path.jump_applied:
        assert np.isin(path.jump_sign, (-1, 1)).all()
    Mx.diagnostics(env.ledger, initial_probability=path.arm.p0, lifetime=path.skeleton.lifetime,
                   release_minus_close_seconds=path.skeleton.ttr_at_close)
    return result


# --------------------------------------------------------------- seeding (S5)
def test_roster_and_path_seed_match_evaluate_hjb():
    roster = event_roster()
    config = json.loads(DYNAMICS_CONFIG.read_text())
    expected = [(g, t) for g in ("broad_based", "sports") for t in sorted(config["groups"][g]["event_tickers"])]
    assert [(g, t) for g, t, _ in roster] == expected and [i for *_, i in roster] == list(range(len(expected)))
    import pandas as pd
    saved = pd.read_csv(EPISODE_RESULTS, usecols=["ticker", "replicate", "seed"]).drop_duplicates()
    index = {t: i for _, t, i in roster}
    for row in saved.sample(60, random_state=0).itertuples():
        assert path_seed(index[row.ticker], row.replicate) == row.seed
    assert SEED == 20261009 and len(STREAMS) == 12


def test_skeleton_reproduces_legacy_streams_and_node_set(calibration):
    sk = build_skeleton("KXNFLGAME-26AUG28MINDEN-DEN", 3, calibration=calibration)
    legacy = TradingEnvironment(EnvironmentConfig(group=sk.group, ticker=sk.ticker))
    legacy.reset(seed=sk.path_seed, options={"seconds_to_close": sk.seconds})
    assert legacy.dynamics.noise_template_ticker == sk.noise_ticker and legacy.spreads.ticker == sk.spreads.ticker
    assert legacy.spread_rng.bit_generator.state != sk.spread_rng_after_init.bit_generator.state  # reset drew a book
    probe = sk.spread_rng()
    sk.spreads.sample(0.0, probe, Decimal("0.01"))
    assert probe.bit_generator.state == legacy.spread_rng.bit_generator.state
    # Children 0-4 equal TradingEnvironment.reset's spawn(5); the rest are distinct streams.
    kids = np.random.SeedSequence(sk.path_seed).spawn(12)
    assert [k.spawn_key for k in kids[:5]] == [k.spawn_key for k in np.random.SeedSequence(sk.path_seed).spawn(5)]
    while not legacy.done:
        legacy.step(None)
    market = [r for r in legacy.ledger if r["kind"] == "market"]
    assert [r["elapsed_seconds"] for r in market] == sk.exec_elapsed.tolist()
    assert [r["market_quantity"] for r in market] == sk.exec_quantity.tolist()
    assert [r["market_sign"] for r in market] == sk.exec_bank_sign.tolist()
    settlement = np.random.default_rng(kids[4])
    assert settlement.random() == sk.u_legacy_settlement
    assert legacy.settlement_rng.bit_generator.state == settlement.bit_generator.state
    # Node set: sorted, every window ends with its window-end node, decision nodes strictly inside windows.
    assert np.all(np.diff(sk.node_elapsed) >= 0) and np.all(sk.node_h >= 0)
    assert np.isclose(sk.node_h.sum(), sk.seconds, atol=1e-6)
    for w in sk.windows:
        kinds = sk.node_kind[w.node_slice]
        assert kinds[-1] == WINDOW_END and (kinds == WINDOW_END).sum() == 1
        assert (kinds == EXECUTION).sum() == w.exec_slice.stop-w.exec_slice.start
        inner = sk.node_elapsed[w.node_slice][kinds == DECISION]
        assert np.all((inner > w.start+1e-6) & (inner < w.end-1e-6))
    assert len(sk.decision_elapsed) == 4*len(sk.windows)       # 60-s grid, 300-s windows
    assert np.array_equal(np.sort(sk.decision_elapsed-sk.start), 60.0*np.array(
        [j for j in range(1, int(sk.seconds//60)) if j % 5]))
    for kind, times in ((PROPOSAL_BID, sk.proposal_bid_elapsed), (PROPOSAL_ASK, sk.proposal_ask_elapsed),
                        (JUMP, sk.jump_elapsed)):
        assert (sk.node_kind == kind).sum() == len(times)
    # W increments have exact variances h (standardized values are N(0,1)).
    z = sk.node_dw[sk.node_h > 0]/np.sqrt(sk.node_h[sk.node_h > 0])
    assert abs(z.mean()) < 5/np.sqrt(len(z)) and abs(z.var()-1) < 6*np.sqrt(2/len(z))
    # Same inputs on rebuild (base-input hash), different replicate -> different inputs.
    assert build_skeleton(sk.ticker, 3, calibration=calibration).base_input_hash() == sk.base_input_hash()
    assert build_skeleton(sk.ticker, 4, calibration=calibration).base_input_hash() != sk.base_input_hash()


def test_proposal_processes_have_rate_one_over_mean_latency(calibration):
    counts = []
    for replicate in range(6):
        sk = build_skeleton("KXCPI-26AUG-T0.5", replicate, calibration=calibration)
        counts += [len(sk.proposal_bid_elapsed), len(sk.proposal_ask_elapsed)]
    mean = sk.seconds/sk.mean_latency
    assert sk.mean_latency == 2.0
    assert abs(np.mean(counts)-mean) < 4*np.sqrt(mean/len(counts))


# --------------------------------------------------------------------- V7
def test_v7_golden_branch_reproduces_trading_environment_full_sweep(calibration, hjb):
    """All templates x 2 seeds x 4 policies: ledgers, observations, rewards, infos, steps (repr-identical)."""
    results = golden_equivalence_sweep(calibration, seeds=(0, 1), hjb=hjb)
    assert len(results) == len(event_roster())*2*4
    assert len({r["ticker"] for r in results}) == len(event_roster())


def test_l0_anchor_reproduces_saved_benchmark_rows(hjb):
    """L0 = TradingEnvironment with Section 5 seeding; bounded subset (replicate 0, all templates, 4 policies)."""
    result = l0_anchor_check(EPISODE_RESULTS, hjb=hjb, replicates=[0])
    assert result["episodes"] == len(event_roster())*4 and result["max_abs_difference"] <= 1e-9


def test_golden_attribution_matches_legacy_diagnostics(calibration, hjb):
    sk = build_skeleton("KXAAAGASM-26AUG31-4.08", 0, calibration=calibration)
    path = build_path(sk, make_arm("golden"))
    for policy in POLICIES:
        env = InformationTradingEnvironment(path)
        records, _ = run_policy(env, policy, hjb, seed=sk.path_seed)
        new = check_episode(env, path, records, policy)
        old = episode_diagnostics(env.ledger, 1000, records[0]["observation"]["probability"], env.start_elapsed, env.elapsed)
        for a, b in (("spread_capture_fills", "spread_capture"), ("fees", "fees"),
                     ("background_price_pnl", "background_price_pnl"), ("flow_price_pnl", "flow_price_pnl"),
                     ("settlement_surprise_pnl", "settlement_surprise_pnl")):
            assert abs(new[a]-old[b]) < 1e-12
        assert new["jump_price_pnl"] == 0 and new["spread_capture_snipes"] == 0


# --------------------------------------------------------- V8, V11, signs
@pytest.mark.parametrize("ticker,replicate,scale", [("KXAAAGASW-26AUG03-4.110", 0, JUMP_SCALE),
                                                    ("KXEPLGAME-26AUG22EVECRY-TIE", 1, JUMP_SCALE)])
def test_v8_hashes_audit_and_attribution_across_arms_policies_and_s(calibration, hjb, ticker, replicate, scale):
    sk = build_skeleton(ticker, replicate, calibration=calibration, jump_rate_scale=scale)
    assert len(sk.jump_elapsed) > 0
    base = sk.base_input_hash()
    golden_state = build_path(sk, make_arm("golden")).spread_state_final
    signal_hashes = {}
    for name in ("golden",)+L_ARMS+M_ARMS:
        path = build_path(sk, arm(name), calibration=calibration)
        # Spread draws only at legacy points: identical consumption of the spread stream in every arm.
        assert path.spread_draws == 1+sk.n_executions+len(sk.windows)
        assert path.spread_state_final == golden_state
        levels = ("S0", "S1") if path.arm.family == "M" else ("S0",)
        walks = set()
        for level in levels:
            for policy in POLICIES:
                env = InformationTradingEnvironment(path, snipe_level=level)
                records, _ = run_policy(env, policy, hjb, seed=sk.path_seed)
                check_episode(env, path, records, policy)
                walks.add(tuple(sorted(env.walk_hashes().items())))
                if level == "S0":
                    assert not any(r["kind"] == "snipe" for r in env.ledger)
        assert len(walks) == 1, name                                        # (b)-(d) across policies and S
        assert dict(walks.pop()) == path.walk_trace_hashes()
        signal_hashes[name] = path.signal_path_hash()
        assert sk.base_input_hash() == base                                 # (a) never mutated by any arm/policy/S
    assert build_skeleton(ticker, replicate, calibration=calibration, jump_rate_scale=scale).base_input_hash() == base
    assert len(set(signal_hashes.values())) == len(signal_hashes)            # every arm is a different public path


def test_none_policy_zero_pnl_and_settlement_rules(calibration, hjb):
    sk = build_skeleton("KXCPICORE-26JUN-T0.2", 2, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    for name in ("golden", "L0b", "L1J", "M0", "M3"):
        path = build_path(sk, arm(name), calibration=calibration)
        s = path.settlement
        if path.arm.family == "M":
            assert s["outcome"] == path.theta == int(sk.u_m < path.arm.p0)
        elif path.arm.family == "L":
            assert s["outcome"] == int(sk.u_l < expit(path.l_after[-1])) and s["uniform"] == sk.u_l
        else:
            assert s["outcome"] == int(sk.u_legacy_settlement < float(displayed_probability(path.l_after[-1])))
        env = InformationTradingEnvironment(path)
        records, _ = run_policy(env, "none", hjb, seed=sk.path_seed)
        check_episode(env, path, records, "none")
        assert records[-1]["info"]["settlement_outcome"] == s["outcome"]


# --------------------------------------------------------------------- V2
def test_v2_reference_posterior_equals_path_at_every_node(calibration):
    worst, count = 0., 0
    for ticker, replicate in (("KXAAAGASW-26AUG03-4.110", 0), ("KXNFLGAME-26AUG22ATLIND-IND", 1),
                              ("KXGDP-26JUL30-T1.0", 2)):
        sk = build_skeleton(ticker, replicate, calibration=calibration, jump_rate_scale=JUMP_SCALE)
        for name in M_ARMS:
            for p0 in (.5, calibration.p0_emp(ticker)):
                path = build_path(sk, with_p0(arm(name), p0), calibration=calibration)
                signals, values = path.signals()
                ref = R.posterior_log_odds_path(p0, signals)
                assert len(ref) == len(values) >= len(sk.node_kind)
                worst = max(worst, float(np.max(np.abs(np.asarray(ref)-np.asarray(values)))))
                count += 1
                if path.jump_applied:
                    assert sum(s["kind"] == "jump" for s in signals) == len(sk.jump_elapsed) > 0
    assert worst < 1e-12 and count == 30


def test_theta_free_background_matches_kernels_bitwise(calibration):
    sk = build_skeleton("KXMLSGAME-26AUG29NYRBPHI-TIE", 0, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    for name in ("L1J", "FGH-LJ"):
        path = build_path(sk, arm(name), calibration=calibration)
        L = path.initial_log_odds
        for i in range(path.node_count):
            new, x, _ = K.background_theta_free(L, sk.node_v_aux[i], path.sigma2, sk.node_h[i], sk.node_dw[i])
            assert float(x) == path.x[i] and float(new) == path.l_bg[i]
            L = path.l_after[i]
        if name == "L1J":
            # FL flow in Theta-free arms uses ImpactPriceProcess arithmetic and S_u signs.
            u = calibration.uninformed_rate(sk.ticker, sk.exec_quantity, sk.exec_phase)
            assert np.array_equal(path.exec_sign, K.uninformed_sign(sk.v_u, u))


# ------------------------------------------------------------- sniping S1
def independent_crossed(side, remaining, price, p, tick="0.01", coefficient="0.07"):
    q, m = Decimal(remaining), Decimal(price)
    fee = (Decimal(coefficient)*q*m*(1-m)/Decimal("0.01")).to_integral_value(rounding=ROUND_CEILING)*Decimal("0.01")
    edge = float(q)*(p-float(m)) if side == "ask" else float(q)*(float(m)-p)
    return edge > float(fee)+float(q)*float(Decimal(tick))


def test_sniping_decisions_fee_point_no_price_update_and_not_market_flow(calibration, hjb):
    sk = build_skeleton("KXAAAGASW-26JUL20-4.060", 1, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    path = build_path(sk, arm("M2", s=20.0), calibration=calibration)     # volatile: many crossings
    totals = {"crossed": 0, "uncrossed": 0}
    for policy in ("fixed", "hjb"):
        envs = {}
        for level in ("S0", "S1"):
            env = InformationTradingEnvironment(path, snipe_level=level, record_proposals=True)
            records, _ = run_policy(env, policy, hjb, seed=sk.path_seed)
            check_episode(env, path, records, policy)
            envs[level] = (env, records)
        s0, s1 = envs["S0"][0], envs["S1"][0]
        assert s0.proposal_log == [] and s0.snipe_count == 0                  # S0 proposal nodes inert
        snipe_rows = [(k, r) for k, r in enumerate(s1.ledger) if r["kind"] == "snipe"]
        crossed = [e for e in s1.proposal_log if e["crossed"]]
        assert len(snipe_rows) == len(crossed) == s1.snipe_count > 0
        for entry in s1.proposal_log:
            assert entry["latent_probability"] == path.p_bg[entry["node"]]
            assert entry["crossed"] == independent_crossed(entry["side"], entry["remaining"], entry["price"],
                                                           entry["latent_probability"])
            totals["crossed" if entry["crossed"] else "uncrossed"] += 1
        for (k, row), entry in zip(snipe_rows, crossed):
            prev = s1.ledger[k-1]
            assert prev["kind"] == "noise" and prev["log_odds"] == row["log_odds"] == path.l_bg[entry["node"]]
            assert row["elapsed_seconds"] == path.elapsed[entry["node"]] and row["quantity"] == entry["remaining"]
            assert Decimal(row["taker_fee"]) == (Decimal("0.07")*Decimal(row["quantity"])*Decimal(row["price"])*(
                1-Decimal(row["price"]))/Decimal("0.01")).to_integral_value(rounding=ROUND_CEILING)*Decimal("0.01")
            assert Decimal(row["fee"]) >= 0 and row["side"] == entry["side"]
        # Not market flow: identical market executions, signed flow and trade counts in S0 and S1.
        keys = ("elapsed_seconds", "market_sign", "market_quantity", "log_odds_before", "log_odds", "impact_logodds")
        market = lambda env: [tuple(r[k] for k in keys) for r in env.ledger if r["kind"] == "market"]
        assert market(s0) == market(s1)
        for a, b in zip(envs["S0"][1][1:], envs["S1"][1][1:]):
            assert a["info"]["market_trades"] == b["info"]["market_trades"]
            assert a["info"]["signed_flow_contracts"] == b["info"]["signed_flow_contracts"]
            assert a["observation"]["log_odds"] == b["observation"]["log_odds"]
            assert (a["observation"]["bid"], a["observation"]["ask"]) == (b["observation"]["bid"], b["observation"]["ask"])
        assert Mx.diagnostics(s1.ledger, initial_probability=.5, lifetime=sk.lifetime,
                              release_minus_close_seconds=sk.ttr_at_close)["snipe_count"] == s1.snipe_count
    assert totals["crossed"] > 0 and totals["uncrossed"] > 0


def test_snipe_rule_uses_the_configured_taker_fee(calibration, hjb):
    from simulator.information.taker_fees import TakerFeeSchedule
    sk = build_skeleton("KXAAAGASW-26JUL20-4.060", 1, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    path = build_path(sk, arm("M0", s=20.0), calibration=calibration)
    counts = {}
    for coefficient in (0.0, 0.07, 5.0):
        env = InformationTradingEnvironment(path, snipe_level="S1", taker_fees=TakerFeeSchedule(coefficient, "test"),
                                            record_proposals=True)
        run_policy(env, "fixed", hjb, seed=sk.path_seed)
        for e in env.proposal_log:
            assert e["crossed"] == independent_crossed(e["side"], e["remaining"], e["price"], e["latent_probability"],
                                                       coefficient=str(coefficient))
        counts[coefficient] = env.snipe_count
    assert counts[0.0] >= counts[0.07] >= counts[5.0]


# ---------------------------------------------------------- observations
def test_observations_have_legacy_keys_plus_arm_and_do_not_leak_future(calibration, hjb):
    sk = build_skeleton("KXAAAGASW-26AUG10-4.010", 0, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    legacy = TradingEnvironment(EnvironmentConfig(group=sk.group, ticker=sk.ticker))
    legacy_keys = set(legacy.reset(seed=sk.path_seed, options={"seconds_to_close": sk.seconds})[0])
    path = build_path(sk, arm("M3"), calibration=calibration)
    env = InformationTradingEnvironment(path, snipe_level="S1")
    records, _ = run_policy(env, "hjb", hjb, seed=sk.path_seed)
    for r in records:
        assert set(r["observation"]) == legacy_keys | set(EXTRA_OBSERVATION_KEYS)
        assert r["observation"]["arm"] == "M3" and r["observation"]["family"] == "M"
        assert r["observation"]["snipe_level"] == "S1"
    # Change Theta and everything after window 5: the first 6 steps' observations must not change.
    cut = path.window_node_slices[5].stop
    future = deepcopy(path)
    future.theta = 1-path.theta
    for i in range(cut, future.node_count):
        future.l_bg[i] += 1.0
        future.l_after[i] += 1.0
        future.p_bg[i] = float(expit(future.l_bg[i]))
    future.exec_sign = -future.exec_sign
    future.exec_sign[[future.refs[i] for i in range(cut) if future.kinds[i] == EXECUTION]] = path.exec_sign[
        [path.refs[i] for i in range(cut) if path.kinds[i] == EXECUTION]]
    other = InformationTradingEnvironment(future, snipe_level="S1")
    changed, _ = run_policy(other, "hjb", hjb, seed=sk.path_seed)
    for a, b in zip(records[:7], changed[:7]):
        assert repr(a["observation"]) == repr(b["observation"])
    assert any(repr(a["observation"]) != repr(b["observation"]) for a, b in zip(records[7:], changed[7:]))


# ------------------------------------------------- 60-s decision interval
def test_sixty_second_decisions_keep_market_windows_books_and_prices(calibration, hjb):
    sk = build_skeleton("KXNFLGAME-26AUG28NYGNYJ-NYJ", 0, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    path = build_path(sk, arm("M3"), calibration=calibration)
    runs = {}
    for interval in (300, 60):
        env = InformationTradingEnvironment(path, snipe_level="S1", decision_interval=interval)
        records, _ = run_policy(env, "skew", hjb, seed=sk.path_seed)
        check_episode(env, path, records, "skew")
        runs[interval] = (env, records)
    (e300, r300), (e60, r60) = runs[300], runs[60]
    assert len(r300)-1 == len(sk.windows) and len(r60)-1 == len(sk.windows)+len(sk.decision_elapsed)
    assert e300.walk_hashes() == e60.walk_hashes() == path.walk_trace_hashes()
    keys = ("elapsed_seconds", "market_sign", "market_quantity", "log_odds")
    assert [tuple(r[k] for k in keys) for r in e300.ledger if r["kind"] == "market"] == \
        [tuple(r[k] for k in keys) for r in e60.ledger if r["kind"] == "market"]
    # At a decision node the observed book is the most recently sampled one (no extra spread draw).
    sampled = sorted(path.books)
    decision_obs = 0
    for record in r60[1:]:
        obs = record["observation"]
        if record["info"]["decision_stop"]:
            decision_obs += 1
            node = path.elapsed.index(obs["elapsed_seconds"])
            assert path.kinds[node] == DECISION and obs["log_odds"] == path.l_bg[node]
            before = [i for i in sampled if i < node]
            book = path.books[before[-1]] if before else path.initial_book
            assert (obs["bid"], obs["ask"]) == (book.bid, book.ask)
    assert decision_obs == len(sk.decision_elapsed)


# ---------------------------------------------------------------- arms
def test_arm_definitions_and_guards(calibration):
    assert make_arm("M3").s == 1 and make_arm("M1-native").s == 1 and make_arm("L1").s == 1
    with pytest.raises(ValueError):
        make_arm("M0")                                   # matched s required
    assert make_arm("M1", s=2.0, constant_alpha=True).flow == "fg_constant"
    spec = {n: (make_arm(n, s=1.0) if n in {"M0", "M1", "M2"} else make_arm(n)) for n in
            ("golden",)+L_ARMS+M_ARMS}
    assert {n: (a.family, a.background, a.flow, a.jumps, a.settlement) for n, a in spec.items()} == {
        "golden": ("legacy", "legacy_bridge", "bank_fl", "none", "legacy"),
        "L0b": ("L", "legacy_bridge", "su_fl", "none", "u_l"),
        "L0c": ("L", "legacy_gaussian", "su_fl", "legacy", "u_l"),
        "L1J": ("L", "theta_free", "su_fl", "theta_free", "u_l"),
        "L1": ("L", "theta_free", "su_fl", "none", "u_l"),
        "FGH-L": ("L", "theta_free", "fg_half_theta_free", "none", "u_l"),
        "FGH-LJ": ("L", "theta_free", "fg_half_theta_free", "theta_free", "u_l"),
        "M0": ("M", "m", "f0", "none", "theta"), "M1": ("M", "m", "fg", "none", "theta"),
        "M2": ("M", "m", "f0", "m", "theta"), "M3": ("M", "m", "fg", "m", "theta"),
        "M1-native": ("M", "m", "fg", "none", "theta")}
    sk = build_skeleton("KXU3-26AUG-T4.1", 0, calibration=calibration)
    golden = build_path(sk, make_arm("golden"))
    with pytest.raises(ValueError):
        InformationTradingEnvironment(golden, snipe_level="S1")
    with pytest.raises(ValueError):
        InformationTradingEnvironment(golden, decision_interval=60)
    with pytest.raises(ValueError):
        InformationTradingEnvironment(golden, replace(sk.config, initial_probability=.4))
    emp = build_path(sk, with_p0(make_arm("M3"), calibration.p0_emp(sk.ticker)), calibration=calibration)
    obs, _ = InformationTradingEnvironment(emp).reset(seed=sk.path_seed)
    assert obs["log_odds"] == emp.initial_log_odds
    # L0b differs from golden only in signs (S_u) and settlement; same pieces and spread points.
    l0b = build_path(sk, make_arm("L0b"), calibration=calibration)
    assert l0b.x == golden.x and l0b.elapsed == golden.elapsed and l0b.kinds == golden.kinds
    assert np.array_equal(l0b.exec_sign, K.uninformed_sign(sk.v_u, calibration.uninformed_rate(
        sk.ticker, sk.exec_quantity, sk.exec_phase)))
    # Zero-sigma_c donors are not floored.
    assert build_path(sk, make_arm("M0", s=3.0), calibration=calibration).sigma2 == 3.0*calibration.sigma_c(sk.noise_ticker)**2
