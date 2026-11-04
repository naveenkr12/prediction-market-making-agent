"""Paper 3 information-arrival calibration: synthetic unit checks (bounded runtime)."""
import numpy as np
import pandas as pd
import pytest

import calibrate_information_arrival as C
from calibrate_market_dynamics import residual_noise_fit


def pairs_frame(y, flow, start="2026-07-01T00:00:00Z", gaps=()):
    """Consecutive 300-s pairs; an index in `gaps` starts a new run (non-consecutive)."""
    left, t = [], pd.Timestamp(start)
    for i in range(len(y)):
        if i in gaps:
            t += pd.Timedelta(seconds=900)
        left.append(t)
        t += pd.Timedelta(seconds=300)
    left = pd.DatetimeIndex(left)
    return pd.DataFrame({"left_time": left, "right_time": left+pd.Timedelta(seconds=300),
                         "signed_flow_contracts": np.asarray(flow, float), "delta_logodds_midpoint": np.asarray(y, float),
                         "duration_seconds": 300})


# ---------------------------------------------------------------- 3.1
def test_effective_innovations_reproduce_legacy_sigma_with_and_without_flow_slope():
    rng = np.random.default_rng(1)
    flow = rng.normal(0, 50, 80)
    y = 2e-4*flow+rng.standard_t(3, 80)*.02
    d = pairs_frame(y, flow)
    stored = residual_noise_fit(d, 5e-5)["sigma_logodds_per_sqrt_second"]
    res = C.effective_innovations(d.sample(frac=1, random_state=3), 5e-5, stored)   # order-insensitive input
    assert res["d"] == 2 and res["n"] == 80
    assert res["eta_tilde"]@res["eta_tilde"]/(80*300) == pytest.approx(stored**2, rel=1e-12)
    np.testing.assert_allclose(res["eta_tilde"], res["eta"]*np.sqrt(80/78))
    const = pairs_frame(y, np.full(80, 3.0))
    res1 = C.effective_innovations(const, 5e-5, residual_noise_fit(const, 5e-5)["sigma_logodds_per_sqrt_second"])
    assert res1["d"] == 1
    np.testing.assert_allclose(res1["eta"], y-y.mean())
    with pytest.raises(AssertionError):
        C.effective_innovations(d, 5e-5, stored*(1+1e-8))
    bad = d.copy(); bad.loc[3, "right_time"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        C.effective_innovations(bad, 5e-5, stored)


def test_pooled_duplicates_are_dropped_and_uniqueness_asserted():
    d = pairs_frame(np.zeros(4), np.zeros(4))
    rows = pd.concat([d.assign(ticker="A", information_class="sports"), d.assign(ticker="A", information_class="pooled"),
                      d.assign(ticker="B", information_class="broad_based"), d.assign(ticker="B", information_class="sports")])
    templates = {"A": {"information_class": "sports"}, "B": {"information_class": "broad_based"}}
    kept = C.own_class_intervals(rows, templates)
    assert len(kept) == 8 and set(zip(kept.ticker, kept.information_class)) == {("A", "sports"), ("B", "broad_based")}
    templates["B"]["information_class"] = "sports"
    rows2 = pd.concat([rows, d.assign(ticker="B", information_class="sports")])
    with pytest.raises(AssertionError):
        C.own_class_intervals(rows2, templates)


# ---------------------------------------------------------------- 3.2/3.3
def test_local_bipower_window_excludes_products_touching_k_and_flags_spike():
    eta = np.ones(40); eta[20] = 100.0
    consecutive = np.r_[False, np.ones(39, bool)]
    sigma2, local = C.local_bipower_variance(eta, consecutive)
    # k=20: product indices i in [8,32] minus {19..22} -> 21 unit products.
    assert sigma2[20] == pytest.approx(np.pi/2, rel=1e-15) and local[20]
    # k=23: i in [11,35] minus {22..25} keeps products 20 and 21 (= 100 each) among 21.
    assert sigma2[23] == pytest.approx(np.pi/2*(19+200)/21, rel=1e-15)
    # k=33: i in [21,39] minus {32..35} keeps product 21 (=100) but not 20 (|20-33| > 12).
    assert sigma2[33] == pytest.approx(np.pi/2*(14+100)/15, rel=1e-15)
    det = C.detect_jumps(eta, consecutive)
    assert np.flatnonzero(det["jump"]).tolist() == [20]
    # All-zero contract has no jumps; zero variance is not floored.
    zero = C.detect_jumps(np.zeros(30), np.r_[False, np.ones(29, bool)])
    assert not zero["jump"].any() and np.all(zero["sigma_hat2"] == 0)


def test_bipower_falls_back_to_contract_level_and_products_need_consecutive_pairs():
    eta = np.arange(1, 21, dtype=float)
    consecutive = np.r_[False, np.array([i % 3 != 0 for i in range(1, 20)])]
    products = C.adjacent_products(eta, consecutive)
    assert np.isnan(products[0]) and np.isnan(products[3]) and products[4] == 5*4
    sigma2, local = C.local_bipower_variance(eta, consecutive)
    assert not local.any()      # < 12 usable products in every window
    np.testing.assert_allclose(sigma2, np.pi/2*np.nanmean(products))
    with pytest.raises(ValueError):
        C.local_bipower_variance(np.ones(5), np.zeros(5, bool))


def test_decomposition_identity_holds_and_is_asserted():
    rng = np.random.default_rng(2)
    eta = rng.normal(0, .02, 100); eta[[10, 50]] = [.5, -.4]
    jump = np.zeros(100, bool); jump[[10, 50]] = True
    s2 = eta@eta/(100*300)
    dec = C.decompose(eta, jump, s2)
    assert dec["sigma_c2"]+dec["jump_variance_rate"] == pytest.approx(s2, rel=1e-14)
    assert dec["sigma_c2"] == pytest.approx((eta@eta-.5**2-.4**2)/(100*300), rel=1e-12)
    with pytest.raises(AssertionError):
        C.decompose(eta, jump, s2*1.001)


# ---------------------------------------------------------------- 3.4
def test_neighbour_pooling_order_and_left_endpoint_ttr_convention():
    exposure = {"B0": 100.0, "B1": 200.0, "B2": 300.0, "B3": 400.0}
    lam = C.pooled_intensity({"B0": 0, "B1": 2, "B2": 13, "B3": 33}, exposure)
    assert lam["B0"]["contributing_buckets"] == ["B0", "B1", "B2"] and lam["B0"]["lambda_per_second"] == 15/600
    assert lam["B1"]["contributing_buckets"] == ["B1", "B0", "B2"] and lam["B1"]["lambda_per_second"] == 15/600
    assert lam["B2"]["contributing_buckets"] == ["B2"] and lam["B3"]["lambda_per_second"] == 33/400
    lam = C.pooled_intensity({"B0": 7, "B1": 0, "B2": 1, "B3": 3}, exposure)
    assert lam["B2"]["contributing_buckets"] == ["B2", "B1", "B3", "B0"]
    assert lam["B3"]["contributing_buckets"] == ["B3", "B2", "B1", "B0"] and lam["B3"]["reached_min_jumps"]
    assert lam["B1"]["contributing_buckets"] == ["B1", "B0"]
    short = C.pooled_intensity({"B0": 1, "B1": 0, "B2": 1, "B3": 1}, exposure)
    assert not short["B0"]["reached_min_jumps"] and short["B0"]["lambda_per_second"] == 3/1000
    # Exposure and jumps use the pair's LEFT endpoint TTR.
    release = pd.Timestamp("2026-07-01T02:00:00Z")
    d = pairs_frame(np.zeros(3), np.zeros(3), start="2026-07-01T00:55:00Z")   # lefts 00:55, 01:00, 01:05
    ttr, bidx = C.pair_ttr_buckets(d, release)
    assert ttr.tolist() == [3900, 3600, 3300] and bidx.tolist() == [1, 1, 0]
    assert C.pair_ttr_buckets(d, pd.Timestamp("2026-07-01T01:00:00Z"))[1].tolist() == [0, 0, -1]


def test_intensity_bootstrap_is_seeded_and_brackets_reasonably():
    rows = pd.DataFrame({"ticker": np.repeat(["A", "B", "C"], 20), "bucket": np.tile(["B2", "B3"], 30),
                         "is_jump": np.r_[np.ones(6, bool), np.zeros(54, bool)], "eta_tilde": np.linspace(.1, .6, 60)})
    a = C.intensity_bootstrap(rows, 50, np.random.default_rng(42))
    b = C.intensity_bootstrap(rows, 50, np.random.default_rng(42))
    assert a == b and a["draws"] == 50
    lo, hi = a["lambda_bootstrap95"]["B2"]
    assert 0 <= lo <= hi


# ---------------------------------------------------------------- 3.5 and sign convention
def test_uninformed_rates_shrinkage_bins_and_sign_convention():
    q = np.r_[np.full(50, 1.0), np.full(50, 10.0), np.full(50, 100.0), np.full(50, 1000.0), np.full(50, 1e4)]
    s = np.where(np.arange(250) % 5 == 0, -1, 1).astype(np.int8)     # 80% YES takers (+1)
    phase = np.zeros(250, int); phase[:10] = 9
    rates = C.uninformed_rates({"T": (q, s, phase)}, n_phases=10)
    assert np.bincount(C.size_bins(q, rates["edges"]), minlength=5).tolist() == [50]*5
    np.testing.assert_allclose(rates["u_class"], .8)
    t = rates["templates"]["T"]
    # Cell (bin 0, phase 9): 10 trades, 8 YES -> (8 + 20*.8)/(10 + 20); empty cells equal the class rate.
    assert t["u_bin_phase"][0, 9] == pytest.approx((8+20*.8)/30)
    assert t["u_bin_phase"][3, 5] == pytest.approx(.8)
    extreme = C.uninformed_rates({"T": (q, np.ones(250, np.int8), phase)}, n_phases=10)
    assert extreme["templates"]["T"]["u_bin_phase"].max() == .98
    with pytest.raises(AssertionError):
        C.uninformed_rates({"T": (q, np.zeros(250, np.int8), phase)}, n_phases=10)


# ---------------------------------------------------------------- 3.6 replay
def synthetic_replay(seed=11, contracts=6, buckets=40):
    rng = np.random.default_rng(seed)
    sizes, us, tb, bc, p_start = [], [], [], [], []
    b = 0
    for c in range(contracts):
        for _ in range(buckets):
            n = int(rng.integers(1, 6))
            sizes.append(rng.lognormal(2, 1.5, n)); us.append(rng.uniform(.4, .8, n)); tb.append(np.full(n, b))
            bc.append(c); p_start.append(rng.uniform(.05, .95)); b += 1
    return C.FlowReplay(np.concatenate(sizes), np.concatenate(us), np.concatenate(tb), np.array(bc), np.array(p_start),
                        seed=7, class_index=0, chunk=16)


def test_replay_objective_at_zero_monotone_coupling_and_determinism():
    replay = synthetic_replay()
    zero = replay.statistic("fg", 0.0, 64)
    assert zero["statistic"] == 0 and np.all(zero["per_contract"] == 0) and zero["mc_se"] == 0
    grid = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    stats = [replay.statistic("fg", x, 64)["statistic"] for x in grid]
    assert stats[0] > 0 and np.all(np.diff(stats) > 0)
    again = replay.statistic("fg", 1e-3, 64)
    assert again["statistic"] == stats[2]                       # frozen uniforms (common random numbers)
    assert replay.statistic("fg", 1e-3, 64, tag=1)["statistic"] != stats[2]   # independent validation uniforms
    # Moments: sign convention +1 YES taker; uninformed YES fraction tracks u at eta = 0.
    slopes, mom = replay.slopes("fg", 0.0, 64)
    assert abs(mom["buys"]/mom["trades"]-replay.u.mean()) < .02 and mom["informed"] == 0
    # FG-half: Delta L = beta * (F - first trade), so the slope is positive and finite.
    half = replay.statistic("fg_half", 1e-4, 64)
    assert half["statistic"] > 0


def test_bracket_bisection_and_replication_doubling_hit_target(monkeypatch):
    monkeypatch.setitem(C.REPLAY, "mc_se_tolerance", 0.01)    # bounded test runtime; production uses 0.25%
    replay = synthetic_replay()
    target = replay.statistic("fg", 2e-3, 400)["statistic"]
    trace = []
    eta, res = C.solve_parameter(replay, "fg", target, 200, trace=trace, label="fg")
    assert abs(res["statistic"]/target-1) < C.REPLAY["bisection_tolerance"]
    assert trace[0]["parameter"] == 0.0 and trace[1]["parameter"] == C.REPLAY["bracket_start"]
    out = C.calibrate_replay_law(replay, "fg", target, [], "fg")
    assert out["mc_se"] < C.REPLAY["mc_se_tolerance"]*target and out["replications"] >= 200
    assert set(out["validation"]) >= {"relative_error", "ci95_relative_error", "passed", "status"}
    with pytest.raises(RuntimeError):
        C.solve_parameter(replay, "constant", 1e9, 32, upper_limit=1-1e-12)


# ---------------------------------------------------------------- training isolation, times, p0
def test_heldout_ticker_access_is_refused():
    splits = pd.DataFrame({"ticker": ["A", "B", "H"], "event_cluster": ["a", "b", "h"], "split": ["train", "train", "test"]})
    guard = C.TrainingGuard(splits)
    guard.require("A", "trades")
    for ticker in ("H", "UNKNOWN"):
        with pytest.raises(C.HeldOutAccessError):
            guard.require(ticker, "trades")
    assert guard.heldout_files_loaded == 0 and guard.loaded == [("trades", "A")]
    crossing = pd.DataFrame({"ticker": ["A", "B"], "event_cluster": ["x", "x"], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        C.TrainingGuard(crossing)


def test_release_time_rules_flags_and_empirical_start_probability():
    template = {"information_class": "broad_based", "open_time": "2026-07-29T00:00:00+00:00", "lifetime_seconds": 64740.0}
    close = "2026-07-29T17:59:00Z"
    md = {"close_time": close, "occurrence_datetime": "2026-07-29T17:33:56Z", "expected_expiration_time": "2026-07-29T18:05:00Z",
          "settlement_ts": "2026-07-29T17:30:00Z", "rules_primary": "If  the\nFed ...", "series_ticker": "KXFED"}
    rec = C.release_record("F", template, md)
    assert rec["early_revelation"] and rec["release_minus_close_seconds"] == -1504
    assert "occurrence_after_settlement_ts" in rec["flags"] and rec["rules_excerpt"] == "If the Fed ..."
    no_occ = C.release_record("F", template, {**md, "occurrence_datetime": None})
    assert no_occ["t_r_source"] == "template_close" and "close_fallback" in no_occ["flags"] and not no_occ["early_revelation"]
    sports = C.release_record("S", {**template, "information_class": "sports"}, md)
    assert sports["t_r_utc"] == sports["t_c_utc"] and "provisional_sports_close" in sports["flags"]
    tape = pd.DataFrame({"trade_id": ["a", "b", "c"], "price": [.004, .3, .995],
                         "timestamp": pd.to_datetime(["2026-07-29T16:00:00Z", "2026-07-29T16:58:59Z", "2026-07-29T17:00:00Z"])})
    p0 = C.empirical_start_probability(tape, pd.Timestamp(close))
    assert p0["p0_emp"] == .3 and p0["p0_source"] == "last_trade_before_episode_start" and not p0["p0_clipped"]
    late = C.empirical_start_probability(tape.iloc[2:], pd.Timestamp(close))
    assert late["p0_source"] == "first_trade_at_or_after_episode_start" and late["p0_sim"] == .99 and late["p0_clipped"]


# ---------------------------------------------------------------- saved artifacts (3.8) and loader
def test_saved_information_calibration_loads_verifies_hashes_and_refuses_tampering(tmp_path):
    import json
    import shutil
    from simulator.information.config import DEFAULT_DIR, InformationCalibration
    if not (DEFAULT_DIR/"information_config.json").exists():
        pytest.skip("Paper 3 calibration outputs not generated")
    cal = InformationCalibration.load()
    report = json.loads((DEFAULT_DIR/"calibration_report.json").read_text())
    assert report["heldout_files_loaded"] == 0 and report["passed"]
    roster = cal.candidate_roster()
    assert len(roster) == len(set(roster)) and not any(cal.early_revelation(t) for t in roster)
    for cls in ("broad_based", "sports"):
        rates = cal.jump_rates(cls)
        assert set(rates) == {"B0", "B1", "B2", "B3"} and all(r >= 0 for r in rates.values())
        assert len(cal.magnitude_bank(cls)) == cal.config["classes"][cls]["jumps"]["magnitude_count"] > 0
        assert cal.eta(cls) > 0 and cal.config["classes"][cls]["flow"]["validation"]["status"]
        for ticker in cal.config["classes"][cls]["jumps"]["supported_tickers"]:
            assert 0 <= cal.sigma_c(ticker) <= cal.template(ticker)["sigma_legacy"]
    ticker = roster[0]
    u = cal.uninformed_rate(ticker, [1.0, 1e6], [9, 0])
    assert np.all((u >= .02) & (u <= .98))
    assert .01 <= cal.p0_emp(ticker) <= .99
    copy = tmp_path/"cal"
    shutil.copytree(DEFAULT_DIR, copy)
    InformationCalibration.load(copy)
    with (copy/"jump_detection.csv").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValueError):
        InformationCalibration.load(copy)
