"""Paper 3 statistics, diagnostics, runner and validator on bounded samples (spec Sections 9-10).

Calibration values are never pinned; jump-bearing episodes use a jump-rate
scale. Statistical checks use bounded samples with generous thresholds.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from simulator.benchmark_metrics import expected_shortfall, hierarchical_samples
from simulator.hjb_policy import HJBReferencePolicy
from simulator.information import bootstrap as B
from simulator.information import evaluate as E
from simulator.information import metrics as Mx
from simulator.information import validate as V
from simulator.information.config import InformationCalibration
from simulator.information.environment import InformationTradingEnvironment, run_policy
from simulator.information.matching import roster_hash
from simulator.information.paths import build_path, make_arm
from simulator.information.skeleton import build_skeleton, event_roster

JUMP_SCALE = 25.0
BROAD, SPORTS = "KXCPICORE-26JUN-T0.2", "KXNFLGAME-26AUG28NYGNYJ-NYJ"


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


# ------------------------------------------------------------- bootstrap
def synthetic_frame(arms, templates=("A", "B", "C"), replicates=4, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    rows = []
    base = {(t, r): rng.normal(size=5) for t in templates for r in range(replicates)}
    for arm in arms:
        for t in templates:
            for r in range(replicates):
                z = base[(t, r)]
                for policy in ("hjb", "fixed", "skew"):
                    rows.append({"arm": arm, "snipe_level": "S0", "policy": policy, "ticker": t,
                                 "group": "broad_based" if t != "C" else "sports", "replicate": r,
                                 "realized_pnl": z[0]+(shift or {}).get(arm, 0.0)+(policy == "hjb")*0.1,
                                 "abs_terminal_inventory": abs(z[1]), "max_drawdown": abs(z[2]),
                                 "fills_total": abs(z[3])*10, "snipe_pnl": -abs(z[4])})
    return pd.DataFrame(rows)


def test_hierarchical_indices_match_the_paper2_resampler():
    cube = np.arange(12.0).reshape(3, 4)
    a = hierarchical_samples(cube, np.random.default_rng(5), 50)
    b = B.resample(cube, B.hierarchical_indices(3, 4, 50, np.random.default_rng(5)))
    assert np.array_equal(a, b)


def test_common_index_contrasts_are_exact_for_identical_and_shifted_cells():
    frame = synthetic_frame(["X", "Y", "Z"], shift={"Y": 0.0, "Z": 0.25})
    rosters = {"R_star": ["A", "B", "C"]}
    c = lambda a, w, p="hjb": ((a, "S0", p), w)
    contrasts = [B.contrast("same", "t", [c("Y", 1), c("X", -1)]), B.contrast("shift", "t", [c("Z", 1), c("X", -1)]),
                 B.contrast("interaction", "t", [c("Z", 1), c("Y", -1), c("X", -1), c("X", 1)])]
    out = B.evaluate_contrasts(frame, contrasts, rosters, draws=200, scopes=("all",)).set_index(["contrast", "statistic"])
    for s in B.STATISTICS+(B.PAIRED,):
        r = out.loc[("same", s)]
        assert r.estimate == 0 and r.bootstrap95_low == 0 and r.bootstrap95_high == 0
    r = out.loc[("shift", "mean_pnl")]
    assert abs(r.estimate-0.25) < 1e-12 and abs(r.bootstrap95_low-0.25) < 1e-12 and abs(r.bootstrap95_high-.25) < 1e-12
    for s in ("var95_loss", "es95_loss"):
        r = out.loc[("shift", s)]
        assert abs(r.estimate+0.25) < 1e-12 and abs(r.bootstrap95_high+0.25) < 1e-12   # loss falls by the shift
    assert abs(out.loc[("shift", B.PAIRED)].estimate+0.25) < 1e-12
    assert abs(out.loc[("interaction", "mean_pnl")].estimate-0.25) < 1e-12


def test_contrast_point_estimates_and_cis_are_marginal_differences():
    frame = synthetic_frame(["X", "Y"], seed=3)
    frame.loc[frame.arm.eq("Y"), "realized_pnl"] *= 2.0
    rosters = {"R_star": ["A", "B", "C"]}
    spec = B.contrast("d", "t", [(("Y", "S0", "fixed"), 1), (("X", "S0", "fixed"), -1)])
    out = B.evaluate_contrasts(frame, [spec], rosters, draws=300, scopes=("all",)).set_index("statistic")
    x = frame[(frame.arm == "X") & (frame.policy == "fixed")].realized_pnl.to_numpy()
    y = frame[(frame.arm == "Y") & (frame.policy == "fixed")].realized_pnl.to_numpy()
    assert abs(out.loc["mean_pnl"].estimate-(y.mean()-x.mean())) < 1e-12
    assert abs(out.loc["es95_loss"].estimate-(expected_shortfall(-y)-expected_shortfall(-x))) < 1e-12
    assert abs(out.loc["var95_loss"].estimate-(np.quantile(-y, .95)-np.quantile(-x, .95))) < 1e-12
    assert out.loc["mean_pnl"].bootstrap95_low <= out.loc["mean_pnl"].estimate <= out.loc["mean_pnl"].bootstrap95_high
    assert set(out.reset_index().kind) == {"difference_of_marginal_statistics", "secondary_paired_difference"}


def test_tier1_contrast_list_covers_section_9_4():
    names = {c["contrast"] for c in B.tier1_contrasts()}
    for required in ("flow:M1-M0", "jump:M2-M0", "jump:M3-M1", "interaction:M3-M2-M1+M0",
                     "sniping_jump:(M2,S1)-(M2,S0)-(M0,S1)+(M0,S0)", "sniping_jump:(M3,S1)-(M3,S0)-(M1,S1)+(M1,S0)",
                     "legacy_chain:L0b-L0", "legacy_chain:L0c-L0b", "legacy_chain:L1J-L0c", "legacy_chain:L1J-L1",
                     "contextual:M1-native-L1", "contextual:M3-L1J", "fg_half:FGH-L-L1", "fg_half:FGH-LJ-L1J",
                     "flow:M1-native-M0-s1", "jump:M2-s1-M0-s1", "flow:M1-emp-M0-emp", "jump:M3-emp-M1-emp"):
        assert required in names, required
    assert all(c["roster"] == "R_star_emp" for c in B.tier1_contrasts() if "emp" in c["contrast"])
    cells = [("M0", "S0", p) for p in ("hjb", "fixed", "skew", "none")]
    policy = B.policy_contrasts(cells, {"M0": ("R_star",)})
    assert {c["contrast"] for c in policy} == {"policy:hjb-fixed@M0,S0", "policy:hjb-skew@M0,S0"}
    l0 = B.policy_contrasts([("L0", "S0", p) for p in ("hjb", "fixed", "skew", "none")], {"L0": ("all35", "R_star")})
    assert {(c["contrast"], c["roster"]) for c in l0} == {(f"policy:hjb-{o}@L0,S0", r) for o in ("fixed", "skew")
                                                         for r in ("all35", "R_star")}


# ------------------------------------------------------------ validators
def test_wald_qr_drops_collinear_columns_and_has_the_right_size_and_power():
    rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(300):
        x = rng.normal(size=(200, 3))
        X = np.c_[x, x[:, 0]+x[:, 1], np.zeros(200), 2*x[:, 2]]
        res = V.wald_qr(X, ["a", "b", "c", "a+b", "zero", "2c"])
        assert res["rank"] == 3 and set(res["dropped"]) >= {"zero"} and len(res["dropped"]) == 3
        rejections += res["p"] < 0.01
    assert rejections <= 10                                   # ~1% expected of 300
    shifted = V.wald_qr(rng.normal(size=(2000, 3))+[0, 0.2, 0], ["a", "b", "c"])
    assert shifted["p"] < 1e-6
    holm = V.holm({"a": 0.001, "b": 0.02, "c": 0.004})
    assert holm["a"]["p_holm"] == pytest.approx(0.003) and holm["c"]["p_holm"] == pytest.approx(0.008)
    assert holm["b"]["p_holm"] == pytest.approx(0.02) and [holm[k]["reject"] for k in "abc"] == [True, False, True]


def test_v1_kernel_closed_forms():
    result = V.v1_kernels()
    assert result["passed"], result["checks"]


@pytest.fixture(scope="module")
def martingale_sample(calibration):
    """Bounded V4-V6 sample: 2 templates x 40 replicates, M arms with s = 2 and jumps forced on."""
    rows = {name: [] for name in ("M0", "M1", "M2", "M3", "L0c")}
    for ticker in (BROAD, SPORTS):
        for replicate in range(40):
            sk = build_skeleton(ticker, 900+replicate, calibration=calibration, jump_rate_scale=JUMP_SCALE)
            for name in rows:
                arm = make_arm(name, s=2.0) if name in ("M0", "M1", "M2") else make_arm(name)
                rows[name].append(V.episode_statistics(build_path(sk, arm, calibration=calibration), calibration))
    return rows


def test_v4_martingale_moments_and_v6_budget_identity_on_a_bounded_sample(martingale_sample):
    names = [f"{c}:{i}" for c in V.CHANNELS for i in V.INSTRUMENTS]
    pvals = {}
    for name in ("M0", "M1", "M2", "M3"):
        rows = martingale_sample[name]
        X = np.vstack([r["moments"] for r in rows])
        res = V.wald_qr(X, names)
        assert res["rank"] < len(names) and not res["nonzero_constant_columns"]
        if name in ("M0", "M2"):
            assert not any(n.startswith("flow") for n in res["kept"])      # F0: no flow updates
        pvals[name] = res["p"]
        ident = np.array([r["identity_error"] for r in rows])
        assert np.max(np.abs(ident)) < 1e-12                  # V6 identity holds per episode
        drift = np.array([r["drift"] for r in rows])
        assert abs(drift.mean()) < 4.5*drift.std(ddof=1)/np.sqrt(len(drift))
        lhs = np.mean([r["p0_uncertainty"]-r["pc_uncertainty"] for r in rows])
        budget = np.mean([sum(r["budget"].values()) for r in rows])
        assert abs(lhs-budget-drift.mean()) < 1e-12
    adjusted = V.holm(pvals, alpha=0.001)
    assert not any(v["reject"] for v in adjusted.values()), pvals
    # A planted drift (bias in one instrument column) is detected.
    X = np.vstack([r["moments"] for r in martingale_sample["M3"]])
    X[:, 1] += 0.5*X[:, 1].std()
    assert V.wald_qr(X, names)["p"] < 1e-4


def test_v5_outcome_calibration_on_a_bounded_sample(calibration):
    """E[phi(p_t)(Theta - p_t)] = 0 at 900/1800/2700 s and close (300 episodes; Theta dominates the noise)."""
    names = [f"{f}@{t}" for f in ("1", "p", "p2") for t in V.V5_TIMES]
    rows = {"M0": [], "M3": []}
    for ticker in (BROAD, SPORTS):
        for replicate in range(150):
            sk = build_skeleton(ticker, 2000+replicate, calibration=calibration, jump_rate_scale=JUMP_SCALE)
            for name in rows:
                arm = make_arm(name, s=2.0) if name == "M0" else make_arm(name)
                rows[name].append(V.episode_statistics(build_path(sk, arm, calibration=calibration), calibration))
    pvals = {name: V.wald_qr(np.vstack([r["v5"] for r in rows[name]]), names)["p"] for name in rows}
    assert not any(v["reject"] for v in V.holm(pvals, alpha=0.001).values()), pvals


# --------------------------------------------------------------- metrics
def test_information_variance_matches_the_gm_closed_form(calibration):
    sk = build_skeleton(SPORTS, 2, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    path = build_path(sk, make_arm("M3"), calibration=calibration)
    info = Mx.information_variance(path, calibration)
    j = np.asarray(path.refs)[np.asarray(path.kinds) == 0]
    p = expit(np.asarray(path.l_bg)[np.asarray(path.kinds) == 0])
    a, u = path.exec_alpha[j], path.exec_u[j]
    q = (1-a)*u+a*p
    closed = np.sum(a**2*p**2*(1-p)**2/(q*(1-q)))
    assert info["info_variance_flow"] == pytest.approx(closed, rel=1e-8)
    assert info["info_variance_jump"] > 0 and info["info_variance_background"] > 0
    m0 = Mx.information_variance(build_path(sk, make_arm("M0", s=1.0), calibration=calibration), calibration)
    assert m0["info_variance_flow"] == 0 and m0["info_variance_jump"] == 0
    moments = Mx.flow_moments(path)
    assert moments["trades"] == sk.n_executions and 0 <= moments["informed_fraction"] <= 1
    assert moments["jumps_applied"] == len(sk.jump_elapsed)


def test_kyle_windows_follow_the_post_trade_endpoint_convention(calibration):
    sk = build_skeleton(SPORTS, 1, calibration=calibration)
    path = build_path(sk, make_arm("M1-native"), calibration=calibration)
    F, dL = Mx.kyle_windows(path)
    traded = [k for k, w in enumerate(sk.windows) if w.exec_slice.stop > w.exec_slice.start]
    assert len(F) == len(traded)
    assert np.allclose(F, np.asarray(path.window_signed_flow)[traded], rtol=0, atol=1e-9)
    k = traded[0]
    nodes = [i for i in range(path.window_node_slices[k].start, path.window_node_slices[k].stop) if path.kinds[i] == 0]
    assert dL[0] == path.l_after[nodes[-1]]-path.l_after[nodes[0]]
    x = np.arange(10.0)
    assert Mx.ols_slope(x, 3*x+1) == pytest.approx(3.0) and Mx.ols_slope(np.ones(5), x[:5]) == 0.0


V12_TEMPLATES = ("KXAAAGASW-26JUL20-4.060", "KXAAAGASW-26AUG03-4.110", "KXCPICORE-26JUN-T0.2", "KXGDP-26JUL30-T1.0")


def _v12_sample(calibration, hjb, monkeypatch=None, skip=0.0, replicates=100):
    """Bounded real-environment V12 sample: volatile M0 (s = 20), fixed policy, 60-s decisions, S1.

    With `skip` > 0 a deliberately biased sniper (test only) ignores that fraction of crossed proposals.
    """
    from simulator.information import environment as ENV
    if skip:
        real, rng = ENV.snipe_profitable, np.random.default_rng(99)
        monkeypatch.setattr(ENV, "snipe_profitable", lambda *a, **k: real(*a, **k) and rng.random() >= skip)
    rows = []
    for ticker in V12_TEMPLATES:
        for replicate in range(replicates):
            sk = build_skeleton(ticker, 7000+replicate, calibration=calibration)
            path = build_path(sk, make_arm("M0", s=20.0), calibration=calibration)
            env = InformationTradingEnvironment(path, snipe_level="S1", decision_interval=60)
            run_policy(env, "fixed", hjb, seed=sk.path_seed)
            res = V.crossing_exposure(path, env.ledger)
            assert res["snipes"] == env.snipe_count
            rows.append({"ticker": ticker, "replicate": replicate, "snipes": res["snipes"],
                         "probes_crossed": res["probes_crossed"]})
    if skip:
        monkeypatch.undo()
    return V.cluster_ratio_test(pd.DataFrame(rows), 2.0)


def test_v12_real_environment_hazard_is_within_a_tight_band(calibration, hjb):
    """Snipe rate while crossed = 1/l: ratio within +-4.5% (SE ~1.4%) and the clustered CI covers 1/l."""
    res = _v12_sample(calibration, hjb)
    assert res["clusters"] == 400 and res["ratio_se"] < 0.02
    assert abs(res["snipe_to_probe_ratio"]-1) < 0.045 and res["ci95"][0] <= 0.5 <= res["ci95"][1]


def test_v12_real_environment_detects_a_ten_percent_biased_sniper(calibration, hjb, monkeypatch):
    res = _v12_sample(calibration, hjb, monkeypatch, skip=0.10)
    assert res["snipe_to_probe_ratio"] < 1-0.045 and res["ci95"][1] < 0.5


def _synthetic_hazard(skip, clusters, rng, lam=0.5, spells=20):
    """Crossed spells (Exp(3 s) lengths); own proposals Poisson(lam) snipe unless skipped; probes Poisson(lam)."""
    rows = []
    for c in range(clusters):
        snipes = probes = 0
        for _ in range(spells):
            length = rng.exponential(3.0)
            t, end = 0.0, length
            while True:                                   # own-side proposals during the spell
                t += rng.exponential(1/lam)
                if t >= length:
                    break
                if rng.random() >= skip:
                    snipes, end = snipes+1, t
                    break
            probes += rng.poisson(lam*end)                # opposite-side probes while crossed
        rows.append({"ticker": "T", "replicate": c, "snipes": snipes, "probes_crossed": probes})
    return V.cluster_ratio_test(pd.DataFrame(rows), 1/lam)


def test_v12_cluster_ratio_test_rejects_a_biased_hazard_synthetic():
    rng = np.random.default_rng(12)
    fair = _synthetic_hazard(0.0, 600, rng)
    assert fair["ci95"][0] <= 0.5 <= fair["ci95"][1] and abs(fair["snipe_to_probe_ratio"]-1) < 0.05
    for skip in (0.3, 0.1):
        biased = _synthetic_hazard(skip, 600, rng)
        assert biased["ci95"][1] < 0.5, (skip, biased)
        assert abs(biased["snipe_to_probe_ratio"]-(1-skip)) < 0.05


# ---------------------------------------------------------------- runner
def test_runner_checks_checkpoints_and_resume(calibration, tmp_path):
    ticker = BROAD
    roster = {"R_star": {"templates": [ticker], "matched_s": {ticker: {"M0": 2.0, "M1": 2.0, "M2": 1.5, "M3": 1.0}}},
              "R_star_emp": {"templates": [ticker], "matched_s": {ticker: {"M0": 2.0, "M1": 2.0, "M2": 1.5, "M3": 1.0}}}}
    roster["roster_sha256"] = roster_hash(roster)
    E._init_worker(None, str(E.SAVED_BENCHMARK))
    names = {"L0", "M0", "M3", "L1J", "M0-emp"}
    plan = [p for p in E.tier1_plan() if p.label in names]
    assert {p.label for p in plan} == names
    task = (ticker, plan, str(tmp_path), 1, E.SEED, roster, "code", True)
    summary = E.run_template(task)
    assert all(not c["resumed"] for c in summary["cells"])
    frames = {p.label: pd.read_csv(tmp_path/"checkpoints"/f"{ticker}__{p.label}.csv") for p in plan}
    assert len(frames["L0"]) == 4 and len(frames["M0"]) == 8 and len(frames["L1J"]) == 4
    assert frames["L0"].anchor_max_abs_difference.max() <= 1e-9
    for label, frame in frames.items():
        assert frame.attribution_error.abs().max() < 1e-8
        assert frame[frame.policy.eq("none")].realized_pnl.eq(0).all()
        assert frame.signal_sha256.nunique() == 1 and frame.settlement_sha256.nunique() == 1
    assert frames["M0"].s.eq(2.0).all() and frames["M0-emp"].p0.eq(calibration.p0_emp(ticker)).all()
    assert frames["M0"][frames["M0"].snipe_level.eq("S0")].snipe_count.eq(0).all()
    assert any((tmp_path/"ledgers").glob("*__M3__S1__hjb_ledger.csv"))
    # Resume: same key -> no recomputation; a changed roster hash -> recomputed.
    again = E.run_template(task)
    assert all(c["resumed"] for c in again["cells"])
    changed = dict(roster, roster_sha256="other")
    redo = E.run_template((ticker, plan[:1], str(tmp_path), 1, E.SEED, changed, "code", False))
    assert not redo["cells"][0]["resumed"]


def test_tier_plans_label_every_cell_once(calibration):
    labels = [p.label for p in E.tier1_plan()]
    assert len(labels) == len(set(labels))
    assert {"L0", "M0", "M1", "M2", "M3", "L0b", "L0c", "L1J", "L1", "M1-native", "FGH-L", "FGH-LJ", "M0-s1", "M2-s1",
            "M0-emp", "M1-emp", "M2-emp", "M3-emp"} == set(labels)
    tier2 = [p.label for p in E.tier2_plan(calibration)]
    assert len(tier2) == len(set(tier2)) and not set(tier2) & set(labels)
    assert {"M3-jx2", "M3-jx0.5", "M0-dec60", "M0-lat30"} <= set(tier2)
    assert set(E.TIER2_NOT_IMPLEMENTED) == {"zero_latency_grid", "s1_thinned", "ttr_clock_arrivals", "early_revelation"}


def test_v12_cluster_robust_ratio_ci():
    rng = np.random.default_rng(4)
    rows = []
    for c in range(400):                       # correlated arms share a cluster's exposure
        exposure = rng.exponential(20.0)
        for arm in ("M0", "M1", "M2"):
            rows.append({"ticker": "T", "replicate": c, "arm": arm, "snipes": rng.poisson(exposure*0.5),
                         "probes_crossed": rng.poisson(exposure*0.5)})
    d = pd.DataFrame(rows)
    res = V.cluster_ratio_test(d, 2.0)
    assert res["clusters"] == 400 and res["ci95"][0] <= 0.5 <= res["ci95"][1]
    s, p = d.snipes.sum(), d.probes_crossed.sum()
    assert res["rate_per_second"] == pytest.approx(s/p/2.0)
    biased = d.assign(snipes=(d.snipes*0.7).round())
    assert V.cluster_ratio_test(biased, 2.0)["ci95"][1] < 0.5


# ---------------------------------------------------- common random numbers
CRN_CASES = [(t, r) for t in ("KXAAAGASW-26AUG10-4.010", "KXNFLGAME-26AUG28NYGNYJ-NYJ") for r in (0, 1)]
CRN_S = {"M0": 2.0, "M1": 2.5, "M2": 1.5}


@pytest.fixture(scope="module", params=CRN_CASES, ids=lambda c: f"{c[0]}-r{c[1]}")
def crn(request, calibration):
    ticker, replicate = request.param
    sk = build_skeleton(ticker, replicate, calibration=calibration, jump_rate_scale=JUMP_SCALE)
    assert len(sk.jump_elapsed) > 0 and sk.n_executions > 0
    paths = {name: build_path(sk, make_arm(name, s=CRN_S[name]) if name in CRN_S else make_arm(name),
                              calibration=calibration)
             for name in ("golden", "L0b", "L1", "L1J", "FGH-L", "FGH-LJ", "M0", "M1", "M2", "M3")}
    return sk, paths


def test_crn_m_background_is_the_shared_brownian_increment(crn):
    sk, paths = crn
    for name in ("M0", "M1", "M2", "M3"):
        p = paths[name]
        assert p.sigma2 > 0 and p.sigma2 == p.arm.s*p.sigma_c**2
        drift = p.sigma2*sk.node_h/2*(2*p.theta-1)
        z = (np.asarray(p.x)-drift)/np.sqrt(p.sigma2)
        assert np.allclose(z, sk.node_dw, rtol=0, atol=1e-12)
    assert len({paths[n].theta for n in ("M0", "M1", "M2", "M3")}) == 1


def test_crn_m_flow_and_jump_signs_are_shared(crn):
    from simulator.information import kernels as K
    sk, paths = crn
    u = paths["M0"].exec_u
    s_u = K.uninformed_sign(sk.v_u, u)
    assert np.array_equal(paths["M0"].exec_sign, paths["M2"].exec_sign) and np.array_equal(paths["M0"].exec_sign, s_u)
    assert np.array_equal(paths["M1"].exec_sign, paths["M3"].exec_sign)
    assert np.array_equal(paths["M1"].exec_informed, paths["M3"].exec_informed)
    uninformed = ~paths["M1"].exec_informed
    assert np.array_equal(paths["M1"].exec_sign[uninformed], s_u[uninformed])    # FG uninformed branch = S_u
    m2, m3 = paths["M2"], paths["M3"]
    assert m2.jump_applied and m3.jump_applied
    assert np.array_equal(m2.jump_sign, m3.jump_sign) and np.array_equal(m2.jump_step, m3.jump_step)
    jumps = [i for i, k in enumerate(m2.kinds) if k == 1]
    assert [m2.elapsed[i] for i in jumps] == [m3.elapsed[i] for i in jumps] == sk.jump_elapsed.tolist()
    assert np.array_equal(np.abs(m2.jump_step), sk.jump_magnitude)


def test_crn_l0_and_l0b_differ_only_in_execution_signs(crn):
    sk, paths = crn
    golden, l0b = paths["golden"], paths["L0b"]          # golden == TradingEnvironment (V7)
    for attr in ("kinds", "refs", "elapsed", "offsets", "x"):
        assert getattr(golden, attr) == getattr(l0b, attr), attr
    assert np.array_equal(golden.exec_sign, sk.exec_bank_sign) and not np.array_equal(golden.exec_sign, l0b.exec_sign)
    assert golden.spread_draws == l0b.spread_draws and golden.spread_state_final == l0b.spread_state_final
    assert sorted(golden.books) == sorted(l0b.books)                   # books sampled at the same nodes
    assert golden.settlement["uniform"] == l0b.settlement["uniform"] == sk.u_legacy_settlement == sk.u_l  # A-3
    assert np.allclose(np.asarray(l0b.exec_increment), l0b.beta*l0b.exec_sign*sk.exec_quantity, rtol=0, atol=1e-15)


def test_crn_theta_free_arms_share_background_noise_and_executions(crn):
    sk, paths = crn
    for name in ("L1", "L1J", "FGH-L", "FGH-LJ"):
        p = paths[name]
        assert p.sigma2 == paths["L1"].sigma2 == p.sigma_c**2
        l_prev = np.r_[p.initial_log_odds, np.asarray(p.l_after)[:-1]]
        theta_tilde = (sk.node_v_aux < expit(l_prev)).astype(int)        # shared L-aux uniforms
        expected = p.sigma2*sk.node_h/2*(2*theta_tilde-1)+np.sqrt(p.sigma2)*sk.node_dw
        assert np.allclose(p.x, expected, rtol=0, atol=1e-15)            # shared W increments
        assert p.kinds == paths["L1"].kinds and p.elapsed == paths["L1"].elapsed
    first_jump = paths["L1J"].kinds.index(1)
    assert paths["L1"].x[:first_jump+1] == paths["L1J"].x[:first_jump+1]   # identical until the first jump
    assert np.array_equal(paths["L1"].exec_sign, paths["L1J"].exec_sign)     # S_u signs, FL flow
    for name in ("FGH-L", "FGH-LJ"):
        execs = [i for i, k in enumerate(paths[name].kinds) if k == 0]
        assert [paths[name].refs[i] for i in execs] == [paths["L1"].refs[i] for i in execs] == list(range(sk.n_executions))
    for name in ("L1J", "FGH-LJ"):                                            # shared jump times and magnitudes
        p = paths[name]
        assert p.jump_applied and np.array_equal(np.abs(p.jump_step), sk.jump_magnitude)
        assert [p.elapsed[i] for i, k in enumerate(p.kinds) if k == 1] == sk.jump_elapsed.tolist()
