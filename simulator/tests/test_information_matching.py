"""Paper 3 variance matching (spec Section 8; V9 bounded).

Calibration values are never pinned: the calibration is loaded through
`config.py`; jump-bearing laws use a jump-rate scale so they do not depend on
calibrated counts.
"""
import json
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from simulator.information import matching as M
from simulator.information.config import InformationCalibration
from simulator.information.paths import build_path, make_arm
from simulator.information.skeleton import build_skeleton

SIGMA_ZERO_TEMPLATE = "KXAAAGASW-26JUL27-4.080"
BUSY_SPORTS = "KXNFLGAME-26AUG28MINDEN-DEN"
DONOR_BROAD = "KXCPI-26AUG-T0.5"


@pytest.fixture(scope="module")
def calibration():
    for attempt in range(10):
        try:
            return InformationCalibration.load()
        except (ValueError, json.JSONDecodeError, FileNotFoundError, KeyError):
            if attempt == 9:
                raise
            time.sleep(30)


# ---------------------------------------------------------------- quadrature
def test_quadrature_matches_scipy_quad_on_1000_random_cases():
    result = M.quadrature_check(1000)
    assert result["cases"] == 1000 and result["passed"] and result["max_abs_error"] <= 1e-9


def test_quadrature_two_orders_agree_and_methods_switch_at_unit_sd():
    ell = np.linspace(-35, 35, 141)
    for V in (1e-8, 0.3, 1.0, 1.0+1e-9, 2.0, 30.0, 900.0):
        values, diff, (method, orders) = M.complement_expectation(ell, V)
        assert diff <= 1e-9 and np.all((values >= 0) & (values <= 1))
        assert method == ("hermite" if np.sqrt(V) <= 1 else "legendre")
    values, diff, (method, _) = M.complement_expectation(ell, 0.0)
    assert method == "exact" and np.array_equal(values, expit(-ell))
    # Identity: E[p'(1-p')] = q E_{Theta=1}[1-p'] = (1-q) E_{Theta=0}[p']  (checked by direct Gauss-Hermite).
    z, w = np.polynomial.hermite_e.hermegauss(200)
    w = w/np.sqrt(2*np.pi)
    for l0, V in ((0.3, 0.5), (-2.0, 0.8)):
        q = expit(l0)
        e1 = expit(l0+V/2+np.sqrt(V)*z)
        e0 = expit(l0-V/2+np.sqrt(V)*z)
        direct = q*np.dot(w, e1*(1-e1))+(1-q)*np.dot(w, e0*(1-e0))
        h, _ = M.h_values(np.array([l0]), V)
        assert abs(direct-h[0]) < 1e-12


def test_h_is_decreasing_in_v_for_every_draw():
    rng = np.random.default_rng(3)
    ell = rng.uniform(-25, 25, 300)
    grid = np.r_[0.0, np.geomspace(1e-6, 2000.0, 160)]
    previous = None
    for V in grid:
        h, diff = M.h_values(ell, V)
        assert diff <= 1e-9
        if previous is not None:
            assert np.all(h <= previous+1e-13)
        previous = h
    assert np.all(previous < 1e-6)                       # V -> infinity: full revelation


# ------------------------------------------------------------- sampler law
def path_components(path):
    sk = path.skeleton
    background = float(np.sum(path.x))
    flow = float(np.sum(path.exec_increment))
    jump = float(np.sum(path.jump_step)) if path.jump_applied else 0.0
    return background, flow, jump


def test_sampler_aligns_exactly_with_paths_for_given_draws(calibration):
    scale = 10.0
    design = M.TemplateDesign.build(BUSY_SPORTS, calibration, jump_rate_scale=scale)
    for replicate in range(4):
        sk = build_skeleton(BUSY_SPORTS, replicate, calibration=calibration, jump_rate_scale=scale)
        for name in ("M0", "M1", "M2", "M3"):
            arm = make_arm(name, s=1.0)
            path = build_path(sk, arm, calibration=calibration)
            aligned = M.aligned_llr(design, sk, arm)
            assert aligned["theta"] == path.theta
            if arm.flow == "fg":
                assert np.array_equal(aligned["per_trade"], path.exec_increment)       # bitwise per trade
            else:
                assert not np.any(path.exec_increment)
            if path.jump_applied:
                assert np.array_equal(aligned["per_jump"], path.jump_step)
            background, flow, jump = path_components(path)
            assert flow == aligned["lam_flow"] and jump == aligned["lam_jump"]
            assert abs(path.l_after[-1]-(aligned["ell"]+background)) < 1e-10
            assert aligned["ell"] == design.initial_log_odds(0.5)+flow+jump
            # Background given Theta: exactly (2 Theta - 1) V/2 + sigma W_T with V = s sigma_c^2 T.
            V = path.sigma2*sk.seconds
            assert path.sigma_c == calibration.sigma_c(sk.noise_ticker)
            assert abs(background-((2*path.theta-1)*V/2+np.sqrt(path.sigma2)*np.sum(sk.node_dw))) < 1e-10


def test_vectorized_sampler_has_the_path_law(calibration):
    """Moments and terminal-LLR law of the vectorized sampler vs build_path on independent skeletons."""
    scale, episodes = 10.0, 300
    design = M.TemplateDesign.build(BUSY_SPORTS, calibration, jump_rate_scale=scale)
    arm = make_arm("M3")
    draws = M.sample_draws(design, arm, 60_000, np.random.default_rng(11))
    rows = []
    for replicate in range(episodes):
        sk = build_skeleton(BUSY_SPORTS, replicate, calibration=calibration, jump_rate_scale=scale)
        path = build_path(sk, arm, calibration=calibration)
        background, flow, jump = path_components(path)
        rows.append({"theta": path.theta, "n_trades": sk.n_executions, "n_jumps": len(sk.jump_elapsed),
                     "flow": flow, "jump": jump, "ell": path.initial_log_odds+flow+jump,
                     "z_background": (background-(2*path.theta-1)*path.sigma2*sk.seconds/2)/np.sqrt(path.sigma2*sk.seconds)})
    paths = pd.DataFrame(rows)
    sampled = pd.DataFrame({"theta": draws.theta, "n_trades": draws.n_trades, "n_jumps": draws.n_jumps,
                            "flow": draws.lam_flow, "jump": draws.lam_jump, "ell": draws.ell})
    assert abs(paths.theta.mean()-0.5) < 4*np.sqrt(.25/episodes) and abs(sampled.theta.mean()-.5) < 4*np.sqrt(.25/60_000)
    zs = []
    for column in ("n_trades", "n_jumps"):
        a, b = paths[column], sampled[column]
        zs.append((a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)))
    for theta in (0, 1):
        a, b = paths[paths.theta.eq(theta)], sampled[sampled.theta.eq(theta)]
        assert len(a) > 100
        for column in ("flow", "jump"):
            zs.append((a[column].mean()-b[column].mean())/np.sqrt(a[column].var()/len(a)+b[column].var()/len(b)))
            # Second moment (variance) via a z-test on squared deviations.
            da, db = (a[column]-b[column].mean())**2, (b[column]-b[column].mean())**2
            zs.append((da.mean()-db.mean())/np.sqrt(da.var()/len(da)+db.var()/len(db)))
        assert stats.ks_2samp(a.ell, b.ell).pvalue > 1e-3
    assert np.all(np.abs(zs) < 4.5), zs
    assert stats.ks_2samp(paths.ell, sampled.ell).pvalue > 1e-3
    # Background: standardized Lambda_B given Theta is N(0,1) (exact variance h at nodes).
    assert abs(paths.z_background.mean()) < 4/np.sqrt(episodes) and abs(paths.z_background.var()-1) < 4*np.sqrt(2/episodes)


def test_donor_law_matches_legacy_selection(calibration):
    design = M.TemplateDesign.build(DONOR_BROAD, calibration)
    assert not design.donor_fixed and len(design.donors) > 1
    draws = M.sample_draws(design, make_arm("M0", s=1.0), 30_000, np.random.default_rng(5))
    counts = np.bincount(draws.donor, minlength=len(design.donors))
    assert stats.chisquare(counts).pvalue > 1e-3
    seen = [build_skeleton(DONOR_BROAD, r, calibration=calibration).noise_ticker for r in range(240)]
    observed = np.array([seen.count(d) for d in design.donors])
    assert observed.sum() == 240 and stats.chisquare(observed).pvalue > 1e-3
    assert set(np.unique(draws.sigma_c2)) <= {calibration.sigma_c(d)**2 for d in design.donors}
    own = M.TemplateDesign.build(BUSY_SPORTS, calibration)
    assert own.donor_fixed and own.donors == [BUSY_SPORTS]


# ------------------------------------------------------- bracket / unmatched
def synthetic_draws(ell, sigma_c2, seconds=3600.0):
    ell = np.asarray(ell, dtype=float)
    n = len(ell)
    return M.Draws(ell=ell, sigma_c2=np.full(n, sigma_c2), theta=np.zeros(n, np.int8), lam_flow=np.zeros(n),
                   lam_jump=np.zeros(n), n_trades=np.zeros(n, int), n_jumps=np.zeros(n, int),
                   donor=np.zeros(n, int), seconds=seconds)


def test_bracket_bisection_and_unmatched_rules():
    draws = synthetic_draws(np.zeros(50), 1e-4)                  # V(s) = 0.36 s
    target = M.rb_uncertainty(synthetic_draws(np.zeros(1), 1e-4), 6.0).mean
    solved = M.solve_scale(draws, target)
    assert solved["status"] == "solved" and abs(solved["s"]-6.0) < 1e-6 and solved["s_hi"] == 8.0
    assert abs(M.rb_uncertainty(draws, solved["s"]).mean/target-1) < 1e-8
    # Needs s > 1,024 -> unmatched at the cap.
    hard = M.rb_uncertainty(synthetic_draws(np.zeros(1), 1e-4), 1500.0).mean
    capped = M.solve_scale(draws, hard)
    assert capped["status"] == "unmatched_cap" and capped["s_hi"] == 1024.0 and capped["s"] is None
    # Target above the arm already at s = 0.
    assert M.solve_scale(draws, 0.3)["status"] == "target_not_above_arm_at_s0"
    # Zero background (sigma_c = 0): the curve is flat.
    flat = synthetic_draws(np.zeros(10), 0.0)
    assert M.rb_uncertainty(flat, 0.0).mean == M.rb_uncertainty(flat, 1024.0).mean == 0.25


def test_sigma_c_zero_template_is_unmatched_and_leaves_r_star(calibration):
    assert calibration.sigma_c(SIGMA_ZERO_TEMPLATE) == 0.0
    design = M.TemplateDesign.build(SIGMA_ZERO_TEMPLATE, calibration)
    assert design.sigma_c_identically_zero
    rows = M.match_template(SIGMA_ZERO_TEMPLATE, calibration, draws=2_000)
    status = {r["arm"]: r["status"] for r in rows}
    assert status == {"M0": "unmatched_sigma_c_zero", "M1": "unmatched_sigma_c_zero",
                      "M2": "unmatched_sigma_c_zero", "M3": "target"}
    report = pd.DataFrame(rows+[{**r, "p0_label": "emp"} for r in rows])
    roster = M.build_roster(report, calibration)
    assert SIGMA_ZERO_TEMPLATE not in roster["R_star"]["templates"]
    assert "unmatched_sigma_c_zero" in roster["R_star"]["excluded"][SIGMA_ZERO_TEMPLATE]


def test_early_revelation_templates_are_excluded_by_rule(calibration):
    early = [t for t in calibration.config["templates"] if calibration.early_revelation(t)]
    assert early
    rows = M.match_template(early[0], calibration, draws=1_000)
    assert {r["status"] for r in rows} == {"excluded_early_revelation"}


def test_small_scale_matching_reaches_the_target(calibration):
    rows = M.match_template(DONOR_BROAD, calibration, draws=20_000)
    by_arm = {r["arm"]: r for r in rows}
    for arm in M.MATCHED_ARMS:
        r = by_arm[arm]
        assert r["status"] in {"matched", "unmatched_validation"}
        assert r["s"] > 0 and abs(r["U_fit"]/r["U_star"]-1) < 1e-6
        if r["status"] == "matched":
            assert abs(r["gap"]) < 0.02 and r["ci_low"] <= 0 <= r["ci_high"]


# ------------------------------------------------------------------- CI
def test_validation_ci_is_the_delta_method_with_target_uncertainty():
    val = M.Estimate(mean=0.101, var=0.04, n=10_000, quad_diff=0.0)
    tgt = M.Estimate(mean=0.100, var=0.09, n=40_000, quad_diff=0.0)
    out = M.validation_gap(val, tgt)
    se = np.sqrt((0.04/10_000)/0.1**2+(0.101**2/0.1**4)*(0.09/40_000))
    assert abs(out["gap"]-0.01) < 1e-12 and abs(out["gap_se"]-se) < 1e-15
    assert abs(out["ci_low"]-(0.01-1.959963984540054*se)) < 1e-15
    assert out["accepted"] == (out["ci_low"] <= 0 <= out["ci_high"])
    # Target uncertainty widens the interval relative to a validation-only interval.
    assert out["gap_se"] > np.sqrt(0.04/10_000)/0.1
    tight = M.validation_gap(M.Estimate(0.103, 1e-6, 10**6, 0.0), M.Estimate(0.1, 1e-6, 10**6, 0.0))
    assert not tight["accepted"]                                  # CI excludes 0
    big = M.validation_gap(M.Estimate(0.125, 1.0, 10, 0.0), M.Estimate(0.1, 1.0, 10, 0.0))
    assert not big["accepted"]                                    # |gap| >= 2%
    # Coverage under the null (independent val and target sets of the same law): about 95%.
    rng = np.random.default_rng(1)
    covered = 0
    for _ in range(2000):
        a, b = rng.exponential(0.1, 400), rng.exponential(0.1, 400)
        ci = M.validation_gap(M.Estimate(a.mean(), a.var(ddof=1), 400, 0), M.Estimate(b.mean(), b.var(ddof=1), 400, 0))
        covered += ci["ci_low"] <= 0 <= ci["ci_high"]
    assert 0.93 < covered/2000 < 0.97


def test_roster_hash_detects_edits(tmp_path):
    roster = {"R_star": {"templates": ["a"], "matched_s": {"a": {"M0": 2.0}}},
              "R_star_emp": {"templates": [], "matched_s": {}}}
    roster["roster_sha256"] = M.roster_hash(roster)
    path = tmp_path/"roster.json"
    path.write_text(json.dumps(roster))
    assert M.load_roster(path)["R_star"]["templates"] == ["a"]
    roster["R_star"]["matched_s"]["a"]["M0"] = 2.5
    path.write_text(json.dumps(roster))
    with pytest.raises(ValueError):
        M.load_roster(path)
