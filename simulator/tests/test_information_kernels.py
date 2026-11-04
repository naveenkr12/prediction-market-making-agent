"""Paper 3 kernels: closed forms (V1), independent reference posterior (V2),
restricted permutation invariance (V3), taker fee and legacy-arithmetic checks."""
from decimal import Decimal

import numpy as np
import pytest
from scipy.special import expit, logit

from simulator.information import kernels as K
from simulator.information import reference as R
from simulator.information.taker_fees import TakerFeeSchedule, snipe_profitable, taker_fee
from simulator.price_impact import ImpactParameters, ImpactPriceProcess


# ------------------------------------------------------------------------- V1
def test_gm_closed_forms_zero_mean_and_variance():
    rng = np.random.default_rng(1)
    p, a, u = rng.uniform(.01, .99, 2000), rng.uniform(0, .95, 2000), rng.uniform(.02, .98, 2000)
    L = logit(p)
    q = (1-a)*u+a*p
    up = expit(L+K.gm_llr(np.ones(2000, int), a, 1-a, u))-p
    down = expit(L+K.gm_llr(-np.ones(2000, int), a, 1-a, u))-p
    np.testing.assert_allclose(up, a*p*(1-p)/q, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(down, -a*p*(1-p)/(1-q), rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(q*up+(1-q)*down, 0, atol=1e-14)
    np.testing.assert_allclose(q*up**2+(1-q)*down**2, a**2*p**2*(1-p)**2/(q*(1-q)), rtol=1e-9, atol=1e-16)
    # Buy probability of the Theta-free sampler is exactly q.
    np.testing.assert_allclose(K.buy_probability_theta_free(L, a, 1-a, u), q, rtol=1e-14)


def test_fg_half_llr_is_plus_minus_beta_c_and_fg_alpha_cap():
    x = np.r_[np.logspace(-8, np.log10(20), 400), 0.0]
    a, om = K.fg_half_alpha_terms(x, 1.0)
    for s in (1, -1):
        generic = K.gm_llr(np.full(x.shape, s), a, om, np.full(x.shape, .5))
        np.testing.assert_allclose(generic, s*x, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(K.fg_half_llr(np.full(x.shape, s), x, 1.0), s*x)
    np.testing.assert_allclose(a, np.tanh(x/2), rtol=1e-15)
    alpha, one_minus = K.fg_alpha_terms(np.array([0, 1, 10, 1e9]), 1e-3)
    np.testing.assert_allclose(alpha[:3], 1-np.exp(-1e-3*np.array([0, 1, 10])), rtol=1e-12)
    assert alpha[0] == 0 and alpha[3] == 1-1e-12 and one_minus[3] == 1e-12
    assert np.all(alpha+one_minus == pytest.approx(1, abs=1e-15))
    # eta = 0 gives zero LLR for every size and u.
    a0, om0 = K.fg_alpha_terms(np.array([1, 1e6]), 0.0)
    assert np.all(K.gm_llr(np.array([1, -1]), a0, om0, np.array([.3, .7])) == 0)


def test_sign_tilt_llr_and_theta_free_jump_martingale():
    a = np.array([1e-6, .05, .7, 3.0, 12.0])
    llr_up = np.log(K.jump_up_probability_m(1, a)/K.jump_up_probability_m(0, a))
    llr_down = np.log((1-K.jump_up_probability_m(1, a))/(1-K.jump_up_probability_m(0, a)))
    np.testing.assert_allclose(llr_up, a, rtol=1e-9)
    np.testing.assert_allclose(llr_down, -a, rtol=1e-9)
    for p in (1e-6, .03, .5, .91, 1-1e-6):
        L = logit(p)
        q = K.jump_up_probability_theta_free(L, a)
        np.testing.assert_allclose(q*expit(L+a)+(1-q)*expit(L-a), p, rtol=1e-12, atol=1e-15)


def test_gaussian_background_llr_equals_observation():
    rng = np.random.default_rng(2)
    sigma2, h = rng.uniform(1e-6, 1e-3, 500), rng.uniform(.1, 300, 500)
    theta = rng.integers(0, 2, 500)
    L1, x = K.background_m(np.zeros(500), theta, sigma2, h, rng.normal(0, np.sqrt(h)))
    np.testing.assert_array_equal(L1, x)
    np.testing.assert_allclose(K.gaussian_llr(x, sigma2, h), x, rtol=1e-9, atol=1e-15)
    # Exact Bayes: conditional density ratio at X equals exp(X).
    v = sigma2*h
    ratio = np.exp(-(x-v/2)**2/(2*v)+(x+v/2)**2/(2*v))
    np.testing.assert_allclose(np.log(ratio), x, rtol=1e-9, atol=1e-14)
    L2, x2 = K.background_legacy(np.ones(3), np.array([4., 0., 1.]), np.array([.5, 9., -2.]))
    np.testing.assert_array_equal(x2, [1., 0., -2.])
    np.testing.assert_array_equal(L2, [2., 1., -1.])


def test_sampler_laws_m_and_theta_free():
    rng = np.random.default_rng(3)
    n = 400_000
    alpha, u, p = .3, .62, .27
    L = np.full(n, logit(p))
    om = 1-alpha
    tol = lambda prob: 5*np.sqrt(prob*(1-prob)/n)
    for theta, target in ((1, alpha+om*u), (0, om*u)):
        s, informed = K.flow_sign_m(np.full(n, theta), rng.random(n), rng.random(n), alpha, u)
        assert abs((s == 1).mean()-target) < tol(target)
        assert abs(informed.mean()-alpha) < tol(alpha)
    q = om*u+alpha*p
    s = K.flow_sign_theta_free(L, rng.random(n), alpha, om, u)
    assert abs((s == 1).mean()-q) < tol(q)
    _, _, theta_tilde = K.background_theta_free(L, rng.random(n), 1e-4, 60, rng.normal(0, np.sqrt(60), n))
    assert abs(theta_tilde.mean()-p) < tol(p)
    a = .8
    qj = p*expit(a)+(1-p)*expit(-a)
    assert abs((K.jump_sign_theta_free(L, rng.random(n), a) == 1).mean()-qj) < tol(qj)
    assert abs((K.jump_sign_m(np.ones(n, int), rng.random(n), a) == 1).mean()-expit(a)) < tol(expit(a))
    assert abs((K.jump_sign_legacy(rng.random(n)) == 1).mean()-.5) < tol(.5)
    su = K.uninformed_sign(rng.random(n), u)
    assert set(np.unique(su)) == {-1, 1} and abs((su == 1).mean()-u) < tol(u)
    # Theta-free marginal martingale in p for flow: E[expit(L + LLR)] = p.
    signs = K.flow_sign_theta_free(L, rng.random(n), alpha, om, u)
    after = K.probability(K.flow_update(L, signs, alpha, om, u)[0])
    assert abs(after.mean()-p) < 5*after.std()/np.sqrt(n)


def test_extreme_llrs_keep_latent_finite_and_count_saturation():
    a, om = K.fg_alpha_terms(np.full(5000, 1e9), 1.0)
    L = 0.0
    for llr in K.gm_llr(np.ones(5000, int), a, om, np.full(5000, .02)):
        L += llr
    assert np.isfinite(L) and L > 1e5
    assert K.saturated(L) and 0 < K.display_probability(L) < 1
    assert K.flow_sign_theta_free(np.array([L, -L]), np.array([.999999, .5]), a[:2], om[:2], .02).tolist() == [1, -1]
    jumped, _ = K.jump_update(np.array([L]), np.array([1]), np.array([40.]))
    assert np.isfinite(jumped).all()
    with pytest.raises(AssertionError):
        K.check_signs([1, 0, -1])
    with pytest.raises(ValueError):
        K.gm_llr([1], [.1], [.9], [1.0])


def test_fl_matches_impact_process_arithmetic_and_settlement_helpers():
    rng = np.random.default_rng(4)
    price = ImpactPriceProcess(.37, ImpactParameters(5.473789936966611e-05))
    L = price.log_odds
    for sign, size in zip(rng.choice([-1, 1], 300), rng.lognormal(2, 2, 300)):
        price.advance(float(np.int8(sign)*size))
        L, _ = K.fl_update(L, np.int8(sign), size, 5.473789936966611e-05)
        assert L == price.log_odds
    assert K.settle_m(.49, .5) == 1 and K.settle_m(.5, .5) == 0
    assert K.settle_l(.2, .2000001) == 1 and K.settle_l(.3, .2) == 0


def test_ttr_buckets_and_piecewise_poisson_jump_times():
    assert K.ttr_bucket_index([0, 3599.9, 3600, 10800, 86399, 86400, 1e9, -1]).tolist() == [0, 0, 1, 2, 2, 3, 3, -1]
    rates = {"B0": 0.01, "B1": 0.0, "B2": 0.002, "B3": 1.0}
    rng = np.random.default_rng(5)
    counts = np.array([len(K.poisson_jump_offsets(5400, 3600, rates, rng)) for _ in range(4000)])
    # TTR runs 5400 -> 1800: 1800 s in B1 (rate 0) and 1800 s in B0.
    assert counts.mean() == pytest.approx(18, rel=.03)
    offsets = np.concatenate([K.poisson_jump_offsets(5400, 3600, rates, rng) for _ in range(200)])
    assert offsets.min() >= 1800 and offsets.max() < 3600
    # After release (TTR <= 0) the reduced-form jump rate is zero.
    late = K.poisson_jump_offsets(100, 3600, {b: 10.0 for b in K.TTR_BUCKETS}, rng)
    assert len(late) > 0 and late.max() < 100


def test_taker_fee_rounding_and_snipe_rule():
    assert taker_fee(10, .5) == Decimal("0.18")          # 0.07*10*.25 = 0.175 -> 0.18
    assert taker_fee(1, .01) == Decimal("0.01")          # tiny fee rounds up to a cent
    assert TakerFeeSchedule(0).fee(5, .4) == 0
    # ask at .40, latent p .50, Q=10: edge 1.00 > fee(.17) + .10
    assert snipe_profitable("ask", 10, .40, .50, "0.01")
    assert not snipe_profitable("ask", 10, .40, .41, "0.01")       # 0.10 < 0.17 + 0.10
    assert snipe_profitable("bid", 10, .60, .45, "0.01")
    assert not snipe_profitable("bid", 10, .60, .61, "0.01")
    # Boundary is strict: edge exactly equal to cost does not trigger.
    assert not snipe_profitable("ask", 4, .5, .75, "0.25", TakerFeeSchedule(0))
    assert snipe_profitable("ask", 4, .5, .875, "0.25", TakerFeeSchedule(0))
    with pytest.raises(ValueError):
        snipe_profitable("mid", 1, .4, .5, "0.01")


# ------------------------------------------------------------------------- V2
def m_episode(rng, arm, n_events=60, flow_law="fg"):
    """Small bounded M-family episode built from kernels; returns (p0, signals, path)."""
    p0 = float(rng.uniform(.05, .95))
    theta = K.settle_m(rng.random(), p0)
    flow_on, jumps_on = arm in ("M1", "M3"), arm in ("M2", "M3")
    eta, beta, alpha0, s = 10**rng.uniform(-6, -3), 10**rng.uniform(-6, -3), rng.uniform(0, .4), rng.uniform(.1, 3)
    sigma_c2 = rng.uniform(1e-6, 5e-4)
    L, path, signals = float(logit(p0)), [], []
    kinds = rng.choice(["background", "flow", "jump"], n_events, p=[.4, .45, .15])
    for kind in kinds:
        if kind == "background":
            h = float(rng.uniform(.05, 120))
            L, x = K.background_m(L, theta, s*sigma_c2, h, rng.normal(0, np.sqrt(h)))
            signals.append({"kind": "background", "x": float(x), "variance": s*sigma_c2*h})
        elif kind == "flow":
            c, u = float(rng.lognormal(2.5, 2)), float(rng.uniform(.02, .98))
            v_i, v_u = rng.random(), rng.random()
            if not flow_on:
                sign = int(K.uninformed_sign(v_u, u))
                signals.append({"kind": "flow", "law": "uninformed", "sign": sign, "size": c})
            else:
                if flow_law == "fg":
                    a, om = K.fg_alpha_terms(c, eta)
                    log = {"law": "fg", "eta": eta, "u": u}
                elif flow_law == "fg_half":
                    a, om = K.fg_half_alpha_terms(c, beta)
                    u, log = .5, {"law": "fg_half", "beta": beta}
                else:
                    a, om = K.constant_alpha_terms(alpha0)
                    log = {"law": "constant", "alpha": alpha0, "u": u}
                sign, _ = K.flow_sign_m(theta, v_i, v_u, a, u)
                if flow_law == "fg_half":
                    L = L+float(K.fg_half_llr(sign, c, beta))
                else:
                    L, _ = K.flow_update(L, sign, a, om, u)
                signals.append({"kind": "flow", "sign": int(sign), "size": c, **log})
        else:
            if not jumps_on:
                continue
            a = float(rng.choice([.02, .1, .4, 1.5]))
            sign = K.jump_sign_m(theta, rng.random(), a)
            L, _ = K.jump_update(L, sign, a)
            signals.append({"kind": "jump", "sign": int(sign), "magnitude": a})
        L = float(L)
        path.append(L)
    return p0, signals, np.asarray(path)


@pytest.mark.parametrize("arm", ["M0", "M1", "M2", "M3"])
@pytest.mark.parametrize("flow_law", ["fg", "fg_half", "constant"])
def test_reference_posterior_matches_kernel_path_at_every_node(arm, flow_law):
    rng = np.random.default_rng([["M0", "M1", "M2", "M3"].index(arm), ["fg", "fg_half", "constant"].index(flow_law)])
    for _ in range(25):
        p0, signals, path = m_episode(rng, arm, flow_law=flow_law)
        ref = np.asarray(R.posterior_log_odds_path(p0, signals))
        assert len(ref) == len(path)
        np.testing.assert_allclose(ref, path, rtol=0, atol=1e-12)


def test_reference_rejects_bad_signals_and_handles_zero_variance():
    assert R.posterior_log_odds_path(.5, [{"kind": "background", "x": 0.0, "variance": 0.0}]) == [0.0]
    with pytest.raises(AssertionError):
        R.posterior_log_odds_path(.5, [{"kind": "jump", "sign": 0, "magnitude": .1}])
    with pytest.raises(ValueError):
        R.posterior_log_odds_path(.5, [{"kind": "mystery"}])
    # Extreme FG LLR (capped alpha) agrees with the kernel and stays finite.
    a, om = K.fg_alpha_terms(1e9, 1.0)
    llr = float(K.gm_llr(1, a, om, .02))
    ref = R.posterior_log_odds_path(.5, [{"kind": "flow", "law": "fg", "sign": 1, "size": 1e9, "eta": 1.0, "u": .02}])
    assert ref[0] == pytest.approx(llr, abs=1e-12) and np.isfinite(ref[0])


# ------------------------------------------------------------------------- V3
def test_restricted_terminal_permutation_invariance_pathwise():
    """Fixed p-independent experiments: terminal L does not depend on signal order."""
    rng = np.random.default_rng(7)
    for arm in ("M1", "M2", "M3"):
        p0, signals, path = m_episode(rng, arm, n_events=200)
        for _ in range(5):
            order = rng.permutation(len(signals))
            permuted = [signals[i] for i in order]
            assert R.posterior_log_odds_path(p0, permuted)[-1] == pytest.approx(path[-1], abs=1e-12)
            # Kernel recomputation of the same signals in permuted order.
            L = float(logit(p0))
            for sig in permuted:
                if sig["kind"] == "background":
                    L += sig["x"]
                elif sig["kind"] == "jump":
                    L = float(K.jump_update(L, sig["sign"], sig["magnitude"])[0])
                elif sig["law"] == "fg":
                    a, om = K.fg_alpha_terms(sig["size"], sig["eta"])
                    L = float(K.flow_update(L, sig["sign"], a, om, sig["u"])[0])
            assert L == pytest.approx(path[-1], abs=1e-12)


def test_restricted_terminal_permutation_invariance_in_law_for_theta_free_generation():
    """Theta-free generation with fixed experiments: the terminal law is order invariant."""
    rng = np.random.default_rng(8)
    n, p0 = 200_000, .35
    sizes = np.array([1., 30., 400., 5., 2500., 12.])
    us = np.array([.3, .6, .7, .5, .45, .8])
    a, om = K.fg_alpha_terms(sizes, 4e-4)
    jumps = np.array([.3, 1.1])

    def terminal(order):
        L = np.full(n, logit(p0))
        for kind, i in order:
            if kind == "f":
                s = K.flow_sign_theta_free(L, rng.random(n), a[i], om[i], us[i])
                L = K.flow_update(L, s, a[i], om[i], us[i])[0]
            else:
                L = K.jump_update(L, K.jump_sign_theta_free(L, rng.random(n), jumps[i]), jumps[i])[0]
        return K.probability(L)

    forward = [("f", i) for i in range(6)]+[("j", 0), ("j", 1)]
    backward = forward[::-1]
    pf, pb = terminal(forward), terminal(backward)
    for stat in (lambda x: x, lambda x: x*(1-x), lambda x: (x > .5).astype(float)):
        a1, b1 = stat(pf), stat(pb)
        se = np.sqrt(a1.var()/n+b1.var()/n)
        assert abs(a1.mean()-b1.mean()) < 5*se
    assert abs(pf.mean()-p0) < 5*pf.std()/np.sqrt(n)
