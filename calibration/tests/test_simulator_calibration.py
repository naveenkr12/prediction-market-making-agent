"""No holdout fitting, endpoint attenuation, scenario ranges and maker signs."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from build_intervals import aggregate_trades
from calibrate_price_impact import (build_profiles, endpoint_factor, make_training_cases,
                                    replay_validation)
from estimate_lambda import VARIANTS, fit_lambda
from simulator.price_impact import ImpactParameters, ImpactPriceProcess, observe_trade_flows


def training_fixture():
    buckets, estimates, tapes = [], [], {}
    for market_number in range(6):
        ticker = f"M{market_number}"
        rows = []
        for i, ts in enumerate(pd.date_range("2026-07-01", periods=65, freq="5min", tz="UTC")):
            sign = 1 if i%2 == 0 else -1
            for j in range(3):
                rows.append({"trade_id": f"{i:03d}-{j}", "timestamp": ts+pd.Timedelta(seconds=j),
                             "price": 0.5+sign*j*0.002, "quantity": float((i%7+1)*(j+1)), "sign": sign})
        tape = pd.DataFrame(rows)
        m = {"ticker": ticker, "event_cluster": f"E{market_number}", "event_ticker": f"E{market_number}",
             "series_ticker": "S", "information_class": "sports", "category": "Sports",
             "result": "yes", "open_time": "2026-07-01", "close_time": "2026-07-02"}
        b = aggregate_trades(tape, m)
        b["split"] = "train" if market_number < 5 else "test"
        buckets.append(b)
        tapes[ticker] = tape
        for variant in VARIANTS:
            estimates.append({"ticker": ticker, "information_class": "sports", "split": b.split.iloc[0],
                              **fit_lambda(b, variant)})
    return pd.concat(buckets, ignore_index=True), pd.DataFrame(estimates), tapes


def test_post_trade_endpoint_factor_and_replay_match_median_target():
    b, estimates, tapes = training_fixture()
    train, targets, cases = make_training_cases(b, estimates, lambda ticker: tapes[ticker])
    assert len(train) == 5*65
    assert endpoint_factor(train[train.ticker.eq("M0")]) == pytest.approx(5/6)
    profiles = build_profiles(targets, bootstrap_draws=30)
    profile = profiles["sports"]
    assert profile["scenarios"]["base"] == pytest.approx(profile["target_lambda_median"]/(5/6))
    assert profile["scenarios"]["low"] <= profile["scenarios"]["base"] <= profile["scenarios"]["high"]
    flows = {(ticker, ts): (d.sign*d.quantity).to_numpy() for ticker, tape in tapes.items()
             for ts, d in tape.groupby(tape.timestamp.dt.floor("5min"))}
    checks, groups = replay_validation(profiles, targets, cases, flows)
    assert checks.split.eq("train").all()
    base = groups[groups.scenario.eq("base") & groups.variant.eq("full")]
    np.testing.assert_allclose(base.simulated_reference_lambda_median, base.empirical_lambda_median)
    np.testing.assert_allclose(base.float_probability_lambda_median, base.empirical_lambda_median)


def test_holdout_values_never_change_calibration_or_trigger_tape_reads():
    b, estimates, tapes = training_fixture()
    loaded = []
    def load(ticker):
        loaded.append(ticker)
        assert ticker != "M5"
        return tapes[ticker]
    _, targets, _ = make_training_cases(b, estimates, load)
    first = build_profiles(targets, bootstrap_draws=30)
    b.loc[b.split.eq("test"), "signed_flow_contracts"] = 1e15
    b.loc[b.split.eq("test"), "delta_logodds"] = -1e15
    estimates.loc[estimates.split.eq("test"), "lambda"] = -1e9
    estimates.loc[estimates.split.eq("test"), "information_class"] = "CHANGED"
    _, changed_targets, _ = make_training_cases(b, estimates, load)
    second = build_profiles(changed_targets, bootstrap_draws=30)
    assert first == second
    assert set(loaded) == {f"M{i}" for i in range(5)}


def test_single_event_unsupported_and_negative_targets_project_only_at_group_level():
    _, estimates, tapes = training_fixture()
    b, _, _ = training_fixture()
    _, targets, _ = make_training_cases(b, estimates, lambda ticker: tapes[ticker])
    unsupported = build_profiles(targets, min_events=100, bootstrap_draws=30)
    with pytest.raises(ValueError, match="No group"):
        replay_validation(unsupported, targets, {}, {})
    targets.loc[targets.ticker.eq("M0"), "information_class"] = "single_name"
    assert build_profiles(targets, bootstrap_draws=30)["single_name"]["status"] == "insufficient_training_events"
    # No positive/significant-contract selection: negative group medians remain
    # in descriptive output and the simulator's beta constraint is explicit.
    targets["lambda"] = -np.abs(targets["lambda"])
    p = build_profiles(targets, min_events=2, bootstrap_draws=30)["sports"]
    assert p["target_lambda_median"] < 0 and p["scenarios"]["base"] == 0
    assert p["beta_boundary_projection"]


def test_price_process_quantity_units_maker_sign_and_zero_control():
    process = ImpactPriceProcess(0.5, ImpactParameters(0.001))
    positive = process.advance(2.5)
    assert positive["log_odds"] == pytest.approx(0.0025)
    assert positive["probability"] > 0.5
    sell = process.advance_maker_fill(-7.5)
    assert sell["log_odds"] == pytest.approx(0.01)  # maker sold YES, price goes up
    buy = process.advance_maker_fill(10)
    assert buy["probability"] == pytest.approx(0.5)
    zero = ImpactPriceProcess(0.5, ImpactParameters(0))
    assert zero.advance(1e12, innovation_logodds=0.2)["log_odds"] == pytest.approx(0.2)
    assert zero.advance(-1e12)["log_odds"] == pytest.approx(0.2)


def test_vector_replay_matches_event_updates_and_latent_state_survives_saturation():
    params = ImpactParameters(0.0001)
    flow = np.array([1.25, -2.5, 3.75])
    latent, prices = observe_trade_flows(0.4, flow, params)
    process = ImpactPriceProcess(0.4, params)
    updates = [process.advance(float(f)) for f in flow]
    np.testing.assert_allclose(latent, [u["log_odds"] for u in updates])
    np.testing.assert_allclose(prices, [u["probability"] for u in updates])
    assert process.advance(1e9)["float_probability_saturated"]
    restored = process.advance(-1e9)
    assert restored["log_odds"] == pytest.approx(logit(0.4)+params.beta*flow.sum(), abs=1e-10)
    assert 0 < restored["probability"] < 1


def test_loader_requires_supported_group_scenario_and_explicit_units(tmp_path):
    path = tmp_path/"config.json"
    config = {"beta_units": "log_odds_per_signed_market_contract", "reference_observation": "post_trade",
              "groups": {"sports": {"status": "calibrated", "scenarios": {"base": 1e-5}},
                         "single_name": {"status": "insufficient_training_events", "scenarios": None}}}
    path.write_text(json.dumps(config))
    assert ImpactParameters.load("sports", path=path).beta == 1e-5
    with pytest.raises(ValueError, match="supported calibration"):
        ImpactParameters.load("single_name", path=path)
    with pytest.raises(ValueError, match="Unknown impact scenario"):
        ImpactParameters.load("sports", "typo", path)
    config["beta_units"] = "cents_per_contract"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="units"):
        ImpactParameters.load("sports", path=path)
    with pytest.raises(ValueError, match="nonnegative"):
        ImpactParameters(-1)
    with pytest.raises(ValueError, match="strictly"):
        ImpactPriceProcess(1, ImpactParameters(0))
