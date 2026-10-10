"""Exposure, quote endpoints, flow separation, sampling and holdout isolation."""
import json

import numpy as np
import pandas as pd
import pytest

from build_intervals import file_hash
from calibrate_market_dynamics import (activity_profile, calibrate, count_moments,
                                      normalize_quotes, quote_intervals, residual_noise_fit)
from simulator.market_dynamics import MarketDynamics, sample_counts
from simulator.price_impact import ImpactPriceProcess


def market(ticker="M0"):
    return {"ticker": ticker, "event_cluster": f"EVENT-{ticker}", "information_class": "sports",
            "open_time": "2026-07-01T00:00:00Z", "close_time": "2026-07-01T03:00:00Z"}


def tape(times, qty, signs):
    return pd.DataFrame({"trade_id": [str(i) for i in range(len(times))],
                         "timestamp": pd.to_datetime(times, utc=True, format="mixed"),
                         "quantity": qty, "sign": signs, "price": .5})


def quotes(times, mids):
    return pd.DataFrame({"end_period_ts": pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns").asi8//10**9,
                         "yes_bid_close_dollars": np.asarray(mids)-.01,
                         "yes_ask_close_dollars": np.asarray(mids)+.01})


def test_listing_exposure_counts_empty_intervals_and_keeps_fractional_marks():
    m = market(); m["close_time"] = "2026-07-01T01:00:00Z"
    t = tape(["2026-07-01T00:00:01Z", "2026-07-01T00:00:10Z", "2026-07-01T00:05:01Z"],
             [1.25, 2.5, 10], [1, -1, 1])
    record, phases, labels = activity_profile(t, m, n_phases=2)
    assert record["arrival_rate_per_second"] == 3/3600
    assert record["empty_full_5min_intervals"] == 10
    assert record["quantity_total"] == 13.75 and record["quantity_median"] == 2.5
    counts = np.array([2, 1]+[0]*10)
    assert record["full_interval_count_mean"] == counts.mean()
    assert record["full_interval_count_variance"] == pytest.approx(counts.var(ddof=1))
    assert record["fano_5min"] == pytest.approx(counts.var(ddof=1)/counts.mean())
    assert labels.tolist() == [0, 0, 0]
    assert phases[1]["trades"] == 0 and phases[1]["arrival_rate_per_second"] == 0


def test_quote_grid_gaps_and_half_open_signed_flow_boundaries():
    ts = pd.to_datetime(["2026-07-01T00:00:00Z", "2026-07-01T00:05:00Z",
                         "2026-07-01T00:10:00Z", "2026-07-01T00:20:00Z"], utc=True)
    q, _ = normalize_quotes(quotes(ts, [.49, .50, .51, .52]), market())
    t = tape(["2026-07-01T00:00:00Z", "2026-07-01T00:04:59Z",
              "2026-07-01T00:05:00Z", "2026-07-01T00:10:00Z"], [1, 2, 4, 100], [1, -1, 1, 1])
    pairs = quote_intervals(q, t)
    assert len(pairs) == 2  # missing 00:15 is never filled
    assert pairs.signed_flow_contracts.tolist() == [-1, 4]
    assert pairs.n_trades.tolist() == [2, 1]
    assert pairs.duration_seconds.eq(300).all()


def test_quote_aliases_crosses_boundaries_and_conflicting_timestamps():
    ts = pd.date_range("2026-07-01", periods=6, freq="5min", tz="UTC")
    raw = quotes(ts, [.5]*6).rename(columns={"yes_bid_close_dollars": "yes_bid_close", "yes_ask_close_dollars": "yes_ask_close"})
    raw.loc[0, ["yes_bid_close", "yes_ask_close"]] = [0, 0]
    raw.loc[1, ["yes_bid_close", "yes_ask_close"]] = [1, 1]
    raw.loc[2, ["yes_bid_close", "yes_ask_close"]] = [.6, .4]
    raw.loc[3, ["yes_bid_close", "yes_ask_close"]] = [49, 51]  # no guessed cents conversion
    duplicate = raw.iloc[[4]].copy()
    duplicate[["yes_bid_close", "yes_ask_close"]] = [.4, .6]  # same midpoint, different quote
    q, stats = normalize_quotes(pd.concat([raw, duplicate], ignore_index=True), market())
    assert stats["invalid_quote_rows"] == 4 and stats["conflicting_timestamp_rows"] == 2
    assert len(q) == 1 and q.timestamp.iloc[0] == ts[-1]


def test_orthogonal_noise_removes_remaining_flow_and_uses_seconds():
    n = 100
    flow = np.linspace(-100, 100, n)
    noise = np.sin(np.arange(n))*.01
    x = np.column_stack([np.ones(n), flow])
    noise -= x@np.linalg.lstsq(x, noise, rcond=None)[0]
    ts = pd.date_range("2026-07-01", periods=n, freq="5min", tz="UTC")
    d = pd.DataFrame({"left_time": ts, "right_time": ts+pd.Timedelta(minutes=5),
                      "signed_flow_contracts": flow, "duration_seconds": 300,
                      "delta_logodds_midpoint": .003+.002*flow+noise})
    fit = residual_noise_fit(d, beta=.0001)
    expected = np.sqrt((noise@noise)/(n-2)/300)
    assert fit["sigma_logodds_per_sqrt_second"] == pytest.approx(expected)
    assert fit["quote_flow_slope"] == pytest.approx(.002)
    assert fit["remaining_flow_slope_after_beta"] == pytest.approx(.0019)
    assert abs(fit["residual_flow_correlation"]) < 1e-10
    assert fit["fixed_impact_residual_flow_correlation"] > .9
    assert fit["variance_decomposition_error"] < 1e-12
    other = residual_noise_fit(d, beta=.02)
    assert other["sigma_logodds_per_sqrt_second"] == pytest.approx(expected)
    assert residual_noise_fit(d.iloc[:49], beta=0)["fit_status"] == "insufficient_quote_pairs"


def fixture():
    metadata, split_rows, tapes, candle_rows = [], [], {}, {}
    for i in range(6):
        ticker = f"M{i}"; m = market(ticker); metadata.append(m)
        split_rows.append({"ticker": ticker, "event_cluster": m["event_cluster"], "split": "train" if i<5 else "test"})
        times = pd.date_range("2026-07-01", periods=90, freq="2min", tz="UTC")
        tapes[ticker] = tape(times, np.tile([1.25, 2.5, 10], 30), np.tile([1, -1, 1], 30))
        qtimes = pd.date_range("2026-07-01", periods=181, freq="min", tz="UTC")
        candle_rows[ticker] = quotes(qtimes, .5+.02*np.sin(np.arange(181)/9))
    impact = {"beta_units": "log_odds_per_signed_market_contract", "reference_observation": "post_trade",
              "training": {"tickers": [f"M{i}" for i in range(5)]},
              "groups": {"sports": {"status": "calibrated", "scenarios": {"zero": 0, "low": 0, "base": 1e-5, "high": 1e-4}}}}
    return pd.DataFrame(metadata), pd.DataFrame(split_rows), impact, tapes, candle_rows


def test_holdout_metadata_trades_and_quotes_do_not_change_dynamics():
    m, splits, impact, tapes, candle_rows = fixture()
    def tl(ticker):
        assert ticker != "M5"
        return tapes[ticker]
    def ql(ticker):
        assert ticker != "M5"
        return candle_rows[ticker]
    first, artifacts = calibrate(m, splits, impact, tl, ql, min_pairs=10, bootstrap_draws=30)
    m.loc[m.ticker.eq("M5"), "open_time"] = "INVALID"
    m.loc[m.ticker.eq("M5"), "information_class"] = "CHANGED"
    tapes["M5"]["quantity"] = 1e99
    candle_rows["M5"]["yes_bid_close_dollars"] = 1e99
    second, _ = calibrate(m, splits, impact, tl, ql, min_pairs=10, bootstrap_draws=30)
    assert first == second
    assert artifacts["activity"].trades.sum() == 450
    assert artifacts["noise_intervals"].split.eq("train").all()


def test_sampler_count_mean_and_burst_variance():
    rng = np.random.default_rng(44)
    counts = sample_counts(4/300, .5, rng, draws=100_000)
    assert counts.mean() == pytest.approx(4, abs=.06)
    assert counts.var() == pytest.approx(4+.5*4**2, abs=.5)
    poisson = sample_counts(4/300, 0, rng, draws=100_000)
    assert poisson.var() == pytest.approx(4, abs=.08)
    assert sample_counts(0, 1000, rng, draws=10).sum() == 0


def test_saved_mark_bank_driver_and_time_scaled_noise(tmp_path):
    m, splits, impact, tapes, candle_rows = fixture()
    config, artifacts = calibrate(m, splits, impact, lambda t: tapes[t], lambda t: candle_rows[t],
                                 min_pairs=10, bootstrap_draws=30)
    impact_path = tmp_path/"impact.json"; impact_path.write_text(json.dumps(impact))
    bank_path = tmp_path/"marks.npz"; np.savez_compressed(bank_path, **artifacts["marks"])
    config.update(mark_bank_file="marks.npz", mark_bank_sha256=file_hash(bank_path), impact_config_file="impact.json",
                  upstream_sha256={"price_impact_config.json": file_hash(impact_path)})
    path = tmp_path/"config.json"; path.write_text(json.dumps(config))
    rng = np.random.default_rng(9)
    dynamics = MarketDynamics("sports", rng, ticker="M0", config_path=path)
    p = ImpactPriceProcess(.5, dynamics.impact)
    window = dynamics.advance_window(p, .5, rng)
    assert window["duration_seconds"] == 300
    assert window["signed_flow_contracts"] == pytest.approx(np.dot(window["quantities"], window["signs"]))
    assert set(window["quantities"]).issubset({1.25, 2.5, 10})
    draws = dynamics.sample_innovation(rng, draws=20_000)
    assert draws.std() == pytest.approx(dynamics.sigma*np.sqrt(300), rel=.03)
    assert abs(draws.mean()) < .03*dynamics.sigma*np.sqrt(300)
    assert abs(window["log_odds"]-dynamics.impact.beta*window["signed_flow_contracts"]-window["innovation_logodds"]) < 1e-12
    remaining_fraction = 1-60/dynamics.template["lifetime_seconds"]
    assert dynamics.sample_window(remaining_fraction, rng)["duration_seconds"] == pytest.approx(60)
    with pytest.raises(ValueError, match="Life fraction"):
        dynamics.sample_window(1, rng)
    impact_path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        MarketDynamics("sports", rng, config_path=path)
