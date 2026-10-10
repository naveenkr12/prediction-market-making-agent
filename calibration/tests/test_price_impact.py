"""Economic definitions, inference under gaps, and saved-tape integration."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from build_intervals import aggregate_trades, assign_event_splits, build_dataset
from estimate_lambda import apply_fdr, calendar_hac_covariance, fit_lambda, run
from validate_results import validate


def market(ticker="A", event="EVENT-A"):
    return {"ticker": ticker, "event_ticker": event, "event_cluster": event,
            "series_ticker": "KXCPI", "category": "Economics", "information_class": "broad_based",
            "result": "yes", "open_time": "2026-07-01T00:00:00Z",
            "close_time": "2026-07-02T00:00:00Z", "rules_primary": "Threshold exceeded",
            "volume_fp": "100.00", "status": "finalized", "market_type": "binary"}


def clean_tape(times, prices, quantities, signs):
    return pd.DataFrame({"trade_id": [f"{i:04d}" for i in range(len(times))],
                         "timestamp": pd.to_datetime(times, utc=True, format="mixed"),
                         "price": prices, "quantity": quantities, "sign": signs})


def regression_buckets(n=80, slope=0.002):
    rng = np.random.default_rng(7)
    flow = np.linspace(-100, 100, n)
    noise = rng.normal(0, 0.01, n)
    x = np.column_stack([np.ones(n), flow])
    noise -= x @ np.linalg.lstsq(x, noise, rcond=None)[0]
    return pd.DataFrame({"bucket_start": pd.date_range("2026-07-01", periods=n, freq="5min", tz="UTC"),
                         "signed_flow_contracts": flow, "delta_logodds": 0.003+slope*flow+noise,
                         "delta_price_cents": 0.4+0.02*flow+noise, "valid_logodds": True,
                         "single_trade_bucket": False, "unambiguous_endpoints": True,
                         "interior_endpoints_30_70": True, "early_80pct_life": True})


def test_signed_fractional_flow_utc_and_within_bucket_prices():
    tape = clean_tape(["2026-07-01T00:04:59Z", "2026-07-01T00:00:01Z",
                       "2026-07-01T00:05:00Z", "2026-07-01T00:15:00Z"],
                      [0.6, 0.4, 0.7, 0.8], [2.5, 1.25, 3, 4], [-1, 1, 1, -1])
    b = aggregate_trades(tape, market())
    assert len(b) == 3  # no synthetic 00:10 bucket
    assert b.n_trades.tolist() == [2, 1, 1]
    assert b.signed_flow_contracts.tolist() == [-1.25, 3, -4]
    assert b.total_contracts.tolist() == [3.75, 3, 4]
    assert b.open_price.tolist() == [0.4, 0.7, 0.8]
    assert b.close_price.tolist() == [0.6, 0.7, 0.8]
    assert b.delta_price_cents.tolist() == pytest.approx([20, 0, 0])
    assert b.delta_logodds.tolist() == pytest.approx([logit(0.6)-logit(0.4), 0, 0])
    assert b.bucket_start.iloc[1] == pd.Timestamp("2026-07-01T00:05:00Z")


def test_boundary_prices_are_recorded_without_epsilon_clipping_and_ties_flagged():
    tape = clean_tape(["2026-07-01T00:00:01Z"]*2+["2026-07-01T00:05:01Z"]*2,
                      [0.4, 0.6, 0, 1], [1]*4, [1, -1, 1, 1])
    b = aggregate_trades(tape, market())
    assert b.ambiguous_open_price.all() and b.ambiguous_close_price.all()
    assert b.valid_logodds.tolist() == [True, False]
    assert pd.isna(b.delta_logodds.iloc[1]) and b.delta_price_cents.iloc[1] == 100


def test_early_life_uses_time_to_99pct_volume_and_excludes_straddling_bucket():
    tape = clean_tape(["2026-07-01T00:00:00Z", "2026-07-01T00:07:00Z",
                       "2026-07-01T00:09:00Z", "2026-07-01T00:10:00Z",
                       "2026-07-01T00:20:00Z"], [0.5]*5, [1, 1, 1, 96, 1], [1]*5)
    b = aggregate_trades(tape, market())
    assert b.volume_99_ts.iloc[0] == pd.Timestamp("2026-07-01T00:10:00Z")
    assert b.early_80pct_cutoff.iloc[0] == pd.Timestamp("2026-07-01T00:08:00Z")
    assert b.early_80pct_life.tolist() == [True, False, False, False]


def test_event_split_keeps_related_contracts_together_by_completion():
    rows = [market(str(i), f"E{i}") for i in range(10)]
    for i, m in enumerate(rows):
        m["close_time"] = f"2026-07-{i+2:02d}T00:00:00Z"
    rows.append({**rows[0], "ticker": "RELATED"})
    splits = assign_event_splits(pd.DataFrame(rows))
    assert splits.groupby("event_cluster").split.nunique().max() == 1
    assert splits.drop_duplicates("event_cluster").split.value_counts().to_dict() == {"train": 6, "validation": 2, "test": 2}
    assert splits.set_index("ticker").loc["RELATED", "split"] == "train"


@pytest.mark.parametrize("slope", [0.002, -0.002])
def test_ols_intercept_original_contract_units_and_negative_slopes(slope):
    d = regression_buckets(slope=slope)
    r = fit_lambda(d)
    assert r["fit_status"] == "ok"
    assert r["lambda"] == pytest.approx(slope, abs=1e-14)
    assert r["intercept"] == pytest.approx(0.003, abs=1e-14)
    assert r["lambda_per_1000_contracts"] == pytest.approx(1000*slope)
    assert fit_lambda(d, response="cents")["lambda"] == pytest.approx(0.02)
    if slope < 0:
        assert "negative_lambda" in r["diagnostic_flags"]


def test_unidentified_and_short_fits_are_not_zero_lambda_estimates():
    assert fit_lambda(regression_buckets(49))["fit_status"] == "insufficient_intervals"
    d = regression_buckets()
    d["signed_flow_contracts"] = 10
    assert fit_lambda(d)["fit_status"] == "constant_signed_flow"
    d = regression_buckets()
    d["delta_logodds"] = 0
    r = fit_lambda(d)
    assert r["fit_status"] == "constant_price_change" and r["lambda"] is None
    d = regression_buckets()
    d.loc[:30, "unambiguous_endpoints"] = False
    assert fit_lambda(d, "unambiguous_endpoints")["fit_status"] == "insufficient_intervals"


def test_calendar_hac_matches_statsmodels_for_regular_buckets():
    d = regression_buckets()
    x = np.column_stack([np.ones(len(d)), d.signed_flow_contracts])
    fit = OLS(d.delta_logodds.to_numpy(), x).fit()
    actual = calendar_hac_covariance(x, fit.resid, d.bucket_start, lags=6)
    expected = cov_hac(fit, nlags=6, use_correction=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-15)


def test_calendar_hac_pairs_actual_time_lags_with_gaps():
    positions = np.array([0, 1, 4, 10, 11, 20])
    x = np.column_stack([np.ones(6), [-2, -1, 0, 1, 2, 4]])
    residuals = np.array([0.2, -0.1, 0.3, -0.2, 0.4, -0.3])
    times = pd.Series(pd.Timestamp("2026-07-01", tz="UTC")+pd.to_timedelta(positions*5, unit="min"))
    scores = np.zeros((21, 2))
    scores[positions] = x*residuals[:, None]
    meat = scores.T@scores
    for lag in range(1, 4):
        cross = scores[lag:].T@scores[:-lag]
        meat += (1-lag/4)*(cross+cross.T)
    bread = np.linalg.inv(x.T@x)
    expected = bread@meat@bread * 6/4
    actual = calendar_hac_covariance(x, residuals, times, lags=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-15)
    regular_times = pd.Series(pd.date_range("2026-07-01", periods=6, freq="5min", tz="UTC"))
    assert not np.allclose(actual, calendar_hac_covariance(x, residuals, regular_times, lags=3))


def test_fdr_families_separate_and_train_adjustment_cannot_use_test_pvalues():
    d = pd.DataFrame({"variant": ["full"]*4, "response": ["logodds"]*3+["cents"],
                      "fit_status": "ok", "p_value_calendar_hac": [0.01, 0.02, 0.0001, 0.04],
                      "split": ["train", "train", "test", "test"]})
    first = apply_fdr(d)
    assert first.q_value_calendar_hac_train.iloc[:2].tolist() == pytest.approx([0.02, 0.02])
    assert first.q_value_calendar_hac_all.iloc[3] == pytest.approx(0.04)
    d.loc[2, "p_value_calendar_hac"] = 0.99
    second = apply_fdr(d)
    np.testing.assert_allclose(first.q_value_calendar_hac_train.iloc[:2], second.q_value_calendar_hac_train.iloc[:2])
    assert second.q_value_calendar_hac_train.iloc[2:].isna().all()


def save_verified_input(root):
    for folder in ["markets", "trades", "collection_status"]:
        (root/folder).mkdir(parents=True)
    markets = []
    for ticker in ["A", "B"]:
        m = market(ticker, f"E-{ticker}")
        rows = []
        for i, ts in enumerate(pd.date_range("2026-07-01", periods=60, freq="5min", tz="UTC")):
            price = 0.51 if i%2 == 0 else 0.49
            for j, p in enumerate([0.5, price]):
                rows.append({"ticker": ticker, "trade_id": f"{i:03d}-{j}",
                             "created_time": (ts+pd.Timedelta(seconds=j)).isoformat(),
                             "yes_price_dollars": str(p), "no_price_dollars": str(1-p),
                             "count_fp": str(i%7+1), "taker_outcome_side": "yes" if i%2 == 0 else "no",
                             "is_block_trade": False})
        m["volume_fp"] = str(sum(float(r["count_fp"]) for r in rows))
        markets.append(m)
        pd.DataFrame(rows).to_parquet(root/"trades"/f"{ticker}.parquet", index=False)
        (root/"collection_status"/f"{ticker}.json").write_text(json.dumps({"tape_complete": True}))
    pd.DataFrame(markets).to_parquet(root/"markets"/"kalshi_selected_contracts.parquet", index=False)


def test_saved_verified_tapes_end_to_end_and_reject_changed_verification(tmp_path, monkeypatch):
    import estimate_lambda
    source, target = tmp_path/"input", tmp_path/"results"
    save_verified_input(source)
    monkeypatch.setattr(estimate_lambda, "make_figure", lambda *a: None)
    summary = run(source, target, bootstrap_draws=20)
    b = pd.read_parquet(target/"five_minute_intervals.parquet")
    r = pd.read_csv(target/"lambda_by_contract.csv")
    assert summary["normalized_trades"] == 240 and summary["active_intervals"] == 120
    assert b.n_trades.sum() == 240 and not b.duplicated(["ticker", "bucket_start"]).any()
    assert summary["primary_fit_status"] == {"ok": 2}
    assert len(r) == 2 and r["lambda"].gt(0).all()
    assert validate(source, target)["passed"]
    manifest = json.loads((target/"aggregation_manifest.json").read_text())
    assert len(manifest["input_sha256"]) == 5
    (source/"collection_status"/"A.json").write_text(json.dumps({"tape_complete": False}))
    with pytest.raises(AssertionError, match="Input changed"):
        validate(source, target)
    with pytest.raises(ValueError, match="no longer eligible"):
        build_dataset(source, tmp_path/"changed")
