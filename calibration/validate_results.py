"""Offline audit of saved five-minute buckets and lambda estimation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, normalize_trades, safe_filename


def check_close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True):
        raise AssertionError(label)


def validate(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    manifest = json.loads((output_dir / "aggregation_manifest.json").read_text())
    summary = json.loads((output_dir / "lambda_summary.json").read_text())
    for path, expected in manifest["input_sha256"].items():
        if file_hash(input_dir / path) != expected:
            raise AssertionError(f"Input changed since estimation: {path}")
    for path, expected in summary["source_sha256"].items():
        if file_hash(PROJECT_ROOT / path) != expected:
            raise AssertionError(f"Source changed since estimation: {path}")
    b = pd.read_parquet(output_dir / "five_minute_intervals.parquet")
    estimates = pd.read_parquet(output_dir / "lambda_estimates.parquet")
    markets = pd.read_parquet(input_dir / "markets" / "kalshi_selected_contracts.parquet")
    assert b.ticker.nunique() == len(markets) == manifest["contracts"]
    assert len(b) == manifest["active_intervals"]
    assert b.n_trades.sum() == manifest["normalized_trades"]
    assert not b.duplicated(["ticker", "bucket_start"]).any()
    assert b.groupby("event_cluster").split.nunique().eq(1).all()
    assert (b.first_trade_ts.ge(b.bucket_start) & b.last_trade_ts.lt(b.bucket_end)).all()
    assert b.last_trade_ts.ge(b.first_trade_ts).all()
    assert b.bucket_end.sub(b.bucket_start).eq(pd.Timedelta(minutes=5)).all()
    assert b.bucket_start.eq(b.bucket_start.dt.floor("5min")).all()
    check_close(b.delta_price_cents, 100*(b.close_price-b.open_price), "Cents conversion")
    valid = b.valid_logodds
    expected_logodds = np.log(b.loc[valid, "close_price"]/(1-b.loc[valid, "close_price"])) - np.log(
        b.loc[valid, "open_price"]/(1-b.loc[valid, "open_price"]))
    check_close(b.loc[valid, "delta_logodds"], expected_logodds, "Log-odds endpoints")
    assert b.loc[~valid, "delta_logodds"].isna().all()
    for ticker, buckets in b.groupby("ticker", sort=True):
        tape, _ = normalize_trades(pd.read_parquet(input_dir/"trades"/f"{safe_filename(ticker)}.parquet"), ticker)
        buckets = buckets.sort_values("bucket_start")
        tape = tape.assign(bucket=tape.timestamp.dt.floor("5min"), signed=tape.quantity*tape.sign)
        manual = tape.groupby("bucket").agg(q=("quantity", "sum"), flow=("signed", "sum"),
                                            opening=("price", "first"), closing=("price", "last"))
        assert buckets.bucket_start.tolist() == manual.index.tolist()
        check_close(buckets.total_contracts, manual.q, f"Quantity: {ticker}")
        check_close(buckets.signed_flow_contracts, manual.flow, f"Signed flow: {ticker}")
        check_close(buckets.open_price, manual.opening, f"First price: {ticker}")
        check_close(buckets.close_price, manual.closing, f"Last price: {ticker}")
    verified_fits = 0
    for row in estimates.to_dict("records"):
        d = b[b.ticker.eq(row["ticker"])].copy()
        if row["response"] == "logodds":
            d = d[d.valid_logodds]
        if row["variant"] != "full":
            d = d[d[row["variant"]]]
        response = "delta_logodds" if row["response"] == "logodds" else "delta_price_cents"
        d = d[np.isfinite(d.signed_flow_contracts) & np.isfinite(d[response])].sort_values("bucket_start")
        assert len(d) == row["n_intervals"]
        if row["fit_status"] != "ok":
            assert pd.isna(row["lambda"])
            continue
        # Independent closed-form slope, intercept, HC1 and scalar HAC checks.
        flow, y = d.signed_flow_contracts.to_numpy(), d[response].to_numpy()
        centered = flow-flow.mean()
        sxx = centered@centered
        slope = centered@(y-y.mean()) / sxx
        intercept = y.mean()-slope*flow.mean()
        residuals = y-intercept-slope*flow
        scores = centered*residuals/sxx
        factor = len(d)/(len(d)-2)
        variance_hc1 = factor*(scores@scores)
        score_by_time = {ts: score for ts, score in zip(d.bucket_start, scores)}
        meat = float(scores@scores)
        lags = int(row["hac_lags_5min"])
        for lag in range(1, lags+1):
            pairs = sum(score*score_by_time.get(ts-pd.Timedelta(minutes=5*lag), 0)
                        for ts, score in score_by_time.items())
            meat += 2*(1-lag/(lags+1))*pairs
        check_close(row["lambda"], slope, "OLS slope")
        check_close(row["intercept"], intercept, "OLS intercept")
        check_close(row["se_hc1"], np.sqrt(variance_hc1), "HC1 standard error")
        check_close(row["se_calendar_hac"], np.sqrt(max(factor*meat, 0)), "Calendar HAC standard error")
        verified_fits += 1
    report = {"passed": True, "contracts": len(markets), "active_intervals": len(b),
              "normalized_trades": int(b.n_trades.sum()), "source_hashes_verified": len(summary["source_sha256"]),
              "input_hashes_verified": len(manifest["input_sha256"]), "fit_rows_checked": len(estimates),
              "fitted_slopes_and_standard_errors_verified": verified_fits,
              "checks": ["Saved inputs/source SHA-256 unchanged", "Unique UTC buckets and related-event splits",
                         "Counts, quantities, signed flow and opening/closing prices reconcile to tapes",
                         "Within-bucket cents and exact log-odds price changes",
                         "Independent closed-form OLS slopes/intercepts and HC1/calendar HAC standard errors"]}
    (output_dir/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.input_dir, args.output_dir), indent=2))
