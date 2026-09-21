"""
Basic data-quality checks on the collected output, per the proposal's
"Data quality and coverage" expected challenge: sparse histories, unit
consistency, and coverage across the requested category/sample size.

Run:  python3 validate.py
Writes: output/logs/validation_report.json and prints a human summary.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

import config
from utils import safe_filename

log = logging.getLogger("validate")


def validate() -> dict:
    report: dict = {"issues": [], "summary": {}}

    markets_path = config.MARKETS_DIR / "kalshi_selected_contracts.parquet"
    if not markets_path.exists():
        report["issues"].append("No contract metadata file found -- run collect.py first.")
        return report

    markets = pd.read_parquet(markets_path)
    report["summary"]["n_contracts_selected"] = len(markets)
    report["summary"]["by_category"] = markets["category"].value_counts().to_dict()
    report["summary"]["by_result"] = markets["result"].value_counts(dropna=False).to_dict()

    n_target = config.TARGET_CONTRACTS_MAX
    if len(markets) < config.TARGET_CONTRACTS_MIN:
        report["issues"].append(
            f"Only {len(markets)} contracts selected, below the target minimum of "
            f"{config.TARGET_CONTRACTS_MIN} -- widen max_series_per_category or the lookback window."
        )

    trades_missing, candles_missing = [], []
    trades_rowcounts, candle_rowcounts = [], []
    price_out_of_bounds = 0
    total_price_checks = 0

    for _, row in markets.iterrows():
        fname = safe_filename(row["ticker"])

        trades_fp = config.TRADES_DIR / f"{fname}.parquet"
        if trades_fp.exists():
            tdf = pd.read_parquet(trades_fp)
            trades_rowcounts.append(len(tdf))
            for col in ("yes_price_dollars", "no_price_dollars"):
                if col in tdf.columns:
                    vals = pd.to_numeric(tdf[col], errors="coerce")
                    total_price_checks += vals.notna().sum()
                    price_out_of_bounds += ((vals < 0) | (vals > 1)).sum()
        else:
            trades_missing.append(row["ticker"])

        candles_fp = config.CANDLES_DIR / f"{fname}.parquet"
        if candles_fp.exists():
            cdf = pd.read_parquet(candles_fp)
            candle_rowcounts.append(len(cdf))
        else:
            candles_missing.append(row["ticker"])

    report["summary"]["contracts_with_trades_file"] = len(markets) - len(trades_missing)
    report["summary"]["contracts_with_candles_file"] = len(markets) - len(candles_missing)
    report["summary"]["median_trade_rows"] = int(pd.Series(trades_rowcounts).median()) if trades_rowcounts else 0
    report["summary"]["median_candle_rows"] = int(pd.Series(candle_rowcounts).median()) if candle_rowcounts else 0
    report["summary"]["contracts_with_zero_trades"] = sum(1 for n in trades_rowcounts if n == 0)
    report["summary"]["price_checks"] = total_price_checks
    report["summary"]["price_out_of_bounds"] = int(price_out_of_bounds)

    if trades_missing:
        report["issues"].append(f"{len(trades_missing)} contracts have no trades file: {trades_missing[:10]}")
    if candles_missing:
        report["issues"].append(f"{len(candles_missing)} contracts have no candlesticks file: {candles_missing[:10]}")
    if price_out_of_bounds:
        report["issues"].append(
            f"{price_out_of_bounds}/{total_price_checks} trade price values fall outside [0,1] dollars -- unit bug."
        )
    zero_trade_share = report["summary"]["contracts_with_zero_trades"] / max(len(markets), 1)
    if zero_trade_share > 0.5:
        report["issues"].append(
            f"{zero_trade_share:.0%} of selected contracts had zero trades in the window -- "
            "sample may be skewed toward illiquid markets; consider filtering by volume."
        )

    return report


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = validate()

    print("\n=== Data validation summary ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print("\n=== Issues ===")
    if not report["issues"]:
        print("  none")
    else:
        for issue in report["issues"]:
            print(f"  - {issue}")

    out_path = config.LOGS_DIR / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
