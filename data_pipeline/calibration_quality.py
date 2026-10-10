"""Paper 2 data eligibility, without fitting any price-impact regression.

Raw trade tapes remain unchanged. Cleaning exclusions and incomplete history
are explicit, and the paper's >=50 active five-minute interval rule is checked
using observations with valid log-odds endpoints.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import safe_filename


BROAD_BASED_SERIES = {
    "KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOREYOY", "KXU3", "KXPAYROLLS",
    "KXFEDDECISION", "KXPCECORE", "KXPCEHEAD", "KXGDP", "GDP", "KXRETAIL",
    "KXUSRETAIL", "KXJOBLESS", "KXJOBLESSCLAIMS", "KXADP", "KXISMPMI",
    "KXISMSERVICES", "KXUSISMSERV", "KXAAAGASW", "KXAAAGASM", "KXAAAGASD",
    "KXGAS", "KXEIACRUDEW", "KXCACPIYOY", "KXUKCPIYOY", "KXJPCPIYOY",
    "KXDECPIPREL", "KXCPIEU", "KXCHCPIYOY",
}

# Reviewed against the saved market descriptions and resolution rules on
# 2026-10-09. These mappings are explicit; category names alone do not suffice.
REVIEWED_SERIES = {
    "KXARMOMINF": ("broad_based", "Reviewed: Argentina national month-on-month inflation statistic"),
    "KXCBDECISIONBRAZIL": ("broad_based", "Reviewed: official Central Bank of Brazil policy-rate decision"),
    "KXH200CHINA": ("single_name", "Reviewed: physical delivery of Nvidia H200 chips; company-specific outcome"),
}


def classify_market(market: dict) -> tuple[str, str]:
    """Conservative, auditable mapping; economics is not automatically broad-based."""
    if market.get("category") == "Sports":
        return "sports", "Kalshi Sports; kept separate from the paper's main two groups"
    if market.get("series_ticker") in REVIEWED_SERIES:
        return REVIEWED_SERIES[market["series_ticker"]]
    if market.get("series_ticker") in BROAD_BASED_SERIES:
        return "broad_based", "Explicit aggregate macroeconomic / US gas-price series allowlist"
    return "needs_review", "No explicit Paper 2 mapping; inspect description and resolution rules"


def event_cluster(market: dict) -> str:
    """Keep related monthly release series together, as well as same-event thresholds.

    This is a conservative grouping aid, not a claim of statistical independence.
    Unknown event identifiers stay missing; they are never invented from a ticker.
    """
    event = market.get("event_ticker") or ""
    series = market.get("series_ticker") or ""
    month = re.search(r"(?:^|-)(\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))(?:-|$)", event)
    family = ({"KXCPI": "US_CPI", "KXCPIYOY": "US_CPI", "KXCPICORE": "US_CPI",
               "KXCPICOREYOY": "US_CPI", "KXU3": "US_JOBS", "KXPAYROLLS": "US_JOBS",
               "KXPCECORE": "US_PCE", "KXPCEHEAD": "US_PCE"}).get(series)
    return f"{family}-{month.group(1)}" if family and month else event


def normalize_trades(raw: pd.DataFrame, ticker: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Normalize legacy/current fields and exclude bad rows and off-book blocks."""
    stats = {"raw_trade_rows": len(raw), "duplicate_trade_ids": 0, "invalid_trade_rows": 0,
             "block_trade_rows": 0, "boundary_trade_rows": 0, "raw_contract_volume": 0.0}
    if raw.empty:
        return pd.DataFrame(columns=["trade_id", "timestamp", "price", "quantity", "sign"]), stats
    d = raw.copy()
    def numeric(primary, legacy, scale=1.0):
        a = pd.to_numeric(d.get(primary, pd.Series(np.nan, index=d.index)), errors="coerce")
        b = pd.to_numeric(d.get(legacy, pd.Series(np.nan, index=d.index)), errors="coerce") / scale
        return a.fillna(b)
    quantity = numeric("count_fp", "count")
    price = numeric("yes_price_dollars", "yes_price", 100)
    no_price = numeric("no_price_dollars", "no_price", 100)
    side = d.get("taker_outcome_side", pd.Series(pd.NA, index=d.index)).replace("", pd.NA)
    side = side.fillna(d.get("taker_side", pd.Series(pd.NA, index=d.index)))
    ts = pd.to_datetime(d.get("created_time", pd.Series(pd.NA, index=d.index)),
                        utc=True, errors="coerce", format="mixed")
    ids = d.get("trade_id", pd.Series(pd.NA, index=d.index))
    missing_id = ids.isna() | ids.eq("")
    duplicates = ids.duplicated(keep="first") & ~missing_id
    valid = (ts.notna() & np.isfinite(price) & price.between(0, 1)
             & np.isfinite(quantity) & quantity.gt(0) & side.isin(["yes", "no"]) & ~missing_id)
    valid &= no_price.isna() | (np.isfinite(no_price) & no_price.between(0, 1)
                                & (price + no_price - 1).abs().le(0.0002))
    if ticker is not None:
        valid &= d.get("ticker", pd.Series("", index=d.index)).eq(ticker)
    block = d.get("is_block_trade", pd.Series(False, index=d.index)).astype(str).str.lower().isin(["true", "1"])
    stats.update(duplicate_trade_ids=int(duplicates.sum()), invalid_trade_rows=int((~valid).sum()),
                 block_trade_rows=int(block.sum()),
                 raw_contract_volume=float(quantity[valid & ~duplicates].sum()))
    keep = valid & ~duplicates & ~block
    normalized = pd.DataFrame({"trade_id": ids, "timestamp": ts, "price": price,
                               "quantity": quantity, "sign": side.map({"yes": 1, "no": -1})})[keep]
    normalized = normalized.sort_values(["timestamp", "trade_id"], kind="stable").reset_index(drop=True)
    stats["boundary_trade_rows"] = int(normalized.price.isin([0, 1]).sum())
    return normalized, stats


def profile_contract(market: dict, raw: pd.DataFrame, *, min_intervals: int = 50,
                     tape_complete: bool | None = None, error: str | None = None) -> dict:
    d, stats = normalize_trades(raw, market.get("ticker"))
    classification, reason = classify_market(market)
    result = {"ticker": market.get("ticker", ""), "event_ticker": market.get("event_ticker") or "",
              "event_cluster": event_cluster(market), "series_ticker": market.get("series_ticker", ""),
              "category": market.get("category", ""), "information_class": classification,
              "classification_reason": reason, "settlement_result": market.get("result", ""),
              "tape_complete": tape_complete, "collection_error": error, **stats,
              "usable_trade_rows": len(d), "active_5min_intervals": 0,
              "usable_logodds_intervals": 0, "multi_trade_intervals": 0,
              "nonzero_price_change_intervals": 0, "unique_trade_prices": 0,
              "single_trade_interval_share": None, "first_trade_ts": None, "last_trade_ts": None,
              "observed_contract_volume": float(d.quantity.sum()), "volume_coverage_ratio": None,
              "outside_lifetime_rows": 0, "metadata_complete": False}
    try:
        reported_volume = float(market.get("volume_fp", market.get("volume", "nan")))
    except (TypeError, ValueError):
        reported_volume = float("nan")
    result["reported_contract_volume"] = reported_volume if np.isfinite(reported_volume) else None
    if np.isfinite(reported_volume) and reported_volume > 0:
        result["volume_coverage_ratio"] = stats["raw_contract_volume"] / reported_volume
    outside = pd.Series(False, index=d.index)
    opened = pd.to_datetime(market.get("open_time"), utc=True, errors="coerce")
    closed = pd.to_datetime(market.get("close_time"), utc=True, errors="coerce")
    if pd.notna(opened):
        outside |= d.timestamp < opened
    if pd.notna(closed):
        outside |= d.timestamp > closed
    result["outside_lifetime_rows"] = int(outside.sum())
    d = d[~outside]
    result["usable_trade_rows"] = len(d)
    result["observed_contract_volume"] = float(d.quantity.sum())
    if not d.empty:
        buckets = d.groupby(d.timestamp.dt.floor("5min")).agg(
            open=("price", "first"), close=("price", "last"), trades=("price", "size"))
        interior_endpoints = buckets.open.between(0, 1, inclusive="neither") & buckets.close.between(0, 1, inclusive="neither")
        result.update(active_5min_intervals=len(buckets), usable_logodds_intervals=int(interior_endpoints.sum()),
                      multi_trade_intervals=int(buckets.trades.ge(2).sum()),
                      nonzero_price_change_intervals=int((buckets.close-buckets.open).abs().gt(1e-12).sum()),
                      unique_trade_prices=int(d.price.nunique()),
                      single_trade_interval_share=float(buckets.trades.eq(1).mean()),
                      first_trade_ts=d.timestamp.min().isoformat(), last_trade_ts=d.timestamp.max().isoformat())
    result["metadata_complete"] = bool(result["event_ticker"] and market.get("rules_primary")
                                       and pd.notna(opened) and pd.notna(closed)
                                       and closed > opened and market.get("result") in {"yes", "no"})
    reasons = []
    if result["usable_logodds_intervals"] < min_intervals:
        reasons.append(f"fewer_than_{min_intervals}_usable_5min_intervals")
    if not result["metadata_complete"]:
        reasons.append("missing_event_rules_or_lifetime_metadata")
    if tape_complete is not True:
        reasons.append("trade_history_not_verified_complete")
    if stats["invalid_trade_rows"]:
        reasons.append("invalid_trade_fields")
    if result["outside_lifetime_rows"]:
        reasons.append("trades_outside_market_lifetime")
    if error:
        reasons.append("collection_failed")
    # Metadata volume is a reconciliation diagnostic; a substantial shortfall is
    # not silently accepted even if the pagination terminated successfully.
    ratio = result["volume_coverage_ratio"]
    if ratio is not None and ratio < 0.95:
        reasons.append("observed_volume_below_95pct_of_metadata_volume")
    if ratio is not None and ratio > 1.05:
        reasons.append("observed_volume_above_105pct_of_metadata_volume")
    result["paper_interval_threshold_met"] = result["usable_logodds_intervals"] >= min_intervals
    result["calibration_eligible"] = not reasons
    result["exclusion_reasons"] = "|".join(reasons)
    return result


def write_report(rows: list[dict], output_dir: Path, *, target_total: int | None = None,
                 min_intervals: int = 50) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "contract_quality.csv", index=False)
    def counts(column, eligible=False):
        if df.empty or column not in df:
            return {}
        subset = df[df.calibration_eligible] if eligible else df
        return {str(k): int(v) for k, v in subset[column].value_counts().items()}
    eligible = df[df.calibration_eligible] if not df.empty else df
    selected = df[df.selected_for_sample & df.calibration_eligible] if "selected_for_sample" in df else eligible
    summary = {
        "contracts_audited": len(df), "total_raw_trades": int(df.raw_trade_rows.sum()) if not df.empty else 0,
        "paper_interval_threshold_met": int(df.paper_interval_threshold_met.sum()) if not df.empty else 0,
        "calibration_eligible": len(eligible), "selected_contracts": len(selected),
        "selected_events": int(selected.event_ticker.nunique()) if not selected.empty else 0,
        "selected_event_clusters": int(selected.event_cluster.nunique()) if not selected.empty else 0,
        "selected_series": int(selected.series_ticker.nunique()) if not selected.empty else 0,
        "selected_raw_trades": int(selected.raw_trade_rows.sum()) if not selected.empty else 0,
        "selected_candle_rows": int(selected.candle_rows.sum()) if "candle_rows" in selected else 0,
        "selected_with_candles": int(selected.candles_ok.eq(True).sum()) if "candles_ok" in selected else 0,
        "selected_but_ineligible": int((df.selected_for_sample & ~df.calibration_eligible).sum())
                                   if "selected_for_sample" in df else 0,
        "by_category": counts("category"), "eligible_by_category": counts("category", True),
        "eligible_by_information_class": counts("information_class", True),
        "target_total": target_total, "target_reached": target_total is not None and len(selected) >= target_total,
        "min_usable_5min_intervals": min_intervals,
        "note": "Eligibility is data readiness, not proof of independent events or reliable lambda estimation. "
                "Sports is separate; needs_review markets must not be treated as broad-based.",
    }
    (output_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def audit_existing(root: Path, report_dir: Path, min_intervals: int = 50) -> dict:
    markets = pd.read_parquet(root / "markets" / "kalshi_selected_contracts.parquet")
    selected_tickers = set(markets.ticker) if "ticker" in markets else set()
    metadata_files = sorted((root / "market_metadata").glob("*.json"))
    # New samples retain all attempted candidates, including rejects. Legacy
    # samples have only the selected parquet and no evidence of complete history.
    records = [json.loads(f.read_text()) for f in metadata_files] if metadata_files else markets.to_dict("records")
    old_report = report_dir / "contract_quality.csv"
    previous_rows = {r["ticker"]: r for r in pd.read_csv(old_report).to_dict("records")} if old_report.exists() else {}
    rows = []
    for market in records:
        f = root / "trades" / f"{safe_filename(market['ticker'])}.parquet"
        raw = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        status_file = root / "collection_status" / f"{safe_filename(market['ticker'])}.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        row = profile_contract(market, raw, min_intervals=min_intervals,
                               tape_complete=status.get("tape_complete"),
                               error=status.get("error") or (None if f.exists() else "missing_trade_file"))
        if metadata_files:
            row["selected_for_sample"] = market["ticker"] in selected_tickers
            for key in ["candle_rows", "candles_ok", "candle_error"]:
                if key in previous_rows.get(market["ticker"], {}):
                    row[key] = previous_rows[market["ticker"]][key]
        rows.append(row)
    if metadata_files and not markets.empty:
        # Keep the selected metadata and quality report in agreement after an
        # explicit classification review. Do not silently remove failed rows.
        by_ticker = {r["ticker"]: r for r in rows}
        for field in ["information_class", "classification_reason", "event_cluster"]:
            markets[field] = markets.ticker.map(lambda t: by_ticker[t][field])
        markets.to_parquet(root / "markets" / "kalshi_selected_contracts.parquet", index=False)
    settings_file = root / "settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    summary_file = report_dir / "quality_summary.json"
    previous_summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    summary = write_report(rows, report_dir, min_intervals=min_intervals, target_total=settings.get("target_total"))
    summary = {**previous_summary, **summary}
    quotas = summary.get("category_quotas", {})
    if quotas:
        summary["balanced_target_reached"] = all(summary.get("eligible_by_category", {}).get(k, 0) >= v
                                               for k, v in quotas.items())
    summary.setdefault("balance_policy", "equal_quotas")
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit saved tapes for Paper 2 readiness; no regressions")
    parser.add_argument("--input-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=config.CALIBRATION_OUTPUT_DIR / "original_sample_audit")
    parser.add_argument("--min-intervals", type=int, default=config.MIN_ACTIVE_5MIN_INTERVALS)
    args = parser.parse_args()
    print(json.dumps(audit_existing(args.input_dir, args.report_dir, args.min_intervals), indent=2))
