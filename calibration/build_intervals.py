"""Build observed five-minute trade buckets for Bartlett & O'Hara equation (1).

Prices are YES probabilities; signed volume is in (possibly fractional)
contracts, +YES taker / -NO taker. Opening/closing prices are the first/last
trades *inside* the same UTC bucket. Empty buckets are not fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data_pipeline"))
from calibration_quality import normalize_trades, profile_contract
from utils import safe_filename

DEFAULT_INPUT = PROJECT_ROOT / "data_pipeline" / "output" / "paper2"
DEFAULT_OUTPUT = DEFAULT_INPUT / "calibration"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aggregate_trades(trades: pd.DataFrame, market: dict) -> pd.DataFrame:
    """Aggregate a normalized, chronological tape; retain single-trade buckets.

    Exact timestamp ties use the normalizer's deterministic trade-ID ordering.
    Endpoints with multiple possible prices are flagged for a separate sensitivity
    run, rather than treating a UUID ordering as verified execution sequence.
    """
    if trades.empty:
        raise ValueError(f"No usable trades for {market.get('ticker')}")
    d = trades.sort_values(["timestamp", "trade_id"], kind="stable").copy()
    if not d.price.between(0, 1).all() or not d.quantity.gt(0).all() or not d.sign.isin([-1, 1]).all():
        raise ValueError("Invalid normalized price, quantity, or taker sign")
    d["bucket_start"] = d.timestamp.dt.floor("5min")
    d["buy_contracts"] = d.quantity.where(d.sign.eq(1), 0.0)
    d["sell_contracts"] = d.quantity.where(d.sign.eq(-1), 0.0)
    d["signed_contracts"] = d.sign * d.quantity
    d["price_quantity"] = d.price * d.quantity
    grouped = d.groupby("bucket_start", sort=True)
    b = grouped.agg(first_trade_ts=("timestamp", "first"), last_trade_ts=("timestamp", "last"),
                    n_trades=("trade_id", "size"), buy_contracts=("buy_contracts", "sum"),
                    sell_contracts=("sell_contracts", "sum"), total_contracts=("quantity", "sum"),
                    signed_flow_contracts=("signed_contracts", "sum"),
                    open_price=("price", "first"), close_price=("price", "last"),
                    high_price=("price", "max"), low_price=("price", "min"),
                    price_quantity_sum=("price_quantity", "sum")).reset_index()
    b["bucket_end"] = b.bucket_start + pd.Timedelta(minutes=5)
    b["vwap_price"] = b.pop("price_quantity_sum") / b.total_contracts
    b["signed_imbalance"] = b.signed_flow_contracts / b.total_contracts
    b["delta_price_dollars"] = b.close_price - b.open_price
    b["delta_price_cents"] = 100 * b.delta_price_dollars
    b["valid_logodds"] = b.open_price.between(0, 1, inclusive="neither") & b.close_price.between(0, 1, inclusive="neither")
    # Boundaries are recorded, not replaced with arbitrary epsilon probabilities.
    b["delta_logodds"] = np.nan
    valid = b.valid_logodds
    b.loc[valid, "delta_logodds"] = (np.log(b.loc[valid, "close_price"])
        - np.log1p(-b.loc[valid, "close_price"]) - np.log(b.loc[valid, "open_price"])
        + np.log1p(-b.loc[valid, "open_price"]))
    b["interior_endpoints_30_70"] = b.open_price.between(0.3, 0.7) & b.close_price.between(0.3, 0.7)
    # Multiple fills at one recorded instant need not have a recoverable sequence.
    endpoint_prices = d.groupby(["bucket_start", "timestamp"]).price.nunique()
    first_keys = pd.MultiIndex.from_arrays([b.bucket_start, b.first_trade_ts])
    last_keys = pd.MultiIndex.from_arrays([b.bucket_start, b.last_trade_ts])
    b["ambiguous_open_price"] = endpoint_prices.reindex(first_keys).to_numpy() > 1
    b["ambiguous_close_price"] = endpoint_prices.reindex(last_keys).to_numpy() > 1
    b["unambiguous_endpoints"] = ~(b.ambiguous_open_price | b.ambiguous_close_price)
    # Paper footnote 20: elapsed time from first trade to the timestamp at which
    # cumulative volume first reaches 99%; take the first 80% of that duration.
    volume_99_ts = d.iloc[np.searchsorted(d.quantity.cumsum().to_numpy(), 0.99*d.quantity.sum())].timestamp
    first = d.timestamp.iloc[0]
    early_cutoff = first + 0.8 * (volume_99_ts-first)
    b["early_80pct_life"] = b.last_trade_ts.le(early_cutoff)
    b["volume_99_ts"] = volume_99_ts
    b["early_80pct_cutoff"] = early_cutoff
    b["single_trade_bucket"] = b.n_trades.eq(1)
    opened = pd.to_datetime(market["open_time"], utc=True)
    closed = pd.to_datetime(market["close_time"], utc=True)
    b["partial_lifetime_bucket"] = b.bucket_start.lt(opened) | b.bucket_end.gt(closed)
    for column in ["ticker", "event_ticker", "event_cluster", "series_ticker", "category", "information_class", "result"]:
        b[column if column != "result" else "settlement_result"] = market.get(column, "")
    if int(b.n_trades.sum()) != len(d) or not np.isclose(b.total_contracts.sum(), d.quantity.sum(), rtol=1e-10):
        raise AssertionError("Trade count/quantity changed during aggregation")
    if not np.allclose(b.buy_contracts-b.sell_contracts, b.signed_flow_contracts, rtol=1e-10, atol=1e-8):
        raise AssertionError("Signed-flow direction or quantity mismatch")
    return b


def assign_event_splits(markets: pd.DataFrame) -> pd.DataFrame:
    """60/20/20 split by completed-event chronology; related clusters stay together."""
    if markets.event_cluster.isna().any() or markets.event_cluster.eq("").any():
        raise ValueError("Missing event cluster; cannot make event-preserving splits")
    d = markets[["ticker", "event_cluster", "close_time"]].copy()
    d["close_time"] = pd.to_datetime(d.close_time, utc=True, format="mixed")
    events = d.groupby("event_cluster", as_index=False).close_time.max().sort_values(["close_time", "event_cluster"])
    n = len(events)
    train_end, validation_end = int(np.floor(0.6*n)), int(np.floor(0.8*n))
    events["split"] = ["train" if i < train_end else "validation" if i < validation_end else "test" for i in range(n)]
    result = d.merge(events[["event_cluster", "split"]], on="event_cluster", validate="many_to_one")
    return result.sort_values(["close_time", "ticker"]).reset_index(drop=True)


def build_dataset(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> tuple[pd.DataFrame, dict]:
    metadata = input_dir / "markets" / "kalshi_selected_contracts.parquet"
    markets = pd.read_parquet(metadata)
    if markets.empty or markets.ticker.duplicated().any():
        raise ValueError("Selected metadata must contain unique contracts")
    splits = assign_event_splits(markets)
    split_lookup = splits.set_index("ticker").split.to_dict()
    (output_dir / "intervals").mkdir(parents=True, exist_ok=True)
    combined, audit, hashes = [], [], {str(metadata.relative_to(input_dir)): file_hash(metadata)}
    if (input_dir / "settings.json").exists():
        hashes["settings.json"] = file_hash(input_dir / "settings.json")
    for market in markets.to_dict("records"):
        fname = safe_filename(market["ticker"])
        tape_path = input_dir / "trades" / f"{fname}.parquet"
        status_path = input_dir / "collection_status" / f"{fname}.json"
        status = json.loads(status_path.read_text())
        raw = pd.read_parquet(tape_path)
        quality = profile_contract(market, raw, tape_complete=status.get("tape_complete"), error=status.get("error"))
        if not quality["calibration_eligible"]:
            raise ValueError(f"Selected contract is no longer eligible: {market['ticker']}: {quality['exclusion_reasons']}")
        clean, exclusion_counts = normalize_trades(raw, market["ticker"])
        buckets = aggregate_trades(clean, market)
        buckets["split"] = split_lookup[market["ticker"]]
        buckets.to_parquet(output_dir / "intervals" / f"{fname}.parquet", index=False)
        combined.append(buckets)
        audit.append({"ticker": market["ticker"], **exclusion_counts,
                      "normalized_trade_rows": len(clean), "active_intervals": len(buckets),
                      "valid_logodds_intervals": int(buckets.valid_logodds.sum()),
                      "single_trade_intervals": int(buckets.single_trade_bucket.sum()),
                      "ambiguous_endpoint_intervals": int((~buckets.unambiguous_endpoints).sum()),
                      "total_contracts": float(buckets.total_contracts.sum()),
                      "buy_contracts": float(buckets.buy_contracts.sum()), "sell_contracts": float(buckets.sell_contracts.sum())})
        hashes[str(tape_path.relative_to(input_dir))] = file_hash(tape_path)
        hashes[str(status_path.relative_to(input_dir))] = file_hash(status_path)
    dataset = pd.concat(combined, ignore_index=True).sort_values(["ticker", "bucket_start"]).reset_index(drop=True)
    if dataset.duplicated(["ticker", "bucket_start"]).any():
        raise AssertionError("Duplicate market/bucket rows")
    dataset.to_parquet(output_dir / "five_minute_intervals.parquet", index=False)
    dataset.to_csv(output_dir / "five_minute_intervals.csv", index=False)
    splits.to_csv(output_dir / "event_splits.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "aggregation_audit.csv", index=False)
    manifest = {"contracts": len(markets), "raw_trades": int(sum(r["raw_trade_rows"] for r in audit)),
                "normalized_trades": int(sum(r["normalized_trade_rows"] for r in audit)),
                "active_intervals": len(dataset), "valid_logodds_intervals": int(dataset.valid_logodds.sum()),
                "single_trade_intervals": int(dataset.single_trade_bucket.sum()),
                "ambiguous_endpoint_intervals": int((~dataset.unambiguous_endpoints).sum()),
                "split_contract_counts": {k: int(v) for k, v in splits.split.value_counts().items()},
                "trade_start_utc": dataset.first_trade_ts.min().isoformat(),
                "trade_end_utc": dataset.last_trade_ts.max().isoformat(), "input_sha256": hashes,
                "source_sha256": {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
                    [Path(__file__), PROJECT_ROOT / "data_pipeline" / "calibration_quality.py",
                     PROJECT_ROOT / "data_pipeline" / "utils.py"]},
                "bucket_definition": "UTC [start,start+5min); active buckets only; signed flow in contracts",
                "price_change": "logit(last YES trade price)-logit(first YES trade price) within same bucket",
                "ties": "deterministic trade-ID ordering, with ambiguous endpoints flagged",
                "early_life_filter": "first 80% of elapsed time from first trade to the 99%-cumulative-volume timestamp",
                "split_note": "Chronological completed-event split; histories can overlap in calendar time. "
                              "Train-only summaries are provided for later calibration; all-sample fits are exploratory."}
    (output_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return dataset, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _, manifest = build_dataset(args.input_dir, args.output_dir)
    print(json.dumps({k: v for k, v in manifest.items() if k != "input_sha256"}, indent=2))
