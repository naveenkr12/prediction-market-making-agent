"""Collect an event-diverse, liquidity-screened Paper 2 sample.

Run from the project root:
    python data_pipeline/collect_calibration.py --end-date 2026-10-09 --target-total 60

The original data_pipeline/output sample and HJB results are never overwritten.
Candidate metadata, complete raw tapes, rejection reasons, and settings are
checkpointed under output/paper2. Re-running with identical settings resumes.
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

import pandas as pd

import config
from calibration_quality import profile_contract, write_report, event_cluster
from contract_selection import discover_calibration_candidates, market_to_contract
from kalshi_client import KalshiClient, KalshiAPIError
from utils import iso_to_unix, safe_filename

log = logging.getLogger("collect_calibration")


def save_json(path: Path, data) -> None:
    """Replace a checkpoint atomically so interrupted runs can be resumed."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def collection_window(end_date: str | None, lookback_days: int) -> tuple[int, int]:
    if lookback_days < 1:
        raise ValueError("lookback-days must be positive")
    # Explicit end-date makes a run reproducible irrespective of the host clock.
    end = (datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
           + timedelta(days=1) - timedelta(seconds=1)) if end_date else datetime.now(timezone.utc)
    return int((end - timedelta(days=lookback_days)).timestamp()), int(end.timestamp())


def run(args: argparse.Namespace, client: KalshiClient | None = None) -> dict:
    if (args.target_total < 1 or args.max_candidates < 1 or args.max_per_event < 1
            or args.max_per_series < 1 or args.min_intervals < 1 or args.min_volume < 0
            or args.max_trade_pages < 1 or args.max_series_per_category < 1 or args.candle_lookback_days < 0):
        raise ValueError("Targets, candidate budget, and sampling caps must be positive")
    start_ts, end_ts = collection_window(args.end_date, args.lookback_days)
    root = args.output_dir
    for folder in ["markets", "market_metadata", "trades", "candlesticks", "collection_status", "reports", "logs"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    settings = {"start_ts": start_ts, "end_ts": end_ts, "lookback_days": args.lookback_days,
                "target_total": args.target_total, "min_intervals": args.min_intervals,
                "min_reported_volume": args.min_volume, "max_per_event": args.max_per_event,
                "max_per_series": args.max_per_series, "max_series_per_category": args.max_series_per_category,
                "max_trade_pages": args.max_trade_pages, "seed": args.seed,
                "candle_lookback_days": args.candle_lookback_days,
                "sampling": "preferred series then seeded discovery; series round robin; volume closest to 50000",
                "categories": config.KALSHI_CATEGORIES}
    settings_path = root / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Output already has different settings. Use a new --output-dir for a different sample.")
    save_json(settings_path, settings)
    client = client or KalshiClient()
    cutoffs = client.get_cutoffs()
    save_json(root / "logs" / "archive_cutoffs.json", cutoffs)
    candidates_file = root / "markets" / "candidate_metadata.json"
    if candidates_file.exists():
        candidates = json.loads(candidates_file.read_text())
        if getattr(args, "expand_discovery", False):
            summary_file = root / "reports" / "quality_summary.json"
            previous = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            shortfalls = previous.get("quota_shortfall", {k: 1 for k in config.KALSHI_CATEGORIES})
            categories_to_expand = [k for k, v in shortfalls.items() if v > 0]
            if categories_to_expand:
                expanded, discovery = discover_calibration_candidates(
                    client, start_ts, end_ts, args.max_series_per_category * 2, args.min_volume, args.seed,
                    root / "markets" / "discovery_cache", categories=categories_to_expand)
                existing_tickers = {m["ticker"] for m in candidates}
                added = [m for m in expanded if m["ticker"] not in existing_tickers]
                candidates.extend(added)
                save_json(candidates_file, candidates)
                discovery_file = root / "logs" / "discovery.json"
                old_discovery = json.loads(discovery_file.read_text()) if discovery_file.exists() else []
                for record in discovery:
                    record["phase"] = "expanded_discovery"
                save_json(discovery_file, old_discovery + discovery)
                expansion_file = root / "logs" / "discovery_expansions.json"
                expansions = json.loads(expansion_file.read_text()) if expansion_file.exists() else []
                expansions.append({"expanded_at": datetime.now(timezone.utc).isoformat(),
                                   "categories": categories_to_expand, "max_series_per_category": args.max_series_per_category * 2,
                                   "new_candidates": len(added)})
                save_json(expansion_file, expansions)
                log.info("Expanded discovery: %d additional candidates for %s", len(added), categories_to_expand)
    else:
        candidates, discovery = discover_calibration_candidates(
            client, start_ts, end_ts, args.max_series_per_category, args.min_volume, args.seed,
            root / "markets" / "discovery_cache")
        save_json(candidates_file, candidates)
        save_json(root / "logs" / "discovery.json", discovery)
    categories = config.KALSHI_CATEGORIES
    quotas = {category: args.target_total // len(categories) + (i < args.target_total % len(categories))
              for i, category in enumerate(categories)}
    accepted, rows = [], []
    category_counts, series_counts, event_counts = Counter(), Counter(), Counter()
    attempts = 0

    def checkpoint():
        meta = pd.DataFrame(accepted)
        meta.to_parquet(root / "markets" / "kalshi_selected_contracts.parquet", index=False)
        summary = write_report(rows, root / "reports", target_total=args.target_total,
                               min_intervals=args.min_intervals)
        summary.update(candidate_pool=len(candidates), attempted_candidates=attempts,
                       selected_by_category=dict(category_counts), category_quotas=quotas,
                       quota_shortfall={k: max(0, v-category_counts[k]) for k, v in quotas.items()},
                       api_requests=client.request_count,
                       discovery_errors=sum(bool(r.get("error")) for r in json.loads(
                           (root / "logs" / "discovery.json").read_text()))
                       if (root / "logs" / "discovery.json").exists() else 0)
        save_json(root / "reports" / "quality_summary.json", summary)
        return summary

    # Balanced quotas are attempted first. An explicitly requested fallback may
    # fill a scarce category's unused slots without relaxing any data-quality cap.
    candidate_rounds = [(True, m) for m in candidates]
    if getattr(args, "allow_unbalanced", False):
        candidate_rounds += [(False, m) for m in candidates]
    attempted_tickers: set[str] = set()
    for respect_quotas, market in candidate_rounds:
        if len(accepted) >= args.target_total or attempts >= args.max_candidates:
            break
        category, series = market["category"], market["series_ticker"]
        cluster = event_cluster(market)
        if ((respect_quotas and category_counts[category] >= quotas[category])
                or series_counts[series] >= args.max_per_series
                or (cluster and event_counts[cluster] >= args.max_per_event)
                or market["ticker"] in attempted_tickers):
            continue
        attempts += 1
        attempted_tickers.add(market["ticker"])
        fname = safe_filename(market["ticker"])
        raw_path = root / "trades" / f"{fname}.parquet"
        status_path = root / "collection_status" / f"{fname}.json"
        save_json(root / "market_metadata" / f"{fname}.json", market)
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        raw = pd.DataFrame()
        if status.get("tape_complete") and raw_path.exists():
            raw = pd.read_parquet(raw_path)
        elif not status or args.retry_failed:
            status = {"ticker": market["ticker"], "tape_complete": False, "error": None,
                      "collected_at": datetime.now(timezone.utc).isoformat(), "archive_cutoffs": cutoffs}
            try:
                opened, closed = iso_to_unix(market.get("open_time", "")), iso_to_unix(market.get("close_time", ""))
                if opened is None or closed is None or opened >= closed:
                    raise ValueError("Missing or invalid market lifetime")
                # Download the full lifetime of markets closing in the selection window.
                # +1 includes the fractional second at the requested closing second.
                tape = client.get_trades(market["ticker"], opened, closed + 1,
                                         include_history=True, max_pages=args.max_trade_pages)
                raw = pd.DataFrame(tape)
                raw.to_parquet(raw_path, index=False)
                status.update(tape_complete=True, query_start_ts=opened, query_end_ts=closed+1,
                              trade_rows=len(raw))
            except (KalshiAPIError, ValueError) as exc:
                status["error"] = str(exc)
            save_json(status_path, status)
        quality = profile_contract(market, raw, min_intervals=args.min_intervals,
                                   tape_complete=status.get("tape_complete", False), error=status.get("error"))
        quality.update(selected_for_sample=False, candle_rows=0, candles_ok=False, candle_error=None)
        if quality["calibration_eligible"]:
            quality["selected_for_sample"] = True
            accepted.append({**dataclasses.asdict(market_to_contract(
                market, {"ticker": series, "title": market.get("series_title", "")}, category)),
                             "metadata_source": market.get("metadata_source", ""),
                             "information_class": quality["information_class"],
                             "classification_reason": quality["classification_reason"],
                             "event_cluster": cluster})
            category_counts[category] += 1
            series_counts[series] += 1
            event_counts[cluster] += 1
            candle_path = root / "candlesticks" / f"{fname}.parquet"
            # Quotes are supplementary. Lambda eligibility does not depend on them.
            if candle_path.exists():
                quality.update(candles_ok=True, candle_rows=len(pd.read_parquet(candle_path)))
            elif args.candle_lookback_days > 0:
                try:
                    closed = iso_to_unix(market["close_time"])
                    opened = iso_to_unix(market["open_time"])
                    candle_start = max(opened, closed - args.candle_lookback_days * 86400)
                    candles = client.get_candlesticks_full(series, market["ticker"], candle_start,
                                                          closed+1, historical=market.get("metadata_source") == "historical")
                    pd.json_normalize(candles, sep="_").to_parquet(candle_path, index=False)
                    quality.update(candles_ok=True, candle_rows=len(candles))
                except KalshiAPIError as exc:
                    quality["candle_error"] = str(exc)
            log.info("ACCEPT %s: %d usable intervals, %d trades; selected %d/%d",
                     market["ticker"], quality["usable_logodds_intervals"], len(raw), len(accepted), args.target_total)
        else:
            log.info("REJECT %s: %s (%d usable intervals)", market["ticker"],
                     quality["exclusion_reasons"], quality["usable_logodds_intervals"])
        rows.append(quality)
        checkpoint()
    summary = checkpoint()
    summary["balance_policy"] = "prefer_equal_allow_shortfall" if getattr(args, "allow_unbalanced", False) else "equal_quotas"
    summary["balanced_target_reached"] = all(category_counts[k] >= v for k, v in quotas.items())
    save_json(root / "reports" / "quality_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=60)
    parser.add_argument("--end-date", help="Inclusive UTC date YYYY-MM-DD; specify for reproducible resume")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=config.CALIBRATION_OUTPUT_DIR)
    parser.add_argument("--min-volume", type=float, default=config.MIN_REPORTED_CONTRACT_VOLUME)
    parser.add_argument("--min-intervals", type=int, default=config.MIN_ACTIVE_5MIN_INTERVALS)
    parser.add_argument("--max-per-event", type=int, default=config.MAX_CONTRACTS_PER_EVENT)
    parser.add_argument("--max-per-series", type=int, default=config.MAX_CONTRACTS_PER_SERIES)
    parser.add_argument("--max-series-per-category", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--max-trade-pages", type=int, default=120)
    parser.add_argument("--candle-lookback-days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--expand-discovery", action="store_true",
                        help="Discover twice as many series for categories below quota; keep eligibility and sampling caps")
    parser.add_argument("--allow-unbalanced", action="store_true",
                        help="After trying equal quotas, fill unused slots from other categories; report the actual split")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = run(args)
    print(json.dumps(summary, indent=2))
    # Make a partial sample explicit to callers/CI rather than silently claiming success.
    raise SystemExit(0 if summary["target_reached"] else 2)


if __name__ == "__main__":
    main()
