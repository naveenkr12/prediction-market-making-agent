"""
Orchestrates the full Kalshi collection run:

  1. select standalone binary contracts (contract_selection.select_contracts)
  2. for each contract: pull its full trade tape + minute candlesticks
  3. write everything to parquet under output/{markets,trades,candlesticks}/
  4. write a run log (output/logs/run_<timestamp>.json) with per-contract
     success/failure so a partial run is easy to diagnose and resume.

Run:  python3 collect.py [--target-total N] [--max-series-per-category N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd

import config
from contract_selection import select_contracts
from kalshi_client import KalshiClient, KalshiAPIError
from utils import iso_to_unix, safe_filename

log = logging.getLogger("collect")


def collect_one_contract(client: KalshiClient, contract) -> dict:
    """Fetch trades + candlesticks for one contract. Returns a status record."""
    ticker = contract.ticker
    fname = safe_filename(ticker)
    open_ts = iso_to_unix(contract.open_time) or config.START_TS
    close_ts = iso_to_unix(contract.close_time) or config.END_TS

    record = {"ticker": ticker, "series_ticker": contract.series_ticker,
              "category": contract.category, "trades_rows": 0, "candle_rows": 0,
              "trades_ok": False, "candles_ok": False, "error": None}

    # -- trades ---------------------------------------------------------
    try:
        trades = client.get_trades(ticker, min_ts=open_ts, max_ts=close_ts)
        if trades:
            df = pd.DataFrame(trades)
            df.to_parquet(config.TRADES_DIR / f"{fname}.parquet", index=False)
        record["trades_rows"] = len(trades)
        record["trades_ok"] = True
    except KalshiAPIError as exc:
        record["error"] = f"trades: {exc}"
        log.warning("trades failed for %s: %s", ticker, exc)

    # -- candlesticks -----------------------------------------------------
    candle_start_ts = open_ts
    if config.CANDLESTICK_LOOKBACK_DAYS is not None:
        candle_start_ts = max(open_ts, close_ts - config.CANDLESTICK_LOOKBACK_DAYS * 86400)

    try:
        candles = client.get_candlesticks_full(
            series_ticker=contract.series_ticker,
            ticker=ticker,
            start_ts=candle_start_ts,
            end_ts=close_ts,
            period_interval=config.CANDLESTICK_PERIOD_INTERVAL,
        )
        if candles:
            df = pd.json_normalize(candles, sep="_")
            df.to_parquet(config.CANDLES_DIR / f"{fname}.parquet", index=False)
        record["candle_rows"] = len(candles)
        record["candles_ok"] = True
    except KalshiAPIError as exc:
        prev = record["error"]
        record["error"] = f"{prev}; candles: {exc}" if prev else f"candles: {exc}"
        log.warning("candles failed for %s: %s", ticker, exc)

    return record


def run(target_total: int, max_series_per_category: int) -> None:
    client = KalshiClient()

    log.info("Selecting up to %d standalone binary contracts across %s ...",
              target_total, config.KALSHI_CATEGORIES)
    contracts = select_contracts(
        client, target_total=target_total, max_series_per_category=max_series_per_category
    )
    log.info("Selected %d contracts.", len(contracts))

    markets_df = pd.DataFrame([dataclasses.asdict(c) for c in contracts])
    markets_path = config.MARKETS_DIR / "kalshi_selected_contracts.parquet"
    markets_df.to_parquet(markets_path, index=False)
    log.info("Wrote contract metadata -> %s", markets_path)

    run_log = []
    t0 = time.time()
    for i, contract in enumerate(contracts, 1):
        log.info("[%d/%d] collecting %s (%s)", i, len(contracts), contract.ticker, contract.category)
        rec = collect_one_contract(client, contract)
        run_log.append(rec)

    elapsed = time.time() - t0
    ok_trades = sum(r["trades_ok"] for r in run_log)
    ok_candles = sum(r["candles_ok"] for r in run_log)
    log.info(
        "Done in %.1fs. trades ok: %d/%d, candlesticks ok: %d/%d",
        elapsed, ok_trades, len(run_log), ok_candles, len(run_log),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.LOGS_DIR / f"run_{stamp}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "started_at": stamp,
                "elapsed_s": elapsed,
                "target_total": target_total,
                "n_selected": len(contracts),
                "trades_ok": ok_trades,
                "candles_ok": ok_candles,
                "records": run_log,
            },
            f,
            indent=2,
        )
    log.info("Run log -> %s", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Kalshi contract data for the capstone.")
    parser.add_argument("--target-total", type=int, default=config.TARGET_CONTRACTS_MAX)
    parser.add_argument("--max-series-per-category", type=int, default=400)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args.target_total, args.max_series_per_category)
