"""
Collect a dense, labelled Kalshi sample for RL training.

For each category in rl_config.CATEGORIES:
  1. list settled binary markets of the configured series that closed in the
     lookback window (chunked by close time so dense series such as hourly
     crypto never hit the pagination cap)
  2. keep one market per event (the most-traded one), rank by volume, cap
     per series, round-robin across series
  3. download the trade tape and minute candles for the episode window
     (final `episode_hours` before close)
  4. accept only contracts passing the density gates; record every decision

Raw files keep Kalshi's own schema, so calibration_quality.normalize_trades,
the paper2 calibration code and BookSnapshot all read them unchanged.
Re-running resumes from the per-contract status files.

Run from data_pipeline/:
    python3 rl_collect_kalshi.py                         # all four categories
    python3 rl_collect_kalshi.py --categories Sports --target 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import rl_config as R
from calibration_quality import normalize_trades
from kalshi_client import KalshiClient, KalshiAPIError
from utils import iso_to_unix, safe_filename

log = logging.getLogger("rl_collect_kalshi")

SETTLED = {"settled", "finalized"}
META_FIELDS = ["ticker", "event_ticker", "title", "subtitle", "yes_sub_title", "no_sub_title",
               "rules_primary", "open_time", "close_time", "expected_expiration_time",
               "settlement_ts", "status", "result", "market_type", "volume_fp",
               "open_interest_fp", "price_level_structure", "notional_value_dollars"]


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)


def list_markets_chunked(client: KalshiClient, series: str, start_ts: int, end_ts: int,
                         historical: bool, chunk_s: int = 7 * 86400) -> list[dict]:
    """Close-time chunks; a chunk that overflows the page cap is split in half."""
    if historical:
        # The archive endpoint ignores close-time filters, so chunking would
        # re-read the whole archive per chunk. Read it once; filter locally.
        try:
            return client.list_markets(series_ticker=series, min_close_ts=start_ts, max_close_ts=end_ts,
                                       mve_filter="exclude", historical=True, max_pages=300)
        except KalshiAPIError as exc:
            log.warning("archive listing for %s skipped: %s", series, exc)
            return []
    out: dict[str, dict] = {}
    stack = [(start_ts, end_ts)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo > chunk_s:
            mid = lo + chunk_s
            stack += [(mid, hi), (lo, mid)]
            continue
        try:
            for m in client.list_markets(series_ticker=series, min_close_ts=lo, max_close_ts=hi,
                                         mve_filter="exclude", historical=historical, max_pages=10):
                out[m["ticker"]] = m
        except KalshiAPIError as exc:
            if "Pagination limit" in str(exc) and hi - lo > 3600:
                half = (lo + hi) // 2
                stack += [(half, hi), (lo, half)]
            else:
                raise
    return list(out.values())


def discover(client: KalshiClient, category: str, start_ts: int, end_ts: int, cache: Path) -> list[dict]:
    spec = R.CATEGORIES[category]
    available = {}
    # Kalshi files some series elsewhere (WTI oil is "Commodities"); the
    # dataset's category label is ours, the series' own category is logged.
    for kalshi_category in [category] + spec.get("kalshi_extra_categories", []):
        available.update({s["ticker"]: s for s in client.list_series(kalshi_category)})
    cutoff = iso_to_unix(client.get_cutoffs().get("market_settled_ts", "")) or 0
    per_series: dict[str, list[dict]] = {}
    for series in spec["kalshi_series"]:
        if series not in available:
            log.warning("%s: series %s not listed under this category; skipped", category, series)
            continue
        cache_file = cache / f"{series}_{start_ts}_{end_ts}.json"
        if cache_file.exists():
            markets = json.loads(cache_file.read_text())
        else:
            markets = list_markets_chunked(client, series, max(start_ts, cutoff), end_ts, historical=False)
            if start_ts < cutoff:
                markets += list_markets_chunked(client, series, start_ts, cutoff, historical=True)
            # Keep only what selection and labelling use; full hourly-crypto
            # listings are otherwise ~1 GB of JSON per series.
            keep = set(META_FIELDS) | {"mve_collection_ticker", "volume"}
            markets = [{k: m[k] for k in keep if k in m} for m in markets
                       if float(m.get("volume_fp") or m.get("volume") or 0) >= R.MIN_REPORTED_VOLUME]
            save_json(cache_file, markets)
        best_by_event: dict[str, dict] = {}
        for m in markets:
            try:
                vol = float(m.get("volume_fp") or m.get("volume") or 0)
            except (TypeError, ValueError):
                continue
            close_ts = iso_to_unix(m.get("close_time", ""))
            if (m.get("market_type") != "binary" or m.get("mve_collection_ticker")
                    or m.get("status") not in SETTLED or m.get("result") not in {"yes", "no"}
                    or close_ts is None or not start_ts <= close_ts <= end_ts
                    or not math.isfinite(vol) or vol < R.MIN_REPORTED_VOLUME):
                continue
            m = dict(m, series_ticker=series, series_title=available[series].get("title", ""),
                     category=category, _volume=vol,
                     _historical=bool(close_ts < cutoff))
            ev = m.get("event_ticker") or m["ticker"]
            if ev not in best_by_event or vol > best_by_event[ev]["_volume"]:
                best_by_event[ev] = m
        ranked = sorted(best_by_event.values(), key=lambda m: (-m["_volume"], m["ticker"]))
        # Spare candidates beyond the cap absorb density rejections.
        per_series[series] = ranked[:spec.get("max_per_series", R.MAX_PER_SERIES) * 3]
        log.info("%s / %s: %d eligible events (%d markets listed)", category, series, len(ranked), len(markets))
    # Round-robin across series so no single series dominates a category.
    ordered, depth = [], 0
    while any(depth < len(v) for v in per_series.values()):
        for series in spec["kalshi_series"]:
            if depth < len(per_series.get(series, [])):
                ordered.append(per_series[series][depth])
        depth += 1
    return ordered


def episode_window(market: dict, hours: float) -> tuple[int, int]:
    opened, closed = iso_to_unix(market["open_time"]), iso_to_unix(market["close_time"])
    return max(opened, closed - int(hours * 3600)), closed


def density(trades: pd.DataFrame, candles: pd.DataFrame, ep_start: int, ep_end: int) -> dict:
    n_bars = max(1, math.ceil((ep_end - ep_start) / R.BAR_SECONDS))
    t_sec = ((trades.timestamp - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
             if len(trades) else pd.Series([], dtype="int64"))
    in_win = (t_sec >= ep_start) & (t_sec <= ep_end)
    trade_bars = set(((t_sec[in_win] - ep_start) // R.BAR_SECONDS).tolist())
    quote_bars = set()
    if len(candles):
        start_of_bar = candles.end_period_ts - R.BAR_SECONDS
        win = (start_of_bar >= ep_start) & (start_of_bar < ep_end)
        quote_bars = set(((start_of_bar[win] - ep_start) // R.BAR_SECONDS).tolist())
    return {"episode_bars": n_bars, "episode_trades": int(in_win.sum()),
            "episode_contracts": float(trades.quantity[in_win].sum()) if len(trades) else 0.0,
            "bars_with_trade": len(trade_bars), "bars_with_quote_update": len(quote_bars),
            "active_bar_share": len(trade_bars | quote_bars) / n_bars}


def collect_contract(client: KalshiClient, m: dict, root: Path) -> dict:
    ticker, fname = m["ticker"], safe_filename(m["ticker"])
    status_path = root / "status" / f"{fname}.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        if st.get("complete"):
            return st
    hours = R.CATEGORIES[m["category"]]["episode_hours"]
    ep_start, ep_end = episode_window(m, hours)
    st = {"ticker": ticker, "category": m["category"], "series_ticker": m["series_ticker"],
          "episode_start_ts": ep_start, "episode_end_ts": ep_end, "episode_hours": hours,
          "complete": False, "error": None, "collected_at": datetime.now(timezone.utc).isoformat()}
    try:
        raw = client.get_trades(ticker, ep_start, ep_end + 1, include_history=True,
                                max_pages=R.MAX_TRADE_PAGES)
        raw = pd.DataFrame(raw)
        raw.to_parquet(root / "trades" / f"{fname}.parquet", index=False)
        # One extra hour before the window gives the opening quote state.
        candles = client.get_candlesticks_full(m["series_ticker"], ticker, ep_start - 3600, ep_end + 60,
                                               period_interval=1, historical=m["_historical"])
        cdf = pd.json_normalize(candles, sep="_") if candles else pd.DataFrame(columns=["end_period_ts"])
        cdf.to_parquet(root / "candlesticks" / f"{fname}.parquet", index=False)
        norm, stats = normalize_trades(raw, ticker) if len(raw) else (pd.DataFrame(
            columns=["timestamp", "quantity"]), {"raw_trade_rows": 0})
        st.update(stats)
        st.update(density(norm, cdf, ep_start, ep_end))
        st["raw_candle_rows"] = len(cdf)
        reasons = []
        if st["episode_trades"] < R.MIN_EPISODE_TRADES:
            reasons.append(f"episode_trades {st['episode_trades']} < {R.MIN_EPISODE_TRADES}")
        if st["active_bar_share"] < R.MIN_ACTIVE_BAR_SHARE:
            reasons.append(f"active_bar_share {st['active_bar_share']:.2f} < {R.MIN_ACTIVE_BAR_SHARE}")
        st.update(accepted=not reasons, rejection_reasons=reasons, complete=True)
        if reasons:
            # Keep the decision (status file), not the unusable raw data.
            for sub in ("trades", "candlesticks"):
                (root / sub / f"{fname}.parquet").unlink(missing_ok=True)
    except (KalshiAPIError, ValueError) as exc:
        st.update(error=str(exc), accepted=False, rejection_reasons=["download_error"])
    save_json(status_path, st)
    return st


def contract_row(m: dict, st: dict) -> dict:
    row = {k: m.get(k, "") or "" for k in META_FIELDS}
    row.update(venue="kalshi", market_id=m["ticker"], category=m["category"],
               series_ticker=m["series_ticker"], series_title=m.get("series_title", ""),
               y_outcome=1 if m["result"] == "yes" else 0, reported_volume=m["_volume"],
               archive_source="historical" if m["_historical"] else "live",
               episode_start_utc=datetime.fromtimestamp(st["episode_start_ts"], timezone.utc).isoformat(),
               episode_end_utc=datetime.fromtimestamp(st["episode_end_ts"], timezone.utc).isoformat(),
               episode_hours=st["episode_hours"],
               **{k: st.get(k) for k in ["episode_bars", "episode_trades", "episode_contracts",
                                         "bars_with_trade", "bars_with_quote_update", "active_bar_share",
                                         "raw_trade_rows", "block_trade_rows", "raw_candle_rows"]})
    return row


def run(categories: list[str], target: int | None, end_ts: int) -> dict:
    config.REQUEST_DELAY_S = R.KALSHI_REQUEST_DELAY_S
    root = R.DATA_DIR / "kalshi"
    for sub in ["markets", "markets/discovery_cache", "trades", "candlesticks", "status", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    start_ts = end_ts - R.LOOKBACK_DAYS * 86400
    client = KalshiClient()
    save_json(root / "logs" / "archive_cutoffs.json", client.get_cutoffs())
    contracts_path = root / "markets" / "contracts.parquet"
    existing = pd.read_parquet(contracts_path) if contracts_path.exists() else pd.DataFrame()
    summary = {"run_started_utc": datetime.now(timezone.utc).isoformat(), "window_start_ts": start_ts,
               "window_end_ts": end_ts, "target_per_category": target, "categories": {}}
    rows = [] if existing.empty else existing[~existing.category.isin(categories)].to_dict("records")
    for category in categories:
        cat_start = end_ts - R.CATEGORIES[category].get("lookback_days", R.LOOKBACK_DAYS) * 86400
        candidates = discover(client, category, cat_start, end_ts, root / "markets" / "discovery_cache")
        save_json(root / "markets" / f"candidates_{category}.json",
                  [{k: c.get(k) for k in ["ticker", "event_ticker", "series_ticker", "_volume", "close_time"]}
                   for c in candidates])
        accepted, per_series, attempts = [], defaultdict(int), []
        cat_target = target or R.CATEGORIES[category].get("kalshi_target", R.KALSHI_TARGET_PER_CATEGORY)
        for m in candidates:
            if len(accepted) >= cat_target:
                break
            if per_series[m["series_ticker"]] >= R.CATEGORIES[category].get("max_per_series", R.MAX_PER_SERIES):
                continue
            st = collect_contract(client, m, root)
            attempts.append({"ticker": m["ticker"], "accepted": st.get("accepted"),
                             "reasons": st.get("rejection_reasons"), "episode_trades": st.get("episode_trades")})
            log.info("%s %s %s: trades=%s active=%.2f %s", category, "ACCEPT" if st.get("accepted") else "REJECT",
                     m["ticker"], st.get("episode_trades"), st.get("active_bar_share") or 0,
                     st.get("rejection_reasons") or "")
            if st.get("accepted"):
                accepted.append(contract_row(m, st))
                per_series[m["series_ticker"]] += 1
        rows += accepted
        summary["categories"][category] = {"target": cat_target, "candidates": len(candidates),
                                           "attempted": len(attempts),
                                           "accepted": len(accepted), "by_series": dict(per_series),
                                           "attempts": attempts}
        pd.DataFrame(rows).to_parquet(contracts_path, index=False)   # checkpoint per category
    summary["api_requests"] = client.request_count
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_json(root / "logs" / f"run_{stamp}.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=list(R.CATEGORIES), choices=list(R.CATEGORIES))
    p.add_argument("--target", type=int, help="accepted contracts per category (default: per-category kalshi_target)")
    p.add_argument("--end-date", help="inclusive UTC date YYYY-MM-DD (default: now)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    end_ts = (int(datetime.fromisoformat(a.end_date).replace(tzinfo=timezone.utc).timestamp()) + 86399
              if a.end_date else int(time.time()))
    s = run(a.categories, a.target, end_ts)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "attempts"} for k, v in s["categories"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
