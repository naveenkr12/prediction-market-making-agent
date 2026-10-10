"""
Select standalone binary contracts across the configured Kalshi categories,
settled within the lookback window.

"Standalone" (a documented simplifying assumption -- the proposal does not
spell this out further) means:
  - market_type == "binary" (a plain Yes/No contract, not a scalar/range
    ladder bucket)
  - not part of a multivariate-event collection (mve_filter="exclude" on
    the /markets query, and mve_collection_ticker is empty on the result)
  - already resolved within the lookback window (status "settled" or
    "finalized", close_time inside [START_TS, END_TS])

Correlated / multi-leg contracts (ParlayMarket, in the literature review)
are explicitly out of scope for the core project.

Implementation note: Kalshi's public /markets endpoint accepts
series_ticker directly, so contracts are pulled one series at a time
(one API call per series) rather than by walking every event -- this is
far cheaper against the unauthenticated rate limit.
"""

from __future__ import annotations

import logging
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import config
from kalshi_client import KalshiClient, KalshiAPIError
from utils import iso_to_unix, safe_filename

log = logging.getLogger("contract_selection")

SETTLED_STATUSES = {"settled", "finalized"}


@dataclass
class SelectedContract:
    ticker: str
    series_ticker: str
    series_title: str
    category: str
    close_time: str
    open_time: str
    status: str
    result: str | None
    market_type: str
    event_ticker: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    settlement_ts: str = ""
    settlement_value_dollars: str = ""
    volume_fp: str = ""


def market_to_contract(market: dict, series: dict, category: str) -> SelectedContract:
    """Preserve event identity and resolution text rather than guessing from tickers."""
    fields = {name: market.get(name, "") or "" for name in
              ["ticker", "close_time", "open_time", "status", "market_type",
               "event_ticker", "title", "subtitle", "yes_sub_title", "rules_primary",
               "rules_secondary", "settlement_ts", "settlement_value_dollars", "volume_fp"]}
    return SelectedContract(**fields, series_ticker=series["ticker"],
                            series_title=series.get("title", ""), category=category,
                            result=market.get("result"))


def discover_calibration_candidates(client: KalshiClient, start_ts: int, end_ts: int,
                                    max_series_per_category: int = 24,
                                    min_volume: float = config.MIN_REPORTED_CONTRACT_VOLUME,
                                    seed: int = 42, cache_dir: Path | None = None,
                                    categories: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Discover live + archived markets, then interleave categories and series.

    The accepted-event and series caps are enforced *after* tape eligibility so
    a failed candidate does not use up an event's place in the final sample.
    """
    rng = random.Random(seed)
    candidates: dict[tuple[str, str], list[dict]] = {}
    discovery: list[dict] = []
    archive_cutoff = iso_to_unix(client.get_cutoffs().get("market_settled_ts", ""))
    for category in categories or config.KALSHI_CATEGORIES:
        series = client.list_series(category)
        by_ticker = {s["ticker"]: s for s in series}
        preferred = [by_ticker[t] for t in config.CALIBRATION_PREFERRED_SERIES.get(category, [])
                     if t in by_ticker]
        remaining = [s for s in series if s["ticker"] not in {p["ticker"] for p in preferred}]
        rng.shuffle(remaining)
        chosen = (preferred + remaining)[:max_series_per_category]
        for s in chosen:
            combined: dict[str, dict] = {}
            for historical in ([False, True] if archive_cutoff is None or start_ts < archive_cutoff
                               else [False]):
                record = {"category": category, "series_ticker": s["ticker"],
                          "source": "historical" if historical else "live", "error": None}
                cache = cache_dir / f"{safe_filename(s['ticker'])}_{record['source']}.json" if cache_dir else None
                try:
                    if cache and cache.exists():
                        markets = json.loads(cache.read_text())
                    else:
                        markets = client.list_markets(series_ticker=s["ticker"],
                                                  min_close_ts=start_ts, max_close_ts=end_ts,
                                                  mve_filter="exclude", historical=historical,
                                                  max_pages=20)
                        if cache:
                            cache.parent.mkdir(parents=True, exist_ok=True)
                            cache.write_text(json.dumps(markets))
                    record["markets_in_window"] = len(markets)
                    for market in markets:
                        market = dict(market)
                        market["metadata_source"] = record["source"]
                        combined.setdefault(market["ticker"], market)
                except KalshiAPIError as exc:
                    record["error"] = str(exc)
                    log.warning("Discovery failed for %s (%s): %s", s["ticker"], record["source"], exc)
                discovery.append(record)
            eligible = []
            for market in combined.values():
                close_ts = iso_to_unix(market.get("close_time", ""))
                try:
                    volume = float(market.get("volume_fp", market.get("volume", 0)))
                except (ValueError, TypeError):
                    continue
                if (market.get("market_type") != "binary" or market.get("mve_collection_ticker")
                        or market.get("status") not in SETTLED_STATUSES
                        or market.get("result") not in {"yes", "no"}
                        or close_ts is None or not start_ts <= close_ts <= end_ts
                        or not math.isfinite(volume) or volume < min_volume):
                    continue
                market.update(series_ticker=s["ticker"], series_title=s.get("title", ""), category=category)
                eligible.append(market)
            # Moderate-volume tapes make a complete download practical. Record this
            # ranking explicitly; it must not be mistaken for random population sampling.
            eligible.sort(key=lambda m: (abs(math.log(max(float(m.get("volume_fp", m.get("volume", 0))), 1)
                                                      / 50_000)), m["ticker"]))
            candidates[category, s["ticker"]] = eligible
            log.info("Discovery %s / %s: %d liquid candidates", category, s["ticker"], len(eligible))
    ordered: list[dict] = []
    keys = list(candidates)
    while any(candidates.values()):
        # One candidate per series per round; category order alternates below.
        rounds = {category: [] for category in config.KALSHI_CATEGORIES}
        for key in keys:
            if candidates[key]:
                rounds[key[0]].append(candidates[key].pop(0))
        while any(rounds.values()):
            for category in config.KALSHI_CATEGORIES:
                if rounds[category]:
                    ordered.append(rounds[category].pop(0))
    return ordered, discovery


def select_contracts(
    client: KalshiClient,
    categories: list[str] | None = None,
    target_total: int = config.TARGET_CONTRACTS_MAX,
    max_series_per_category: int = 400,
    seed: int = 42,
) -> list[SelectedContract]:
    categories = categories or config.KALSHI_CATEGORIES
    selected: list[SelectedContract] = []
    per_category_target = max(1, target_total // len(categories))
    rng = random.Random(seed)

    for category in categories:
        series_list = client.list_series(category)
        rng.shuffle(series_list)
        found_this_category = 0
        series_scanned = 0

        log.info(
            "category=%s: %d series available, sampling up to %d",
            category, len(series_list), max_series_per_category,
        )

        for series in series_list[:max_series_per_category]:
            if found_this_category >= per_category_target:
                break
            series_scanned += 1
            series_ticker = series["ticker"]

            try:
                markets = client.list_markets(
                    series_ticker=series_ticker,
                    min_close_ts=config.START_TS,
                    max_close_ts=config.END_TS,
                    mve_filter="exclude",
                )
            except Exception as exc:  # noqa: BLE001 - log and keep scanning
                log.warning("skipping series %s: %s", series_ticker, exc)
                continue

            for market in markets:
                if market.get("market_type") != "binary":
                    continue
                if market.get("mve_collection_ticker"):
                    continue
                if market.get("status") not in SETTLED_STATUSES:
                    continue

                selected.append(market_to_contract(market, series, category))
                found_this_category += 1
                if found_this_category >= per_category_target:
                    break

        log.info(
            "category=%s: scanned %d series -> %d standalone settled contracts",
            category, series_scanned, found_this_category,
        )

    rng.shuffle(selected)
    return selected[:target_total]
