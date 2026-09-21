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
import random
from dataclasses import dataclass

import config
from kalshi_client import KalshiClient

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

                selected.append(
                    SelectedContract(
                        ticker=market["ticker"],
                        series_ticker=series_ticker,
                        series_title=series.get("title", ""),
                        category=category,
                        close_time=market.get("close_time", ""),
                        open_time=market.get("open_time", ""),
                        status=market.get("status", ""),
                        result=market.get("result"),
                        market_type=market.get("market_type", ""),
                    )
                )
                found_this_category += 1
                if found_this_category >= per_category_target:
                    break

        log.info(
            "category=%s: scanned %d series -> %d standalone settled contracts",
            category, series_scanned, found_this_category,
        )

    rng.shuffle(selected)
    return selected[:target_total]
