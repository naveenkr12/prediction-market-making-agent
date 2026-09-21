"""
Thin client for Kalshi's public Trade API v2.

All endpoints used here are unauthenticated, public market-data endpoints
(confirmed against https://docs.kalshi.com as of Sep 2026):

  GET /series                                          - list series, filterable by category
  GET /events                                           - list events, filterable by series/status
  GET /markets                                          - list markets, filterable by event/status/close-time
  GET /markets/trades                                    - public trade tape
  GET /series/{series_ticker}/markets/{ticker}/candlesticks - minute/hour/day OHLC

No API key is required for any of the above. We still rate-limit ourselves
and back off on 429/5xx to be a good citizen.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests

import config

log = logging.getLogger("kalshi_client")


class KalshiAPIError(RuntimeError):
    pass


class KalshiClient:
    def __init__(self, base_url: str = config.KALSHI_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "capstone-market-making-research/1.0 "
                "(academic project; contact via Georgia Tech QCF)",
            }
        )

    # -- low-level ----------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                if attempt > config.MAX_RETRIES:
                    raise KalshiAPIError(f"GET {path} failed after {attempt} attempts: {exc}") from exc
                sleep_s = config.BACKOFF_BASE_S ** attempt
                log.warning("network error on %s (attempt %d): %s -> retry in %.1fs", path, attempt, exc, sleep_s)
                time.sleep(sleep_s)
                continue

            if resp.status_code == 200:
                time.sleep(config.REQUEST_DELAY_S)
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt > config.MAX_RETRIES:
                    raise KalshiAPIError(f"GET {path} gave {resp.status_code} after {attempt} attempts")
                sleep_s = config.BACKOFF_BASE_S ** attempt
                log.warning(
                    "HTTP %d on %s (attempt %d) -> retry in %.1fs", resp.status_code, path, attempt, sleep_s
                )
                time.sleep(sleep_s)
                continue

            # Non-retryable error
            raise KalshiAPIError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:300]}")

    def _paginate(
        self, path: str, params: dict[str, Any], items_key: str
    ) -> Iterator[dict[str, Any]]:
        cursor = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            data = self._get(path, page_params)
            items = data.get(items_key, [])
            yield from items
            cursor = data.get("cursor")
            if not cursor or not items:
                return

    # -- public surface -------------------------------------------------

    def list_series(self, category: str) -> list[dict[str, Any]]:
        """All series in a discovery category (e.g. 'Economics', 'Sports')."""
        data = self._get("/series", {"category": category})
        return data.get("series", [])

    def list_events(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        with_nested_markets: bool = False,
        min_close_ts: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "with_nested_markets": with_nested_markets}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        return list(self._paginate("/events", params, "events"))

    def list_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        mve_filter: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
            params.setdefault("mve_filter", mve_filter or "exclude")
        if status:
            params["status"] = status
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if mve_filter and "mve_filter" not in params:
            params["mve_filter"] = mve_filter
        return list(self._paginate("/markets", params, "markets"))

    def get_trades(
        self, ticker: str, min_ts: int | None = None, max_ts: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": config.PAGE_LIMIT}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        return list(self._paginate("/markets/trades", params, "trades"))

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = config.CANDLESTICK_PERIOD_INTERVAL,
    ) -> list[dict[str, Any]]:
        """One page. Kalshi caps a single request at 5000 candlesticks, i.e. at
        period_interval=1 (minute bars) that's ~3.47 days -- use
        get_candlesticks_full for a full market lifetime."""
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        data = self._get(path, params)
        return data.get("candlesticks", [])

    def get_candlesticks_full(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = config.CANDLESTICK_PERIOD_INTERVAL,
        max_candles_per_page: int = 5000,
    ) -> list[dict[str, Any]]:
        """Chunks the [start_ts, end_ts] window so each request stays under
        Kalshi's 5000-candlestick cap, and concatenates the pages."""
        if end_ts <= start_ts:
            return []
        chunk_seconds = max_candles_per_page * period_interval * 60
        out: list[dict[str, Any]] = []
        chunk_start = start_ts
        while chunk_start < end_ts:
            chunk_end = min(chunk_start + chunk_seconds, end_ts)
            out.extend(
                self.get_candlesticks(series_ticker, ticker, chunk_start, chunk_end, period_interval)
            )
            chunk_start = chunk_end
        return out
