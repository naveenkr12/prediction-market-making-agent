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
from utils import iso_to_unix

log = logging.getLogger("kalshi_client")


class KalshiAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class KalshiClient:
    def __init__(self, base_url: str = config.KALSHI_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._cutoffs: dict[str, Any] | None = None
        self.request_count = 0
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
                self.request_count += 1
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
            raise KalshiAPIError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)

    def _paginate(
        self, path: str, params: dict[str, Any], items_key: str,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        cursor = None
        seen_cursors: set[str] = set()
        pages = 0
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            data = self._get(path, page_params)
            pages += 1
            items = data.get(items_key, [])
            yield from items
            cursor = data.get("cursor")
            if not cursor:
                return
            if cursor in seen_cursors or not items:
                raise KalshiAPIError(f"Pagination did not advance for {path}; refusing an incomplete tape")
            seen_cursors.add(cursor)
            if max_pages is not None and pages >= max_pages:
                raise KalshiAPIError(f"Pagination limit ({max_pages}) reached for {path}; data incomplete")

    # -- public surface -------------------------------------------------

    def get_cutoffs(self) -> dict[str, Any]:
        """Snapshot the public archive cutoffs once per collection run."""
        if self._cutoffs is None:
            self._cutoffs = self._get("/historical/cutoff")
        return self._cutoffs

    def get_market(self, ticker: str) -> dict[str, Any]:
        try:
            return self._get(f"/markets/{ticker}")["market"]
        except KalshiAPIError as exc:
            if exc.status_code != 404:
                raise
            return self._get(f"/historical/markets/{ticker}")["market"]

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
        historical: bool = False,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
            if not historical:
                params.setdefault("mve_filter", mve_filter or "exclude")
        if status and not historical:
            params["status"] = status
        if min_close_ts and not historical:
            params["min_close_ts"] = min_close_ts
        if max_close_ts and not historical:
            params["max_close_ts"] = max_close_ts
        # Historical API treats series/event/tickers/mve_filter as mutually
        # exclusive. Apply MVE exclusion locally when querying a specific series.
        if mve_filter and "mve_filter" not in params and not (historical and (series_ticker or event_ticker)):
            params["mve_filter"] = mve_filter
        path = "/historical/markets" if historical else "/markets"
        markets = list(self._paginate(path, params, "markets", max_pages=max_pages))
        if historical:
            # Historical listing does not accept the live endpoint's time/status filters.
            markets = [m for m in markets if
                       (min_close_ts is None or (iso_to_unix(m.get("close_time", "")) or 0) >= min_close_ts)
                       and (max_close_ts is None or (iso_to_unix(m.get("close_time", "")) or 0) <= max_close_ts)]
            if mve_filter == "exclude":
                markets = [m for m in markets if not m.get("mve_collection_ticker")]
        return markets

    def get_trades(
        self, ticker: str, min_ts: int | None = None, max_ts: int | None = None,
        include_history: bool = True, max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"ticker": ticker, "limit": config.PAGE_LIMIT}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if not include_history:
            return list(self._paginate("/markets/trades", params, "trades", max_pages=max_pages))
        cutoff = iso_to_unix(self.get_cutoffs().get("trades_created_ts", ""))
        if cutoff is None:
            raise KalshiAPIError("Missing trade archive cutoff; cannot verify full-history routing")
        trades: list[dict[str, Any]] = []
        # Read live first, then historical; overlap by one second and deduplicate.
        if max_ts is None or max_ts >= cutoff:
            live = dict(params)
            live["min_ts"] = max(min_ts or 0, cutoff - 1)
            trades.extend(self._paginate("/markets/trades", live, "trades", max_pages=max_pages))
        if min_ts is None or min_ts < cutoff:
            old = dict(params)
            old["max_ts"] = min(max_ts if max_ts is not None else cutoff, cutoff)
            trades.extend(self._paginate("/historical/trades", old, "trades", max_pages=max_pages))
        unique: dict[str, dict[str, Any]] = {}
        for trade in trades:
            key = trade.get("trade_id")
            if not key:
                raise KalshiAPIError("Trade without trade_id; cannot safely deduplicate archive overlap")
            unique[key] = trade
        return sorted(unique.values(), key=lambda t: t.get("created_time", ""))

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = config.CANDLESTICK_PERIOD_INTERVAL,
        historical: bool = False,
    ) -> list[dict[str, Any]]:
        """One page. Kalshi caps a single request at 5000 candlesticks, i.e. at
        period_interval=1 (minute bars) that's ~3.47 days -- use
        get_candlesticks_full for a full market lifetime."""
        path = (f"/historical/markets/{ticker}/candlesticks" if historical else
                f"/series/{series_ticker}/markets/{ticker}/candlesticks")
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
        historical: bool = False,
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
                self.get_candlesticks(series_ticker, ticker, chunk_start, chunk_end,
                                     period_interval, historical=historical)
            )
            chunk_start = chunk_end
        # Adjacent requests have inclusive bounds; do not double-count boundary bars.
        return sorted({c["end_period_ts"]: c for c in out}.values(), key=lambda c: c["end_period_ts"])
