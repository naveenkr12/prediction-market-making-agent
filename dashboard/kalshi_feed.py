"""Read-only client for Kalshi production market data.

Only GET requests to market-data paths on the production host are possible:
there is no method for any other verb, and every path is checked against an
allowlist, so an order or account request cannot be sent by accident.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

import requests

from . import config

log = logging.getLogger("dashboard.feed")

ALLOWED_PREFIXES = ("/markets", "/series", "/events", "/exchange")


class FeedError(RuntimeError):
    pass


def check_path(base_url: str, path: str) -> None:
    host = urlparse(base_url).hostname
    if host != urlparse(config.PROD_REST).hostname:
        raise FeedError(f"Refusing host {host}: only production market data is used")
    if not path.startswith(ALLOWED_PREFIXES) or "/portfolio" in path:
        raise FeedError(f"Refusing path {path}: not a market-data endpoint")


def parse_ts(iso: str | None) -> float | None:
    if not iso:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def effective_close(market: dict) -> float:
    """Sports markets list a late backstop close_time and close early when the
    game ends; expected_expiration_time is the better clock when present."""
    return parse_ts(market.get("expected_expiration_time")) or parse_ts(market["close_time"])


def market_tick(market: dict) -> str:
    steps = [r.get("step") for r in market.get("price_ranges") or [] if r.get("step")]
    # The coarsest step is valid at every price.
    return max(steps, key=float) if steps else "0.01"


class KalshiFeed:
    def __init__(self, base_url: str = config.PROD_REST):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "capstone-market-making-dashboard/1.0"})
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.request_count = 0
        self.errors = 0

    def _get(self, path: str, params=None) -> dict:
        check_path(self.base_url, path)
        for attempt in range(1, 6):
            with self._lock:
                wait = config.REQUEST_SPACING_S - (time.time() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.time()
            self.request_count += 1
            try:
                r = self.session.get(self.base_url + path, params=params, timeout=15)
            except requests.RequestException as exc:
                err = str(exc)
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise FeedError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
                err = f"HTTP {r.status_code}"
            self.errors += 1
            log.warning("%s on %s (attempt %d)", err, path, attempt)
            time.sleep(min(30, 2 ** attempt))
        raise FeedError(f"GET {path} failed after retries")

    def market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")["market"]

    def discover(self, series: list[str], close_within_h: float, min_volume: float, max_markets: int) -> list[dict]:
        now = time.time()
        best: dict[str, dict] = {}
        for s in series:
            try:
                markets = self._get("/markets", {"series_ticker": s, "status": "open", "limit": 1000,
                                                 "mve_filter": "exclude"}).get("markets", [])
            except FeedError as exc:
                log.warning("discovery %s: %s", s, exc)
                continue
            for m in markets:
                close = effective_close(m)
                vol = float(m.get("volume_24h_fp") or 0)
                if (m.get("market_type") != "binary" or vol < min_volume
                        or not now < close <= now + close_within_h * 3600):
                    continue
                m["series_ticker"] = s
                ev = m.get("event_ticker") or m["ticker"]
                if ev not in best or vol > float(best[ev].get("volume_24h_fp") or 0):
                    best[ev] = m
        ranked = sorted(best.values(), key=lambda m: -float(m.get("volume_24h_fp") or 0))
        return ranked[:max_markets]

    def books(self, tickers: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(tickers), 20):
            data = self._get("/markets/orderbooks", [("tickers", t) for t in tickers[i:i + 20]])
            for ob in data.get("orderbooks", []):
                out[ob["ticker"]] = {"orderbook_fp": ob["orderbook_fp"]}
        return out

    def trades(self, ticker: str, min_ts: int, max_pages: int = 5) -> list[dict]:
        """Trades at or after min_ts (Unix seconds), oldest first."""
        rows, cursor = [], None
        for _ in range(max_pages):
            params = {"ticker": ticker, "min_ts": int(min_ts), "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets/trades", params)
            rows += data.get("trades", [])
            cursor = data.get("cursor")
            if not cursor:
                break
        for t in rows:
            t["ts"] = parse_ts(t["created_time"])
        return sorted(rows, key=lambda t: (t["ts"], t["trade_id"]))
