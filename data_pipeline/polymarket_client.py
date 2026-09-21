"""
Thin client for Polymarket's public, unauthenticated read APIs -- used only
as the OPTIONAL external comparison source the proposal mentions.

  Gamma API     https://gamma-api.polymarket.com   - market/event metadata
  CLOB API      https://clob.polymarket.com        - order book, price history
  Data API      https://data-api.polymarket.com    - trade tape

Confirmed against https://docs.polymarket.com as of Sep 2026. None of these
require an API key for read access.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

import config

log = logging.getLogger("polymarket_client")


class PolymarketAPIError(RuntimeError):
    pass


class PolymarketClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "capstone-market-making-research/1.0"}
        )

    def _get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                if attempt > config.MAX_RETRIES:
                    raise PolymarketAPIError(f"GET {path} failed after {attempt} attempts: {exc}") from exc
                time.sleep(config.BACKOFF_BASE_S ** attempt)
                continue

            if resp.status_code == 200:
                time.sleep(config.REQUEST_DELAY_S)
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt > config.MAX_RETRIES:
                    raise PolymarketAPIError(f"GET {path} gave {resp.status_code} after {attempt} attempts")
                sleep_s = config.BACKOFF_BASE_S ** attempt
                log.warning("HTTP %d on %s (attempt %d) -> retry in %.1fs", resp.status_code, path, attempt, sleep_s)
                time.sleep(sleep_s)
                continue
            raise PolymarketAPIError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:300]}")

    # -- Gamma: market/event discovery --------------------------------------

    def list_closed_markets(
        self, limit: int = 100, offset: int = 0, tag: str | None = None, order: str = "endDate"
    ) -> list[dict]:
        params: dict[str, Any] = {
            "closed": "true", "limit": limit, "offset": offset, "order": order, "ascending": "false",
        }
        if tag:
            params["tag"] = tag
        data = self._get(config.POLYMARKET_GAMMA_URL, "/markets", params)
        return data if isinstance(data, list) else data.get("markets", [])

    # -- CLOB: price history --------------------------------------------

    def get_price_history(self, token_id: str, interval: str = "max", fidelity: int = 60) -> list[dict]:
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        data = self._get(config.POLYMARKET_CLOB_URL, "/prices-history", params)
        return data.get("history", [])

    # -- Data API: trade tape --------------------------------------------

    def get_trades(self, condition_id: str, limit: int = 500) -> list[dict]:
        params = {"market": condition_id, "limit": limit, "takerOnly": "true"}
        data = self._get(config.POLYMARKET_DATA_API_URL, "/trades", params)
        return data if isinstance(data, list) else []
