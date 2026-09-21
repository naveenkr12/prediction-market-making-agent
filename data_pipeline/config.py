"""
Central configuration for the prediction-market data pipeline.

Matches the "Data needed and feasibility" section of the capstone proposal:
- Primary source: Kalshi (public API, no auth needed for market data).
- Optional comparison source: Polymarket.
- Target: 50-100 standalone contracts across economics and sports,
  over a recent three-month window.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# ---------------------------------------------------------------------------
# Kalshi — public REST API (no authentication required for market data)
# https://docs.kalshi.com
# ---------------------------------------------------------------------------
KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

# ---------------------------------------------------------------------------
# Polymarket — public REST APIs (no authentication required)
# https://docs.polymarket.com
# ---------------------------------------------------------------------------
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"   # markets / events metadata
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"          # order book, price history
POLYMARKET_DATA_API_URL = "https://data-api.polymarket.com"  # trade tape

# ---------------------------------------------------------------------------
# Collection window & sample size
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 90  # "a recent three-month window"
END_TS = int(dt.datetime.now(dt.timezone.utc).timestamp())
START_TS = END_TS - LOOKBACK_DAYS * 86400

TARGET_CONTRACTS_MIN = 50
TARGET_CONTRACTS_MAX = 100

# Kalshi series "category" values to sample from (economics + sports, per proposal).
KALSHI_CATEGORIES = ["Economics", "Sports"]

# Candlestick granularity in minutes: 1 (minute bars, per proposal's "minute bid/ask candles").
CANDLESTICK_PERIOD_INTERVAL = 1

# Kalshi caps a single candlesticks request at 5000 bars, and many markets are
# listed weeks or months before they close (long, thinly-traded pre-market
# window). Pulling full lifetime minute bars for every contract is possible
# (get_candlesticks_full chunks around the cap) but slow at scale. Default to
# the most recent N days before close -- the active price-discovery window
# that matters for market-making microstructure -- and widen this if you have
# the time/compute budget to pull full lifetimes (set to None for no cap).
CANDLESTICK_LOOKBACK_DAYS = 14

# ---------------------------------------------------------------------------
# Networking / politeness
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_S = 20
MAX_RETRIES = 6
BACKOFF_BASE_S = 2.0           # exponential backoff base for 429 / 5xx
REQUEST_DELAY_S = 1.2          # unauthenticated public endpoints throttle aggressively; stay polite
PAGE_LIMIT = 1000              # Kalshi max page size for trades

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MARKETS_DIR = OUTPUT_DIR / "markets"        # one parquet per venue: selected contract metadata
TRADES_DIR = OUTPUT_DIR / "trades"          # one parquet per contract: trade tape
CANDLES_DIR = OUTPUT_DIR / "candlesticks"   # one parquet per contract: minute OHLC
POLYMARKET_DIR = OUTPUT_DIR / "polymarket"  # optional comparison source
LOGS_DIR = OUTPUT_DIR / "logs"

for d in (MARKETS_DIR, TRADES_DIR, CANDLES_DIR, POLYMARKET_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
