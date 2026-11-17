"""
Settings for the RL training dataset (Kalshi + Polymarket).

Neither venue publishes historical order-book depth for free. The dataset
therefore has two parts:

  1. Historical (available now): full trade tapes + top-of-book.
     Kalshi minute candles carry the best YES bid/ask; Polymarket's price
     history carries a minute midpoint only (no bid/ask).
  2. Live L2 (going forward): rl_record_orderbooks.py polls full-depth books
     for currently open markets in the same categories at a fixed cadence.

Time alignment with Feil & Nendel (Paper 1): the paper uses a unitless
horizon T=1 ("chosen to illustrate qualitative effects rather than to
calibrate the model to a specific contract") and 1000 Monte Carlo steps.
Every bar here carries tau = elapsed / episode_length in [0, 1], so
hjb_policy's t = T * tau mapping applies unchanged, and rl_loader.paper_grid
resamples any episode onto the paper's uniform n-step grid.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "PM_RL_DATA_DIR",
    "/Users/som/Desktop/Georgia Tech/Other Courses/Practice of QCF/Work/Data"))

# Canonical bar size. One minute is the finest resolution at which either
# venue publishes historical quotes (Kalshi candles, Polymarket fidelity=1).
BAR_SECONDS = 60
# Paper 1's Monte Carlo resolution; rl_loader.paper_grid defaults to it.
PAPER_N_STEPS = 1000

LOOKBACK_DAYS = 60
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}

# Episode = the final `episode_hours` of each contract's life, clipped to its
# open time. Chosen to cover each category's active trading window:
#   Sports     - pre-game plus in-play for a typical game (2-4 h)
#   Crypto     - hourly price contracts live about one hour
#   Financials - one US cash session plus the pre-open
#   Economics  - the day leading into a data release / decision
CATEGORIES = {
    "Sports": {
        "episode_hours": 6,
        # Games are the densest contracts (~30k Kalshi trades per episode) and
        # most of the dataset's volume: four months of games, more per league.
        "lookback_days": 120,
        "max_per_series": 60,
        "kalshi_target": 560,
        "polymarket_target": 250,
        "kalshi_series": ["KXMLBGAME", "KXNFLGAME", "KXNCAAFGAME", "KXNBAGAME", "KXNHLGAME",
                          "KXWNBAGAME", "KXEPLGAME", "KXUCLGAME", "KXMLSGAME", "KXATPMATCH",
                          "KXWTAMATCH"],
        "polymarket_tags": ["mlb", "nfl", "ncaa-football", "nba", "nhl", "wnba", "epl",
                            "champions-league", "mls", "tennis"],
    },
    "Crypto": {
        "episode_hours": 2,
        # Hourly strike ladders list ~250k-400k markets per series in 60 days;
        # two weeks still yields hundreds of liquid hourly events.
        "lookback_days": 30,
        # Only hourly BTC thresholds (KXBTCD) are dense in the pilot (~8k trades
        # per episode); ETH/SOL/XRP and range ladders had <100. Each hourly
        # contract is a separate event, so a larger series cap is safe.
        "max_per_series": 60,
        "kalshi_target": 120,
        "polymarket_target": 60,
        "kalshi_series": ["KXBTCD", "KXETHD", "KXBTC", "KXETH", "KXSOLD", "KXXRPD"],
        "polymarket_tags": ["crypto", "bitcoin", "ethereum"],
    },
    "Financials": {
        "episode_hours": 8,
        "kalshi_extra_categories": ["Commodities"],   # Kalshi files WTI oil there
        "lookback_days": 120,
        "max_per_series": 30,
        "kalshi_target": 120,
        "polymarket_target": 60,
        "kalshi_series": ["KXINXU", "KXINX", "KXNASDAQ100U", "KXNASDAQ100", "KXWTI",
                          "KXWTIW", "KXEURUSD", "KXUSDJPY"],
        "polymarket_tags": ["stocks", "finance", "commodities"],
    },
    "Economics": {
        "episode_hours": 24,
        # Releases are monthly: 60 days held only 28 liquid events in the pilot,
        # so take a full year and keep every release that passes the gates.
        "lookback_days": 365,
        "max_per_series": 25,
        "kalshi_target": 120,
        "polymarket_target": 40,
        "kalshi_series": ["KXFEDDECISION", "KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOREYOY",
                          "KXPAYROLLS", "KXU3", "KXGDP", "KXPCECORE", "KXJOBLESSCLAIMS",
                          "KXAAAGASW", "KXAAAGASM"],
        "polymarket_tags": ["fed-rates", "economy", "economic-policy"],
        # Polymarket's economy tags also carry earnings, crypto races, Fed-chair
        # politics and "mention" markets. Keep macro releases and central-bank
        # rate decisions only: an allowlisted series, or a matching question.
        "polymarket_series_allow": ["fomc", "bank-of-japan", "ecb", "bank-of-england",
                                    "bank-of-canada", "us-cpi", "cpi", "inflation", "jobs-report",
                                    "unemployment", "gdp", "nonfarm-payrolls"],
        "polymarket_title_regex": (r"interest rates?|rate (cut|hike)|\bbps\b|\bCPI\b|inflation|"
                                   r"unemployment|jobs report|payrolls?|\bGDP\b|recession"),
    },
}

# Sample sizes and density gates. A contract is kept only if its episode
# window is dense enough to train on; rejections are recorded with reasons.
# Per-category "kalshi_target" / "polymarket_target" / "max_per_series" above
# override these defaults.
KALSHI_TARGET_PER_CATEGORY = 40
POLYMARKET_TARGET_PER_CATEGORY = 20
MAX_PER_SERIES = 12            # spread each category across several series

# Polymarket tags are coarse: its "economy" tag also carries company
# market-cap races. These series are labelled by what they resolve on.
POLYMARKET_SERIES_CATEGORY = {"largest-company": "Financials",
                              "second-largest-company": "Financials",
                              "third-largest-company": "Financials"}
MIN_EPISODE_TRADES = 200       # usable trades inside the episode window
MIN_ACTIVE_BAR_SHARE = 0.25    # share of bars with a trade or a quote update
MIN_REPORTED_VOLUME = 5_000    # discovery prefilter (contracts), not a quality gate
MAX_TRADE_PAGES = 200          # Kalshi pages of 1000 trades per contract

# Pacing. Bursts of 25 unauthenticated Kalshi requests returned 200s in
# testing; the client still backs off on 429/5xx.
KALSHI_REQUEST_DELAY_S = 0.3
POLYMARKET_REQUEST_DELAY_S = 0.3

# Live L2 recorder
RECORD_INTERVAL_S = 10
RECORD_UNIVERSE_REFRESH_S = 1800
RECORD_MAX_MARKETS_PER_VENUE = 60
RECORD_CLOSING_WITHIN_HOURS = 48
