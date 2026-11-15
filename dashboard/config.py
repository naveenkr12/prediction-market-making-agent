"""Settings for the live paper-quoting dashboard.

Data: Kalshi production market data through public, unauthenticated GET
endpoints. Nothing here places orders; fills are simulated from real trades.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROD_REST = "https://external-api.kalshi.com/trade-api/v2"

# Polling and decision cadence. The strategies were built for one decision
# every 300 s over the final hour before close (simulator.trading_env).
BOOK_POLL_S = 5
TRADE_POLL_S = 5
DECISION_S = 300
EPISODE_S = 3600
STATUS_POLL_S = 60
REQUEST_SPACING_S = 0.15        # minimum gap between API requests

# Execution scenario, matching the simulator's "base" fill scenario.
LATENCY_S = 1.0                 # a new quote can only fill trades after this delay
PARTICIPATION = 0.10            # our share of trade volume left after the queue ahead
QUEUE_PROXY = 50                # queue ahead used only when the book has no depth data
QUOTE_SIZE = 10                 # contracts per side (hjb_baseline delta)
INVENTORY_LIMIT = 100
INITIAL_CASH = 1000

# Market selection: liquid open binary markets whose expected close is soon.
MAX_MARKETS = 8
CLOSE_WITHIN_HOURS = 12
MIN_VOLUME_24H = 1000
SERIES = ["KXNFLGAME", "KXNCAAFGAME", "KXNBAGAME", "KXNHLGAME", "KXMLBGAME", "KXEPLGAME",
          "KXUCLGAME", "KXATPMATCH", "KXWTAMATCH", "KXBTCD", "KXETHD", "KXINXU", "KXINX",
          "KXNASDAQ100", "KXWTI", "KXFEDDECISION", "KXCPI", "KXAAAGASW"]

PPO_MODEL = ROOT / "rl_agent" / "models" / "ppo_capstone_broad_v2.zip"

HISTORY_POINTS = 4000           # per market chart history (about 5.5 h at 5 s)
RUNS_DIR = Path(__file__).resolve().parent / "runs"
