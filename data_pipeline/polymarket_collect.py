"""
OPTIONAL comparison-source collector, per the proposal ("Polymarket remains
an optional external comparison"). Pulls a handful of closed, high-volume
binary markets: trade tape (Data API) + price history (CLOB API).

Run:  python3 polymarket_collect.py [--n-markets 15]
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

import config
from polymarket_client import PolymarketClient, PolymarketAPIError
from utils import safe_filename

log = logging.getLogger("polymarket_collect")


def run(n_markets: int) -> None:
    client = PolymarketClient()

    log.info("Fetching top %d closed markets by volume from Gamma API ...", n_markets)
    markets = client.list_closed_markets(limit=n_markets, order="volume")

    meta_rows = []
    for i, m in enumerate(markets, 1):
        question = m.get("question", "")
        condition_id = m.get("conditionId")
        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
        except json.JSONDecodeError:
            token_ids = []
        if not condition_id or not token_ids:
            continue

        fname = safe_filename(m.get("id", condition_id))
        log.info("[%d/%d] %s", i, len(markets), question[:70])

        try:
            trades = client.get_trades(condition_id, limit=1000)
            if trades:
                pd.DataFrame(trades).to_parquet(config.POLYMARKET_DIR / f"{fname}_trades.parquet", index=False)
        except PolymarketAPIError as exc:
            log.warning("trades failed for %s: %s", question, exc)
            trades = []

        try:
            hist = client.get_price_history(token_ids[0])
            if hist:
                pd.DataFrame(hist).to_parquet(config.POLYMARKET_DIR / f"{fname}_price_history.parquet", index=False)
        except PolymarketAPIError as exc:
            log.warning("price history failed for %s: %s", question, exc)
            hist = []

        meta_rows.append(
            {
                "id": m.get("id"),
                "question": question,
                "conditionId": condition_id,
                "endDate": m.get("endDate"),
                "volume": m.get("volume"),
                "n_trades": len(trades),
                "n_price_points": len(hist),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(config.POLYMARKET_DIR / "polymarket_selected_markets.parquet", index=False)
    log.info("Done. %d/%d markets collected with data.", (meta_df["n_trades"] > 0).sum() if len(meta_df) else 0, len(markets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional Polymarket comparison-data collector.")
    parser.add_argument("--n-markets", type=int, default=15)
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args.n_markets)
