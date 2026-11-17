"""
Record live full-depth order books (L2) for open Kalshi and Polymarket
markets in the RL categories, at a fixed cadence.

Neither venue offers historical depth, so this is the only way to obtain
real L2 for training and execution studies: record now, label at settlement.
Read-only public endpoints; no orders are ever sent.

  Kalshi      GET  /markets/orderbooks?tickers=..   (batched, unauthenticated)
  Polymarket  POST /books                           (batched YES-token books)

Every N seconds one snapshot per market is appended to
  DATA_DIR/live_orderbooks/<venue>/<UTC date>/<market id>.jsonl.gz
Kalshi rows use exactly the schema of capture_orderbook.py. Polymarket rows
carry the same `payload.orderbook_fp` shape (YES bids in yes_dollars; YES asks
converted to NO bids, 1 - ask, in no_dollars), so
simulator.execution.BookSnapshot.from_kalshi(row["payload"]) parses both.
A book identical to the previous one is stored as {"unchanged": true,
"payload": null} to save space while keeping the observation time.

The tracked universe (most active market per event, closing within
RECORD_CLOSING_WITHIN_HOURS) is refreshed every RECORD_UNIVERSE_REFRESH_S and
logged to live_orderbooks/universe.jsonl with its labels. After markets
settle, run rl_collect_kalshi / rl_collect_polymarket to attach trade tapes
and outcomes.

Run from data_pipeline/ (Ctrl+C stops cleanly):
    python3 rl_record_orderbooks.py                       # runs until stopped
    python3 rl_record_orderbooks.py --duration-hours 6 --interval 10
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
import rl_config as R
from rl_collect_polymarket import to_ts
from utils import safe_filename

log = logging.getLogger("rl_record_orderbooks")
KALSHI = config.KALSHI_BASE_URL
HEADERS = {"Accept": "application/json", "User-Agent": "capstone-market-making-research/1.0"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request(session: requests.Session, method: str, url: str, **kw):
    for attempt in range(1, 5):
        try:
            r = session.request(method, url, timeout=config.REQUEST_TIMEOUT_S, **kw)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{method} {url} -> HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            log.warning("network error %s (attempt %d)", exc, attempt)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{method} {url} failed after retries")


def kalshi_universe(s: requests.Session, per_category: int) -> list[dict]:
    horizon = int(time.time()) + R.RECORD_CLOSING_WITHIN_HOURS * 3600
    out = []
    for category, spec in R.CATEGORIES.items():
        best: dict[str, dict] = {}
        for series in spec["kalshi_series"]:
            cursor = None
            for _ in range(5):
                params = {"series_ticker": series, "status": "open", "max_close_ts": horizon,
                          "limit": 1000, "mve_filter": "exclude"}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = request(s, "GET", KALSHI + "/markets", params=params)
                except RuntimeError as exc:
                    log.warning("kalshi universe %s: %s", series, exc)
                    break
                for m in data.get("markets", []):
                    if m.get("market_type") != "binary":
                        continue
                    vol = float(m.get("volume_24h_fp") or m.get("volume_fp") or 0)
                    ev = m.get("event_ticker") or m["ticker"]
                    if vol > 0 and (ev not in best or vol > best[ev]["volume_24h"]):
                        best[ev] = {"venue": "kalshi", "market_id": m["ticker"], "category": category,
                                    "series_ticker": series, "event_ticker": ev, "title": m.get("title", ""),
                                    "yes_sub_title": m.get("yes_sub_title", ""), "open_time": m.get("open_time"),
                                    "close_time": m.get("close_time"), "volume_24h": vol}
                cursor = data.get("cursor")
                if not cursor:
                    break
                time.sleep(R.KALSHI_REQUEST_DELAY_S)
        out += sorted(best.values(), key=lambda m: -m["volume_24h"])[:per_category]
    return out


def polymarket_universe(s: requests.Session, per_category: int) -> list[dict]:
    horizon = datetime.fromtimestamp(time.time() + R.RECORD_CLOSING_WITHIN_HOURS * 3600, timezone.utc)
    out, seen = [], set()
    for category, spec in R.CATEGORIES.items():
        picks = []
        for tag in spec["polymarket_tags"]:
            try:
                events = request(s, "GET", config.POLYMARKET_GAMMA_URL + "/events", params={
                    "tag_slug": tag, "closed": "false", "active": "true", "limit": 100, "order": "volume24hr",
                    "ascending": "false", "end_date_max": horizon.strftime("%Y-%m-%dT%H:%M:%SZ")})
            except RuntimeError as exc:
                log.warning("polymarket universe %s: %s", tag, exc)
                continue
            for e in events:
                if e["id"] in seen:
                    continue
                best = None
                for m in e.get("markets", []):
                    try:
                        outcomes, tokens = json.loads(m.get("outcomes") or "[]"), json.loads(m.get("clobTokenIds") or "[]")
                    except json.JSONDecodeError:
                        continue
                    if (len(outcomes) == 2 and len(tokens) == 2 and m.get("enableOrderBook")
                            and m.get("acceptingOrders") and not m.get("closed")):
                        vol = float(m.get("volume24hr") or 0)
                        if vol > 0 and (best is None or vol > best["volume_24h"]):
                            best = {"venue": "polymarket", "market_id": m["conditionId"], "category": category,
                                    "series_ticker": tag, "event_ticker": e.get("slug", ""),
                                    "title": m.get("question", ""), "yes_sub_title": outcomes[0],
                                    "no_sub_title": outcomes[1], "yes_token_id": tokens[0], "no_token_id": tokens[1],
                                    "open_time": m.get("startDate"), "close_time": m.get("endDate"),
                                    "volume_24h": vol}
                if best:
                    seen.add(e["id"])
                    picks.append(best)
            time.sleep(R.POLYMARKET_REQUEST_DELAY_S)
        out += sorted(picks, key=lambda m: -m["volume_24h"])[:per_category]
    return out


def poll_kalshi(s, markets):
    rows = []
    for i in range(0, len(markets), 20):
        chunk = markets[i:i + 20]
        started = now_iso()
        data = request(s, "GET", KALSHI + "/markets/orderbooks", params=[("tickers", m["market_id"]) for m in chunk])
        received = now_iso()
        for ob in data.get("orderbooks", []):
            rows.append((ob["ticker"], {"ticker": ob["ticker"], "request_started_utc": started,
                                        "available_at_utc": received, "source": "public_rest_orderbooks_batch",
                                        "sequence_number": None,
                                        "payload": {"orderbook_fp": ob["orderbook_fp"]}}))
    return rows


def poll_polymarket(s, markets):
    rows, by_token = [], {m["yes_token_id"]: m for m in markets}
    tokens = list(by_token)
    for i in range(0, len(tokens), 50):
        started = now_iso()
        books = request(s, "POST", config.POLYMARKET_CLOB_URL + "/books",
                        json=[{"token_id": t} for t in tokens[i:i + 50]])
        received = now_iso()
        for b in books:
            m = by_token.get(str(b.get("asset_id")))
            if not m:
                continue
            fp = {"yes_dollars": [[f"{float(l['price']):.4f}", f"{float(l['size']):.2f}"] for l in b.get("bids", [])],
                  "no_dollars": [[f"{1 - float(l['price']):.4f}", f"{float(l['size']):.2f}"] for l in b.get("asks", [])]}
            rows.append((m["market_id"], {"ticker": m["market_id"], "token_id": m["yes_token_id"],
                                          "request_started_utc": started, "available_at_utc": received,
                                          "exchange_timestamp_ms": b.get("timestamp"), "book_hash": b.get("hash"),
                                          "tick_size": b.get("tick_size"), "source": "public_clob_books_batch",
                                          "sequence_number": None, "payload": {"orderbook_fp": fp}}))
    return rows


class Writer:
    def __init__(self, root: Path):
        self.root, self.last_hash = root, {}

    def write(self, venue: str, market_id: str, category: str, row: dict) -> bool:
        h = hashlib.sha1(json.dumps(row["payload"], sort_keys=True).encode()).hexdigest()
        unchanged = self.last_hash.get((venue, market_id)) == h
        self.last_hash[(venue, market_id)] = h
        row = dict(row, venue=venue, category=category, unchanged=unchanged,
                   payload=None if unchanged else row["payload"])
        day = row["available_at_utc"][:10]
        path = self.root / venue / day / f"{safe_filename(market_id)}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at") as f:
            f.write(json.dumps(row) + "\n")
        return not unchanged


def run(duration_hours: float | None, interval: float, max_per_venue: int) -> None:
    root = R.DATA_DIR / "live_orderbooks"
    root.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(HEADERS)
    writer = Writer(root)
    per_cat = max(1, max_per_venue // len(R.CATEGORIES))
    deadline = time.time() + duration_hours * 3600 if duration_hours else None
    next_refresh, universe, n_polls = 0.0, {"kalshi": [], "polymarket": []}, 0
    while deadline is None or time.time() < deadline:
        t0 = time.time()
        if t0 >= next_refresh:
            universe = {"kalshi": kalshi_universe(s, per_cat), "polymarket": polymarket_universe(s, per_cat)}
            with open(root / "universe.jsonl", "a") as f:
                f.write(json.dumps({"refreshed_utc": now_iso(), **universe}) + "\n")
            log.info("universe: %d kalshi, %d polymarket markets", len(universe["kalshi"]), len(universe["polymarket"]))
            next_refresh = t0 + R.RECORD_UNIVERSE_REFRESH_S
        changed = 0
        for venue, poll in (("kalshi", poll_kalshi), ("polymarket", poll_polymarket)):
            # Kalshi close_time is binding. Polymarket endDate is often the game's
            # scheduled start and trading continues in-play, so Polymarket markets
            # stay tracked until a universe refresh finds them closed.
            live = [m for m in universe[venue] if venue == "polymarket"
                    or (to_ts(m.get("close_time")) or 1e18) > time.time()]
            cats = {m["market_id"]: m["category"] for m in live}
            try:
                for mid, row in poll(s, live):
                    changed += writer.write(venue, mid, cats.get(mid, ""), row)
            except RuntimeError as exc:
                log.warning("%s poll failed: %s", venue, exc)
        n_polls += 1
        if n_polls % 30 == 1:
            log.info("poll %d: %d changed books written", n_polls, changed)
        time.sleep(max(0.0, interval - (time.time() - t0)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration-hours", type=float, help="stop after this long (default: run until Ctrl+C)")
    p.add_argument("--interval", type=float, default=R.RECORD_INTERVAL_S, help="seconds between snapshots")
    p.add_argument("--max-markets-per-venue", type=int, default=R.RECORD_MAX_MARKETS_PER_VENUE)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    if a.interval < 2:
        raise SystemExit("--interval below 2 s is not polite to public endpoints")
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(a.duration_hours, a.interval, a.max_markets_per_venue)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
