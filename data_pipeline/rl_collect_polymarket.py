"""
Collect a dense, labelled Polymarket sample in the same categories as Kalshi.

  Gamma API  /events?tag_slug=...   resolved events per category tag
  Data API   /trades                taker trade tape (offset is capped at
                                    10,000, so the window is walked backwards
                                    in time with `end`)
  CLOB API   /prices-history        1-minute YES midpoint (Polymarket
                                    publishes no historical bid/ask)

YES is outcome index 0 (for games, the first-listed team). Trades are written
in Kalshi's raw trade schema with YES prices and a YES/NO taker side, so
calibration_quality.normalize_trades and the bar builder treat both venues
alike. Wallet addresses, user names and profile fields are dropped.

Run from data_pipeline/:
    python3 rl_collect_polymarket.py
    python3 rl_collect_polymarket.py --categories Sports --target 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config
import rl_config as R
from calibration_quality import normalize_trades
from rl_collect_kalshi import density, save_json
from utils import safe_filename

log = logging.getLogger("rl_collect_polymarket")
GAMMA, DATA, CLOB = config.POLYMARKET_GAMMA_URL, config.POLYMARKET_DATA_API_URL, config.POLYMARKET_CLOB_URL
OFFSET_CAP, PAGE = 10_000, 1000


class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json",
                               "User-Agent": "capstone-market-making-research/1.0"})
        self.request_count = 0

    def get(self, base: str, path: str, params: dict | None = None):
        for attempt in range(1, config.MAX_RETRIES + 2):
            self.request_count += 1
            try:
                r = self.s.get(base + path, params=params, timeout=config.REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                err = str(exc)
            else:
                if r.status_code == 200:
                    time.sleep(R.POLYMARKET_REQUEST_DELAY_S)
                    return r.json()
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"GET {path} -> HTTP {r.status_code}: {r.text[:200]}")
                err = f"HTTP {r.status_code}"
            log.warning("%s on %s (attempt %d)", err, path, attempt)
            time.sleep(config.BACKOFF_BASE_S ** attempt)
        raise RuntimeError(f"GET {path} failed after retries")


def to_ts(value) -> int | None:
    if not value:
        return None
    t = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(t) else int(t.timestamp())


def binary_market(m: dict) -> dict | None:
    """Two-outcome order-book market with a clean 1/0 resolution."""
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        tokens = json.loads(m.get("clobTokenIds") or "[]")
        prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2 or not m.get("enableOrderBook"):
        return None
    if sorted(prices) != [0.0, 1.0]:
        return None   # unresolved or split resolution
    close_ts = to_ts(m.get("closedTime")) or to_ts(m.get("umaEndDate")) or to_ts(m.get("endDate"))
    open_ts = (to_ts(m.get("acceptingOrdersTimestamp")) or to_ts(m.get("startDate"))
               or to_ts(m.get("createdAt")))
    if not close_ts or not open_ts or open_ts >= close_ts:
        return None
    return {"outcomes": outcomes, "tokens": tokens, "y": int(prices[0] == 1.0),
            "open_ts": open_ts, "close_ts": close_ts, "volume": float(m.get("volumeNum") or 0)}


def topical(spec: dict, series_slug: str, question: str) -> bool:
    """Category allowlist where tags are too broad (see rl_config Economics)."""
    if "polymarket_series_allow" not in spec:
        return True
    return (series_slug in spec["polymarket_series_allow"]
            or bool(re.search(spec["polymarket_title_regex"], question, re.IGNORECASE)))


def discover(c: Client, category: str, start_ts: int, end_ts: int, seen_events: set) -> list[dict]:
    spec = R.CATEGORIES[category]
    found = []
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for tag in spec["polymarket_tags"]:
        offset = 0
        while offset < 2000:
            try:
                events = c.get(GAMMA, "/events", {"tag_slug": tag, "closed": "true", "limit": 100,
                                                  "offset": offset, "order": "volume", "ascending": "false",
                                                  "end_date_min": iso(start_ts), "end_date_max": iso(end_ts)})
            except RuntimeError as exc:
                log.warning("tag %s: %s", tag, exc)
                break
            if not events:
                break
            for e in events:
                if e["id"] in seen_events:
                    continue
                best = None
                for m in e.get("markets", []):
                    b = binary_market(m)
                    if b and start_ts <= b["close_ts"] <= end_ts and b["volume"] >= R.MIN_REPORTED_VOLUME:
                        if best is None or b["volume"] > best[1]["volume"]:
                            best = (m, b)
                series_slug = (e.get("series") or [{}])[0].get("slug", "") if e.get("series") else ""
                if best and not topical(spec, series_slug, best[0].get("question", "")):
                    best = None
                if best and R.POLYMARKET_SERIES_CATEGORY.get(series_slug, category) == category:
                    seen_events.add(e["id"])
                    m, b = best
                    found.append({"category": category, "tag": tag, "event_id": e["id"],
                                  "event_slug": e.get("slug", ""), "event_title": e.get("title", ""),
                                  "series_slug": series_slug,
                                  "market": m, **b})
            offset += 100
        log.info("%s / tag %s: %d candidate markets so far", category, tag, len(found))
    found.sort(key=lambda f: -f["volume"])
    return found


def fetch_trades(c: Client, cid: str, start: int, end: int) -> tuple[list[dict], bool]:
    """Walk backwards in time so the offset cap is never the binding limit."""
    rows, seen, complete = [], set(), True
    while True:
        oldest, offset, n_page = None, 0, PAGE
        while n_page == PAGE and offset < OFFSET_CAP:
            page = c.get(DATA, "/trades", {"market": cid, "takerOnly": "true", "limit": PAGE,
                                           "offset": offset, "start": start, "end": end})
            n_page = len(page)
            for t in page:
                key = (t.get("transactionHash"), t.get("asset"), t.get("side"), t.get("price"),
                       t.get("size"), t.get("timestamp"))
                if key not in seen:
                    seen.add(key)
                    rows.append(t)
                ts = int(t["timestamp"])
                oldest = ts if oldest is None else min(oldest, ts)
            offset += PAGE
        if n_page < PAGE or oldest is None:
            return rows, complete
        if oldest >= end:            # >10k trades inside one second: cannot page further
            return rows, False
        end = oldest                 # overlap that second; duplicates are removed by key


def to_kalshi_schema(rows: list[dict], cid: str, yes_token: str) -> pd.DataFrame:
    out = []
    for t in rows:
        is_yes_token = str(t.get("asset")) == yes_token
        price = float(t["price"])
        yes_price = price if is_yes_token else 1 - price
        buy = str(t.get("side")).upper() == "BUY"
        taker_yes = buy == is_yes_token        # buy YES or sell NO = YES taker
        key = "|".join(str(t.get(k)) for k in ["transactionHash", "asset", "side", "price", "size", "timestamp"])
        out.append({"trade_id": hashlib.sha1(key.encode()).hexdigest()[:20], "ticker": cid,
                    "created_time": datetime.fromtimestamp(int(t["timestamp"]), timezone.utc).isoformat(),
                    "yes_price_dollars": f"{yes_price:.4f}", "no_price_dollars": f"{1 - yes_price:.4f}",
                    "count_fp": f"{float(t['size']):.2f}", "taker_outcome_side": "yes" if taker_yes else "no",
                    "taker_side": "yes" if taker_yes else "no", "is_block_trade": False,
                    "pm_side": t.get("side"), "pm_outcome": t.get("outcome"),
                    "pm_outcome_index": t.get("outcomeIndex"), "pm_asset": str(t.get("asset")),
                    "transaction_hash": t.get("transactionHash")})
    return pd.DataFrame(out)


def fetch_price_history(c: Client, token: str, start: int, end: int) -> pd.DataFrame:
    pts = []
    chunk = 86400
    for lo in range(start, end, chunk):
        h = c.get(CLOB, "/prices-history", {"market": token, "startTs": lo, "endTs": min(end, lo + chunk),
                                            "fidelity": 1})
        pts += h.get("history", [])
    df = pd.DataFrame(pts, columns=["t", "p"]).drop_duplicates("t").sort_values("t")
    return df.rename(columns={"t": "ts", "p": "yes_mid"}).reset_index(drop=True)


def collect_contract(c: Client, cand: dict, root: Path) -> dict:
    m = cand["market"]
    cid, fname = m["conditionId"], safe_filename(m["conditionId"])
    status_path = root / "status" / f"{fname}.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("complete"):
        return json.loads(status_path.read_text())
    hours = R.CATEGORIES[cand["category"]]["episode_hours"]
    ep_end = cand["close_ts"]
    ep_start = max(cand["open_ts"], ep_end - int(hours * 3600))
    st = {"ticker": cid, "category": cand["category"], "episode_start_ts": ep_start, "episode_end_ts": ep_end,
          "episode_hours": hours, "complete": False, "error": None,
          "collected_at": datetime.now(timezone.utc).isoformat()}
    try:
        rows, tape_complete = fetch_trades(c, cid, ep_start, ep_end)
        raw = to_kalshi_schema(rows, cid, cand["tokens"][0])
        raw.to_parquet(root / "trades" / f"{fname}.parquet", index=False)
        ph = fetch_price_history(c, cand["tokens"][0], ep_start - 3600, ep_end + 60)
        ph.to_parquet(root / "price_history" / f"{fname}.parquet", index=False)
        norm, stats = normalize_trades(raw, cid) if len(raw) else (pd.DataFrame(
            columns=["timestamp", "quantity"]), {"raw_trade_rows": 0})
        st.update(stats)
        # No historical quote updates on Polymarket: activity counts trades only.
        st.update(density(norm, pd.DataFrame(columns=["end_period_ts"]), ep_start, ep_end))
        st.update(tape_complete=tape_complete, price_history_rows=len(ph))
        reasons = []
        if not tape_complete:
            reasons.append("tape_incomplete")
        if st["episode_trades"] < R.MIN_EPISODE_TRADES:
            reasons.append(f"episode_trades {st['episode_trades']} < {R.MIN_EPISODE_TRADES}")
        if st["active_bar_share"] < R.MIN_ACTIVE_BAR_SHARE:
            reasons.append(f"active_bar_share {st['active_bar_share']:.2f} < {R.MIN_ACTIVE_BAR_SHARE}")
        if len(ph) == 0:
            reasons.append("no_price_history")
        st.update(accepted=not reasons, rejection_reasons=reasons, complete=True)
        if reasons:
            # Keep the decision (status file), not the unusable raw data.
            for sub in ("trades", "price_history"):
                (root / sub / f"{fname}.parquet").unlink(missing_ok=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        st.update(error=str(exc), accepted=False, rejection_reasons=["download_error"])
    save_json(status_path, st)
    return st


def contract_row(cand: dict, st: dict) -> dict:
    m = cand["market"]
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat()
    return {"venue": "polymarket", "market_id": m["conditionId"], "ticker": m["conditionId"],
            "category": cand["category"], "series_ticker": cand["series_slug"] or cand["tag"],
            "series_title": cand["tag"], "event_ticker": cand["event_slug"], "title": m.get("question", ""),
            "subtitle": cand["event_title"], "yes_sub_title": cand["outcomes"][0],
            "no_sub_title": cand["outcomes"][1], "rules_primary": (m.get("description") or "")[:4000],
            "open_time": iso(cand["open_ts"]), "close_time": iso(cand["close_ts"]),
            "expected_expiration_time": m.get("endDate", ""), "settlement_ts": m.get("umaEndDate", "") or "",
            "status": "resolved", "result": "yes" if cand["y"] else "no", "market_type": "binary",
            "y_outcome": cand["y"], "reported_volume": cand["volume"], "volume_fp": str(cand["volume"]),
            "yes_token_id": cand["tokens"][0], "no_token_id": cand["tokens"][1],
            "price_tick": str(m.get("orderPriceMinTickSize", "")), "neg_risk": bool(m.get("negRisk")),
            "slug": m.get("slug", ""), "archive_source": "live",
            "episode_start_utc": iso(st["episode_start_ts"]), "episode_end_utc": iso(st["episode_end_ts"]),
            "episode_hours": st["episode_hours"],
            **{k: st.get(k) for k in ["episode_bars", "episode_trades", "episode_contracts", "bars_with_trade",
                                      "bars_with_quote_update", "active_bar_share", "raw_trade_rows",
                                      "block_trade_rows", "price_history_rows"]}}


def run(categories: list[str], target: int | None, end_ts: int) -> dict:
    root = R.DATA_DIR / "polymarket"
    for sub in ["markets", "trades", "price_history", "status", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    start_ts = end_ts - R.LOOKBACK_DAYS * 86400
    c = Client()
    contracts_path = root / "markets" / "contracts.parquet"
    existing = pd.read_parquet(contracts_path) if contracts_path.exists() else pd.DataFrame()
    rows = [] if existing.empty else existing[~existing.category.isin(categories)].to_dict("records")
    seen_events: set = set()
    summary = {"run_started_utc": datetime.now(timezone.utc).isoformat(), "window_start_ts": start_ts,
               "window_end_ts": end_ts, "target_per_category": target, "categories": {}}
    for category in categories:
        cat_start = end_ts - R.CATEGORIES[category].get("lookback_days", R.LOOKBACK_DAYS) * 86400
        cands = discover(c, category, cat_start, end_ts, seen_events)
        save_json(root / "markets" / f"candidates_{category}.json",
                  [{"condition_id": x["market"]["conditionId"], "question": x["market"].get("question"),
                    "tag": x["tag"], "volume": x["volume"], "close_ts": x["close_ts"]} for x in cands])
        accepted, per_series, attempts = [], {}, []
        spec = R.CATEGORIES[category]
        cat_target = target or spec.get("polymarket_target", R.POLYMARKET_TARGET_PER_CATEGORY)
        for cand in cands:
            if len(accepted) >= cat_target:
                break
            key = cand["series_slug"] or cand["tag"]
            if per_series.get(key, 0) >= spec.get("max_per_series", R.MAX_PER_SERIES):
                continue
            st = collect_contract(c, cand, root)
            attempts.append({"market_id": cand["market"]["conditionId"], "accepted": st.get("accepted"),
                             "reasons": st.get("rejection_reasons"), "episode_trades": st.get("episode_trades")})
            log.info("%s %s %s: trades=%s active=%.2f %s", category, "ACCEPT" if st.get("accepted") else "REJECT",
                     cand["market"].get("question", "")[:50], st.get("episode_trades"),
                     st.get("active_bar_share") or 0, st.get("rejection_reasons") or "")
            if st.get("accepted"):
                accepted.append(contract_row(cand, st))
                per_series[key] = per_series.get(key, 0) + 1
        rows += accepted
        summary["categories"][category] = {"target": cat_target, "candidates": len(cands),
                                           "attempted": len(attempts),
                                           "accepted": len(accepted), "by_series": per_series,
                                           "attempts": attempts}
        pd.DataFrame(rows).to_parquet(contracts_path, index=False)
    summary["api_requests"] = c.request_count
    save_json(root / "logs" / f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=list(R.CATEGORIES), choices=list(R.CATEGORIES))
    p.add_argument("--target", type=int, help="accepted contracts per category (default: per-category polymarket_target)")
    p.add_argument("--end-date", help="inclusive UTC date YYYY-MM-DD (default: now)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    end_ts = (int(datetime.fromisoformat(a.end_date).replace(tzinfo=timezone.utc).timestamp()) + 86399
              if a.end_date else int(time.time()))
    s = run(a.categories, a.target, end_ts)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "attempts"} for k, v in s["categories"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
