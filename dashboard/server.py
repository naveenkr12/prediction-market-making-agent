"""Live paper-quoting dashboard.

Run from the repository root:
    python -m dashboard.server                    # auto-picks liquid markets closing soon
    python -m dashboard.server --tickers KXNFLGAME-26NOV16BUFKC-KC
    python -m dashboard.server --decision-seconds 60 --max-markets 4

Then open http://127.0.0.1:8050. Market data comes from Kalshi's public
production API; no orders are ever sent. Every quote, fill and settlement is
appended to dashboard/runs/<start time>/ledger.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import config, strategies                      # noqa: E402
from dashboard.engine import Engine                           # noqa: E402
from dashboard.kalshi_feed import FeedError, KalshiFeed       # noqa: E402

log = logging.getLogger("dashboard.server")
STATIC = Path(__file__).resolve().parent / "static"


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self):
            path = unquote(self.path.split("?")[0])
            if path in ("/", "/index.html"):
                path = "/static/index.html"
            if path == "/api/state":
                return self._json(engine.snapshot())
            if path.startswith("/api/market/"):
                detail = engine.market_detail(path.rsplit("/", 1)[1])
                return self._json(detail) if detail else self._json({"error": "unknown market"}, 404)
            if path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if STATIC in f.parents and f.is_file():
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "text/plain")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args):   # keep the console for engine logs
            pass

    return Handler


def run_engine(engine: Engine, stop: threading.Event):
    while not stop.is_set():
        try:
            engine.tick()
        except Exception:                # keep the dashboard alive; the error is logged
            log.exception("engine tick failed")
        stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", help="explicit market tickers (default: auto-discover)")
    ap.add_argument("--max-markets", type=int, default=config.MAX_MARKETS)
    ap.add_argument("--close-within-hours", type=float, default=config.CLOSE_WITHIN_HOURS)
    ap.add_argument("--decision-seconds", type=float, default=config.DECISION_S,
                    help="seconds between strategy decisions (training used 300)")
    ap.add_argument("--no-ppo", action="store_true", help="skip loading the PPO agent")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    feed = KalshiFeed()
    strats = strategies.build(include_ppo=not a.no_ppo)
    for s in strats:
        log.info("strategy %-6s %s %s", s.name, "ready" if s.available else "UNAVAILABLE", s.reason)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engine = Engine(feed, strats, decision_s=a.decision_seconds, run_dir=config.RUNS_DIR / stamp)

    if a.tickers:
        markets = []
        for t in a.tickers:
            try:
                markets.append(feed.market(t))
            except FeedError as exc:
                log.error("cannot load %s: %s", t, exc)
    else:
        log.info("discovering liquid markets closing within %.0f h ...", a.close_within_hours)
        markets = feed.discover(config.SERIES, a.close_within_hours, config.MIN_VOLUME_24H, a.max_markets)
    if not markets:
        raise SystemExit("No markets found. Try --close-within-hours 48 or pass --tickers.")
    for m in markets:
        engine.add_market(m)
        log.info("tracking %s  %s", m["ticker"], m.get("title", "")[:70])

    stop = threading.Event()
    threading.Thread(target=run_engine, args=(engine, stop), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine))
    log.info("dashboard at http://127.0.0.1:%d  (paper only, no orders sent)", a.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
