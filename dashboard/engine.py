"""Paper-quoting engine: real Kalshi books and trades, simulated maker fills.

For every market and strategy the engine keeps resting quotes and an account.
Fills use the simulator's own rules (simulator.execution.RestingOrder.match and
simulator.accounting.BinaryAccount), with one live adaptation: a real trade's
own price is the execution limit. A YES taker trade at price P can fill our
ask only if the ask is at or below P; a NO taker trade can fill our bid only
if the bid is at or above P. Volume first clears the queue ahead of us
(resting size at our price and better when the quote was placed), then we get
at most PARTICIPATION of the remainder. Book changes alone never create fills.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from simulator.accounting import BinaryAccount
from simulator.execution import BookSnapshot, QuoteAction, RestingOrder
from simulator.fees import decimal, floor_grid

from . import config
from .kalshi_feed import FeedError, effective_close, market_tick, parse_ts
from .strategies import Context

log = logging.getLogger("dashboard.engine")
LOT = Decimal("0.01")


@dataclass
class LiveOrder:
    order: RestingOrder
    active_from: float
    placed_at: float
    queue_at_placement: Decimal


@dataclass
class StrategyState:
    account: BinaryAccount
    orders: dict = field(default_factory=dict)          # side -> LiveOrder
    quotes: deque = field(default_factory=lambda: deque(maxlen=config.HISTORY_POINTS))
    fills: list = field(default_factory=list)
    last_info: dict = field(default_factory=dict)
    order_number: int = 0
    note: str = ""


@dataclass
class MarketState:
    meta: dict
    ticker: str
    title: str
    subtitle: str
    series: str
    open_ts: float
    close_ts: float
    tick: str
    book: BookSnapshot | None = None
    book_ts: float = 0.0
    healthy: bool = False
    last_mark: float | None = None       # last two-sided mid, used while the book is unusable
    history: deque = field(default_factory=lambda: deque(maxlen=config.HISTORY_POINTS))
    trades: deque = field(default_factory=lambda: deque(maxlen=600))
    seen_ids: deque = field(default_factory=lambda: deque(maxlen=5000))
    last_trade_ts: float = 0.0
    last_decision: float = 0.0
    status: str = "open"
    result: str | None = None
    strategies: dict = field(default_factory=dict)       # name -> StrategyState


def num(x):
    return float(x) if isinstance(x, Decimal) else x


class Engine:
    def __init__(self, feed, strategies, decision_s=config.DECISION_S, run_dir: Path | None = None):
        self.feed, self.strategies, self.decision_s = feed, strategies, decision_s
        self.markets: dict[str, MarketState] = {}
        self.lock = threading.RLock()
        self.last_book_poll = self.last_trade_poll = self.last_status_poll = 0.0
        self.started = time.time()
        self.events = deque(maxlen=200)
        self.ledger_path = None
        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.ledger_path = run_dir / "ledger.jsonl"

    # ----------------------------------------------------------- setup
    def add_market(self, m: dict, now: float | None = None):
        now = now or time.time()
        ms = MarketState(meta=m, ticker=m["ticker"], title=m.get("title", ""), subtitle=m.get("yes_sub_title", ""),
                         series=m.get("series_ticker", m.get("event_ticker", "")), open_ts=parse_ts(m["open_time"]),
                         close_ts=effective_close(m), tick=market_tick(m), last_trade_ts=now)
        for s in self.strategies:
            ms.strategies[s.name] = StrategyState(
                account=BinaryAccount(config.INITIAL_CASH, config.INVENTORY_LIMIT, "0.01"),
                note="" if s.available else s.reason)
        with self.lock:
            self.markets[ms.ticker] = ms
        self._log("market_added", ticker=ms.ticker, title=ms.title, close_ts=ms.close_ts, tick=ms.tick)

    # ----------------------------------------------------------- main loop
    def tick(self, now: float | None = None):
        now = now or time.time()
        if now - self.last_book_poll >= config.BOOK_POLL_S:
            self.last_book_poll = now
            self._poll_books(now)
        if now - self.last_trade_poll >= config.TRADE_POLL_S:
            self.last_trade_poll = now
            self._poll_trades(now)
        for ms in list(self.markets.values()):
            if ms.status == "open" and now - ms.last_decision >= self.decision_s:
                self.decide(ms, now)
        if now - self.last_status_poll >= config.STATUS_POLL_S:
            self.last_status_poll = now
            self._poll_status(now)

    def _poll_books(self, now):
        open_tickers = [t for t, m in self.markets.items() if m.status == "open"]
        if not open_tickers:
            return
        try:
            books = self.feed.books(open_tickers)
        except FeedError as exc:
            self._log("error", where="books", error=str(exc))
            return
        for t, payload in books.items():
            self.update_book(t, payload, now)

    def update_book(self, ticker, payload, now):
        ms = self.markets[ticker]
        with self.lock:
            try:
                ms.book = BookSnapshot.from_kalshi(payload, now)
                ms.healthy = True
            except ValueError:
                # One side empty: no reliable mid, so all strategies stop quoting.
                ms.healthy = False
                for name in ms.strategies:
                    self._cancel_all(ms, name, now, "book one-sided")
            ms.book_ts = now
            if ms.healthy:
                b = ms.book
                ms.last_mark = (b.bid + b.ask) / 2
                ms.history.append((now, b.bid, b.ask, ms.last_mark))

    def _poll_trades(self, now):
        for t, ms in list(self.markets.items()):
            if ms.status != "open":
                continue
            try:
                trades = self.feed.trades(t, int(ms.last_trade_ts))
            except FeedError as exc:
                self._log("error", where="trades", ticker=t, error=str(exc))
                continue
            self.process_trades(t, trades)

    def process_trades(self, ticker, trades):
        ms = self.markets[ticker]
        with self.lock:
            for tr in trades:
                if tr["trade_id"] in ms.seen_ids:
                    continue
                ms.seen_ids.append(tr["trade_id"])
                ms.last_trade_ts = max(ms.last_trade_ts, tr["ts"])
                ms.trades.append({"ts": tr["ts"], "price": float(tr["yes_price_dollars"]),
                                  "qty": float(tr["count_fp"]), "side": tr.get("taker_outcome_side")})
                if tr.get("is_block_trade") or tr.get("taker_outcome_side") not in ("yes", "no"):
                    continue
                for name, ss in ms.strategies.items():
                    self._match(ms, name, ss, tr)

    def _match(self, ms, name, ss, tr):
        sign = 1 if tr["taker_outcome_side"] == "yes" else -1    # YES taker lifts asks; NO taker hits bids
        side = "ask" if sign == 1 else "bid"
        lo = ss.orders.get(side)
        if lo is None or tr["ts"] < lo.active_from or lo.order.remaining <= 0:
            return
        order, queue_before = lo.order, lo.order.queue_ahead
        qty = order.match(sign, decimal(tr["count_fp"]), decimal(tr["yes_price_dollars"]),
                          decimal(config.PARTICIPATION), LOT)
        acct = ss.account
        capacity = acct.inventory_limit - acct.inventory if side == "bid" else acct.inventory_limit + acct.inventory
        qty = floor_grid(min(qty, capacity), LOT)
        if qty <= 0:
            return
        try:
            booked = acct.fill(side, order.price, qty, order.rounding_accumulator)
        except ValueError as exc:
            self._log("fill_rejected", ticker=ms.ticker, strategy=name, side=side, error=str(exc))
            return
        order.remaining -= qty
        order.rounding_accumulator = booked["accumulator"]
        fill = {"ts": tr["ts"], "side": side, "price": float(order.price), "qty": float(qty),
                "trade_price": float(tr["yes_price_dollars"]), "trade_qty": float(tr["count_fp"]),
                "queue_before": float(queue_before), "queue_after": float(order.queue_ahead),
                "fee": float(booked["fee"]), "inventory": float(acct.inventory), "cash": float(acct.cash)}
        ss.fills.append(fill)
        self._log("fill", ticker=ms.ticker, strategy=name, trade_id=tr["trade_id"], **fill)

    # ----------------------------------------------------------- decisions
    def decide(self, ms: MarketState, now: float):
        with self.lock:
            if not ms.healthy or ms.book is None or now - ms.book_ts > 60:
                # Pull quotes, but do not use up the decision slot: quote again
                # as soon as a usable book arrives instead of a full interval later.
                for name in ms.strategies:
                    self._cancel_all(ms, name, now, "book unavailable or stale")
                return
            prev_decision = ms.last_decision
            ms.last_decision = now
            since = max(prev_decision, now - self.decision_s)
            recent = tuple(t for t in ms.trades if t["ts"] >= since)
            for s in self.strategies:
                ss = ms.strategies[s.name]
                if not s.available:
                    continue
                ctx = Context(ms.book.bid, ms.book.ask, float(ss.account.inventory), now, ms.open_ts, ms.close_ts, ms.tick,
                              ticker=ms.ticker, trades=recent)
                try:
                    action, info = s.quote(ctx)
                except Exception as exc:   # a strategy failing must not stop the others
                    ss.note = f"error: {exc}"
                    self._cancel_all(ms, s.name, now, "strategy error")
                    self._log("strategy_error", ticker=ms.ticker, strategy=s.name, error=str(exc))
                    continue
                ss.note = ""
                ss.last_info = {**info, "episode_fraction": round(ctx.episode_fraction, 4),
                                "probability": round(ctx.probability, 4)}
                self.place(ms, s.name, action, now)

    def place(self, ms: MarketState, name: str, action: QuoteAction, now: float):
        ss = ms.strategies[name]
        requests = {side: (getattr(action, side) if getattr(action, side) is not None else 0.5,
                           getattr(action, side + "_size")) for side in ("bid", "ask")}
        admitted, _ = ss.account.admit_quotes(requests)
        for side in ("bid", "ask"):
            price, qty = getattr(action, side), admitted[side]
            current = ss.orders.get(side)
            if price is None or qty <= 0:
                if current:
                    self._cancel(ms, name, side, now, "no quote")
                continue
            p = decimal(price)
            if current and current.order.price == p and current.order.remaining > 0:
                continue                      # unchanged price keeps its queue position
            if current:
                self._cancel(ms, name, side, now, "replaced")
            ss.order_number += 1
            queue = ms.book.queue_ahead(side, p, config.QUEUE_PROXY)
            ss.orders[side] = LiveOrder(RestingOrder(ss.order_number, side, p, qty, queue),
                                        active_from=now + config.LATENCY_S, placed_at=now, queue_at_placement=queue)
            self._log("quote", ticker=ms.ticker, strategy=name, side=side, price=float(p), qty=float(qty),
                      queue_ahead=float(queue))
        bid, ask = ss.orders.get("bid"), ss.orders.get("ask")
        ss.quotes.append((now, float(bid.order.price) if bid else None, float(ask.order.price) if ask else None))

    def _cancel(self, ms, name, side, now, reason):
        lo = ms.strategies[name].orders.pop(side, None)
        if lo:
            self._log("cancel", ticker=ms.ticker, strategy=name, side=side, reason=reason,
                      remaining=float(lo.order.remaining))

    def _cancel_all(self, ms, name, now, reason):
        ss = ms.strategies[name]
        if ss.orders:
            for side in list(ss.orders):
                self._cancel(ms, name, side, now, reason)
            ss.quotes.append((now, None, None))

    # ----------------------------------------------------------- settlement
    def _poll_status(self, now):
        # Every unsettled market, not only those near their expected close:
        # games and other early-close markets can stop trading sooner.
        for t, ms in list(self.markets.items()):
            if ms.status == "settled":
                continue
            try:
                m = self.feed.market(t)
            except FeedError:
                continue
            self.update_status(t, m, now)

    def update_status(self, ticker, m, now):
        ms = self.markets[ticker]
        with self.lock:
            status = m.get("status", "")
            if status in ("closed", "determined") and ms.status == "open":
                ms.status = "closed"
                for name in ms.strategies:
                    self._cancel_all(ms, name, now, "market closed")
            if status in ("settled", "finalized") and m.get("result") in ("yes", "no") and ms.status != "settled":
                for name in ms.strategies:
                    self._cancel_all(ms, name, now, "market settled")
                ms.status, ms.result = "settled", m["result"]
                outcome = 1 if m["result"] == "yes" else 0
                for name, ss in ms.strategies.items():
                    if not ss.account.settled:
                        r = ss.account.settle(outcome)
                        self._log("settlement", ticker=ticker, strategy=name, outcome=outcome,
                                  realized_pnl=float(r["realized_pnl"]))

    # ----------------------------------------------------------- views
    def _mark(self, ms):
        # Keep marking at the last known mid when the book is briefly one-sided
        # or empty (e.g. between close and settlement), so PnL never vanishes.
        return ms.last_mark

    def _strategy_view(self, s, ms, ss):
        mark = self._mark(ms)
        acct = ss.account
        equity = float(acct.cash) if acct.settled else (float(acct.equity(mark)) if mark is not None else None)
        orders = {side: {"price": float(lo.order.price), "remaining": float(lo.order.remaining),
                         "queue_ahead": float(lo.order.queue_ahead), "queue_at_placement": float(lo.queue_at_placement),
                         "active": time.time() >= lo.active_from} for side, lo in ss.orders.items()}
        return {"name": s.name, "label": s.label, "available": s.available, "note": ss.note,
                "orders": orders, "inventory": float(acct.inventory), "cash": float(acct.cash),
                "fees": float(acct.total_fees), "fills": len(ss.fills),
                "pnl": None if equity is None else equity - config.INITIAL_CASH, "info": ss.last_info}

    def snapshot(self):
        with self.lock:
            markets = []
            for ms in self.markets.values():
                b = ms.book if ms.healthy else None
                markets.append({"ticker": ms.ticker, "title": ms.title, "subtitle": ms.subtitle, "series": ms.series,
                                "close_ts": ms.close_ts, "status": ms.status, "result": ms.result,
                                "bid": b.bid if b else None, "ask": b.ask if b else None,
                                "book_age_s": time.time() - ms.book_ts if ms.book_ts else None,
                                "healthy": ms.healthy, "trades_seen": len(ms.seen_ids),
                                "next_decision_s": max(0, ms.last_decision + self.decision_s - time.time()),
                                "strategies": [self._strategy_view(s, ms, ms.strategies[s.name]) for s in self.strategies]})
            totals = {}
            for s in self.strategies:
                pnls = [v for m in markets for st in m["strategies"] if st["name"] == s.name
                        for v in [st["pnl"]] if v is not None]
                fills = sum(st["fills"] for m in markets for st in m["strategies"] if st["name"] == s.name)
                totals[s.name] = {"label": s.label, "available": s.available, "pnl": sum(pnls), "fills": fills}
            return {"now": time.time(), "started": self.started, "decision_s": self.decision_s,
                    "requests": self.feed.request_count, "feed_errors": self.feed.errors,
                    "markets": markets, "totals": totals, "events": list(self.events)[-30:],
                    "settings": {"participation": config.PARTICIPATION, "latency_s": config.LATENCY_S,
                                 "quote_size": config.QUOTE_SIZE, "episode_s": config.EPISODE_S}}

    def market_detail(self, ticker):
        with self.lock:
            ms = self.markets.get(ticker)
            if ms is None:
                return None
            levels = None
            if ms.book is not None and ms.healthy:
                bids = sorted(ms.book.yes_bids, key=lambda x: -x[0])[:10]
                asks = sorted(ms.book.yes_asks, key=lambda x: x[0])[:10]
                levels = {"bids": bids, "asks": asks}
            return {"ticker": ticker, "history": list(ms.history), "trades": list(ms.trades)[-300:], "levels": levels,
                    "quotes": {n: list(ss.quotes) for n, ss in ms.strategies.items()},
                    "fills": {n: ss.fills[-50:] for n, ss in ms.strategies.items()}}

    def _log(self, kind, **values):
        row = {"ts": time.time(), "kind": kind, **{k: num(v) for k, v in values.items()}}
        self.events.append(row)
        if self.ledger_path:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
