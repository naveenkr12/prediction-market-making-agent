"""Paper engine: fills follow the simulator rules; nothing can place an order."""
from decimal import Decimal

import pytest

from dashboard import config
from dashboard.engine import Engine
from dashboard.kalshi_feed import FeedError, KalshiFeed, check_path
from dashboard.strategies import HJB, Context, FixedSpread, InventorySkew, PPOAgent
from simulator.execution import QuoteAction

T0 = 1_800_000_000.0


def book(bids, asks):
    """bids/asks: [(yes price, size)]; asks are stored as NO bids at 1 - price."""
    return {"orderbook_fp": {"yes_dollars": [[f"{p:.4f}", f"{q:.2f}"] for p, q in bids],
                             "no_dollars": [[f"{1 - p:.4f}", f"{q:.2f}"] for p, q in asks]}}


def trade(i, ts, side, price, qty, block=False):
    return {"trade_id": f"t{i}", "ts": ts, "taker_outcome_side": side, "yes_price_dollars": f"{price:.4f}",
            "count_fp": f"{qty:.2f}", "is_block_trade": block}


class FakeFeed:
    request_count = errors = 0


class Fixed:
    """Test strategy that quotes whatever it is told."""
    name, label, available, reason = "fixed", "fixed", True, ""

    def __init__(self, bid, ask, size=10):
        self.action = QuoteAction(bid, ask, size, size)

    def quote(self, ctx):
        return self.action, {}


def make_engine(strategy, bids=((0.40, 50),), asks=((0.44, 30),)):
    e = Engine(FakeFeed(), [strategy], decision_s=300)
    e.add_market({"ticker": "TEST", "title": "t", "open_time": "2027-01-15T00:00:00Z",
                  "close_time": "2027-01-15T12:00:00Z", "price_ranges": [{"step": "0.01"}]}, now=T0)
    e.update_book("TEST", book(bids, asks), T0)
    e.decide(e.markets["TEST"], T0)
    return e, e.markets["TEST"].strategies["fixed"]


def test_join_best_queues_behind_existing_size_and_respects_latency():
    e, ss = make_engine(Fixed(0.40, 0.44))
    assert ss.orders["bid"].order.queue_ahead == 50 and ss.orders["ask"].order.queue_ahead == 30
    e.process_trades("TEST", [trade(1, T0 + 0.5, "yes", 0.44, 500)])     # before latency: ignored
    assert ss.fills == []
    e.process_trades("TEST", [trade(2, T0 + 2, "yes", 0.44, 20)])        # clears 20 of 30 queue
    assert ss.fills == [] and ss.orders["ask"].order.queue_ahead == 10
    e.process_trades("TEST", [trade(3, T0 + 3, "yes", 0.45, 60)])        # 10 queue, then 10% of 50
    assert ss.fills[-1]["qty"] == 5.0 and ss.account.inventory == Decimal("-5")


def test_improving_the_best_has_no_queue_and_wrong_side_never_fills():
    e, ss = make_engine(Fixed(0.41, 0.43))
    assert ss.orders["bid"].order.queue_ahead == 0
    e.process_trades("TEST", [trade(1, T0 + 5, "yes", 0.42, 100)])       # YES taker below our ask
    assert ss.fills == []
    e.process_trades("TEST", [trade(2, T0 + 6, "no", 0.41, 100)])        # NO taker hits our bid
    assert ss.fills[-1]["side"] == "bid" and ss.fills[-1]["qty"] == 10.0  # min(remaining 10, 10% of 100)


def test_duplicate_and_block_trades_are_ignored():
    e, ss = make_engine(Fixed(0.41, 0.43))
    e.process_trades("TEST", [trade(1, T0 + 5, "no", 0.41, 50)])
    e.process_trades("TEST", [trade(1, T0 + 5, "no", 0.41, 50), trade(2, T0 + 6, "no", 0.41, 50, block=True)])
    assert len(ss.fills) == 1


def test_unchanged_price_keeps_queue_and_a_new_price_resets_it():
    strat = Fixed(0.40, 0.44)
    e, ss = make_engine(strat)
    e.process_trades("TEST", [trade(1, T0 + 2, "no", 0.40, 20)])         # queue 50 -> 30
    e.update_book("TEST", book([(0.40, 50)], [(0.44, 30)]), T0 + 300)
    e.decide(e.markets["TEST"], T0 + 300)
    assert ss.orders["bid"].order.queue_ahead == 30                      # kept
    strat.action = QuoteAction(0.39, 0.44, 10, 10)
    e.update_book("TEST", book([(0.40, 50), (0.39, 80)], [(0.44, 30)]), T0 + 600)
    e.decide(e.markets["TEST"], T0 + 600)
    assert ss.orders["bid"].order.queue_ahead == 130                     # 0.40 and 0.39 levels ahead


def test_stale_book_pulls_quotes_at_decision():
    e, ss = make_engine(Fixed(0.40, 0.44))
    e.decide(e.markets["TEST"], T0 + 300)                                # last book is 300 s old
    assert ss.orders == {}


def test_one_sided_book_pulls_quotes():
    e, ss = make_engine(Fixed(0.40, 0.44))
    e.update_book("TEST", {"orderbook_fp": {"yes_dollars": [["0.4000", "10.00"]], "no_dollars": []}}, T0 + 10)
    assert ss.orders == {} and not e.markets["TEST"].healthy


def test_settlement_pays_inventory():
    e, ss = make_engine(Fixed(0.41, 0.43))
    e.process_trades("TEST", [trade(1, T0 + 5, "no", 0.41, 100)])        # bought 10 at 0.41
    e.update_status("TEST", {"status": "settled", "result": "yes"}, T0 + 9999)
    assert ss.account.settled and ss.orders == {}
    fee = Decimal(str(ss.fills[0]["fee"]))
    assert ss.account.cash == Decimal(config.INITIAL_CASH) - Decimal("4.1") - fee + 10


def test_feed_refuses_orders_and_other_hosts():
    with pytest.raises(FeedError):
        check_path(config.PROD_REST, "/portfolio/events/orders")
    with pytest.raises(FeedError):
        check_path("https://external-api.demo.kalshi.co/trade-api/v2", "/markets")
    check_path(config.PROD_REST, "/markets/orderbooks")
    assert not any(hasattr(KalshiFeed, m) for m in ("post", "delete", "put", "_post"))


@pytest.mark.parametrize("cls", [HJB, FixedSpread, InventorySkew, PPOAgent])
def test_strategies_return_post_only_quotes(cls):
    s = cls()
    if not s.available:
        pytest.skip(s.reason)
    for now in (T0 - 7200, T0 - 1800, T0):                               # before, inside and at end of episode
        ctx = Context(0.40, 0.44, 0.0, now, T0 - 86400, T0)
        action, _ = s.quote(ctx)
        assert action.bid is None or action.bid < 0.44
        assert action.ask is None or action.ask > 0.40
        assert action.bid_size <= 10 and action.ask_size <= 10
