"""Post-only YES quotes and a volume-conserving queue scenario.

Historical minute candles do not identify queue depth. Proxy depth and capture
are configurable assumptions. L2 snapshots can supply initial queue volume;
without ordered deltas/cancellations this still is not an exchange replay.
"""
from dataclasses import dataclass
from decimal import Decimal
import math
from .fees import decimal, floor_grid


@dataclass(frozen=True)
class BookSnapshot:
    bid: float
    ask: float
    yes_bids: tuple = ()
    yes_asks: tuple = ()
    timestamp: str | None = None

    def __post_init__(self):
        if not (math.isfinite(self.bid) and math.isfinite(self.ask) and 0 <= self.bid < self.ask <= 1):
            raise ValueError("Require an uncrossed two-sided YES book")
        for levels in (self.yes_bids, self.yes_asks):
            for p, qty in levels:
                if not 0 <= decimal(p) <= 1 or decimal(qty) <= 0:
                    raise ValueError("Invalid depth level")

    @classmethod
    def from_kalshi(cls, payload, timestamp=None):
        """Current orderbook_fp dollars/count_fp; asks = 1 - NO bids."""
        book = payload["orderbook_fp"]
        bids = tuple((float(p), float(q)) for p, q in book["yes_dollars"] if decimal(q)>0)
        asks = tuple((float(1-decimal(p)), float(q)) for p, q in book["no_dollars"] if decimal(q)>0)
        if not bids or not asks:
            raise ValueError("Cannot infer a two-sided book from empty depth")
        return cls(max(p for p, _ in bids), min(p for p, _ in asks), bids, asks, timestamp)

    def queue_ahead(self, side, price, proxy):
        levels = self.yes_bids if side == "bid" else self.yes_asks
        if levels:
            return sum((decimal(q) for p, q in levels if
                       (decimal(p)>=price if side=="bid" else decimal(p)<=price)), Decimal(0))
        # Improving the same-side top-of-book earns priority in this scenario.
        improves = price>decimal(self.bid) if side=="bid" else price<decimal(self.ask)
        return Decimal(0) if improves else decimal(proxy)


@dataclass(frozen=True)
class QuoteAction:
    bid: float | None = None
    ask: float | None = None
    bid_size: float = 0
    ask_size: float = 0

    def validate(self, book, tick, lot):
        for side in ("bid", "ask"):
            p, qty = getattr(self, side), decimal(getattr(self, side+"_size"))
            if qty < 0 or qty != floor_grid(qty, lot):
                raise ValueError("Quote sizes must be nonnegative lot multiples")
            if p is None:
                if qty:
                    raise ValueError("Positive quantity requires a price")
                continue
            p = decimal(p)
            if not 0 < p < 1 or p != floor_grid(p, tick):
                raise ValueError("Quote price is off tick or outside (0,1)")
            if qty and ((side=="bid" and p>=decimal(book.ask)) or (side=="ask" and p<=decimal(book.bid))):
                raise ValueError("Quote would take liquidity; environment is post-only")
        if self.bid is not None and self.ask is not None and decimal(self.bid)>=decimal(self.ask):
            raise ValueError("Own quotes must be uncrossed")


@dataclass
class RestingOrder:
    order_id: int
    side: str
    price: Decimal
    remaining: Decimal
    queue_ahead: Decimal
    rounding_accumulator: Decimal = Decimal(0)

    def match(self, market_sign, quantity, execution_limit, participation, lot):
        expected_sign = -1 if self.side=="bid" else 1
        if market_sign != expected_sign or self.remaining<=0:
            return Decimal(0)
        limit, volume = decimal(execution_limit), decimal(quantity)
        eligible = self.price>=limit if self.side=="bid" else self.price<=limit
        if not eligible:
            return Decimal(0)
        ahead = min(self.queue_ahead, volume)
        self.queue_ahead -= ahead
        # Consume queue volume first; cap our share of the remaining execution.
        return max(Decimal(0), floor_grid(min(self.remaining, (volume-ahead)*decimal(participation)), lot))
