"""Taker fee used only by the stale-quote sniping decision rule (spec Section 6.4).

taker_fee(Q, P) = ceil_cent(coefficient * Q * P * (1 - P)), coefficient 0.07 by
default (a configurable scenario, label recorded). The sniper is hypothetical;
its fee only decides whether crossing the maker's stale quote is profitable.
The maker's own fill still pays the maker fee via `BinaryAccount.fill`.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

from ..fees import decimal

CENT = Decimal("0.01")


@dataclass(frozen=True)
class TakerFeeSchedule:
    coefficient: float = 0.07
    label: str = "kalshi_general_taker_0.07"

    def __post_init__(self):
        if decimal(self.coefficient) < 0:
            raise ValueError("Taker fee coefficient must be nonnegative")

    def fee(self, quantity, price) -> Decimal:
        q, p = decimal(quantity), decimal(price)
        if q <= 0 or not 0 < p < 1:
            raise ValueError("Taker fee needs positive quantity and price in (0,1)")
        raw = decimal(self.coefficient)*q*p*(1-p)
        return (raw/CENT).to_integral_value(rounding=ROUND_CEILING)*CENT


def taker_fee(quantity, price, coefficient: float = 0.07) -> Decimal:
    return TakerFeeSchedule(coefficient).fee(quantity, price)


def snipe_profitable(side: str, quantity, maker_price, latent_probability: float, tick,
                     schedule: TakerFeeSchedule = TakerFeeSchedule()) -> bool:
    """Crossing test at the node's exact unclipped p (not the displayed price).

    ask a_m: Q(p - a_m) > taker_fee(Q, a_m) + Q*tick   (sniper buys YES from the maker)
    bid b_m: Q(b_m - p) > taker_fee(Q, b_m) + Q*tick   (sniper sells YES to the maker)
    """
    if side not in {"bid", "ask"}:
        raise ValueError("Side must be bid or ask")
    p = float(latent_probability)
    if not 0 <= p <= 1:
        raise ValueError("Latent probability must lie in [0,1]")
    q, m = float(decimal(quantity)), float(decimal(maker_price))
    if q <= 0:
        return False
    edge = q*(p-m) if side == "ask" else q*(m-p)
    return edge > float(schedule.fee(quantity, maker_price))+q*float(decimal(tick))
