"""Decimal fee accounting, including per-order rounding rebates.

Rules checked 2026-10-09 against Kalshi's July 7 schedule and fee_rounding
documentation. Multipliers are explicit scenarios, not historical fee estimates.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR


def decimal(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Money and quantities must be finite")
    return result


def floor_grid(value, grid):
    return (value / grid).to_integral_value(rounding=ROUND_FLOOR) * grid


@dataclass(frozen=True)
class FeeSchedule:
    maker_multiplier: float = 1.0
    balance_precision: str = "0.0001"
    label: str = "standard_maker_fee_scenario"

    def __post_init__(self):
        if decimal(self.maker_multiplier) < 0 or self.balance_precision not in {"0.0001", "0.01"}:
            raise ValueError("Invalid multiplier or account balance precision")

    @property
    def grid(self):
        return decimal(self.balance_precision)

    def reserve_bound(self, price, quantity, lot):
        # Worst case: one execution per minimum lot, with no rebate. Covers
        # fee rounding, even for a zero multiplier and tiny fractional fills.
        n = quantity / lot
        return decimal("0.0175") * decimal(self.maker_multiplier) * quantity * price * (1-price) + n * (self.grid + decimal("0.000001"))

    def charge(self, price, quantity, maker_side, accumulator=Decimal(0)):
        p, qty = decimal(price), decimal(quantity)
        if not 0 < p < 1 or qty <= 0 or maker_side not in {"bid", "ask"}:
            raise ValueError("Invalid maker fill")
        revenue = qty*p * (-1 if maker_side == "bid" else 1)
        model = decimal("0.0175") * decimal(self.maker_multiplier) * qty*p*(1-p)
        trade_fee = model.quantize(decimal("0.000001"), rounding=ROUND_CEILING)
        aligned = floor_grid(revenue-trade_fee, self.grid)
        rounding = revenue-trade_fee-aligned
        carried = decimal(accumulator)+rounding
        rebate = min(floor_grid(carried, self.grid), floor_grid(trade_fee+rounding, self.grid))
        net = trade_fee+rounding-rebate
        return {"revenue": revenue, "trade_fee": trade_fee, "rounding_fee": rounding,
                "rebate": rebate, "fee": net, "cash_change": revenue-net,
                "accumulator": carried-rebate}
