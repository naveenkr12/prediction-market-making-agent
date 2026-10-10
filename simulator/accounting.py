"""Single-market fully collateralized binary accounting in signed YES units."""
from decimal import Decimal
from .fees import FeeSchedule, decimal, floor_grid


class BinaryAccount:
    def __init__(self, cash=1000, inventory_limit=100, lot="0.01", fees=None):
        self.fees = fees or FeeSchedule()
        self.cash, self.initial_cash = decimal(cash), decimal(cash)
        self.inventory, self.total_fees = Decimal(0), Decimal(0)
        self.inventory_limit, self.lot = decimal(inventory_limit), decimal(lot)
        self.settled = False
        if (self.cash < 0 or self.cash != floor_grid(self.cash, self.fees.grid) or
                self.lot <= 0 or self.inventory_limit <= 0 or
                self.lot != floor_grid(self.lot, self.fees.grid)):
            raise ValueError("Invalid cash precision, inventory limit or lot")

    @property
    def collateral(self):
        return max(-self.inventory, Decimal(0)) if not self.settled else Decimal(0)

    @property
    def available_cash(self):
        return self.cash-self.collateral

    def equity(self, probability):
        return self.cash+self.inventory*decimal(probability)

    def admit_quotes(self, requests):
        """Reserve both sides conservatively, without netting future fills.

        Returns admitted quantities. Bid has admission priority if funds are
        limited. Inventory-closing orders may reserve excess cash; this avoids
        any dependence on the unknown ordering of bid/ask fills.
        """
        if self.settled:
            raise RuntimeError("Market is settled")
        budget, admitted, reserved = self.available_cash, {}, Decimal(0)
        for side in ("bid", "ask"):
            p, desired = requests[side]
            p, desired = decimal(p), decimal(desired)
            capacity = self.inventory_limit-self.inventory if side == "bid" else self.inventory_limit+self.inventory
            desired = floor_grid(min(desired, capacity), self.lot)
            unit_cost = p if side == "bid" else 1-p
            unit = unit_cost+self.fees.reserve_bound(p, Decimal(1), self.lot)
            qty = max(Decimal(0), floor_grid(min(desired, budget/unit), self.lot))
            cost = qty*unit
            budget -= cost
            reserved += cost
            admitted[side] = qty
        return admitted, reserved

    def fill(self, side, price, quantity, accumulator=Decimal(0)):
        if self.settled:
            raise RuntimeError("Market is settled")
        qty = decimal(quantity)
        if qty <= 0 or qty != floor_grid(qty, self.lot):
            raise ValueError("Fill must be a positive lot multiple")
        fee = self.fees.charge(price, qty, side, accumulator)
        delta = qty if side == "bid" else -qty
        new_q, new_cash = self.inventory+delta, self.cash+fee["cash_change"]
        if abs(new_q) > self.inventory_limit or new_cash < max(-new_q, Decimal(0)):
            raise ValueError("Fill exceeds inventory or collateral budget")
        before = {"cash_before": self.cash, "inventory_before": self.inventory}
        self.cash, self.inventory = new_cash, new_q
        self.total_fees += fee["fee"]
        return {**before, **fee, "cash": self.cash, "inventory": self.inventory,
                "inventory_change": delta, "collateral": self.collateral}

    def settle(self, outcome):
        if self.settled:
            raise RuntimeError("Settlement cannot be applied twice")
        if outcome not in (0, 1):
            raise ValueError("Outcome must be binary")
        old_q, payout = self.inventory, self.inventory*decimal(outcome)
        self.cash += payout
        self.inventory = Decimal(0)
        self.settled = True
        return {"terminal_inventory": old_q, "settlement_adjustment": payout,
                "settlement_outcome": int(outcome), "cash": self.cash, "inventory": self.inventory,
                "collateral": self.collateral, "realized_pnl": self.cash-self.initial_cash}
