"""Simple reference policies for environment checks, not trained strategies."""
from .execution import QuoteAction
from .fees import decimal, floor_grid


def quote_policy(observation, size=10, inventory_skew=0., tick="0.01", half_spread=.01):
    """Quote around the reference, skew away from inventory, enforce post-only."""
    tick = decimal(tick)
    p = observation["probability"]-inventory_skew*observation["inventory"]
    bid = floor_grid(decimal(p-half_spread), tick)
    ask = -floor_grid(-decimal(p+half_spread), tick)
    bid = min(bid, -floor_grid(-decimal(observation["ask"]), tick)-tick)
    ask = max(ask, floor_grid(decimal(observation["bid"]), tick)+tick)
    bid, ask = max(tick, bid), min(1-tick, ask)
    bid_active = bid<decimal(observation["ask"]) and bid<ask
    ask_active = ask>decimal(observation["bid"]) and bid<ask
    return QuoteAction(float(bid) if bid_active else None, float(ask) if ask_active else None,
                       size if bid_active else 0, size if ask_active else 0)
