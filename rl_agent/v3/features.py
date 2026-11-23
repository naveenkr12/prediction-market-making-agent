"""Observation features shared by the simulator wrapper and the live dashboard.

Everything here is computable from a live Kalshi feed: the displayed book
(bid/ask), our own inventory, the clock to close, and the public trades
printed since the previous decision. Nothing uses the simulator's latent
reference price, impact beta or volatility, so the trained policy sees the
same kind of information live as in training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

OBS_DIM = 12
INVENTORY_LIMIT = 100.0


@dataclass
class FeatureTracker:
    """Per-market rolling state; call ``update`` once per decision."""
    start_mid: float | None = None
    last_mid: float | None = None
    cum_flow: float = 0.0
    cum_volume: float = 0.0
    windows: int = 0
    last_flow: float = 0.0
    last_volume: float = 0.0
    last_trades: int = 0
    last_dmid: float = 0.0
    history: list = field(default_factory=list)

    def reset(self):
        self.__init__()

    def record_window(self, signs, quantities):
        """Trades printed since the last decision: sign +1 = YES taker, -1 = NO taker."""
        signs = np.asarray(signs, dtype=float)
        qty = np.asarray(quantities, dtype=float)
        self.last_flow = float(np.dot(signs, qty)) if len(qty) else 0.0
        self.last_volume = float(qty.sum()) if len(qty) else 0.0
        self.last_trades = int(len(qty))
        self.cum_flow += self.last_flow
        self.cum_volume += self.last_volume
        self.windows += 1

    def observe(self, bid: float, ask: float, inventory: float, episode_fraction: float) -> np.ndarray:
        mid = 0.5 * (bid + ask)
        if self.start_mid is None:
            self.start_mid = mid
        self.last_dmid = 0.0 if self.last_mid is None else mid - self.last_mid
        self.last_mid = mid
        avg_vol = self.cum_volume / max(1, self.windows)
        vec = np.array([
            2.0 * float(np.clip(episode_fraction, 0, 1)) - 1.0,
            2.0 * mid - 1.0,
            np.clip((ask - bid) / 0.10, 0, 3) - 1.0,
            np.clip(inventory / INVENTORY_LIMIT, -1, 1),
            np.tanh(self.last_flow / 500.0),
            np.log1p(self.last_volume) / 8.0,
            np.log1p(self.last_trades) / 4.0,
            np.tanh(self.cum_flow / 2000.0),
            np.log1p(avg_vol) / 8.0,
            np.clip(self.last_dmid * 10.0, -1, 1),
            np.clip((mid - self.start_mid) * 5.0, -1, 1),
            4.0 * mid * (1.0 - mid),
        ], dtype=np.float32)
        return np.clip(vec, -3, 3)


# --- action mapping ---------------------------------------------------------
# a in [-1,1]^4 -> (center shift, half width, bid size, ask size), all relative
# to the displayed mid. a = 0 is a symmetric 2.75c-wide-each-side, 10-lot quote.
SHIFT_RANGE = 0.04
MIN_HALF, MAX_HALF = 0.005, 0.05
MAX_SIZE = 20.0


def decode_action(a, mid: float, inventory: float, inventory_limit: float = INVENTORY_LIMIT):
    a = np.clip(np.asarray(a, dtype=float), -1, 1)
    center = mid + SHIFT_RANGE * a[0]
    half = MIN_HALF + (MAX_HALF - MIN_HALF) * 0.5 * (a[1] + 1)
    bid_size = MAX_SIZE * 0.5 * (a[2] + 1)
    ask_size = MAX_SIZE * 0.5 * (a[3] + 1)
    bid_size = 0.0 if bid_size < 1.0 else min(round(bid_size), max(0.0, inventory_limit - inventory))
    ask_size = 0.0 if ask_size < 1.0 else min(round(ask_size), max(0.0, inventory_limit + inventory))
    return center - half, center + half, float(bid_size), float(ask_size)
