"""Adapters from a live book to the repository's quoting strategies.

Each strategy receives the same context and returns a simulator QuoteAction.
Conventions follow the training code exactly: one episode is the last
EPISODE_S seconds before close; 10 contracts per side, capped by inventory
headroom; quotes are tick-rounded and post-only via the simulator helpers.
Before the final hour the episode clock is held at its start (t = 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.execution import QuoteAction
from simulator.hjb_policy import HJBReferencePolicy, executable_action
from simulator.policies import quote_policy

from . import config


@dataclass
class Context:
    bid: float
    ask: float
    inventory: float
    now: float
    open_ts: float
    close_ts: float
    tick: str = "0.01"

    @property
    def probability(self) -> float:
        # Live proxy for the simulator's latent reference price.
        return (self.bid + self.ask) / 2

    @property
    def episode_fraction(self) -> float:
        return float(np.clip((self.now - (self.close_ts - config.EPISODE_S)) / config.EPISODE_S, 0, 1))

    def observation(self) -> dict:
        return {"probability": self.probability, "bid": self.bid, "ask": self.ask, "inventory": self.inventory}


def headroom(ctx: Context) -> tuple[float, float]:
    q, lim = ctx.inventory, config.INVENTORY_LIMIT
    return min(config.QUOTE_SIZE, max(0.0, lim - q)), min(config.QUOTE_SIZE, max(0.0, lim + q))


class FixedSpread:
    name, label = "fixed", "Fixed spread"
    available, reason = True, ""

    def quote(self, ctx: Context):
        return quote_policy(ctx.observation(), size=config.QUOTE_SIZE, inventory_skew=0.0, tick=ctx.tick), {}


class InventorySkew:
    name, label = "skew", "Inventory skew"
    available, reason = True, ""

    def quote(self, ctx: Context):
        return quote_policy(ctx.observation(), size=config.QUOTE_SIZE, inventory_skew=0.0005, tick=ctx.tick), {}


class HJB:
    name, label = "hjb", "HJB (Paper 1)"
    available, reason = True, ""

    def __init__(self):
        self.policy = HJBReferencePolicy()

    def quote(self, ctx: Context):
        lifetime = ctx.close_ts - ctx.open_ts
        start = max(0.0, lifetime - config.EPISODE_S)
        elapsed = float(np.clip(ctx.now - ctx.open_ts, start, lifetime))
        obs = {**ctx.observation(), "elapsed_seconds": elapsed}
        action, info = self.policy.action(obs, start_elapsed=start, lifetime=lifetime,
                                          inventory_limit=config.INVENTORY_LIMIT, tick=ctx.tick, lot="0.01")
        return action, {"hjb_time": info["hjb_time"]}


class PPOAgent:
    """Naveen's calibrated-simulator agent (rl_agent/calibrated_env.py contract)."""
    name, label = "ppo", "PPO agent"
    OFFSET_CENTER, OFFSET_HALF_RANGE = 0.03, 0.17

    def __init__(self, model_path=config.PPO_MODEL):
        self.model, self.available, self.reason = None, False, ""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path), device="cpu")
            self.available = True
        except ImportError:
            self.reason = "stable-baselines3 not installed (pip install -r rl_agent/requirements-rl.txt)"
        except Exception as exc:   # missing or unreadable model file
            self.reason = f"model not loaded: {exc}"

    def quote(self, ctx: Context):
        p = ctx.probability
        vec = np.array([2 * ctx.episode_fraction - 1, 2 * p - 1, ctx.inventory / config.INVENTORY_LIMIT],
                       dtype=np.float32)
        a, _ = self.model.predict(vec, deterministic=True)
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        bid = p - (self.OFFSET_CENTER + self.OFFSET_HALF_RANGE * a[0])
        ask = p + (self.OFFSET_CENTER + self.OFFSET_HALF_RANGE * a[1])
        bid_size, ask_size = headroom(ctx)
        action, diag = executable_action(bid, ask, bid_size, ask_size, ctx.observation(), tick=ctx.tick, lot="0.01")
        return action, {"raw_action": [round(float(x), 4) for x in a]}


def build(include_ppo: bool = True) -> list:
    out = [HJB(), FixedSpread(), InventorySkew()]
    if include_ppo:
        out.append(PPOAgent())
    return out


NO_QUOTE = QuoteAction()
