"""Gymnasium env for the v3 market-making agent (calibrated simulator).

Changes from ``rl_agent/calibrated_env.py`` (v2) and why:

* Observation = ``features.FeatureTracker``: book mid and spread, inventory,
  time to close and public trade flow. These can all be computed live, so the
  agent never sees the simulator's latent price, which a live feed doesn't show.
* Action = skew, width and a separate size per side (0-20 lots), so the
  agent can quote one side only or not at all.
* Training reward = the part of PnL the quotes can influence: spread captured
  on each fill (vs. the reference at fill time), minus fees, plus the impact
  moves on inventory held (adverse selection and flow drift). The zero-mean
  diffusion noise (q * dp from background innovations) and settlement noise
  (q * (Y - p)) are removed as control variates. Their means are zero, so the
  optimum is unchanged, but the variance of the gradient drops by ~2 orders of
  magnitude. A mean-variance penalty (lambda/2 * Var) for diffusion and
  settlement risk is added back in expectation.
  Evaluation always uses true realized cash PnL.
* Domain randomization over market group, impact and fill scenarios, starting
  probability, and a YES/NO mirror of the taker flow. The mirror stops the
  agent from memorising that the simulated sports flow buys YES; it has to
  read the direction from the observed flow.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.trading_env import EnvironmentConfig, TradingEnvironment  # noqa: E402
from simulator.hjb_policy import executable_action  # noqa: E402
from rl_agent.v3.features import OBS_DIM, FeatureTracker, decode_action  # noqa: E402

EPISODE_SECONDS = 3600.0
SIGMA_REF = 0.008            # log-odds / sqrt(s); calibrated range 0.0044-0.011
FILL_SCENARIOS = {"conservative": (100.0, 0.05), "base": (50.0, 0.10), "optimistic": (0.0, 1.0)}


def sample_scenario(rng: np.random.Generator) -> dict:
    return {
        "group": "sports" if rng.random() < 0.6 else "broad_based",
        "impact": "high" if rng.random() < 0.15 else "base",
        "fill": "conservative" if rng.random() < 0.3 else "base",
        "flip": bool(rng.random() < 0.5),
        "p0": float(rng.uniform(0.15, 0.85)),
    }


def make_config(scn: dict, decision_seconds: float = 300.0) -> EnvironmentConfig:
    queue, part = FILL_SCENARIOS[scn["fill"]]
    return EnvironmentConfig(group=scn["group"], impact_scenario=scn["impact"],
                             queue_ahead_contracts=queue, participation=part,
                             decision_seconds=decision_seconds)


def episode_fraction(env: TradingEnvironment) -> float:
    span = env.lifetime - env.start_elapsed
    return 1.0 if span <= 0 else float(np.clip((env.elapsed - env.start_elapsed) / span, 0, 1))


class SimAgentDriver:
    """Owns one TradingEnvironment episode plus the live-compatible feature state."""

    def __init__(self, scenario: dict, seed: int, decision_seconds: float = 300.0):
        self.scenario = scenario
        self.env = TradingEnvironment(make_config(scenario, decision_seconds))
        self.raw, _ = self.env.reset(seed=seed, options={"seconds_to_close": EPISODE_SECONDS,
                                                          "initial_probability": scenario["p0"]})
        if scenario["flip"]:
            self.env.dynamics.signs = -self.env.dynamics.signs
        self.tracker = FeatureTracker()
        self.ledger_pos = len(self.env.ledger)

    def obs(self) -> np.ndarray:
        r = self.raw
        return self.tracker.observe(r["bid"], r["ask"], r["inventory"], episode_fraction(self.env))

    def quote(self, action):
        r = self.raw
        mid = 0.5 * (r["bid"] + r["ask"])
        bid, ask, bs, as_ = decode_action(action, mid, r["inventory"], self.env.config.inventory_limit)
        q_action, _ = executable_action(bid, ask, bs, as_, r, tick=self.env.config.price_tick,
                                        lot=self.env.config.quantity_lot)
        return q_action

    def step(self, quote_action, risk_lambda: float):
        env = self.env
        q_before, elapsed_before = self.raw["inventory"], env.elapsed
        self.raw, _, terminated, truncated, info = env.step(quote_action)
        rows = env.ledger[self.ledger_pos:]
        self.ledger_pos = len(env.ledger)
        spread = fees = impact = 0.0
        signs, qty = [], []
        for row in rows:
            k = row["kind"]
            if k == "fill":
                px, p, n = float(row["price"]), float(row["probability"]), float(row["quantity"])
                spread += (px - p) * n if row["side"] == "ask" else (p - px) * n
                fees += float(row["fee"])
            elif k == "market":
                impact += float(row["inventory"]) * (float(row["probability"]) - float(row["probability_before"]))
                signs.append(row["market_sign"])
                qty.append(row["market_quantity"])
        self.tracker.record_window(signs, qty)
        # Expected-variance penalty (mean-variance utility), using only observables.
        dt = max(0.0, env.elapsed - elapsed_before)
        p_now = self.raw["probability"]
        pq = p_now * (1 - p_now)
        q_hold = q_before if terminated else self.raw["inventory"]
        var = q_hold ** 2 * (pq ** 2) * SIGMA_REF ** 2 * dt
        if terminated:
            pre = [r for r in rows if r["kind"] != "settlement"]
            term_q = float(pre[-1]["inventory"]) if pre else q_before
            var += term_q ** 2 * pq
        reward = spread - fees + impact - 0.5 * risk_lambda * var
        parts = {"spread": spread, "fees": fees, "impact": impact, "risk": 0.5 * risk_lambda * var}
        return reward, terminated or truncated, parts


class V3QuotingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, risk_lambda: float = 0.002, scenario: dict | None = None, seed: int = 0,
                 decision_seconds: float = 300.0):
        super().__init__()
        self.risk_lambda = risk_lambda
        self.decision_seconds = decision_seconds
        self.fixed_scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-3, 3, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.driver = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scn = self.fixed_scenario or sample_scenario(self.rng)
        self.driver = SimAgentDriver(scn, int(self.rng.integers(2**31)), self.decision_seconds)
        return self.driver.obs(), {"scenario": scn}

    def step(self, action):
        d = self.driver
        reward, done, parts = d.step(d.quote(action), self.risk_lambda)
        return d.obs(), float(reward), bool(done), False, parts
