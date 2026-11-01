"""Gymnasium wrapper around ``simulator/trading_env.py`` for RL training.

Plain English
-------------
``simulator.TradingEnvironment`` is the capstone's calibrated, event-driven
maker environment (real Kalshi-calibrated arrivals, price impact, queueing,
fees, fractional settlement). It is *not* a Gym env (no spaces, dict
observations, variable-length "windows" per step) and it is not vectorised.
This module wraps one instance of it as a standard ``gymnasium.Env`` so
Stable-Baselines3 can train PPO against it, and keeps the race with the HJB
reference policy fair by construction:

  * Observation: ``[episode_time_fraction, 2*probability-1, inventory/Q]``,
    the *same* state ``simulator.hjb_policy.HJBReferencePolicy`` conditions
    on (see its ``action()``). The agent is given no extra information
    (no queue depth, no own-order state, no raw cash) that HJB does not use.
  * Action: two numbers in [-1, 1], mapped to bid/ask distances from the
    current probability, exactly as ``rl_agent/paper1_env.py`` does for the
    toy world. The mapped prices are pushed through
    ``simulator.hjb_policy.executable_action`` -- the *same* tick-rounding /
    post-only / crossing-guard helper the HJB adapter itself uses -- so both
    policies are projected onto the exchange's quote grid identically.
  * Size: fixed at ``min(delta, headroom)`` per side, headroom being how much
    inventory room is left before ``inventory_limit``. This mirrors exactly
    how ``HJBReferencePolicy.action`` sizes its own quotes. The agent only
    ever chooses *where* to quote, not how much -- same decision problem HJB
    solves, so a PnL/VaR/ES gap is attributable to quoting skill, not to one
    side having a sizing advantage.

Reward shaping vs. evaluation metric (read this before trusting a number)
---------------------------------------------------------------------------
``EnvironmentConfig`` ships with *zero* running/terminal inventory penalty by
default: ``env.step()`` reward is then raw mark-to-market PnL, which is a
weak training signal on its own (the dominant source of variance is the
Bernoulli settlement draw, which no quoting policy controls), so some
inventory-risk shaping is useful for training the way HJB's own objective
shapes its (differently-scaled) problem.

A first version of this file reused the frozen HJB paper's own ``gamma`` /
``gamma_T`` (hjb_baseline/params.py: 4e-3, 1e-3) unchanged. That is wrong by
construction, not just "a bit strong": those coefficients multiply a running
integral of ``inventory**2 * dt``, and the paper's ``dt`` lives on a
normalized clock with ``T = 1`` for a whole episode, while
``TradingEnvironment`` integrates over *real seconds* (``T`` here is a
3,600-second hour). Reusing the same coefficient inflates the running
penalty by a factor of ~3,600 relative to what HJB's own derivation implies,
which, measured empirically against 100 HJB-policy episodes in this
environment, means the running penalty term averages ~144,000 (raw
``inventory**2 * seconds`` integral) against a realized-PnL scale of a few
dollars -- the penalty swamps any achievable spread capture, so a PPO agent
trained against it learns the trivial "quote so wide you almost never fill"
policy (empirically: ~0.03 fills per episode, vs. HJB's ~3.7). That is a
reward-shaping artifact, not a market-making result, and it is documented
here rather than silently fixed in a later commit so the mistake and the fix
are both visible.

The constants below are rescaled from the same empirical measurement
(100 HJB episodes, this environment, ``EnvironmentConfig()`` defaults):
running integral mean ~144,000, terminal ``q_T**2`` mean ~98, realized PnL
mean ~-0.4 / std ~4.1. ``RUN_GAMMA`` and ``TERM_GAMMA`` are chosen so each
term contributes, on average *at HJB's own inventory behaviour*, roughly
$0.4-0.5 of shaped penalty -- a real but secondary nudge towards flatter
inventory, not a term that can dominate every other consideration. This is
a judgment call, not a derived constant; ``rl_agent/calibrate_reward.py``
reproduces the measurement if the environment or scenario cell changes.

This changes the reward the agent *optimises*, not the ledger. At
*evaluation* time (see ``evaluate_capstone.py``) we read ``realized_pnl``
(cash, no penalty) and the existing ``simulator.benchmark_metrics`` helpers
(VaR/ES/inventory diagnostics), exactly like ``simulator/evaluate_hjb.py``
reports HJB -- so RL and HJB are judged on the same, penalty-free scorecard.

Not a claim of optimality or completeness
------------------------------------------
HJB is a frozen published-model policy applied to an impacted/queued
environment it was never derived for (see ``simulator/hjb_policy.py``'s own
docstring: "a reference benchmark, not an optimum"). This wrapper's job is
only to give an RL agent a fair shot at the *same* game, in the *same*
calibrated environment the rest of the capstone already validated. One
scenario cell, one fill model, one fee multiplier -- not the full scenario
grid ``simulator/evaluate_hjb.py`` sweeps.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.trading_env import EnvironmentConfig, TradingEnvironment  # noqa: E402
from simulator.hjb_policy import executable_action, legacy_modules  # noqa: E402

# Same fixed per-side clip size the HJB adapter uses (hjb_baseline's par.delta).
_HJB_PARAMS = legacy_modules()["params"].DEFAULT
DELTA = float(_HJB_PARAMS.delta)          # 10 contracts

# Training-only risk shaping, rescaled to THIS environment's real-second /
# dollar units (see the module docstring for the measurement and the bug this
# replaces). Do not reuse hjb_baseline's gamma/gamma_T directly here.
TRAIN_GAMMA = 3e-6       # running inventory_penalty_per_contract2_second
TRAIN_GAMMA_T = 5e-3     # terminal_penalty_per_contract2

# Agent outputs a in [-1, 1] per side -> offset = CENTER + HALF_RANGE * a.
# Calibrated against two reference scales measured in this environment:
# HJB's own raw quote offsets sit in [0.014, 0.031] (tight, ~2.5c median),
# while observed bid/ask spreads run median ~0.08, p90 ~0.22 (so a passive,
# spread-matching quote needs an offset up to ~0.20). The range below spans
# both: near-HJB tightness is reachable close to the action-space centre, and
# fully passive quoting is reachable near its edge. executable_action still
# projects every quote to post-only/on-tick regardless of what is proposed.
OFFSET_CENTER, OFFSET_HALF_RANGE = 0.03, 0.17


def _episode_fraction(env: TradingEnvironment) -> float:
    span = env.lifetime - env.start_elapsed
    if span <= 0:
        return 1.0
    return float(np.clip((env.elapsed - env.start_elapsed) / span, 0.0, 1.0))


def _obs_vector(env: TradingEnvironment, obs: dict) -> np.ndarray:
    return np.array([
        2.0 * _episode_fraction(env) - 1.0,
        2.0 * obs["probability"] - 1.0,
        obs["inventory"] / env.config.inventory_limit,
    ], dtype=np.float32)


class CapstoneQuotingEnv(gym.Env):
    """One ``TradingEnvironment`` episode (default: last simulated hour before close)."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig | None = None, seconds_to_close: float = 3600.0,
                 reward_mode: str = "shaped"):
        super().__init__()
        if reward_mode not in ("shaped", "raw"):
            raise ValueError("reward_mode must be 'shaped' or 'raw'")
        train_cfg = config or EnvironmentConfig()
        if reward_mode == "shaped":
            train_cfg = EnvironmentConfig(**{
                **{f.name: getattr(train_cfg, f.name) for f in train_cfg.__dataclass_fields__.values()},
                "inventory_penalty_per_contract2_second": TRAIN_GAMMA,
                "terminal_penalty_per_contract2": TRAIN_GAMMA_T,
            })
        self.config = train_cfg
        self.seconds_to_close = seconds_to_close
        self.env = TradingEnvironment(self.config)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._raw_obs = None

    def _action_to_quotes(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        p = self._raw_obs["probability"]
        bid_px = p - (OFFSET_CENTER + OFFSET_HALF_RANGE * a[0])
        ask_px = p + (OFFSET_CENTER + OFFSET_HALF_RANGE * a[1])
        q = self._raw_obs["inventory"]
        headroom_bid = min(DELTA, max(0.0, self.config.inventory_limit - q))
        headroom_ask = min(DELTA, max(0.0, self.config.inventory_limit + q))
        action_obj, diagnostics = executable_action(
            bid_px, ask_px, headroom_bid, headroom_ask, self._raw_obs,
            tick=self.config.price_tick, lot=self.config.quantity_lot)
        return action_obj, diagnostics

    def reset(self, *, seed=None, options=None):
        opts = {"seconds_to_close": self.seconds_to_close, **(options or {})}
        self._raw_obs, info = self.env.reset(seed=seed, options=opts)
        return _obs_vector(self.env, self._raw_obs), info

    def step(self, action):
        action_obj, diagnostics = self._action_to_quotes(action)
        self._raw_obs, reward, terminated, truncated, info = self.env.step(action_obj)
        info = {**info, **diagnostics}
        return _obs_vector(self.env, self._raw_obs), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()
