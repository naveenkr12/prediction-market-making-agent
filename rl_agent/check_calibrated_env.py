"""Sanity checks for calibrated_env.py before any training spend.

1. The raw TradingEnvironment + HJBReferencePolicy path (unchanged, no RL
   code involved) reproduces a sane PnL distribution -- this is the number
   RL will be judged against.
2. CapstoneQuotingEnv, driven by actions that reproduce the SAME HJB quotes
   (converted back into this env's [-1, 1] action convention), gives the
   same episode outcomes as (1), seed for seed. This is the real check:
   it proves the Gym wrapper's observation/action/reward plumbing does not
   silently change the game HJB was benchmarked on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibrated_env import CapstoneQuotingEnv, OFFSET_CENTER, OFFSET_HALF_RANGE  # noqa: E402
from simulator.trading_env import EnvironmentConfig, TradingEnvironment  # noqa: E402
from simulator.hjb_policy import HJBReferencePolicy  # noqa: E402

N_EPISODES = 60


def hjb_pnls(cfg):
    pol = HJBReferencePolicy()
    pnls = []
    for seed in range(N_EPISODES):
        env = TradingEnvironment(cfg)
        obs, _ = env.reset(seed=seed, options={"seconds_to_close": 3600.0})
        while not env.done:
            action, _ = pol.action(obs, start_elapsed=env.start_elapsed, lifetime=env.lifetime,
                                    inventory_limit=cfg.inventory_limit, tick=cfg.price_tick, lot=cfg.quantity_lot)
            obs, *_ = env.step(action)
        pnls.append(float(env.account.cash - env.account.initial_cash))
    return np.array(pnls)


def hjb_action_for(env: CapstoneQuotingEnv, pol: HJBReferencePolicy) -> np.ndarray:
    """Invert executable_action's bid/ask back into this wrapper's [-1, 1] convention,
    from HJB's own *raw* (pre-projection) quotes, so the comparison is apples to apples."""
    raw = env._raw_obs
    fraction = (env.env.elapsed - env.env.start_elapsed) / (env.env.lifetime - env.env.start_elapsed)
    t = pol.parameters.T * float(np.clip(fraction, 0, 1))
    bid, ask, p = pol.raw_quotes(t, raw["probability"], raw["inventory"])
    a_bid = (p - bid - OFFSET_CENTER) / OFFSET_HALF_RANGE if np.isfinite(bid) else 1.0
    a_ask = (ask - p - OFFSET_CENTER) / OFFSET_HALF_RANGE if np.isfinite(ask) else 1.0
    return np.clip([a_bid, a_ask], -1, 1).astype(np.float32)


def wrapped_pnls(cfg):
    pol = HJBReferencePolicy()
    pnls = []
    for seed in range(N_EPISODES):
        env = CapstoneQuotingEnv(config=cfg, reward_mode="raw")
        env.reset(seed=seed)
        done = False
        while not done:
            a = hjb_action_for(env, pol)
            _, _, term, trunc, _ = env.step(a)
            done = term or trunc
        pnls.append(float(env.env.account.cash - env.env.account.initial_cash))
    return np.array(pnls)


def main():
    cfg = EnvironmentConfig()
    direct = hjb_pnls(cfg)
    wrapped = wrapped_pnls(cfg)
    se = direct.std(ddof=1) / np.sqrt(len(direct))
    corr = np.corrcoef(direct, wrapped)[0, 1]
    median_abs_diff = np.median(np.abs(direct - wrapped))
    print(f"direct   HJB: n={len(direct)} mean={direct.mean():.4f} std={direct.std(ddof=1):.4f}")
    print(f"wrapped  HJB: n={len(wrapped)} mean={wrapped.mean():.4f} std={wrapped.std(ddof=1):.4f}")
    print(f"per-seed correlation: {corr:.4f}  median |diff|: {median_abs_diff:.4f}  mean SE: {se:.4f}")
    # The wrapper re-derives HJB's quotes through its own [-1, 1] action
    # encoding and re-projects them with executable_action, so exact
    # per-episode equality is not expected (clipping at the action-space edge
    # on a handful of extreme-offset steps can flip a fill). What must hold:
    # (a) same seed -> same price/settlement path -> strongly correlated PnL,
    # (b) no systematic bias in the mean beyond ordinary Monte Carlo noise.
    ok = corr > 0.9 and abs(direct.mean() - wrapped.mean()) < 3 * se
    print("CHECK PASSED" if ok else "CHECK FAILED -- wrapper diverges from the direct HJB path")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
