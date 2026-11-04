"""
The two players from Paper 1's Table 2, rewritten as policies for paper1_env.

  hjb_policy()     the paper's optimal strategy, read from the team's saved
                   production value function V (151 x 121 x 21 grid).
  myopic_policy()  the paper's benchmark: maximise instantaneous expected
                   profit, ignore inventory (z = 0 in the same maximisation).

Both call the team's original functions in hjb_baseline/monte_carlo.py, so
they quote exactly as in the verified Table 2 reproduction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import paper1_env  # noqa: F401  (puts hjb_baseline on the import path)
from paper1_env import HJB_DIR, action_to_quotes
from hjb_solver import Grid
import monte_carlo as MC

DATA = HJB_DIR / "output" / "data"


def load_hjb(grid_path: Path = DATA / "grid_production.npz", value_path: Path = DATA / "V_production.npy"):
    g = np.load(grid_path)
    grid = Grid(**{k: g[k].copy() for k in ("t", "p", "q", "pi")})
    V = np.load(value_path)
    if V.shape != (len(grid.t), len(grid.p), len(grid.q)):
        raise ValueError(f"V shape {V.shape} does not match grid")
    interp = RegularGridInterpolator((grid.t, grid.p, grid.q), V, bounds_error=False, fill_value=None)
    return grid, interp


def hjb_policy():
    grid, interp = load_hjb()

    def policy(m):
        return MC._batch_optimal_quotes(m.t, m.p, m.q, m.par, grid, interp)
    return policy


def myopic_policy():
    grid, _ = load_hjb()

    def policy(m):
        return MC._myopic_quotes(m.t, m.p, grid.pi, m.par)
    return policy


def rl_policy(model):
    """Wrap a trained Stable-Baselines3 model (deterministic actions)."""
    def policy(m):
        action, _ = model.predict(m.observation(), deterministic=True)
        return action_to_quotes(action, m.p)
    return policy
