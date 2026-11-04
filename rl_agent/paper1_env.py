"""
Paper 1 (Feil & Nendel, 2026) market as a game an RL agent can play.

Plain English
-------------
One episode = one contract's life, from t = 0 to t = T = 1, cut into
`n_steps` decision points (1,000 by default, as in the team's Table 2
reproduction). At every decision point the player sees

    (time t, price p, inventory q)

and posts a bid and an ask. Then, exactly as in Section 4.4 of the paper:

  1. market sell orders hit the bid with Poisson intensity Lambda^b(t,p,bid),
     market buy orders lift the ask with intensity Lambda^a(t,p,ask);
     every fill is Delta = 10 contracts, inventory is capped at |q| <= Q = 100;
  2. the price moves:  dp = p(1-p) sigma_bar(t,p) dW   (Euler step, clipped
     to [0.001, 0.999]);
  3. at T the contract settles: Y ~ Bernoulli(p_T), PnL = X_T + q_T * Y.

Nothing here is new modelling. Every formula is imported from the team's
verified reproduction in ../hjb_baseline/ (params, intensities, price
process), so the RL agent and the HJB strategy play exactly the same game.

Reward (what the agent is trained to maximise) = the paper's own objective

    E[ X_T + q_T p_T - gamma * integral q^2 varsigma^2 dt - gamma_T q_T^2 p_T(1-p_T) ].

Two equivalent ways to hand it out step by step:

  reward_mode="spread" (default, used for training)
      r_n = Delta*(ask - p)*sells + Delta*(p - bid)*buys - gamma q^2 varsigma^2 dt
      (+ terminal penalty). This is the paper's reduced form (their
      V = Upsilon - x - q p): the mark-to-market term q*dp is dropped because
      p is a martingale, so it adds zero in expectation for EVERY policy but a
      lot of noise. Same objective, much faster learning.

  reward_mode="mtm"
      r_n = change in X + q p  minus the same penalties. Sums to the
      literal objective path by path; noisier.

Common random numbers: the price path and the settlement draw depend only on
the seed, never on the policy (in Paper 1 the maker's trades do not move the
price). So HJB, myopic and RL evaluated with the same seed face identical
prices and identical outcomes. Only their own fills differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HJB_DIR = Path(__file__).resolve().parents[1] / "hjb_baseline"
if str(HJB_DIR) not in sys.path:
    sys.path.insert(0, str(HJB_DIR))

from params import DEFAULT, ModelParams  # noqa: E402  (team's Table 1 parameters)
import intensities as I  # noqa: E402
import price_process as PP  # noqa: E402

P_LO, P_HI = 1e-3, 1 - 1e-3          # price projection used in the paper (Sec. 4.4)
QUOTE_LO, QUOTE_HI = 1e-6, 1 - 1e-6  # same quote range as the HJB quote grid

# The agent outputs two numbers in [-1, 1]. They become distances from p:
#     bid = p - offset_b,  ask = p + offset_a,  offset = CENTER + HALF_RANGE * action
# so offsets live in [-0.03, 0.09]. HJB's optimal offsets sit around 0.02-0.03
# plus an inventory skew of up to a few cents, well inside this range.
OFFSET_CENTER, OFFSET_HALF_RANGE = 0.03, 0.06


def action_to_quotes(action: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map agent actions of shape (n, 2) in [-1, 1] to (bid, ask) prices."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    off_b = OFFSET_CENTER + OFFSET_HALF_RANGE * a[:, 0]
    off_a = OFFSET_CENTER + OFFSET_HALF_RANGE * a[:, 1]
    bid = np.clip(p - off_b, QUOTE_LO, QUOTE_HI)
    ask = np.clip(p + off_a, QUOTE_LO, QUOTE_HI)
    return bid, ask


class PaperOneMarket:
    """Vectorised Paper 1 market: runs `n_paths` independent episodes at once."""

    def __init__(self, n_paths: int, par: ModelParams = DEFAULT, n_steps: int = 1000,
                 p0: float = 0.5, enforce_bound: bool = True, reward_mode: str = "spread"):
        if reward_mode not in ("spread", "mtm"):
            raise ValueError("reward_mode must be 'spread' or 'mtm'")
        self.n_paths, self.par, self.n_steps, self.p0 = n_paths, par, n_steps, p0
        self.enforce_bound, self.reward_mode = enforce_bound, reward_mode
        self.dt = par.T / n_steps
        self.t_grid = np.linspace(0.0, par.T, n_steps + 1)

    # ------------------------------------------------------------------ state
    def reset(self, seed: int | None = None) -> np.ndarray:
        # Separate random streams: prices, fills, settlement. Prices and the
        # settlement uniform therefore never depend on what the policy does.
        price_ss, fill_ss, settle_ss = np.random.SeedSequence(seed).spawn(3)
        self.price_rng = np.random.default_rng(price_ss)
        self.fill_rng = np.random.default_rng(fill_ss)
        self.settle_u = np.random.default_rng(settle_ss).random(self.n_paths)
        n = self.n_paths
        self.n = 0
        self.p = np.full(n, self.p0)
        self.q = np.zeros(n)
        self.x = np.zeros(n)
        self.running_penalty = np.zeros(n)
        self.spread_pnl = np.zeros(n)
        self.n_buys = np.zeros(n)
        self.n_sells = np.zeros(n)
        return self.observation()

    @property
    def t(self) -> float:
        return self.t_grid[self.n]

    def observation(self) -> np.ndarray:
        """What a player sees: [t/T, 2p - 1, q/Q], each roughly in [-1, 1]."""
        return np.column_stack([
            np.full(self.n_paths, self.t / self.par.T),
            2.0 * self.p - 1.0,
            self.q / self.par.Q,
        ]).astype(np.float32)

    # ------------------------------------------------------------------- step
    def step(self, bid: np.ndarray, ask: np.ndarray):
        """Post (bid, ask) for one step. NaN on a side = no quote on that side.

        Returns (reward, done). After the final step, `self.result` holds the
        settled outcome of every path.
        """
        par, dt, t, p = self.par, self.dt, self.t, self.p
        bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
        bid_on, ask_on = ~np.isnan(bid), ~np.isnan(ask)
        if self.enforce_bound:
            bid_on &= self.q < par.Q - 1e-9
            ask_on &= self.q > -par.Q + 1e-9
        bid_safe = np.where(bid_on, bid, 0.5)
        ask_safe = np.where(ask_on, ask, 0.5)

        # 1. fills (Poisson counts over dt, each of size Delta)
        lam_b = np.where(bid_on, I.lambda_bid(t, p, bid_safe, par), 0.0)
        lam_a = np.where(ask_on, I.lambda_ask(t, p, ask_safe, par), 0.0)
        buys = self.fill_rng.poisson(np.clip(lam_b, 0, None) * dt).astype(float)
        sells = self.fill_rng.poisson(np.clip(lam_a, 0, None) * dt).astype(float)
        if self.enforce_bound:  # cannot breach the inventory limit
            buys = np.minimum(buys, np.floor(np.clip((par.Q - self.q) / par.delta, 0, None)))
            sells = np.minimum(sells, np.floor(np.clip((self.q + par.Q) / par.delta, 0, None)))
        spread_gain = par.delta * (sells * (ask_safe - p) + buys * (p - bid_safe))
        wealth_before = self.x + self.q * p
        self.x += par.delta * (sells * ask_safe - buys * bid_safe)
        self.q += par.delta * (buys - sells)
        self.n_buys += buys
        self.n_sells += sells
        self.spread_pnl += spread_gain

        # 2. running inventory penalty, gamma q^2 varsigma(t,p)^2 dt
        varsigma = p * (1 - p) * PP.sigma_bar(t, p, par)
        penalty = par.gamma * self.q ** 2 * varsigma ** 2 * dt
        self.running_penalty += penalty

        # 3. price moves (identical for every policy given the seed)
        dW = self.price_rng.normal(0.0, np.sqrt(dt), size=self.n_paths)
        self.p = np.clip(p + varsigma * dW, P_LO, P_HI)
        self.n += 1

        if self.reward_mode == "spread":
            reward = spread_gain - penalty
        else:
            reward = (self.x + self.q * self.p) - wealth_before - penalty

        done = self.n >= self.n_steps
        if done:
            terminal = par.gamma_T * self.q ** 2 * self.p * (1 - self.p)
            reward = reward - terminal
            y = (self.settle_u < self.p).astype(float)
            self.result = {
                "pnl": self.x + self.q * y,                         # what Table 2 reports
                "objective": self.x + self.q * self.p - self.running_penalty - terminal,
                "q_T": self.q.copy(),
                "p_T": self.p.copy(),
                "Y": y,
                "spread_pnl": self.spread_pnl.copy(),
                "fills": self.n_buys + self.n_sells,
            }
        return reward, done


def simulate(policy, n_paths: int = 10_000, seed: int = 0, enforce_bound: bool = True,
             n_steps: int = 1000, par: ModelParams = DEFAULT) -> dict:
    """Play `policy(market) -> (bid, ask)` over `n_paths` episodes. Returns market.result."""
    market = PaperOneMarket(n_paths, par=par, n_steps=n_steps, enforce_bound=enforce_bound)
    market.reset(seed)
    done = False
    while not done:
        bid, ask = policy(market)
        _, done = market.step(bid, ask)
    return market.result


def table2_stats(result: dict) -> dict:
    """The five Table 2 columns, defined exactly as in hjb_baseline/monte_carlo.py."""
    pnl = result["pnl"]
    q05 = np.quantile(pnl, 0.05)
    return {
        "PnL mean": pnl.mean(),
        "PnL std": pnl.std(),
        "mean |q_T|": np.abs(result["q_T"]).mean(),
        "VaR 5%": -q05,
        "ES 5%": -pnl[pnl <= q05].mean(),
        "objective": result["objective"].mean(),
    }


# ---------------------------------------------------------------------------
# Stable-Baselines3 wrapper. SB3 normally runs many copies of a small env;
# here one object already holds many paths, so we expose it directly as a
# VecEnv. All paths finish together after n_steps, then all reset.
# ---------------------------------------------------------------------------
try:
    from gymnasium import spaces
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv
except ImportError:  # check_env.py works without the RL libraries
    VecEnv = object


class Paper1VecEnv(VecEnv):
    def __init__(self, n_envs: int = 64, seed: int = 0, **market_kwargs):
        self.market = PaperOneMarket(n_envs, **market_kwargs)
        self._seed_seq = np.random.SeedSequence(seed)
        obs_space = spaces.Box(-1.5, 1.5, shape=(3,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None
        self._ep_return = np.zeros(n_envs)

    def _new_seed(self) -> int:
        return int(self._seed_seq.spawn(1)[0].generate_state(1)[0])

    def reset(self):
        self._ep_return[:] = 0.0
        return self.market.reset(self._new_seed())

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        bid, ask = action_to_quotes(self._actions, self.market.p)
        reward, done = self.market.step(bid, ask)
        self._ep_return += reward
        obs = self.market.observation()
        dones = np.full(self.num_envs, done)
        infos = [{} for _ in range(self.num_envs)]
        if done:
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode"] = {"r": float(self._ep_return[i]), "l": self.market.n_steps,
                                       "pnl": float(self.market.result["pnl"][i])}
            obs = self.reset()
        return obs, reward.astype(np.float32), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
