"""RL v3 as a player inside Rithvik's information simulator (stress test).

Same notebook + bookmark logic as rl_agent/v3/env_v3.py, but for playing only (no training reward).
Run via: STRESS_V3=1 python -m simulator.information.evaluate ...
"""
import torch
from stable_baselines3 import PPO

from rl_agent.v3.features import FeatureTracker, decode_action
from rl_agent.v3.env_v3 import episode_fraction
from simulator.hjb_policy import executable_action

torch.set_num_threads(1)  # each parallel worker uses one core


class V3Policy:
    def __init__(self, model_path):
        self.model = PPO.load(str(model_path), device="cpu")

    def reset(self, env):
        self.tracker = FeatureTracker()       # fresh notebook each episode
        self.ledger_pos = len(env.ledger)     # bookmark in the trade diary
        self.first_decision = True

    def action(self, obs, env):
        rows = env.ledger[self.ledger_pos:]   # only the new diary rows since last decision
        self.ledger_pos = len(env.ledger)

        # Public market trades only (own fills and snipes are not visible flow, as in training)
        signs = [row["market_sign"] for row in rows if row["kind"] == "market"]
        qty = [row["market_quantity"] for row in rows if row["kind"] == "market"]

        if not self.first_decision:            # env_v3 never records before the first quote
            self.tracker.record_window(signs, qty)
        self.first_decision = False

        vec = self.tracker.observe(obs["bid"], obs["ask"], obs["inventory"], episode_fraction(env))
        a, _ = self.model.predict(vec, deterministic=True)

        mid = 0.5 * (obs["bid"] + obs["ask"])
        bid, ask, bs, as_ = decode_action(a, mid, obs["inventory"], env.config.inventory_limit)
        q_action, _ = executable_action(bid, ask, bs, as_, obs, tick=env.config.price_tick,
                                        lot=env.config.quantity_lot)
        return q_action
