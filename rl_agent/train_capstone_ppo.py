"""Train a PPO quoting agent inside the CALIBRATED simulator (not the toy
Paper 1 world rl_agent/paper1_env.py uses).

    python rl_agent/train_capstone_ppo.py                       # default run
    python rl_agent/train_capstone_ppo.py --timesteps 200000     # quick run
    python rl_agent/train_capstone_ppo.py --group sports         # other scenario cell

Episodes here are the simulator's own hour-before-close window (~12-40
quote decisions each, not Paper 1's 1,000), so "timesteps" is a much smaller
unit of experience than in rl_agent/train_ppo.py; see calibrated_env.py's
module docstring for the observation/action/reward conventions, which were
chosen to keep this a fair comparison with HJBReferencePolicy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from calibrated_env import CapstoneQuotingEnv  # noqa: E402
from simulator.trading_env import EnvironmentConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--group", default="broad_based", choices=["broad_based", "sports"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "models" / "ppo_capstone"))
    a = ap.parse_args()

    cfg = EnvironmentConfig(group=a.group)

    def env_fn():
        return CapstoneQuotingEnv(config=cfg, reward_mode="shaped")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    env = make_vec_env(env_fn, n_envs=a.n_envs, seed=a.seed)
    model = PPO(
        "MlpPolicy", env,
        n_steps=256,              # ~ a handful of full episodes per env per update
        batch_size=512,
        n_epochs=10,
        gamma=1.0,                # the shaped reward is already an undiscounted running objective
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[128, 128]), log_std_init=-1.0),
        seed=a.seed, device="cpu", verbose=1,
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(out.parent / "logs"), ["stdout", "csv"]))

    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, progress_bar=False)
    model.save(out)
    print(f"\nSaved trained agent to {out}.zip after {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
