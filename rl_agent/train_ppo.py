"""
Step 3: train the RL agent (PPO) inside the Paper 1 game.

    python rl_agent/train_ppo.py                      # default run
    python rl_agent/train_ppo.py --timesteps 200000   # quick crash test

What happens: 64 games run side by side. After every full game (1,000 steps
each, so 64,000 decisions), PPO nudges the network towards the quotes that
earned more reward. Progress prints every update; `ep_rew_mean` is the
average episode objective and should climb towards HJB's ~12.2.

Output: rl_agent/models/ppo_paper1.zip (the trained "brain") and a
progress CSV in rl_agent/models/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from paper1_env import Paper1VecEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "models" / "ppo_paper1"))
    a = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    env = Paper1VecEnv(n_envs=a.n_envs, seed=a.seed)
    model = PPO(
        "MlpPolicy", env,
        n_steps=1000,            # one full game per env per update
        batch_size=8000,
        n_epochs=10,
        gamma=1.0,               # paper's objective is undiscounted
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
