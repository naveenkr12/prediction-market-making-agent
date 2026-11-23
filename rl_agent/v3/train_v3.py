"""Train the v3 market-making agent (PPO, Stable-Baselines3).

    python -m rl_agent.v3.train_v3 --timesteps 3000000 --out rl_agent/v3/models/ppo_v3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from rl_agent.v3.env_v3 import V3QuotingEnv  # noqa: E402


def make(seed, risk_lambda):
    return lambda: V3QuotingEnv(risk_lambda=risk_lambda, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--risk-lambda", type=float, default=0.002)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="continue training from this .zip")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(ROOT / "rl_agent/v3/models/ppo_v3"))
    a = ap.parse_args()
    torch.set_num_threads(1)

    fns = [make(a.seed * 1000 + i, a.risk_lambda) for i in range(a.n_envs)]
    venv = SubprocVecEnv(fns) if a.subproc else DummyVecEnv(fns)
    # Normalise rewards only (obs are already scaled); keeps PPO's value loss sane.
    venv = VecNormalize(VecMonitor(venv), norm_obs=False, norm_reward=True, gamma=1.0, clip_reward=20.0)
    model = PPO(
        "MlpPolicy", venv,
        n_steps=512, batch_size=1024, n_epochs=10,
        gamma=1.0, gae_lambda=0.95,          # 12-step finite-horizon episodes
        learning_rate=a.lr, clip_range=0.2, max_grad_norm=0.5,
        ent_coef=a.ent_coef, target_kl=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-0.5),
        seed=a.seed, device="cpu", verbose=0,
    )
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu", custom_objects={"learning_rate": a.lr, "ent_coef": a.ent_coef})
    t0 = time.time()
    chunk = min(200_000, a.timesteps)
    done = 0
    target = model.num_timesteps + a.timesteps
    while model.num_timesteps < target:
        model.learn(total_timesteps=min(chunk, target - model.num_timesteps), reset_num_timesteps=False)
        done = model.num_timesteps
        ep = [e["r"] for e in model.ep_info_buffer] if model.ep_info_buffer else []
        print(f"{done:>9,} steps  {time.time()-t0:6.0f}s  ep_reward(train, raw) {sum(ep)/max(1,len(ep)):+.3f}", flush=True)
        model.save(a.out)
    print(f"saved {a.out}.zip after {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
