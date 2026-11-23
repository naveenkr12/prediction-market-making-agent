"""Deterministic rollout diagnostics: reward parts, realized PnL, action stats."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from rl_agent.v3.env_v3 import SimAgentDriver, sample_scenario
from rl_agent.v3.features import decode_action

def main(path, n=300, lam=0.002, start=700_000):
    m = PPO.load(path, device="cpu"); rows = []; acts = []
    for seed in range(start, start + n):
        scn = sample_scenario(np.random.default_rng(seed)); d = SimAgentDriver(scn, seed)
        obs = d.obs(); done = False; parts = dict(spread=0, fees=0, impact=0, risk=0)
        while not done:
            a, _ = m.predict(obs, deterministic=True)
            mid = .5*(d.raw["bid"]+d.raw["ask"])
            b, k, bs, as_ = decode_action(a, mid, d.raw["inventory"])
            acts.append(dict(shift=.5*(b+k)-mid, half=.5*(k-b), bs=bs, as_=as_, flow=obs[4], cflow=obs[7], q=obs[3], tau=obs[0], group=scn["group"]))
            _, done, p = d.step(d.quote(a), lam); obs = d.obs()
            for kk in parts: parts[kk] += p[kk]
        rows.append(dict(group=scn["group"], flip=scn["flip"], pnl=float(d.env.account.cash-d.env.account.initial_cash), **parts))
    df = pd.DataFrame(rows); A = pd.DataFrame(acts)
    print(df.groupby("group")[["pnl","spread","fees","impact","risk"]].mean().round(3))
    print("pnl mean %.3f sd %.3f" % (df.pnl.mean(), df.pnl.std()))
    print(A.groupby("group")[["shift","half","bs","as_"]].mean().round(4))
    print("corr(shift, cum flow) %.2f  corr(shift, q) %.2f" % (A["shift"].corr(A.cflow), A["shift"].corr(A.q)))
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
