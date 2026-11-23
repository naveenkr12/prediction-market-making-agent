"""Grid-search a mid-anchored, inventory-skewed heuristic on held-out tuning seeds.

This is the strongest simple baseline: the RL agent should beat *this*, not
just untuned defaults. Uses the same randomized scenario distribution as training.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rl_agent.v3.env_v3 import SimAgentDriver, sample_scenario
from rl_agent.v3.evaluate_v3 import mid_heuristic

def run(args):
    half, skew, size, seed = args
    d = SimAgentDriver(sample_scenario(np.random.default_rng(seed)), seed)
    done = False
    while not done:
        _, done, _ = d.step(mid_heuristic(d.raw, half, skew, size, d.env.config), 0.0)
    return half, skew, size, seed, float(d.env.account.cash - d.env.account.initial_cash)

if __name__ == "__main__":
    grid = list(itertools.product([0.01, 0.02, 0.03, 0.05], [0.0, 0.002, 0.004], [10.0, 20.0]))
    jobs = [(*g, s) for g in grid for s in range(600_000, 601_000)]
    from multiprocessing import Pool
    with Pool(1) as p:
        rows = p.map(run, jobs, chunksize=100)
    df = pd.DataFrame(rows, columns=["half", "skew", "size", "seed", "pnl"])
    s = df.groupby(["half", "skew", "size"]).pnl.agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(s.round(3).head(12).to_string())
    best = s.index[0]
    out = ROOT / "rl_agent/v3/results"; out.mkdir(parents=True, exist_ok=True)
    (out / "tuned_heuristic.json").write_text(json.dumps({"half": best[0], "skew": best[1], "size": best[2]}))
    s.to_csv(out / "tuned_heuristic_grid.csv")
