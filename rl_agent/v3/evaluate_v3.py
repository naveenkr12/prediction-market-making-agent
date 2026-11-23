"""Paired evaluation of the v3 agent against every baseline on identical seeds.

Score = true realized cash PnL after settlement (no shaping, no control
variates). Each seed fixes the scenario and the market path, so every policy
faces the same markets, flow and settlement draws.

    python -m rl_agent.v3.evaluate_v3 --model rl_agent/v3/models/ppo_v3 --n 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.execution import QuoteAction  # noqa: E402
from simulator.hjb_policy import HJBReferencePolicy, executable_action  # noqa: E402
from simulator.policies import quote_policy  # noqa: E402
from rl_agent.v3.env_v3 import SimAgentDriver, sample_scenario  # noqa: E402

CELLS = {
    "sports_std": {"group": "sports", "impact": "base", "fill": "base", "flip": False, "p0": 0.5},
    "broad_std": {"group": "broad_based", "impact": "base", "fill": "base", "flip": False, "p0": 0.5},
    "randomized": None,
}
TUNED = {"half": 0.03, "skew": 0.002, "size": 10.0}   # overwritten by tune_heuristic.py output if present
_G = {}


def _init(model_path, v2_path):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    _G["hjb"] = HJBReferencePolicy()
    _G["v3"] = PPO.load(model_path, device="cpu") if model_path else None
    _G["v2"] = PPO.load(v2_path, device="cpu") if v2_path and Path(v2_path).exists() else None
    tuned = ROOT / "rl_agent/v3/results/tuned_heuristic.json"
    if tuned.exists():
        TUNED.update(json.loads(tuned.read_text()))


def mid_heuristic(raw, half, skew, size, cfg):
    mid = 0.5 * (raw["bid"] + raw["ask"]) - skew * raw["inventory"]
    q = raw["inventory"]
    a, _ = executable_action(mid - half, mid + half, min(size, max(0, 100 - q)), min(size, max(0, 100 + q)),
                             raw, tick=cfg.price_tick, lot=cfg.quantity_lot)
    return a


def policy_action(name, d: SimAgentDriver, obs_vec):
    env, raw, cfg = d.env, d.raw, d.env.config
    if name == "none":
        return QuoteAction()
    if name == "fixed":
        return quote_policy(raw, inventory_skew=0.0, tick=cfg.price_tick)
    if name == "skew":
        return quote_policy(raw, inventory_skew=0.0005, tick=cfg.price_tick)
    if name == "hjb":
        return _G["hjb"].action(raw, start_elapsed=env.start_elapsed, lifetime=env.lifetime,
                                inventory_limit=cfg.inventory_limit, tick=cfg.price_tick, lot=cfg.quantity_lot)[0]
    if name == "tuned":
        return mid_heuristic(raw, TUNED["half"], TUNED["skew"], TUNED["size"], cfg)
    if name == "v2":
        from rl_agent.calibrated_env import CapstoneQuotingEnv, _obs_vector
        act, _ = _G["v2"].predict(_obs_vector(env, raw), deterministic=True)
        shim = CapstoneQuotingEnv.__new__(CapstoneQuotingEnv)
        shim.config, shim._raw_obs = cfg, raw
        return shim._action_to_quotes(act)[0]
    if name == "v3":
        act, _ = _G["v3"].predict(obs_vec, deterministic=True)
        return d.quote(act)
    raise ValueError(name)


def run(job):
    cell, policy, seed = job
    scn = CELLS[cell] or sample_scenario(np.random.default_rng(seed))
    d = SimAgentDriver(scn, seed)
    done, n_quotes, ctrl = False, 0, 0.0
    obs = d.obs()
    while not done:
        a = policy_action(policy, d, obs)
        n_quotes += int((a.bid_size or 0) > 0) + int((a.ask_size or 0) > 0)
        _, done, parts = d.step(a, 0.0)
        ctrl += parts["spread"] - parts["fees"] + parts["impact"]
        obs = d.obs()
    env = d.env
    fills = [r for r in env.ledger if r["kind"] == "fill"]
    return {"cell": cell, "policy": policy, "seed": seed, "group": scn["group"], "flip": scn["flip"],
            "ticker": env.dynamics.ticker,
            "pnl": float(env.account.cash - env.account.initial_cash),
            # Same expectation as pnl (zero-mean diffusion/settlement noise removed), far lower variance.
            "exp_pnl": ctrl,
            "fills": len(fills), "contracts": sum(float(r["quantity"]) for r in fills),
            "max_abs_q": max(abs(float(r["inventory"])) for r in env.ledger),
            "fees": float(env.account.total_fees), "quote_sides": n_quotes}


def summarize(df: pd.DataFrame, ref="hjb", boot=2000):
    out = []
    rng = np.random.default_rng(0)
    for (cell, pol), f in df.groupby(["cell", "policy"]):
        x = f.sort_values("seed").pnl.to_numpy()
        row = {"cell": cell, "policy": pol, "n": len(x), "mean_pnl": x.mean(), "std_pnl": x.std(ddof=1),
               "sharpe_ep": x.mean() / x.std(ddof=1) if x.std() > 0 else 0.0,
               "es95_loss": -np.mean(np.sort(x)[: max(1, len(x) // 20)]),
               "hit_rate": (x > 0).mean(), "fills": f.fills.mean(), "contracts": f.contracts.mean(),
               "max_abs_q": f.max_abs_q.mean()}
        e = f.sort_values("seed").exp_pnl.to_numpy()
        be = rng.choice(e, size=(boot, len(e))).mean(1)
        row.update({"exp_pnl": e.mean(), "exp_ci_lo": np.quantile(be, .025), "exp_ci_hi": np.quantile(be, .975),
                    "noise_mean_check": (x - e).mean()})
        ref_rows = df[(df.cell == cell) & (df.policy == ref)].sort_values("seed")
        if pol != ref and len(ref_rows) == len(x):
            for col, key in (("pnl", "diff"), ("exp_pnl", "exp_diff")):
                diff = f.sort_values("seed")[col].to_numpy() - ref_rows[col].to_numpy()
                bm = rng.choice(diff, size=(boot, len(diff))).mean(1)
                row.update({f"{key}_vs_{ref}": diff.mean(), f"{key}_lo": np.quantile(bm, .025),
                            f"{key}_hi": np.quantile(bm, .975)})
        se_b = rng.choice(x, size=(boot, len(x))).mean(1)
        row.update({"mean_ci_lo": np.quantile(se_b, .025), "mean_ci_hi": np.quantile(se_b, .975)})
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "rl_agent/v3/models/ppo_v3.zip"))
    ap.add_argument("--v2", default=str(ROOT / "rl_agent/models/ppo_capstone_broad_v2.zip"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed-start", type=int, default=500_000)   # never used in training
    ap.add_argument("--policies", nargs="*", default=["none", "fixed", "skew", "hjb", "tuned", "v2", "v3"])
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()
    jobs = [(c, p, s) for c in a.cells for p in a.policies for s in range(a.seed_start, a.seed_start + a.n)]
    with Pool(a.workers, initializer=_init, initargs=(a.model, a.v2)) as pool:
        rows = pool.map(run, jobs, chunksize=50)
    df = pd.DataFrame(rows)
    out = ROOT / "rl_agent/v3/results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"episodes_{a.tag}.csv", index=False)
    s = summarize(df)
    s.to_csv(out / f"summary_{a.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
