"""Bounded complete episodes with fixed/skew/no-quote reference policies."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .execution import QuoteAction
from .market_dynamics import file_hash
from .policies import quote_policy
from .trading_env import EnvironmentConfig, TradingEnvironment
from .validate_trading_env import audit_ledger, validate_run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/"data_pipeline/output/paper2/calibration/simulator/trading_environment"
FILL_SCENARIOS = {"conservative": (100,.05), "base": (50,.1), "optimistic": (0,1)}


def run(output_dir=DEFAULT_OUTPUT, episodes=3, seed=42, hours_to_close=1.,
        groups=("broad_based","sports"), impacts=("zero","base","high"),
        fills=("conservative","base","optimistic"), policies=("none","fixed","skew"), fee_multiplier=1.):
    if episodes<1 or seed<0 or not np.isfinite(hours_to_close) or hours_to_close<=0:
        raise ValueError("Invalid demo length or seed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest = {"generated_at_utc":datetime.now(timezone.utc).isoformat(), "episodes": [],
        "scope":"Synthetic training-template scenarios; not held-out results or historical execution PnL.",
        "fee_schedule_effective_date":"2026-07-07", "historical_fee_profiles_reconstructed":False,
        "fee_rule_sources":["https://kalshi.com/docs/kalshi-fee-schedule.pdf",
                            "https://docs.kalshi.com/getting_started/fee_rounding"],
        "hours_to_close":hours_to_close, "source_sha256":{}, "config_sha256":{}}
    summary = []
    for group in groups:
        for impact in impacts:
            for fill in fills:
                queue, participation = FILL_SCENARIOS[fill]
                for policy in policies:
                    for index in range(episodes):
                        cfg = EnvironmentConfig(group=group,impact_scenario=impact,queue_ahead_contracts=queue,
                            participation=participation,maker_fee_multiplier=fee_multiplier)
                        env = TradingEnvironment(cfg)
                        obs,initial_info = env.reset(seed=seed+index)
                        seconds = min(hours_to_close*3600,env.lifetime)
                        obs,initial_info = env.reset(seed=seed+index,options={"seconds_to_close":seconds})
                        initial_latent = env.price.log_odds
                        while not env.done:
                            action = QuoteAction() if policy=="none" else quote_policy(obs,
                                inventory_skew=.0005 if policy=="skew" else 0.,tick=cfg.price_tick)
                            obs,_,_,_,info = env.step(action)
                        arguments = {"initial_cash":cfg.initial_cash,"initial_log_odds":initial_latent,
                            "beta":env.dynamics.impact.beta,"inventory_limit":cfg.inventory_limit,
                            "maker_fee_multiplier":fee_multiplier,"balance_precision":cfg.balance_precision,
                            "quantity_lot":cfg.quantity_lot,"participation":cfg.participation}
                        audit = audit_ledger(env.ledger,**arguments)
                        assert env.account.settled, "Demo hit max_steps; increase cap for this episode length"
                        episode_id = f"{group}_{impact}_{fill}_{policy}_{seed+index}"
                        ledger_file, steps_file = episode_id+"_ledger.csv", episode_id+"_steps.csv"
                        pd.DataFrame(env.ledger).to_csv(output_dir/ledger_file,index=False)
                        pd.DataFrame(env.steps).to_csv(output_dir/steps_file,index=False)
                        entry = {"episode_id":episode_id,"ledger_file":ledger_file,"steps_file":steps_file,
                            "audit_arguments":arguments,"environment_config":asdict(cfg),"initial_info":initial_info,
                            "maker_fills":audit["maker_fills"]}
                        manifest["episodes"].append(entry)
                        summary.append({"episode_id":episode_id,"group":group,"impact_scenario":impact,
                            "fill_scenario":fill,"policy":policy,"seed":seed+index,"ticker":env.dynamics.ticker,
                            "market_trades":audit["market_trades"],"maker_fills":audit["maker_fills"],
                            "maker_contracts":sum(float(r["quantity"]) for r in env.ledger if r["kind"]=="fill"),
                            "fees":float(env.account.total_fees),"realized_pnl":float(env.account.cash-env.account.initial_cash),
                            "reward":env.total_reward,"terminal_inventory_before_settlement":info["terminal_inventory"],
                            "settlement_outcome":info["settlement_outcome"],"cash":obs["cash"],
                            "max_abs_inventory":max(abs(float(r["inventory"])) for r in env.ledger),
                            "source_noise_extrapolated_any":any(s["volatility_extrapolated"] for s in env.steps),
                            "source_spread_extrapolated_any":any(s["spread_extrapolated"] for s in env.steps),
                            "saturated_market_updates":sum(r.get("float_probability_saturated",False) for r in env.ledger)})
                        for path in (env.dynamics_path,env.execution_path):
                            manifest["config_sha256"][str(path)] = file_hash(path)
    source = [ROOT/"simulator"/name for name in
        ["accounting.py","fees.py","execution.py","trading_env.py","spreads.py","policies.py","price_impact.py",
         "market_dynamics.py","demo_trading_env.py","validate_trading_env.py"]]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)):file_hash(p) for p in source}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    data = pd.DataFrame(summary)
    data.to_csv(output_dir/"episode_summary.csv",index=False)
    data.groupby(["group","impact_scenario","fill_scenario","policy"],as_index=False).agg(
        episodes=("episode_id","size"),mean_pnl=("realized_pnl","mean"),mean_fills=("maker_fills","mean"),
        mean_fees=("fees","mean"),worst_pnl=("realized_pnl","min")).to_csv(output_dir/"scenario_summary.csv",index=False)
    validation = validate_run(output_dir)
    return {"passed":validation["passed"],"episodes":len(data),"market_trades":int(data.market_trades.sum()),
        "maker_fills":int(data.maker_fills.sum()),"heldout_raw_files_loaded":0,"output_dir":str(output_dir)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--hours-to-close",type=float,default=1.)
    parser.add_argument("--groups",nargs="+",choices=["broad_based","sports","pooled"],default=["broad_based","sports"])
    parser.add_argument("--impact-scenarios",nargs="+",choices=["zero","low","base","high"],default=["zero","base","high"])
    parser.add_argument("--fill-scenarios",nargs="+",choices=list(FILL_SCENARIOS),default=list(FILL_SCENARIOS))
    parser.add_argument("--policies",nargs="+",choices=["none","fixed","skew"],default=["none","fixed","skew"])
    parser.add_argument("--maker-fee-multiplier",type=float,default=1.)
    args=parser.parse_args()
    print(json.dumps(run(args.output_dir,args.episodes,args.seed,args.hours_to_close,args.groups,
        args.impact_scenarios,args.fill_scenarios,args.policies,args.maker_fee_multiplier),indent=2))
