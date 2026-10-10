"""Paired profitability/risk evaluation of the frozen HJB reference policy.

Equal weight to each available training event; synthetic full-settlement
episodes. No held-out tape/quotes are loaded, no HJB parameter is retuned.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .benchmark_metrics import (balanced_cube,episode_diagnostics,hierarchical_samples,summarize_cell)
from .demo_trading_env import FILL_SCENARIOS
from .execution import QuoteAction
from .hjb_policy import HJBReferencePolicy
from .market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, file_hash
from .policies import quote_policy
from .spreads import DEFAULT_CONFIG as SPREAD_CONFIG
from .trading_env import EnvironmentConfig,TradingEnvironment
from .validate_trading_env import audit_ledger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/"data_pipeline/output/paper2/calibration/simulator/hjb_benchmark"
POLICIES = ("hjb","fixed","skew","none")
CELL = ["group","impact_scenario","fill_scenario","policy"]


def path_hashes(env):
    driver, price = hashlib.sha256(),hashlib.sha256()
    for row in env.ledger:
        if row["kind"] not in {"noise","market"}:
            continue
        item = {k:row[k] for k in ("kind","elapsed_seconds")}
        item.update({k:row[k] for k in ("innovation_logodds","market_sign","market_quantity") if k in row})
        driver.update(json.dumps(item,sort_keys=True,allow_nan=False).encode())
        item.update(probability=row["probability"],log_odds=row["log_odds"])
        price.update(json.dumps(item,sort_keys=True,allow_nan=False).encode())
    # Check the observed external books as well as the price/reference path.
    price.update(json.dumps([(s["bid"],s["ask"]) for s in env.steps],allow_nan=False).encode())
    return driver.hexdigest(),price.hexdigest()


def cached_environment(cfg, prototype, supported):
    """Reuse immutable loaded banks without changing random-stream semantics.

    Explicit ticker avoids source-selection randomness. The only spread RNG
    consumed by initialization is the supported-bank draw for a borrowed
    ticker, which must still be consumed for every policy on the paired path.
    MarketDynamics and SpreadSampler sampling methods do not mutate the banks.
    """
    def spread_factory(dynamics,rng):
        if dynamics.ticker not in supported:
            selected = str(rng.choice(supported))
            if selected!=prototype.spreads.ticker:
                raise AssertionError("Cached spread source differs from uncached reset")
        return prototype.spreads
    return TradingEnvironment(cfg,dynamics_factory=lambda rng:prototype.dynamics,spread_factory=spread_factory)


def comparison_rows(data, bootstrap_draws, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for (group,impact,fill),cell in data.groupby(CELL[:-1],sort=True):
        target = cell[cell.policy.eq("hjb")].set_index(["ticker","replicate"])
        for name in ("fixed","skew","none"):
            reference = cell[cell.policy.eq(name)].set_index(["ticker","replicate"])
            if len(reference)!=len(target):
                raise AssertionError("Incomplete paired policy panel")
            difference = target.realized_pnl-reference.realized_pnl
            frame = difference.rename("difference").reset_index()
            cube = balanced_cube(frame,"difference")
            boot = hierarchical_samples(cube,rng,bootstrap_draws).mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            rows.append({"group":group,"impact_scenario":impact,"fill_scenario":fill,"reference_policy":name,
                "paired_episodes":len(frame),"mean_pnl_hjb_minus_reference":float(difference.mean()),
                "median_pnl_hjb_minus_reference":float(difference.median()),"bootstrap95_low":float(lo),"bootstrap95_high":float(hi),
                "hjb_pnl_win_fraction":float(difference.gt(0).mean()),
                "mean_abs_terminal_inventory_difference":float((target.abs_terminal_inventory-reference.abs_terminal_inventory).mean()),
                "mean_drawdown_difference":float((target.max_drawdown-reference.max_drawdown).mean())})
    return pd.DataFrame(rows)


def impact_comparisons(data, bootstrap_draws, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for (group,fill,policy),cell in data.groupby(["group","fill_scenario","policy"],sort=True):
        zero = cell[cell.impact_scenario.eq("zero")].set_index(["ticker","replicate"])
        if zero.empty:
            continue
        for impact in ("low","base","high"):
            selected = cell[cell.impact_scenario.eq(impact)].set_index(["ticker","replicate"])
            if selected.empty:
                continue
            difference = selected.realized_pnl-zero.realized_pnl
            cube = balanced_cube(difference.rename("difference").reset_index(),"difference")
            lo,hi = np.quantile(hierarchical_samples(cube,rng,bootstrap_draws).mean(axis=1),[.025,.975])
            rows.append({"group":group,"fill_scenario":fill,"policy":policy,"impact_scenario":impact,
                "paired_episodes":len(difference),"mean_pnl_impact_minus_zero":float(difference.mean()),
                "bootstrap95_low":float(lo),"bootstrap95_high":float(hi),
                "mean_abs_terminal_inventory_difference":float((selected.abs_terminal_inventory-zero.abs_terminal_inventory).mean()),
                "same_driver_fraction":float(selected.driver_sha256.eq(zero.driver_sha256).mean()),
                "same_settlement_outcome_fraction":float(selected.settlement_outcome.eq(zero.settlement_outcome).mean())})
    return pd.DataFrame(rows)


def plot_results(summary, output_dir):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"capstone-matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = summary[summary.fill_scenario.eq("base")&summary.policy.ne("none")]
    groups = sorted(selected.group.unique())
    if not groups:
        return
    fig,axes = plt.subplots(len(groups),3,figsize=(13,4*len(groups)),squeeze=False)
    colors = {"hjb":"#245879","fixed":"#9b6338","skew":"#4c7765"}
    labels = {"hjb":"HJB reference","fixed":"Fixed spread","skew":"Inventory skew"}
    impacts = [s for s in ("zero","low","base","high") if s in selected.impact_scenario.unique()]
    x = np.arange(len(impacts))
    for row,group in enumerate(groups):
        for policy in ("hjb","fixed","skew"):
            frame = selected[selected.group.eq(group)&selected.policy.eq(policy)].set_index("impact_scenario").loc[impacts]
            for col,(metric,title,units) in enumerate([
                ("mean_pnl","Net mean PnL","Dollars per one-hour episode"),
                ("es95_loss","Worst 5% average loss (ES95)","Dollars per one-hour episode"),
                ("mean_abs_terminal_inventory","Mean inventory before settlement","Absolute YES contracts")]):
                y = frame[metric].to_numpy()
                axes[row,col].plot(x,y,marker="o",label=labels[policy],color=colors[policy])
                if metric in {"mean_pnl","es95_loss"}:
                    axes[row,col].fill_between(x,frame[metric+"_bootstrap95_low"].to_numpy(),
                        frame[metric+"_bootstrap95_high"].to_numpy(),color=colors[policy],alpha=.1)
                axes[row,col].set(title=group.replace("_"," ")+": "+title,xlabel="Impact scenario",ylabel=units)
                axes[row,col].set_xticks(x,impacts)
                axes[row,col].axhline(0,color=".6",linewidth=.6)
                axes[row,col].spines[["right","top"]].set_visible(False)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Training-template simulation • base fill scenario • frozen HJB parameters",fontsize=13)
    fig.text(.01,.01,"Shading: hierarchical event/replicate bootstrap 95% ranges; not real-market profit confidence intervals.",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(output_dir/"hjb_profit_and_risk.png",dpi=180)
    plt.close(fig)


def run(output_dir=DEFAULT_OUTPUT, replicates=20, seed=20261009, hours_to_close=1., bootstrap_draws=1000,
        groups=("broad_based","sports"), impacts=("zero","base","high"),
        fills=("conservative","base","optimistic"), tickers=None, time_mapping="episode"):
    if (not isinstance(replicates,int) or replicates<1 or seed<0 or bootstrap_draws<20 or
            not np.isfinite(hours_to_close) or hours_to_close<=0 or
            not groups or not impacts or not fills or len(set(groups))!=len(groups) or
            len(set(impacts))!=len(impacts) or len(set(fills))!=len(fills)):
        raise ValueError("Invalid benchmark design")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    dynamics = json.loads(DYNAMICS_CONFIG.read_text())
    spread = json.loads(SPREAD_CONFIG.read_text())
    selection = {}
    for group in groups:
        profile = dynamics["groups"][group]
        if profile["status"]!="calibrated":
            raise ValueError("Unsupported training group")
        selection[group] = sorted(t for t in profile["event_tickers"] if tickers is None or t in tickers)
        if not selection[group]:
            raise ValueError(f"No training templates selected for {group}")
    if tickers is not None and set(tickers)!=set().union(*map(set,selection.values())):
        raise ValueError("Requested ticker outside selected training groups")
    hjb = HJBReferencePolicy(time_mapping=time_mapping)
    manifest = {"generated_at_utc":datetime.now(timezone.utc).isoformat(),"hjb":hjb.provenance(),
        "design":{"replicates_per_event":replicates,"seed":seed,"hours_to_close":hours_to_close,
            "initial_probability":.5,"initial_cash":1000,"inventory_limit":100,"order_size":10,
            "policies":list(POLICIES),"groups":list(groups),"impact_scenarios":list(impacts),
            "fill_scenarios":list(fills),"training_event_tickers":selection,"bootstrap_draws":bootstrap_draws,
            "fees":{"maker_multiplier":1,"balance_precision":"0.0001","historical_profiles_reconstructed":False},
            "objective":"Realized cash PnL and unpenalized risk diagnostics; no environment inventory penalty"},
        "scope":"Training-template synthetic Monte Carlo comparison; no held-out historical execution or real trading profitability claim.",
        "episodes":[],"source_sha256":{},"config_sha256":{},"data_sha256":{}}
    summaries = []
    total_worlds = sum(map(len,selection.values()))*replicates*len(impacts)
    completed = 0
    last_progress = time.monotonic()
    event_index = 0
    for group,tickers_in_group in selection.items():
        for ticker in tickers_in_group:
            for replicate in range(replicates):
                path_seed = int(np.random.SeedSequence([seed,event_index,replicate]).generate_state(1)[0])
                for impact in impacts:
                    prototype_config = EnvironmentConfig(group=group,ticker=ticker,impact_scenario=impact)
                    prototype = TradingEnvironment(prototype_config)
                    seconds = min(hours_to_close*3600,dynamics["event_templates"][ticker]["lifetime_seconds"])
                    prototype.reset(seed=path_seed,options={"seconds_to_close":seconds})
                    for fill in fills:
                        queue,participation = FILL_SCENARIOS[fill]
                        paired_hash,paired_outcome = None,None
                        for policy in POLICIES:
                            cfg = EnvironmentConfig(group=group,ticker=ticker,impact_scenario=impact,
                                queue_ahead_contracts=queue,participation=participation)
                            env = cached_environment(cfg,prototype,spread["groups"][group]["supported_tickers"])
                            obs,initial_info = env.reset(seed=path_seed,options={"seconds_to_close":seconds})
                            initial_p,initial_l = obs["probability"],obs["log_odds"]
                            actions = []
                            while not env.done:
                                if policy=="hjb":
                                    action,diagnostics = hjb.action(obs,start_elapsed=env.start_elapsed,lifetime=env.lifetime,
                                        inventory_limit=cfg.inventory_limit,tick=cfg.price_tick,lot=cfg.quantity_lot)
                                    actions.append({"elapsed_seconds":env.elapsed,**diagnostics})
                                else:
                                    action = QuoteAction() if policy=="none" else quote_policy(obs,size=hjb.parameters.delta,
                                        inventory_skew=.0005 if policy=="skew" else 0.,tick=cfg.price_tick)
                                obs,_,terminated,truncated,info = env.step(action)
                            if not env.account.settled or truncated:
                                raise AssertionError("Benchmark must reach genuine settlement")
                            arguments = {"initial_cash":cfg.initial_cash,"initial_log_odds":initial_l,
                                "beta":env.dynamics.impact.beta,"inventory_limit":cfg.inventory_limit,
                                "maker_fee_multiplier":cfg.maker_fee_multiplier,"balance_precision":cfg.balance_precision,
                                "quantity_lot":cfg.quantity_lot,"participation":cfg.participation}
                            audit = audit_ledger(env.ledger,**arguments)
                            attribution = episode_diagnostics(env.ledger,cfg.initial_cash,initial_p,env.start_elapsed,env.elapsed)
                            driver_hash,price_hash = path_hashes(env)
                            outcome = info["settlement_outcome"]
                            if paired_hash is None:
                                paired_hash,paired_outcome = price_hash,outcome
                            if price_hash!=paired_hash or outcome!=paired_outcome:
                                raise AssertionError("Policies did not see identical market/settlement paths")
                            if policy=="none" and (env.account.cash!=env.account.initial_cash or audit["maker_fills"]):
                                raise AssertionError("No-quote control must have zero PnL")
                            pnl = float(env.account.cash-env.account.initial_cash)
                            if abs(env.total_reward-pnl)>1e-8:
                                raise AssertionError("Benchmark reward and realized PnL differ")
                            episode_id = f"{group}_{ticker}_{replicate}_{impact}_{fill}_{policy}"
                            terminal_q = float(info["terminal_inventory"])
                            row = {"episode_id":episode_id,"group":group,"ticker":ticker,
                                "event_cluster":env.dynamics.template["event_cluster"],"replicate":replicate,"seed":path_seed,
                                "impact_scenario":impact,"fill_scenario":fill,"policy":policy,"initial_cash":cfg.initial_cash,
                                "episode_seconds":seconds,"realized_pnl":pnl,"cash":float(env.account.cash),
                                "terminal_inventory":terminal_q,"abs_terminal_inventory":abs(terminal_q),
                                "settlement_outcome":outcome,"closing_probability":env.price.probability,
                                "market_trades":audit["market_trades"],"driver_sha256":driver_hash,"price_path_sha256":price_hash,
                                "beta":env.dynamics.impact.beta,"sigma":env.dynamics.sigma,
                                "noise_template_ticker":env.dynamics.noise_template_ticker,"spread_template_ticker":env.spreads.ticker,
                                "volatility_extrapolated_any":initial_info["volatility_extrapolated"] or any(s["volatility_extrapolated"] for s in env.steps),
                                "spread_extrapolated_any":initial_info["spread_extrapolated"] or any(s["spread_extrapolated"] for s in env.steps),
                                "hjb_decisions":len(actions),"hjb_post_only_projections":sum(a["bid_post_only_adjusted"] or a["ask_post_only_adjusted"] for a in actions),
                                "hjb_probability_clips":sum(a["hjb_probability_clipped"] for a in actions),
                                "hjb_crossed_quote_cancellations":sum(a["crossed_quotes_disabled"] for a in actions),
                                "minimum_available_cash":min(s["available_cash"] for s in env.steps),
                                "inventory_limit_reached":any(abs(float(r["inventory"]))>=cfg.inventory_limit-1e-9 for r in env.ledger),
                                **attribution}
                            summaries.append(row)
                            # Persist complete representative ledgers, not thousands
                            # of redundant files. All episodes are audited in memory.
                            if replicate==0 and ticker==tickers_in_group[0]:
                                ledger_file,steps_file = episode_id+"_ledger.csv",episode_id+"_steps.csv"
                                pd.DataFrame(env.ledger).to_csv(output_dir/ledger_file,index=False)
                                pd.DataFrame(env.steps).to_csv(output_dir/steps_file,index=False)
                                if actions:
                                    pd.DataFrame(actions).to_csv(output_dir/(episode_id+"_hjb_actions.csv"),index=False)
                                manifest["episodes"].append({"episode_id":episode_id,"ledger_file":ledger_file,"steps_file":steps_file,
                                    "audit_arguments":arguments,"environment_config":asdict(cfg),"initial_info":initial_info,
                                    "initial_probability":initial_p,"start_elapsed":env.start_elapsed,"end_elapsed":env.elapsed,
                                    "maker_fills":audit["maker_fills"]})
                    completed += 1
                    if time.monotonic()-last_progress>15:
                        print(f"Evaluated {completed}/{total_worlds} market scenarios; {len(summaries)} policy episodes",flush=True)
                        last_progress = time.monotonic()
            event_index += 1
    data = pd.DataFrame(summaries)
    data.to_csv(output_dir/"episode_results.csv",index=False)
    rng = np.random.default_rng(seed)
    risk = []
    for key,frame in data.groupby(CELL,sort=True):
        risk.append({**dict(zip(CELL,key)),**summarize_cell(frame,rng,bootstrap_draws)})
    risk = pd.DataFrame(risk)
    risk.to_csv(output_dir/"risk_summary.csv",index=False)
    comparison_rows(data,bootstrap_draws,seed+1).to_csv(output_dir/"paired_policy_comparison.csv",index=False)
    impact_comparisons(data,bootstrap_draws,seed+2).to_csv(output_dir/"paired_impact_comparison.csv",index=False)
    data.groupby(CELL+["ticker"],as_index=False).agg(episodes=("episode_id","size"),mean_pnl=("realized_pnl","mean"),
        mean_abs_terminal_inventory=("abs_terminal_inventory","mean"),mean_drawdown=("max_drawdown","mean"),
        mean_fills=("maker_fills","mean")).to_csv(output_dir/"event_summary.csv",index=False)
    source_names = ["hjb_policy.py","benchmark_metrics.py","evaluate_hjb.py","validate_hjb_benchmark.py",
        "trading_env.py","market_dynamics.py","price_impact.py","spreads.py","execution.py","accounting.py",
        "fees.py","policies.py","validate_trading_env.py","demo_trading_env.py"]
    manifest["source_sha256"] = {str((ROOT/"simulator"/name).relative_to(ROOT)):file_hash(ROOT/"simulator"/name) for name in source_names}
    manifest["source_sha256"].update(manifest["hjb"]["source_sha256"])
    for path in (DYNAMICS_CONFIG,SPREAD_CONFIG):
        manifest["config_sha256"][str(path)] = file_hash(path)
    for config_path,fields in ((DYNAMICS_CONFIG,("mark_bank_file","impact_config_file")),(SPREAD_CONFIG,("bank_file",))):
        config = json.loads(config_path.read_text())
        for name in fields:
            path = (config_path.parent/config[name]).resolve()
            manifest["data_sha256"][str(path)] = file_hash(path)
    for field in ("grid_file","value_file"):
        path = manifest["hjb"][field]
        manifest["data_sha256"][path] = file_hash(path)
    manifest.update(episodes_evaluated=len(data),market_scenarios=total_worlds,heldout_raw_files_loaded=0,
        episode_ledgers_audited=len(data),persisted_representative_ledgers=len(manifest["episodes"]),
        limitations=["Synthetic training-template Monte Carlo; not independent historical executions.",
            "Same 20 broad-based/15 sports event templates reused across replicates/scenarios; equal-event weights.",
            "Bootstrap resamples templates and paired replicates, not lambda/noise calibration uncertainty or market-wide dependence.",
            "Assumed queue/capture, maker fee multiplier 1, initial probability .5, immediate Bernoulli settlement at close.",
            "HJB frozen paper-model risk/intensity parameters are not recalibrated to impacted price/partial fills.",
            "Policy decisions occur every five minutes, with tick rounding and post-only projection recorded."])
    plot_results(risk,output_dir)
    artifacts = ["episode_results.csv","risk_summary.csv","paired_policy_comparison.csv","paired_impact_comparison.csv","event_summary.csv"]
    if (output_dir/"hjb_profit_and_risk.png").exists():
        artifacts.append("hjb_profit_and_risk.png")
    artifacts += [entry[key] for entry in manifest["episodes"] for key in ("ledger_file","steps_file")]
    manifest["artifact_sha256"] = {name:file_hash(output_dir/name) for name in artifacts}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    from .validate_hjb_benchmark import validate
    validation = validate(output_dir)
    base = risk[risk.policy.eq("hjb")&risk.impact_scenario.eq("base")&risk.fill_scenario.eq("base")]
    return {"passed":validation["passed"],"episodes":len(data),"market_scenarios":total_worlds,
        "training_events":sum(map(len,selection.values())),"base_hjb":base.to_dict("records"),"output_dir":str(output_dir)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates",type=int,default=20)
    parser.add_argument("--seed",type=int,default=20261009)
    parser.add_argument("--hours-to-close",type=float,default=1.)
    parser.add_argument("--bootstrap-draws",type=int,default=1000)
    parser.add_argument("--groups",nargs="+",choices=["broad_based","sports"],default=["broad_based","sports"])
    parser.add_argument("--impact-scenarios",nargs="+",choices=["zero","low","base","high"],default=["zero","base","high"])
    parser.add_argument("--fill-scenarios",nargs="+",choices=list(FILL_SCENARIOS),default=list(FILL_SCENARIOS))
    parser.add_argument("--tickers",nargs="+")
    parser.add_argument("--time-mapping",choices=["episode","lifetime"],default="episode")
    args=parser.parse_args()
    print(json.dumps(run(args.output_dir,args.replicates,args.seed,args.hours_to_close,args.bootstrap_draws,
        args.groups,args.impact_scenarios,args.fill_scenarios,args.tickers,args.time_mapping),indent=2))
