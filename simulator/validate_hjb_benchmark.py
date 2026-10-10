"""Independent persisted benchmark panel/statistics and representative audits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .market_dynamics import file_hash
from .validate_trading_env import validate_run


def _risk(pnl):
    # Separate implementation, using ordered scalar losses for exact tail mass.
    pnl = np.asarray(pnl,dtype=float)
    losses = sorted((-float(v) for v in pnl),reverse=True)
    mass = .05*len(losses)
    whole = int(np.floor(mass+1e-12))
    es = sum(losses[:whole])
    if mass-whole>1e-12:
        es += (mass-whole)*losses[whole]
    return {"mean_pnl":float(pnl.mean()),"median_pnl":float(np.median(pnl)),
        "std_pnl":float(pnl.std(ddof=1)) if len(pnl)>1 else 0.,
        "var95_loss":float(np.quantile(-pnl,.95,method="linear")),"es95_loss":es/mass,
        "loss_probability":float(np.mean(pnl<0))}


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    design = manifest["design"]
    episodes = pd.read_csv(directory/"episode_results.csv")
    summary = pd.read_csv(directory/"risk_summary.csv")
    pairs = pd.read_csv(directory/"paired_policy_comparison.csv")
    expected = sum(len(v) for v in design["training_event_tickers"].values())*design["replicates_per_event"]*len(
        design["impact_scenarios"])*len(design["fill_scenarios"])*len(design["policies"])
    assert len(episodes)==expected==manifest["episodes_evaluated"]==manifest["episode_ledgers_audited"]
    assert not episodes.episode_id.duplicated().any()
    assert not episodes.select_dtypes(include="number").isna().any().any()
    assert episodes.pnl_attribution_error.abs().max()<1e-8
    assert episodes.minimum_available_cash.min()>=-1e-8
    assert episodes.max_abs_inventory.max()<=design["inventory_limit"]+1e-9
    assert episodes.realized_pnl.min()>=-design["initial_cash"]-1e-9
    assert np.allclose(episodes.cash-design["initial_cash"],episodes.realized_pnl,atol=1e-9)
    keys = ["group","ticker","replicate"]
    for (group,ticker),frame in episodes.groupby(keys[:2]):
        assert ticker in design["training_event_tickers"][group]
        assert frame.replicate.nunique()==design["replicates_per_event"]
    for _,frame in episodes.groupby(keys+["impact_scenario"]):
        assert frame.price_path_sha256.nunique()==1 and frame.settlement_outcome.nunique()==1
        assert frame.market_trades.nunique()==1
    for _,frame in episodes.groupby(keys):
        assert frame.driver_sha256.nunique()==1, "Impact sensitivity changed exogenous marks/innovations"
    none = episodes[episodes.policy.eq("none")]
    assert none.realized_pnl.eq(0).all() and none.maker_fills.eq(0).all()
    cells = ["group","impact_scenario","fill_scenario","policy"]
    checked = 0
    for _,saved in summary.iterrows():
        frame = episodes
        for key in cells:
            frame = frame[frame[key].eq(saved[key])]
        assert len(frame)==saved.episodes
        assert frame.ticker.nunique()==saved.training_events
        stats = _risk(frame.realized_pnl)
        for key, value in stats.items():
            assert abs(value-saved[key])<1e-8, f"Risk statistic mismatch: {key}"
        for metric in ("mean_pnl","var95_loss","es95_loss"):
            assert saved[metric+"_bootstrap95_low"]<=saved[metric+"_bootstrap95_high"]
        assert saved.es95_loss>=saved.var95_loss-1e-8
        checked += 1
    expected_cells = len(design["groups"])*len(design["impact_scenarios"])*len(design["fill_scenarios"])*len(design["policies"])
    assert len(summary)==checked==expected_cells
    for _,row in pairs.iterrows():
        frame = episodes[episodes.group.eq(row.group)&episodes.impact_scenario.eq(row.impact_scenario)&episodes.fill_scenario.eq(row.fill_scenario)]
        target = frame[frame.policy.eq("hjb")].set_index(["ticker","replicate"])
        reference = frame[frame.policy.eq(row.reference_policy)].set_index(["ticker","replicate"])
        difference = target.realized_pnl-reference.realized_pnl
        assert len(difference)==row.paired_episodes
        assert abs(float(difference.mean())-row.mean_pnl_hjb_minus_reference)<1e-8
    # Generic validator re-reads exact-money CSVs, settlement, all financial
    # transitions, source hashes and representative reward telescoping.
    representatives = validate_run(directory)
    for sample in manifest["episodes"]:
        row = episodes[episodes.episode_id.eq(sample["episode_id"])].iloc[0]
        audit = next(r for r in representatives["checks"] if r["episode_id"]==sample["episode_id"])
        assert abs(float(audit["terminal_cash"])-row.cash)<1e-8
        assert abs(float(audit["fees"])-row.fees)<1e-8
    for path,checksum in manifest["data_sha256"].items():
        assert file_hash(path)==checksum, f"Frozen input changed: {path}"
    for relative,checksum in manifest.get("artifact_sha256",{}).items():
        assert file_hash(directory/relative)==checksum, f"Benchmark artifact changed: {relative}"
    result = {"passed":True,"episodes":len(episodes),"risk_cells_verified":checked,
        "paired_policy_comparisons_verified":len(pairs),"representative_ledgers_verified":len(manifest["episodes"]),
        "every_episode_audited_during_run":True,"no_quote_control_zero_pnl":True,
        "shared_driver_across_impact_scenarios":True,"shared_price_and_settlement_across_policies_and_fill_scenarios":True,
        "heldout_raw_files_loaded":manifest["heldout_raw_files_loaded"]}
    (directory/"validation_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    print(json.dumps(validate(parser.parse_args().directory),indent=2))
