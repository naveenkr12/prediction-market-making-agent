"""Training-only marginal spread bank; queue/capture remain scenario inputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, safe_filename
from calibrate_market_dynamics import normalize_quotes


def build_spread_bank(dynamics, markets, quote_loader, min_quotes=50):
    training = set(dynamics["training"]["tickers"])
    selected = markets[markets.ticker.isin(training)]
    if set(selected.ticker)!=training or min_quotes<1:
        raise ValueError("Missing training metadata or invalid support threshold")
    bank, templates, audit = {}, {}, []
    for index, market in enumerate(selected.to_dict("records")):
        ticker = market["ticker"]
        quotes, counts = normalize_quotes(quote_loader(ticker), market)
        usable = quotes[(quotes.bid>0)&(quotes.ask<1)&(quotes.ask>quotes.bid)]
        counts["interior_positive_spread_rows"] = len(usable)
        counts["spread_status"] = "supported" if len(usable)>=min_quotes else "insufficient_quotes"
        audit.append(counts)
        if len(usable)<min_quotes:
            continue
        key = f"spread_{index:03}"
        bank[key] = (logit(usable.ask.to_numpy())-logit(usable.bid.to_numpy())).astype(float)
        templates[ticker] = {"bank_key": key, "rows": len(usable),
            "start_utc": usable.timestamp.min().isoformat(), "end_utc": usable.timestamp.max().isoformat(),
            "median_spread_probability": float(usable.spread.median())}
    groups = {}
    for group, profile in dynamics["groups"].items():
        if profile["status"]!="calibrated":
            continue
        supported = sorted(set(profile["event_tickers"])&templates.keys())
        if len(supported)<5:
            raise ValueError(f"Insufficient spread events for {group}")
        groups[group] = {"supported_tickers": supported}
    return {"schema_version": 1, "training_tickers": sorted(training), "min_quotes": min_quotes,
        "spread_units": "log_odds_ask_minus_bid", "templates": templates, "groups": groups,
        "assumptions": ["Spreads sampled independently from marginal trailing-window training quotes.",
            "Midpoint, lifetime dependence, replenishment and queue depth are not estimated by this bank.",
            "Queue ahead, market-volume participation and tick size are explicit scenario inputs."]}, bank, pd.DataFrame(audit)


def run(raw_input=DEFAULT_INPUT, calibration_dir=DEFAULT_OUTPUT, output_dir=None):
    output_dir = output_dir or calibration_dir/"simulator"/"execution"
    dynamics_path = calibration_dir/"simulator"/"market_dynamics"/"market_dynamics_config.json"
    dynamics = json.loads(dynamics_path.read_text())
    splits_path = calibration_dir/"event_splits.csv"
    splits = pd.read_csv(splits_path)
    train = set(splits.loc[splits.split.eq("train"), "ticker"])
    if train!=set(dynamics["training"]["tickers"]) or file_hash(splits_path)!=dynamics["upstream_sha256"]["event_splits.csv"]:
        raise ValueError("Training split changed")
    markets_path = raw_input/"markets"/"kalshi_selected_contracts.parquet"
    if file_hash(markets_path)!=dynamics["raw_input_sha256"]["markets/kalshi_selected_contracts.parquet"]:
        raise ValueError("Training metadata changed")
    hashes, loaded = {}, []
    def load(ticker):
        if ticker not in train:
            raise AssertionError("Held-out quote access")
        path = raw_input/"candlesticks"/f"{safe_filename(ticker)}.parquet"
        rel = str(path.relative_to(raw_input))
        checksum = file_hash(path)
        if checksum!=dynamics["raw_input_sha256"].get(rel):
            raise ValueError(f"Quotes changed: {ticker}")
        hashes[rel] = checksum
        loaded.append(ticker)
        return pd.read_parquet(path)
    config, bank, audit = build_spread_bank(dynamics, pd.read_parquet(markets_path), load)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/"spread_bank.npz", **bank)
    audit.to_csv(output_dir/"training_spread_audit.csv", index=False)
    config.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), bank_file="spread_bank.npz",
        bank_sha256=file_hash(output_dir/"spread_bank.npz"), raw_quote_sha256=hashes,
        dynamics_config_file=os.path.relpath(dynamics_path.resolve(), output_dir.resolve()),
        dynamics_config_sha256=file_hash(dynamics_path),
        source_sha256={str(p.relative_to(PROJECT_ROOT)):file_hash(p) for p in
            [Path(__file__), PROJECT_ROOT/"calibration"/"calibrate_market_dynamics.py"]})
    (output_dir/"execution_config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    report = {"passed": set(loaded)==train, "training_quote_files": len(loaded),
        "supported_events": len(config["templates"]), "heldout_quote_files_loaded": 0,
        "usable_spread_rows": sum(t["rows"] for t in config["templates"].values()),
        "groups": {g:len(p["supported_tickers"]) for g,p in config["groups"].items()}}
    (output_dir/"calibration_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_input, args.calibration_dir, args.output_dir), indent=2))
