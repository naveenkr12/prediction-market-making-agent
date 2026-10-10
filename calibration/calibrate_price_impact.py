"""Train-only, scenario-based calibration of a log-odds market-flow jump.

Choose beta so the median five-minute lambda in a post-trade reference-price
replay matches the training median. Low/high scenarios envelope sampling,
per-market uncertainty, endpoint filters, and influential-bucket sensitivities.
This is a reduced-form modelling choice, not causal identification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, normalize_trades, safe_filename
from estimate_lambda import VARIANTS, fit_lambda, interval_subset

sys.path.insert(0, str(PROJECT_ROOT))
from simulator.price_impact import ImpactParameters, observe_trade_flows

CALIBRATION_VARIANTS = [*VARIANTS, "drop_max_leverage"]


def endpoint_factor(d: pd.DataFrame) -> float | None:
    """OLS slope of after-first flow on total flow, with an intercept."""
    flow = d.signed_flow_contracts.to_numpy(dtype=float)
    centered = flow-flow.mean()
    denominator = float(centered@centered)
    if denominator <= 1e-24:
        return None
    effective = d.after_first_signed_flow.to_numpy(dtype=float)
    return float(centered@(effective-effective.mean())/denominator)


def make_training_cases(buckets: pd.DataFrame, estimates: pd.DataFrame,
                        tape_loader, min_intervals: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filter before statistics/raw-tape loading; recompute training fits only."""
    train = buckets[buckets.split.eq("train")].copy()
    saved = estimates[estimates.split.eq("train") & estimates.response.eq("logodds")].copy()
    if train.empty or saved.empty:
        raise ValueError("No training events")
    if train.groupby("event_cluster").split.nunique().gt(1).any():
        raise ValueError("Inconsistent event splits")
    if saved.duplicated(["ticker", "variant"]).any():
        raise ValueError("Duplicate saved training fits")
    if set(saved.ticker) != set(train.ticker):
        raise ValueError("Training contracts differ between intervals and estimates")
    augmented, records, cases = [], [], {}
    for ticker, d in train.groupby("ticker", sort=True):
        d = d.sort_values("bucket_start").reset_index(drop=True)
        tape = tape_loader(ticker).sort_values(["timestamp", "trade_id"], kind="stable").copy()
        tape["bucket_start"] = tape.timestamp.dt.floor("5min")
        tape["signed"] = tape.quantity*tape.sign
        first = tape.groupby("bucket_start").signed.first()
        totals = tape.groupby("bucket_start").signed.sum()
        if d.bucket_start.tolist() != first.index.tolist() or int(d.n_trades.sum()) != len(tape):
            raise ValueError(f"Training tape/bucket mismatch: {ticker}")
        if not np.allclose(d.signed_flow_contracts, totals.to_numpy(), rtol=1e-10, atol=1e-8):
            raise ValueError(f"Training signed-flow mismatch: {ticker}")
        d["first_trade_signed_contracts"] = first.to_numpy()
        d["after_first_signed_flow"] = d.signed_flow_contracts-d.first_trade_signed_contracts
        augmented.append(d)
        metadata = {key: d[key].iloc[0] for key in ["event_cluster", "information_class", "series_ticker", "category"]}
        market_saved = saved[saved.ticker.eq(ticker)].set_index("variant")
        if not set(VARIANTS).issubset(market_saved.index):
            raise ValueError(f"Missing training sensitivity fits: {ticker}")
        for variant in CALIBRATION_VARIANTS:
            if variant == "drop_max_leverage":
                subset = interval_subset(d, "full", "logodds")
                if len(subset):
                    # With intercept plus one regressor, max leverage is the
                    # largest absolute distance from the flow mean.
                    imax = (subset.signed_flow_contracts-subset.signed_flow_contracts.mean()).abs().idxmax()
                    subset = subset.drop(index=imax).reset_index(drop=True)
                hac_lags = int(market_saved.loc["full", "hac_lags_5min"])
                fit = fit_lambda(subset, min_intervals=min_intervals, hac_lags=hac_lags)
                fit["variant"] = variant
            else:
                subset = interval_subset(d, variant, "logodds")
                old = market_saved.loc[variant]
                fit = fit_lambda(d, variant, min_intervals=min_intervals, hac_lags=int(old.hac_lags_5min))
                if fit["fit_status"] != old.fit_status or fit["n_intervals"] != old.n_intervals:
                    raise ValueError(f"Saved training fit status changed: {ticker}/{variant}")
                if fit["fit_status"] == "ok" and not np.isclose(fit["lambda"], old["lambda"], rtol=1e-9, atol=1e-12):
                    raise ValueError(f"Saved training lambda changed: {ticker}/{variant}")
            factor = endpoint_factor(subset) if fit["fit_status"] == "ok" else None
            records.append({"ticker": ticker, **metadata, "split": "train", **fit,
                            "endpoint_flow_factor": factor})
            cases[(ticker, variant)] = subset
    return pd.concat(augmented, ignore_index=True), pd.DataFrame(records), cases


def bootstrap_beta(d: pd.DataFrame, draws: int, rng) -> tuple[list[float], int]:
    """Whole-event bootstrap of the projected median-lambda/median-factor ratio."""
    clusters = [g[["lambda", "endpoint_flow_factor"]].to_numpy(dtype=float)
                for _, g in d.groupby("event_cluster", sort=True)]
    samples, invalid = [], 0
    for _ in range(draws):
        selected = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        factor = float(np.median(selected[:, 1]))
        if factor <= 1e-12:
            invalid += 1
            continue
        samples.append(max(0.0, float(np.median(selected[:, 0])))/factor)
    if not samples:
        raise ValueError("All bootstrap draws have nonpositive endpoint factors")
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])], invalid


def build_profiles(targets: pd.DataFrame, min_events: int = 5,
                   bootstrap_draws: int = 2000, seed: int = 42) -> dict:
    profiles = {}
    names = [*sorted(targets.information_class.unique()), "pooled"]
    for index, name in enumerate(names):
        group = targets if name == "pooled" else targets[targets.information_class.eq(name)]
        full = group[group.variant.eq("full") & group.fit_status.eq("ok")]
        event_count = int(full.event_cluster.nunique())
        profile = {"status": "insufficient_training_events", "training_contracts": len(full),
                   "training_event_clusters": event_count, "minimum_events": min_events,
                   "scenarios": None, "sensitivities": []}
        profiles[name] = profile
        if event_count < min_events:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        for variant in CALIBRATION_VARIANTS:
            d = group[group.variant.eq(variant) & group.fit_status.eq("ok")]
            n_events = int(d.event_cluster.nunique())
            factor = float(d.endpoint_flow_factor.median()) if len(d) else None
            row = {"variant": variant, "contracts": len(d), "event_clusters": n_events,
                   "status": "insufficient_training_events", "lambda_mean": None, "lambda_median": None,
                   "endpoint_factor_median": factor, "beta": None, "bootstrap95_beta": None,
                   "median_market_hac_bounds_beta": None, "bootstrap_invalid_factor_draws": None}
            profile["sensitivities"].append(row)
            if n_events < min_events:
                continue
            if factor is None or not np.isfinite(factor) or factor <= 1e-12:
                row["status"] = "nonpositive_endpoint_factor"
                continue
            median = float(d["lambda"].median())
            beta = max(0.0, median)/factor
            sampling, invalid = bootstrap_beta(d, bootstrap_draws, rng)
            # This is a stress envelope of individual uncertainty bounds, not
            # a confidence interval for the group or a multiple-testing claim.
            hac_bounds = [max(0.0, float(d[col].median()))/factor for col in
                          ["ci95_low_calendar_hac", "ci95_high_calendar_hac"]]
            row.update(status="calibrated", lambda_mean=float(d["lambda"].mean()), lambda_median=median,
                       beta=beta, bootstrap95_beta=sampling, median_market_hac_bounds_beta=hac_bounds,
                       bootstrap_invalid_factor_draws=invalid)
        full_summary = profile["sensitivities"][0]
        if full_summary["status"] != "calibrated":
            profile["status"] = full_summary["status"]
            continue
        bounds = [value for row in profile["sensitivities"] if row["status"] == "calibrated"
                  for value in [row["beta"], *row["bootstrap95_beta"], *row["median_market_hac_bounds_beta"]]]
        beta = full_summary["beta"]
        profile.update(status="calibrated", scenarios={"zero": 0.0, "low": min(bounds), "base": beta, "high": max(bounds)},
                       target_lambda_median=full_summary["lambda_median"],
                       target_lambda_mean=full_summary["lambda_mean"],
                       endpoint_flow_factor_median=full_summary["endpoint_factor_median"],
                       base_beta_bootstrap95=full_summary["bootstrap95_beta"],
                       beta_boundary_projection=bool(full_summary["lambda_median"] < 0),
                       base_p50_move_100_contracts_pp=float(100*(expit(100*beta)-0.5)))
    return profiles


def replay_validation(profiles: dict, targets: pd.DataFrame, cases: dict, flow_by_bucket: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic post-trade replay: validate units and endpoint mapping.

    Each bucket is a separate zero-innovation segment initialized at its saved
    opening price. It is not a full-life simulation or held-out performance test.
    """
    if not any(profile["status"] == "calibrated" for profile in profiles.values()):
        raise ValueError("No group has sufficient training support for a calibrated impact parameter")
    rows, checks, cache = [], [], {}
    for group, profile in profiles.items():
        if profile["status"] != "calibrated":
            continue
        selected = targets[targets.fit_status.eq("ok")]
        if group != "pooled":
            selected = selected[selected.information_class.eq(group)]
        for scenario, beta in profile["scenarios"].items():
            for target in selected.to_dict("records"):
                d = cases[(target["ticker"], target["variant"])].copy()
                deltas, float_deltas, saturated = [], [], 0
                for bucket in d.itertuples():
                    key = (target["ticker"], bucket.bucket_start, beta)
                    if key not in cache:
                        latent, prices = observe_trade_flows(bucket.open_price,
                            flow_by_bucket[(target["ticker"], bucket.bucket_start)], ImpactParameters(beta, group, scenario))
                        cache[key] = (float(latent[-1]-latent[0]), float(logit(prices[-1])-logit(prices[0])),
                                      int(((prices == np.nextafter(1.0, 0.0)) |
                                      (prices == np.nextafter(0.0, 1.0))).sum()))
                    delta, float_delta, clipped = cache[key]
                    deltas.append(delta)
                    float_deltas.append(float_delta)
                    saturated += clipped
                flow = d.signed_flow_contracts.to_numpy(dtype=float)
                centered = flow-flow.mean()
                y = np.asarray(deltas)
                realized = float(centered@(y-y.mean())/(centered@centered))
                float_y = np.asarray(float_deltas)
                float_realized = float(centered@(float_y-float_y.mean())/(centered@centered))
                expected = beta*target["endpoint_flow_factor"]
                if not np.isclose(realized, expected, rtol=1e-8, atol=1e-12):
                    raise AssertionError("Replayed lambda does not match the endpoint mapping")
                rows.append({"information_class": group, "scenario": scenario, "beta": beta,
                             "ticker": target["ticker"], "variant": target["variant"], "split": "train",
                             "n_intervals": len(d), "empirical_lambda": target["lambda"],
                             "simulated_reference_lambda": realized, "expected_reference_lambda": expected,
                             "float_probability_lambda": float_realized,
                             "float_probability_lambda_error": abs(float_realized-realized),
                             "absolute_mapping_error": abs(realized-expected), "saturated_trade_probabilities": saturated})
    replay = pd.DataFrame(rows)
    for (group, scenario, variant), d in replay.groupby(["information_class", "scenario", "variant"], sort=True):
        empirical = float(d.empirical_lambda.median())
        realized = float(d.simulated_reference_lambda.median())
        float_realized = float(d.float_probability_lambda.median())
        if scenario == "base" and variant == "full" and not np.isclose(realized, max(0.0, empirical), rtol=1e-8, atol=1e-12):
            raise AssertionError("Base group median lambda does not reproduce the training target")
        if scenario == "base" and variant == "full" and not np.isclose(float_realized, max(0.0, empirical), rtol=1e-7, atol=1e-12):
            raise AssertionError("Base median lambda from float probabilities does not reproduce the training target")
        checks.append({"information_class": group, "scenario": scenario, "variant": variant,
                       "contracts": len(d), "empirical_lambda_median": empirical,
                       "empirical_lambda_mean": float(d.empirical_lambda.mean()),
                       "simulated_reference_lambda_median": realized,
                       "simulated_reference_lambda_mean": float(d.simulated_reference_lambda.mean()),
                       "float_probability_lambda_median": float_realized,
                       "max_float_probability_lambda_error": float(d.float_probability_lambda_error.max()),
                       "max_absolute_mapping_error": float(d.absolute_mapping_error.max()),
                       "saturated_trade_probabilities": int(d.saturated_trade_probabilities.sum())})
    return replay, pd.DataFrame(checks)


def run(raw_input: Path = DEFAULT_INPUT, calibration_dir: Path = DEFAULT_OUTPUT,
        output_dir: Path | None = None, min_events: int = 5, bootstrap_draws: int = 2000, seed: int = 42) -> dict:
    if min_events < 2 or bootstrap_draws < 20 or seed < 0:
        raise ValueError("Require at least two events, twenty bootstrap draws and a nonnegative seed")
    output_dir = output_dir or calibration_dir/"simulator"
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets_path = calibration_dir/"five_minute_intervals.parquet"
    estimates_path = calibration_dir/"lambda_estimates.parquet"
    buckets, estimates = pd.read_parquet(buckets_path), pd.read_parquet(estimates_path)
    if buckets.groupby("event_cluster").split.nunique().gt(1).any():
        raise ValueError("An event cluster crosses data splits")
    manifest = json.loads((calibration_dir/"aggregation_manifest.json").read_text())
    hashes, tapes, flow_by_bucket = {}, {}, {}
    def load(ticker):
        path = raw_input/"trades"/f"{safe_filename(ticker)}.parquet"
        relative = str(path.relative_to(raw_input))
        checksum = file_hash(path)
        if checksum != manifest["input_sha256"].get(relative):
            raise ValueError(f"Training tape changed since estimation: {ticker}")
        hashes[relative] = checksum
        tape, _ = normalize_trades(pd.read_parquet(path), ticker)
        tapes[ticker] = tape
        for ts, d in tape.groupby(tape.timestamp.dt.floor("5min")):
            flow_by_bucket[(ticker, ts)] = (d.sign*d.quantity).to_numpy(dtype=float)
        return tape
    training_buckets, targets, cases = make_training_cases(buckets, estimates, load)
    profiles = build_profiles(targets, min_events, bootstrap_draws, seed)
    replay, replay_groups = replay_validation(profiles, targets, cases, flow_by_bucket)
    training_buckets.to_parquet(output_dir/"training_impact_intervals.parquet", index=False)
    targets.to_csv(output_dir/"training_impact_targets.csv", index=False)
    replay.to_csv(output_dir/"replay_contract_checks.csv", index=False)
    replay_groups.to_csv(output_dir/"replay_group_checks.csv", index=False)
    table = [{"information_class": group, "scenario": name, "beta_logodds_per_contract": value,
              "training_contracts": profile["training_contracts"],
              "target_full_lambda_median": profile["target_lambda_median"],
              "p50_move_100_contracts_pp": float(100*(expit(100*value)-0.5))}
             for group, profile in profiles.items() if profile["status"] == "calibrated"
             for name, value in profile["scenarios"].items()]
    pd.DataFrame(table).to_csv(output_dir/"price_impact_scenarios.csv", index=False)
    config = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "beta_units": "log_odds_per_signed_market_contract", "reference_observation": "post_trade",
              "update_rule": "log_odds_next = log_odds + innovation_logodds + beta * signed_market_contracts",
              "endpoint_mapping": "factor = Cov(total_flow, total_flow - first_trade_signed_quantity) / Var(total_flow); "
                                  "beta = max(0, median(training_lambda)) / median(training_factor)",
              "scenario_rule": "Base matches full training median; low/high envelope all available sensitivity point "
                               "calibrations, event-bootstrap 95% ranges and median individual HAC bounds. These are stress scenarios.",
              "training": {"split": "train", "contracts": training_buckets.ticker.nunique(),
                           "event_clusters": training_buckets.event_cluster.nunique(), "intervals": len(training_buckets),
                           "trades": int(training_buckets.n_trades.sum()), "tickers": sorted(tapes),
                           "trade_start_utc": training_buckets.first_trade_ts.min().isoformat(),
                           "trade_end_utc": training_buckets.last_trade_ts.max().isoformat(),
                           "min_events": min_events, "min_intervals_per_fit": 50,
                           "bootstrap_draws": bootstrap_draws, "seed": seed},
              "groups": profiles, "training_tape_sha256": hashes,
              "source_file_sha256": {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
                    [Path(__file__), PROJECT_ROOT/"simulator"/"price_impact.py", PROJECT_ROOT/"calibration"/"estimate_lambda.py",
                     PROJECT_ROOT/"calibration"/"build_intervals.py", PROJECT_ROOT/"data_pipeline"/"calibration_quality.py",
                     PROJECT_ROOT/"data_pipeline"/"utils.py"]},
              "upstream_file_sha256": {"five_minute_intervals.parquet": file_hash(buckets_path),
                                       "lambda_estimates.parquet": file_hash(estimates_path)},
              "limitations": ["Transaction-price lambda does not identify causal reference-price impact or an informed-trader fraction.",
                    "Post-trade reference observations are an explicit modelling proxy; actual execution quotes need separate validation.",
                    "Base targets an equal-weight median, not every contract's coefficient or the pooled OLS slope.",
                    "Low/high ranges are uncertainty stress scenarios, not confidence bounds for a causal structural parameter.",
                    "Event bootstraps do not eliminate cross-event dependence or convenience-sample selection.",
                    "Single-name has insufficient training events for its own parameter; pooled use must be explicit.",
                    "Replay checks reset each bucket and use zero innovations: no full-life, out-of-sample or PnL performance claim.",
                    "Arrival rates, independent volatility and inventory/risk parameters are not calibrated in this step."]}
    (output_dir/"price_impact_config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    report = {"passed": True, "training_contracts": int(training_buckets.ticker.nunique()),
              "training_intervals": len(training_buckets), "training_trades": int(training_buckets.n_trades.sum()),
              "heldout_raw_tapes_loaded": 0, "attempted_training_fits": len(targets),
              "successful_training_fits": int(targets.fit_status.eq("ok").sum()),
              "replayed_contract_variant_scenarios": len(replay),
              "max_lambda_mapping_error": float(replay.absolute_mapping_error.max()),
              "checks": ["Training raw tapes match verified hashes", "Saved training estimates reproduce from intervals",
                         "Post-trade reference replay matches analytical endpoint factors",
                         "Base group median reproduces projected full training median",
                         "Base median also matches when log odds are recomputed from float probabilities",
                         "No validation/test raw tapes loaded or used to choose parameters"]}
    (output_dir/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return {**report, "scenarios": {name: p["scenarios"] for name, p in profiles.items()}, "output_dir": str(output_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_input, args.calibration_dir, args.output_dir,
                         args.min_events, args.bootstrap_draws, args.seed), indent=2))
