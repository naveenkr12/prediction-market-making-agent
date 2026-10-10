"""Training-only market trade arrivals, paired size/side marks and residual noise.

Full listing exposure includes quiet time. Quotes sample actual UTC endpoints;
no forward fills or fabricated quote changes. Residual volatility is a trailing
quote-window proxy conditional on the already chosen impact scenario.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, normalize_trades, safe_filename

SCENARIOS = ["zero", "low", "base", "high"]
NOISE_VARIANTS = {"midpoint_5min": (300, False), "midpoint_10min": (600, False),
                  "interior_midpoint_5min": (300, True)}


def count_moments(timestamps: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Count moments on full UTC five-minute intervals, with implicit zeros."""
    lo, hi = start.ceil("5min"), end.floor("5min")
    n = max(0, int((hi-lo).total_seconds()/300))
    counts = timestamps.groupby(timestamps.dt.floor("5min")).size()
    counts = counts[(counts.index >= lo) & (counts.index < hi)]
    total, squared = int(counts.sum()), float(np.dot(counts, counts))
    mean = total/n if n else 0.0
    variance = max(0.0, (squared-n*mean*mean)/(n-1)) if n > 1 else None
    fano = variance/mean if variance is not None and mean > 0 else None
    alpha = max(0.0, (variance-mean)/(mean*mean)) if variance is not None and mean > 0 else 0.0
    return {"full_5min_intervals": n, "full_interval_trade_count": total,
            "full_interval_count_square_sum": squared, "full_interval_count_mean": mean,
            "full_interval_count_variance": variance, "fano_5min": fano,
            "gamma_poisson_alpha": alpha, "empty_full_5min_intervals": n-len(counts)}


def activity_profile(tape: pd.DataFrame, market: dict, n_phases: int = 10) -> tuple[dict, list[dict], np.ndarray]:
    opened, closed = pd.to_datetime(market["open_time"], utc=True), pd.to_datetime(market["close_time"], utc=True)
    duration = (closed-opened).total_seconds()
    if duration <= 0 or tape.empty or not tape.timestamp.ge(opened).all() or not tape.timestamp.le(closed).all():
        raise ValueError(f"Invalid training lifetime/tape: {market['ticker']}")
    phase = np.minimum(((tape.timestamp-opened).dt.total_seconds().to_numpy()/duration*n_phases).astype(int), n_phases-1)
    record = {"ticker": market["ticker"], "event_cluster": market["event_cluster"],
              "information_class": market["information_class"], "split": "train", "trades": len(tape),
              "unique_trade_timestamps": int(tape.timestamp.nunique()), "open_time": opened.isoformat(),
              "close_time": closed.isoformat(), "lifetime_seconds": duration,
              "arrival_rate_per_second": len(tape)/duration, "arrival_rate_per_minute": 60*len(tape)/duration,
              "quantity_total": float(tape.quantity.sum()), "quantity_mean": float(tape.quantity.mean()),
              "quantity_median": float(tape.quantity.median()), "quantity_p95": float(tape.quantity.quantile(.95)),
              "quantity_max": float(tape.quantity.max()), "yes_trade_fraction": float(tape.sign.eq(1).mean()),
              "yes_quantity_fraction": float(tape.quantity[tape.sign.eq(1)].sum()/tape.quantity.sum()),
              **count_moments(tape.timestamp, opened, closed)}
    rows = []
    for j in range(n_phases):
        d = tape[phase == j]
        lo, hi = opened+(closed-opened)*j/n_phases, opened+(closed-opened)*(j+1)/n_phases
        rows.append({"ticker": market["ticker"], "phase": j, "life_fraction_start": j/n_phases,
                     "life_fraction_end": (j+1)/n_phases, "exposure_seconds": duration/n_phases,
                     "trades": len(d), "arrival_rate_per_second": len(d)/(duration/n_phases),
                     "quantity_mean": float(d.quantity.mean()) if len(d) else None,
                     "quantity_median": float(d.quantity.median()) if len(d) else None,
                     "yes_trade_fraction": float(d.sign.eq(1).mean()) if len(d) else None,
                     **count_moments(d.timestamp, lo, hi)})
    if sum(r["trades"] for r in rows) != len(tape):
        raise AssertionError("Phase assignment loses trades")
    return record, rows, phase


def normalize_quotes(raw: pd.DataFrame, market: dict, max_spread: float = .20) -> tuple[pd.DataFrame, dict]:
    """Saved flattened dollar-probability fields, including the historical aliases."""
    d = pd.DataFrame(index=raw.index)
    for side in ["bid", "ask"]:
        current, alias = f"yes_{side}_close_dollars", f"yes_{side}_close"
        values = raw[current] if current in raw else raw[alias] if alias in raw else pd.Series(np.nan, index=raw.index)
        d[side] = pd.to_numeric(values, errors="coerce")
    ts = pd.to_numeric(raw.get("end_period_ts", pd.Series(np.nan, index=raw.index)), errors="coerce")
    d["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    d["midpoint"] = (d.bid+d.ask)/2
    d["spread"] = d.ask-d.bid
    opened, closed = pd.to_datetime(market["open_time"], utc=True), pd.to_datetime(market["close_time"], utc=True)
    valid = (d.bid.between(0, 1) & d.ask.between(0, 1) & d.bid.le(d.ask) &
             d.midpoint.between(0, 1, inclusive="neither") & d.timestamp.ge(opened) & d.timestamp.le(closed))
    invalid = int((~valid).sum())
    d = d[valid].copy()
    conflict = d.groupby("timestamp")[["bid", "ask"]].nunique().max(axis=1).gt(1)
    conflicts = set(conflict[conflict].index)
    conflicting_rows = int(d.timestamp.isin(conflicts).sum())
    d = d[~d.timestamp.isin(conflicts)].drop_duplicates("timestamp").sort_values("timestamp")
    wide = int(d.spread.gt(max_spread+1e-12).sum())
    d = d[d.spread.le(max_spread+1e-12)].reset_index(drop=True)
    stats = {"ticker": market["ticker"], "raw_quote_rows": len(raw), "invalid_quote_rows": invalid,
             "conflicting_timestamp_rows": conflicting_rows, "wide_spread_rows": wide,
             "usable_quote_rows": len(d), "max_spread": max_spread,
             "quote_start_utc": d.timestamp.min().isoformat() if len(d) else None,
             "quote_end_utc": d.timestamp.max().isoformat() if len(d) else None}
    return d, stats


def quote_intervals(quotes: pd.DataFrame, tape: pd.DataFrame, seconds: int = 300,
                    interior: bool = False) -> pd.DataFrame:
    """Consecutive observed quote endpoints with flow in [left,right)."""
    times = pd.DatetimeIndex(quotes.timestamp).as_unit("ns").asi8
    selected = times % (seconds*10**9) == 0
    d = quotes[selected].copy().reset_index(drop=True)
    out = pd.DataFrame({"left_time": d.timestamp.shift(), "right_time": d.timestamp,
                        "open_midpoint": d.midpoint.shift(), "close_midpoint": d.midpoint})
    keep = out.right_time.sub(out.left_time).eq(pd.Timedelta(seconds=seconds))
    if interior:
        keep &= out.open_midpoint.between(.3, .7) & out.close_midpoint.between(.3, .7)
    out = out[keep].reset_index(drop=True)
    trade_times = pd.DatetimeIndex(tape.timestamp).as_unit("ns").asi8
    prefix = np.r_[0.0, np.cumsum((tape.sign*tape.quantity).to_numpy(dtype=float))]
    left = np.searchsorted(trade_times, pd.DatetimeIndex(out.left_time).as_unit("ns").asi8, side="left")
    right = np.searchsorted(trade_times, pd.DatetimeIndex(out.right_time).as_unit("ns").asi8, side="left")
    out["signed_flow_contracts"] = prefix[right]-prefix[left]
    out["n_trades"] = right-left
    out["duration_seconds"] = seconds
    out["delta_logodds_midpoint"] = logit(out.close_midpoint.to_numpy())-logit(out.open_midpoint.to_numpy())
    return out


def residual_noise_fit(intervals: pd.DataFrame, beta: float, min_pairs: int = 50) -> dict:
    n = len(intervals)
    record = {"fit_status": "insufficient_quote_pairs", "quote_pairs": n, "beta": beta,
              "sigma_logodds_per_sqrt_second": None, "residual_drift_per_second": None,
              "raw_sigma_logodds_per_sqrt_second": None, "residual_flow_correlation": None,
              "fixed_impact_sigma_logodds_per_sqrt_second": None, "fixed_impact_residual_flow_correlation": None,
              "quote_flow_slope": None, "remaining_flow_slope_after_beta": None,
              "variance_decomposition_error": None,
              "residual_lag1_correlation": None, "zero_flow_pairs": int(intervals.signed_flow_contracts.eq(0).sum()),
              "p50_sigma_5min_linear_pp": None, "gaussian_excess_kurtosis": None}
    if n < min_pairs:
        return record
    dt = intervals.duration_seconds.to_numpy(dtype=float)
    if not np.all(dt == dt[0]) or dt[0] <= 0:
        raise ValueError("Noise fit requires fixed positive observed quote durations")
    flow = intervals.signed_flow_contracts.to_numpy(dtype=float)
    raw = intervals.delta_logodds_midpoint.to_numpy(dtype=float)
    residual = raw-beta*flow
    fc, yc = flow-flow.mean(), raw-raw.mean()
    sxx = float(fc@fc)
    slope = float(fc@yc/sxx) if sxx > 1e-24 else None
    remaining = slope-beta if slope is not None else 0.0
    centered = yc-(slope*fc if slope is not None else 0.0)
    k = 2 if slope is not None else 1
    sigma = float(np.sqrt((centered@centered)/(n-k)/dt[0]))
    drift = float((raw.mean()-(slope*flow.mean() if slope is not None else beta*flow.mean()))/dt[0])
    fixed_corr = float(np.corrcoef(residual, flow)[0, 1]) if np.std(residual)>1e-12 and np.std(flow)>1e-12 else None
    corr = float(np.corrcoef(centered, flow)[0, 1]) if np.std(centered)>1e-12 and np.std(flow)>1e-12 else None
    decomposition = abs(float(yc@yc)-(float(slope*slope*sxx) if slope is not None else 0.0)-float(centered@centered))
    # Pair only adjacent residual windows; do not correlate across missing bars.
    adjacent = intervals.left_time.iloc[1:].to_numpy() == intervals.right_time.iloc[:-1].to_numpy()
    now, previous = centered[1:][adjacent], centered[:-1][adjacent]
    lag1 = float(np.corrcoef(now, previous)[0, 1]) if len(now)>2 and np.std(now)>1e-12 and np.std(previous)>1e-12 else None
    record.update(fit_status="ok", sigma_logodds_per_sqrt_second=sigma, residual_drift_per_second=drift,
                  raw_sigma_logodds_per_sqrt_second=float(np.std(raw, ddof=1)/np.sqrt(dt[0])),
                  fixed_impact_sigma_logodds_per_sqrt_second=float(np.std(residual, ddof=1)/np.sqrt(dt[0])),
                  fixed_impact_residual_flow_correlation=fixed_corr, quote_flow_slope=slope,
                  remaining_flow_slope_after_beta=remaining, variance_decomposition_error=decomposition,
                  residual_flow_correlation=corr, residual_lag1_correlation=lag1,
                  p50_sigma_5min_linear_pp=25*sigma*np.sqrt(300),
                  gaussian_excess_kurtosis=float(pd.Series(centered).kurt()) if sigma>0 else None)
    return record


def median_bootstrap(d: pd.DataFrame, column: str, draws: int, rng) -> list[float]:
    clusters = [g[column].to_numpy(dtype=float) for _, g in d.groupby("event_cluster", sort=True)]
    values = [np.median(np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]))
              for _ in range(draws)]
    return [float(v) for v in np.quantile(values, [.025, .975])]


def calibrate(markets: pd.DataFrame, splits: pd.DataFrame, impact: dict, trade_loader, quote_loader,
              n_phases: int = 10, min_events: int = 5, min_pairs: int = 50,
              bootstrap_draws: int = 2000, seed: int = 42) -> tuple[dict, dict]:
    """Filter before loading raw data/statistics. Load no held-out tapes or quotes."""
    if not 1 <= n_phases <= 100 or min_events < 2 or min_pairs < 3 or bootstrap_draws < 20 or seed < 0:
        raise ValueError("Invalid phase count or calibration support settings")
    train_ids = sorted(splits.loc[splits.split.eq("train"), "ticker"].unique())
    selected = markets[markets.ticker.isin(train_ids)].copy().sort_values("ticker")
    if set(selected.ticker) != set(train_ids) or selected.ticker.duplicated().any():
        raise ValueError("Training metadata does not match event split")
    if train_ids != sorted(impact["training"]["tickers"]):
        raise ValueError("Dynamics and price-impact training events differ")
    if splits.groupby("event_cluster").split.nunique().gt(1).any():
        raise ValueError("An event crosses data splits")
    activity, phases, audits, noise, intervals, marks, templates = [], [], [], [], [], {}, {}
    tapes, quote_data = {}, {}
    for index, market in enumerate(selected.to_dict("records")):
        ticker = market["ticker"]
        tape = trade_loader(ticker).sort_values(["timestamp", "trade_id"], kind="stable").reset_index(drop=True)
        raw_quotes = quote_loader(ticker)
        quotes, audit = normalize_quotes(raw_quotes, market)
        quote_data[ticker] = quotes
        tapes[ticker] = tape
        record, phase_rows, phase = activity_profile(tape, market, n_phases)
        activity.append(record); phases.extend(phase_rows); audits.append(audit)
        key = f"event_{index:03d}"
        marks[f"{key}_quantity"] = tape.quantity.to_numpy(dtype=float)
        marks[f"{key}_sign"] = tape.sign.to_numpy(dtype=np.int8)
        marks[f"{key}_phase"] = phase.astype(np.int8)
        templates[ticker] = {"information_class": market["information_class"], "event_cluster": market["event_cluster"],
                             "mark_bank_key": key, "open_time": record["open_time"], "close_time": record["close_time"],
                             "lifetime_seconds": record["lifetime_seconds"], "arrival_rate_per_second": record["arrival_rate_per_second"],
                             "phases": phase_rows, "quote_support": audit}
    a, ph = pd.DataFrame(activity), pd.DataFrame(phases)
    profiles = {}
    all_noise = []
    for index, (group, price_profile) in enumerate(impact["groups"].items()):
        tickers = train_ids if group == "pooled" else sorted(a[a.information_class.eq(group)].ticker)
        profile = {"status": "insufficient_training_events", "training_contracts": len(tickers),
                   "event_tickers": tickers, "arrival": None, "volatility": {}}
        profiles[group] = profile
        if price_profile["status"] != "calibrated" or len(set(a[a.ticker.isin(tickers)].event_cluster)) < min_events:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        d = a[a.ticker.isin(tickers)]
        rate_ci = median_bootstrap(d, "arrival_rate_per_second", bootstrap_draws, rng)
        base = float(d.arrival_rate_per_second.median())
        profile.update(status="calibrated", arrival={"rate_per_second_mean": float(d.arrival_rate_per_second.mean()),
                      "rate_per_second_median": base, "pooled_exposure_rate_per_second": float(d.trades.sum()/d.lifetime_seconds.sum()),
                      "bootstrap95_median_rate": rate_ci, "rate_scales": {"low": rate_ci[0]/base if base else 0,
                      "base": 1.0, "high": rate_ci[1]/base if base else 1},
                      "contract_quantity_mean_median": float(d.quantity_mean.median()),
                      "contract_quantity_mean_equal_event": float(d.quantity_mean.mean()),
                      "trade_quantity_mean_pooled": float(d.quantity_total.sum()/d.trades.sum()),
                      "contract_quantity_median_median": float(d.quantity_median.median()),
                      "fano_5min_median": float(d.fano_5min.median())})
        group_noise = []
        for ticker in tickers:
            for variant, (seconds, interior) in NOISE_VARIANTS.items():
                pairs = quote_intervals(quote_data[ticker], tapes[ticker], seconds, interior)
                if variant == "midpoint_5min":
                    intervals.append(pairs.assign(ticker=ticker, information_class=group, split="train"))
                    fit = residual_noise_fit(pairs, price_profile["scenarios"]["base"], min_pairs)
                    if fit["fit_status"] == "ok" and "noise" not in templates[ticker]:
                        flow = pairs.signed_flow_contracts.to_numpy(dtype=float)
                        y = pairs.delta_logodds_midpoint.to_numpy(dtype=float)
                        residual = y-y.mean()-(fit["quote_flow_slope"] or 0.0)*(flow-flow.mean())
                        scale = float(np.std(residual))
                        key = f"{templates[ticker]['mark_bank_key']}_noise_standardized"
                        marks[key] = residual/scale if scale > 1e-12 else np.zeros(len(residual))
                        templates[ticker]["noise"] = {"standardized_residual_key": key,
                            "sigma_logodds_per_sqrt_second": fit["sigma_logodds_per_sqrt_second"],
                            "quote_pairs": len(pairs), "pair_start_utc": pairs.left_time.min().isoformat(),
                            "pair_end_utc": pairs.right_time.max().isoformat()}
                for scenario in SCENARIOS:
                    beta = float(price_profile["scenarios"][scenario])
                    row = {"ticker": ticker, "event_cluster": templates[ticker]["event_cluster"], "information_class": group,
                           "impact_scenario": scenario, "variant": variant, "split": "train",
                           **residual_noise_fit(pairs, beta, min_pairs)}
                    group_noise.append(row)
        nd = pd.DataFrame(group_noise)
        all_noise.extend(group_noise)
        for scenario in SCENARIOS:
            # The orthogonal noise sample is identical across beta scenarios;
            # use the same bootstrap draws so finite Monte Carlo variation does
            # not create artificial differences between their uncertainty bands.
            noise_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 100]))
            scoped = nd[nd.impact_scenario.eq(scenario)]
            variant_summaries = []
            for variant in NOISE_VARIANTS:
                v = scoped[scoped.variant.eq(variant) & scoped.fit_status.eq("ok")]
                row = {"variant": variant, "supported_events": int(v.event_cluster.nunique()),
                       "status": "insufficient_quote_support", "sigma_mean": None, "sigma_median": None, "bootstrap95_sigma": None}
                if v.event_cluster.nunique() >= min_events:
                    row.update(status="calibrated", sigma_mean=float(v.sigma_logodds_per_sqrt_second.mean()),
                               sigma_median=float(v.sigma_logodds_per_sqrt_second.median()),
                               bootstrap95_sigma=median_bootstrap(v, "sigma_logodds_per_sqrt_second", bootstrap_draws, noise_rng))
                variant_summaries.append(row)
            main = variant_summaries[0]
            volatility = {"status": main["status"], "sensitivities": variant_summaries, "scenarios": None}
            profile["volatility"][scenario] = volatility
            if main["status"] == "calibrated":
                bounds = [value for row in variant_summaries if row["status"] == "calibrated"
                          for value in [row["sigma_median"], *row["bootstrap95_sigma"]]]
                full = scoped[scoped.variant.eq("midpoint_5min") & scoped.fit_status.eq("ok")]
                volatility.update(scenarios={"low": min(bounds), "base": main["sigma_median"], "high": max(bounds)},
                                  supported_events=main["supported_events"], sigma_mean=main["sigma_mean"],
                                  sigma_median=main["sigma_median"],
                                  supported_tickers=sorted(full.ticker),
                                  strong_fixed_impact_residual_flow_correlations=int(full.fixed_impact_residual_flow_correlation.abs().gt(.3).sum()),
                                  residual_drift_per_second_median=float(full.residual_drift_per_second.median()))
    if not any(p["status"] == "calibrated" and p["volatility"].get("base", {}).get("status") == "calibrated" for p in profiles.values()):
        raise ValueError("No group has enough training arrival and quote support")
    config = {"schema_version": 1, "clock_units": "seconds", "count_window_seconds": 300,
              "sigma_units": "log_odds_per_sqrt_second", "n_lifetime_phases": n_phases,
              "arrival_model": "Phase-conditioned gamma-Poisson counts; one uniformly selected training-event template per episode",
              "marks_model": "Empirical paired contract quantity and taker sign; phase-specific with explicit all-event fallback",
              "noise_model": "Empirical standardized innovations (Gaussian alternative) from quote changes after removing their fitted linear flow association; "
                             "fixed-beta residual sigma/correlation and fitted intercept are diagnostics",
              "quote_observation": "End-of-period YES bid/ask midpoint, max spread 0.20; exact UTC endpoints, no forward fill",
              "training": {"tickers": train_ids, "contracts": len(selected), "trades": int(a.trades.sum()),
                           "min_events": min_events, "min_quote_pairs": min_pairs, "bootstrap_draws": bootstrap_draws, "seed": seed},
              "groups": profiles, "event_templates": templates,
              "limitations": ["Market trade-execution arrivals are not quote-conditioned maker fill intensities or parent-order counts.",
                 "Volatility is estimated in the saved trailing quote window, usually three days; application earlier in life is extrapolation.",
                 "Noise is linearly orthogonal to fitted training flow, not identified pure fundamental volatility or proof of independence.",
                 "Quote-based flow slopes can differ from the previously chosen transaction-based beta; that gap is retained as a diagnostic.",
                 "Residual-flow and serial correlations diagnose departures from independent Gaussian noise.",
                 "Sampling trades independently within a phase does not reproduce every order-flow autocorrelation or batch execution.",
                 "Low/high scenarios describe training-event sampling and sampling-grid sensitivity, not causal confidence bounds.",
                 "The full quoting, cash/inventory, settlement and policy-evaluation environment is subsequent work."]}
    artifacts = {"activity": a, "phases": ph, "quote_audit": pd.DataFrame(audits), "noise": pd.DataFrame(all_noise),
                 "noise_intervals": pd.concat(intervals, ignore_index=True), "marks": marks}
    return config, artifacts


def run(raw_input: Path = DEFAULT_INPUT, calibration_dir: Path = DEFAULT_OUTPUT,
        output_dir: Path | None = None, bootstrap_draws: int = 2000, seed: int = 42) -> dict:
    if bootstrap_draws < 20 or seed < 0:
        raise ValueError("Require twenty bootstrap draws and a nonnegative seed")
    output_dir = output_dir or calibration_dir/"simulator"/"market_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)
    impact_path = calibration_dir/"simulator"/"price_impact_config.json"
    impact = json.loads(impact_path.read_text())
    markets_path = raw_input/"markets"/"kalshi_selected_contracts.parquet"
    splits_path = calibration_dir/"event_splits.csv"
    hashes = {"markets/kalshi_selected_contracts.parquet": file_hash(markets_path)}
    loaded = {"trades": [], "quotes": []}
    def load_tape(ticker):
        path = raw_input/"trades"/f"{safe_filename(ticker)}.parquet"
        rel = str(path.relative_to(raw_input)); checksum = file_hash(path)
        if checksum != impact["training_tape_sha256"].get(rel):
            raise ValueError(f"Training tape changed: {ticker}")
        hashes[rel] = checksum; loaded["trades"].append(ticker)
        tape, stats = normalize_trades(pd.read_parquet(path), ticker)
        if stats["invalid_trade_rows"]:
            raise ValueError(f"Invalid training trades: {ticker}")
        return tape
    def load_quotes(ticker):
        path = raw_input/"candlesticks"/f"{safe_filename(ticker)}.parquet"
        hashes[str(path.relative_to(raw_input))] = file_hash(path); loaded["quotes"].append(ticker)
        return pd.read_parquet(path)
    config, artifacts = calibrate(pd.read_parquet(markets_path), pd.read_csv(splits_path), impact,
                                   load_tape, load_quotes, bootstrap_draws=bootstrap_draws, seed=seed)
    for name in ["activity", "phases", "quote_audit", "noise"]:
        artifacts[name].to_csv(output_dir/f"training_{name}.csv", index=False)
    artifacts["noise_intervals"].to_parquet(output_dir/"training_noise_intervals.parquet", index=False)
    np.savez_compressed(output_dir/"trade_mark_bank.npz", **artifacts["marks"])
    config.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), mark_bank_file="trade_mark_bank.npz",
                  impact_config_file=os.path.relpath(impact_path.resolve(), output_dir.resolve()), raw_input_sha256=hashes,
                  upstream_sha256={"price_impact_config.json": file_hash(impact_path), "event_splits.csv": file_hash(splits_path)},
                  source_sha256={str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
                                [Path(__file__), PROJECT_ROOT/"data_pipeline"/"calibration_quality.py",
                                 PROJECT_ROOT/"simulator"/"market_dynamics.py", PROJECT_ROOT/"simulator"/"price_impact.py"]})
    config["mark_bank_sha256"] = file_hash(output_dir/"trade_mark_bank.npz")
    (output_dir/"market_dynamics_config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    summary_rows = []
    for group, p in config["groups"].items():
        if p["status"] != "calibrated":
            continue
        for impact_scenario, v in p["volatility"].items():
            if v["status"] != "calibrated":
                continue
            summary_rows.append({"information_class": group, "impact_scenario": impact_scenario,
                "training_contracts": p["training_contracts"], "volatility_supported_contracts": v["supported_events"],
                "trade_rate_per_minute_median": 60*p["arrival"]["rate_per_second_median"],
                "trade_rate_per_minute_mean": 60*p["arrival"]["rate_per_second_mean"],
                "trade_rate_per_minute_pooled_exposure": 60*p["arrival"]["pooled_exposure_rate_per_second"],
                "median_contract_trade_size": p["arrival"]["contract_quantity_median_median"],
                "mean_trade_size_equal_event": p["arrival"]["contract_quantity_mean_equal_event"],
                "mean_trade_size_pooled": p["arrival"]["trade_quantity_mean_pooled"],
                "sigma_low": v["scenarios"]["low"], "sigma_base": v["scenarios"]["base"], "sigma_high": v["scenarios"]["high"],
                "p50_sigma_5min_linear_pp": 25*v["scenarios"]["base"]*np.sqrt(300),
                "strong_fixed_impact_residual_flow_correlations": v["strong_fixed_impact_residual_flow_correlations"]})
    pd.DataFrame(summary_rows).to_csv(output_dir/"market_dynamics_summary.csv", index=False)
    report = {"passed": True, "training_contracts": config["training"]["contracts"], "training_trades": config["training"]["trades"],
              "training_quote_files": len(loaded["quotes"]), "heldout_raw_files_loaded": 0,
              "phase_trade_reconciliation": int(artifacts["phases"].trades.sum()),
              "noise_fit_rows": len(artifacts["noise"]), "successful_noise_fits": int(artifacts["noise"].fit_status.eq("ok").sum()),
              "base_group_summary": [r for r in summary_rows if r["impact_scenario"] == "base"]}
    if set(loaded["trades"]) != set(config["training"]["tickers"]) or set(loaded["quotes"]) != set(config["training"]["tickers"]):
        raise AssertionError("Unexpected raw-file access")
    if report["phase_trade_reconciliation"] != report["training_trades"]:
        raise AssertionError("Trade phase reconciliation failed")
    (output_dir/"calibration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.raw_input, args.calibration_dir, args.output_dir, args.bootstrap_draws, args.seed), indent=2))
