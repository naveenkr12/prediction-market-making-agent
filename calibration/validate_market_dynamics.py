"""Independently reconcile training activity, saved marks and quote-noise fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_intervals import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT, file_hash, normalize_trades, safe_filename
from calibrate_market_dynamics import NOISE_VARIANTS, normalize_quotes, quote_intervals


def close(a, b, label):
    if not np.allclose(a, b, rtol=1e-8, atol=1e-11, equal_nan=True):
        raise AssertionError(label)


def validate(raw_input: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT/"simulator"/"market_dynamics") -> dict:
    c = json.loads((output_dir/"market_dynamics_config.json").read_text())
    for path, expected in c["raw_input_sha256"].items():
        assert file_hash(raw_input/path) == expected, path
    for path, expected in c["source_sha256"].items():
        assert file_hash(PROJECT_ROOT/path) == expected, path
    impact_path = (output_dir/c["impact_config_file"]).resolve()
    assert file_hash(impact_path) == c["upstream_sha256"]["price_impact_config.json"]
    assert file_hash(impact_path.parent.parent/"event_splits.csv") == c["upstream_sha256"]["event_splits.csv"]
    assert file_hash(output_dir/c["mark_bank_file"]) == c["mark_bank_sha256"]
    a, noise = pd.read_csv(output_dir/"training_activity.csv"), pd.read_csv(output_dir/"training_noise.csv")
    assert sorted(a.ticker) == c["training"]["tickers"] and a.split.eq("train").all()
    assert noise.split.eq("train").all()
    metadata = pd.read_parquet(raw_input/"markets"/"kalshi_selected_contracts.parquet").set_index("ticker")
    tapes, quotes, count_checks = {}, {}, 0
    bank = np.load(output_dir/c["mark_bank_file"], allow_pickle=False)
    for row in a.to_dict("records"):
        ticker = row["ticker"]
        t, _ = normalize_trades(pd.read_parquet(raw_input/"trades"/f"{safe_filename(ticker)}.parquet"), ticker)
        tapes[ticker] = t
        market = metadata.loc[ticker].to_dict(); market["ticker"] = ticker
        quotes[ticker], _ = normalize_quotes(pd.read_parquet(raw_input/"candlesticks"/f"{safe_filename(ticker)}.parquet"), market)
        opened, closed = pd.to_datetime(market["open_time"], utc=True), pd.to_datetime(market["close_time"], utc=True)
        duration = (closed-opened).total_seconds()
        close(row["arrival_rate_per_second"], len(t)/duration, "Full lifetime trade rate")
        close(row["quantity_mean"], t.quantity.mean(), "Quantity mean")
        close(row["quantity_median"], t.quantity.median(), "Quantity median")
        template = c["event_templates"][ticker]; key = template["mark_bank_key"]
        close(bank[f"{key}_quantity"], t.quantity, "Empirical quantities changed")
        close(bank[f"{key}_sign"], t.sign, "Empirical taker signs changed")
        phases = np.minimum(((t.timestamp-opened).dt.total_seconds().to_numpy()/duration*c["n_lifetime_phases"]).astype(int), c["n_lifetime_phases"]-1)
        close(bank[f"{key}_phase"], phases, "Phase labels changed")
        for p in template["phases"]:
            lo = opened+pd.Timedelta(seconds=p["phase"]*duration/c["n_lifetime_phases"])
            hi = opened+pd.Timedelta(seconds=(p["phase"]+1)*duration/c["n_lifetime_phases"])
            d = t[phases == p["phase"]]
            close(p["arrival_rate_per_second"], len(d)/(duration/c["n_lifetime_phases"]), "Phase exposure")
            starts = pd.date_range(lo.ceil("5min"), hi.floor("5min"), freq="5min", inclusive="left")
            observed = d.groupby(d.timestamp.dt.floor("5min")).size().reindex(starts, fill_value=0)
            assert len(observed) == p["full_5min_intervals"]
            close(p["full_interval_count_mean"], observed.mean() if len(observed) else 0, "Implicit zero counts")
            if len(observed)>1:
                close(p["full_interval_count_variance"], observed.var(ddof=1), "Count variance")
            count_checks += 1
        if "noise" in template:
            z = bank[template["noise"]["standardized_residual_key"]]
            close(z.mean(), 0, "Innovation centering")
            close(z.var(), 1 if z.std()>1e-12 else 0, "Innovation normalization")
    checked, maximum_correlation = 0, 0.0
    cache = {}
    for row in noise.to_dict("records"):
        key = (row["ticker"], row["variant"])
        if key not in cache:
            seconds, interior = NOISE_VARIANTS[row["variant"]]
            pairs = quote_intervals(quotes[row["ticker"]], tapes[row["ticker"]], seconds, interior)
            tape = tapes[row["ticker"]]
            manual = (tape.quantity*tape.sign).groupby(tape.timestamp.dt.floor(f"{seconds}s")).sum()
            close(pairs.signed_flow_contracts, manual.reindex(pd.DatetimeIndex(pairs.left_time), fill_value=0), "Quote interval signed flow")
            cache[key] = pairs
        d = cache[key]
        assert len(d) == row["quote_pairs"]
        if row["fit_status"] != "ok":
            assert pd.isna(row["sigma_logodds_per_sqrt_second"])
            continue
        flow, y = d.signed_flow_contracts.to_numpy(), d.delta_logodds_midpoint.to_numpy()
        spread = np.std(flow)
        x = np.column_stack([np.ones(len(d)), (flow-flow.mean())/spread]) if spread>1e-12 else np.ones((len(d), 1))
        parameters = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y-x@parameters
        sigma = np.sqrt((residual@residual)/(len(d)-x.shape[1])/d.duration_seconds.iloc[0])
        close(row["sigma_logodds_per_sqrt_second"], sigma, "OLS residual sigma and time units")
        fixed = y-row["beta"]*flow
        close(row["fixed_impact_sigma_logodds_per_sqrt_second"], np.std(fixed, ddof=1)/np.sqrt(d.duration_seconds.iloc[0]), "Fixed-impact residual diagnostic")
        if len(parameters)>1:
            close(row["quote_flow_slope"], parameters[1]/spread, "Quote flow slope")
        if np.isfinite(row["residual_flow_correlation"]):
            maximum_correlation = max(maximum_correlation, abs(row["residual_flow_correlation"]))
        checked += 1
    bank.close()
    assert maximum_correlation < 1e-8
    report = {"passed": True, "training_contracts": len(a), "training_trades": int(a.trades.sum()),
              "phase_exposure_and_count_checks": count_checks, "independent_noise_fit_checks": checked,
              "max_orthogonal_residual_flow_correlation": maximum_correlation,
              "raw_file_hash_checks": len(c["raw_input_sha256"]), "source_hash_checks": len(c["source_sha256"]),
              "checks": ["Full listing exposure and phase rates", "Empty five-minute bins included in count moments",
                         "Paired fractional size/sign/phase bank matches source tapes", "Observed quote intervals and signed flow",
                         "Independent OLS residual volatility in seconds", "Centered normalized innovation samples",
                         "Raw/source/impact/split/mark-bank hashes"]}
    (output_dir/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT/"simulator"/"market_dynamics")
    args = parser.parse_args()
    print(json.dumps(validate(args.raw_input, args.output_dir), indent=2))
