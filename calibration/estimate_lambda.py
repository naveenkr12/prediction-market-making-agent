"""Estimate market-level Kyle lambda from observed five-minute signed flow.

Point estimates follow Bartlett & O'Hara (2026), equation (1). HC1 errors,
calendar-aware Newey-West errors, BH false-discovery adjustments, influence
diagnostics, and event splits are additional analysis safeguards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from importlib.metadata import version
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests

from build_intervals import build_dataset, file_hash, DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT

VARIANTS = ["full", "interior_endpoints_30_70", "early_80pct_life", "unambiguous_endpoints"]
RESPONSES = {"logodds": "delta_logodds", "cents": "delta_price_cents"}
Z95 = float(norm.ppf(0.975))


def calendar_hac_covariance(x: np.ndarray, residuals: np.ndarray,
                            bucket_starts: pd.Series, lags: int = 6) -> np.ndarray:
    """Bartlett Newey-West covariance at actual five-minute calendar lags.

    statsmodels' standard cov_hac assumes consecutive equally spaced rows.
    Active trade buckets have gaps, so pair scores only when their timestamps
    differ by exactly k*5 minutes. Missing time slots contribute zero scores,
    never synthetic observations to the regression. Small-sample factor n/(n-k).
    """
    if lags < 0:
        raise ValueError("HAC lags must be non-negative")
    n, k = x.shape
    if n <= k:
        raise ValueError("Not enough residual degrees of freedom")
    # pandas may store datetime values in microseconds; explicitly convert to ns.
    times = pd.DatetimeIndex(pd.to_datetime(bucket_starts, utc=True)).as_unit("ns").asi8 // (300*10**9)
    if len(times) != n or np.any(np.diff(times) <= 0):
        raise ValueError("HAC rows require unique, chronological five-minute buckets")
    scores = x * np.asarray(residuals)[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags+1):
        target = times-lag
        idx = np.searchsorted(times, target)
        safe_idx = np.minimum(idx, n-1)
        match = (idx < n) & (times[safe_idx] == target)
        cross = scores[match].T @ scores[idx[match]]
        weight = 1-lag/(lags+1)
        meat += weight*(cross+cross.T)
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ meat @ bread * n/(n-k)
    return (covariance+covariance.T)/2


def interval_subset(buckets: pd.DataFrame, variant: str, response: str) -> pd.DataFrame:
    if variant not in VARIANTS or response not in RESPONSES:
        raise ValueError("Unknown response or sensitivity variant")
    mask = pd.Series(True, index=buckets.index)
    if response == "logodds":
        mask &= buckets.valid_logodds
    if variant != "full":
        mask &= buckets[variant]
    mask &= np.isfinite(buckets.signed_flow_contracts) & np.isfinite(buckets[RESPONSES[response]])
    return buckets[mask].sort_values("bucket_start").reset_index(drop=True)


def fit_lambda(buckets: pd.DataFrame, variant: str = "full", response: str = "logodds",
               min_intervals: int = 50, hac_lags: int = 6) -> dict:
    d = interval_subset(buckets, variant, response)
    record = {"variant": variant, "response": response,
              "lambda_units": "log_odds_per_net_contract" if response == "logodds" else "cents_per_net_contract",
              "n_intervals": len(d), "fit_status": "insufficient_intervals", "lambda": None,
              "intercept": None, "lambda_per_1000_contracts": None, "r_squared": None,
              "se_hc1": None, "ci95_low_hc1": None, "ci95_high_hc1": None, "p_value_hc1": None,
              "se_calendar_hac": None, "ci95_low_calendar_hac": None, "ci95_high_calendar_hac": None,
              "p_value_calendar_hac": None, "hac_lags_5min": hac_lags,
              "single_trade_interval_share": float(d.single_trade_bucket.mean()) if len(d) else None,
              "ambiguous_endpoint_share": float((~d.unambiguous_endpoints).mean()) if len(d) else None,
              "nonzero_price_change_intervals": int(d[RESPONSES[response]].abs().gt(1e-12).sum()),
              "flow_std_contracts": None, "max_leverage": None, "max_cooks_distance": None,
              "max_abs_flow_bucket_share": None, "lambda_drop_max_leverage": None,
              "leave_one_out_relative_change": None, "diagnostic_flags": ""}
    if len(d) < min_intervals:
        return record
    flow = d.signed_flow_contracts.to_numpy(dtype=float)
    y = d[RESPONSES[response]].to_numpy(dtype=float)
    flow_mean, scale = float(flow.mean()), float(flow.std())
    record["flow_std_contracts"] = scale
    if scale <= 1e-12:
        record["fit_status"] = "constant_signed_flow"
        return record
    if float(y.std()) <= 1e-12:
        record["fit_status"] = "constant_price_change"
        return record
    # Centering/scaling is numerical only. Convert the slope back to original
    # contract units; no dollar-volume weighting or quantity truncation is used.
    x = np.column_stack([np.ones(len(d)), (flow-flow_mean)/scale])
    result = OLS(y, x).fit()
    hc1 = result.get_robustcov_results(cov_type="HC1", use_t=False)
    hac_cov = calendar_hac_covariance(x, result.resid, d.bucket_start, hac_lags)
    slope = float(result.params[1]/scale)
    se_hc1 = float(hc1.bse[1]/scale)
    se_hac = float(np.sqrt(max(hac_cov[1, 1], 0))/scale)
    record.update(fit_status="ok", **{"lambda": slope},
                  intercept=float(result.params[0]-slope*flow_mean),
                  lambda_per_1000_contracts=1000*slope, r_squared=float(result.rsquared))
    for tag, se in [("hc1", se_hc1), ("calendar_hac", se_hac)]:
        record[f"se_{tag}"] = se
        record[f"ci95_low_{tag}"] = slope-Z95*se
        record[f"ci95_high_{tag}"] = slope+Z95*se
        record[f"p_value_{tag}"] = float(2*norm.sf(abs(slope/se))) if se > 0 else None
    bread = np.linalg.inv(x.T @ x)
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    imax = int(np.argmax(leverage))
    record["max_leverage"] = float(leverage[imax])
    mse = float(result.mse_resid)
    if mse > 0:
        cooks = result.resid**2/(2*mse) * leverage / np.maximum((1-leverage)**2, 1e-24)
        record["max_cooks_distance"] = float(cooks.max())
    denominator = float(np.abs(flow).sum())
    record["max_abs_flow_bucket_share"] = float(np.abs(flow).max()/denominator) if denominator else None
    keep = np.arange(len(d)) != imax
    if np.std(flow[keep]) > 1e-12:
        reduced = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        dropped = float(reduced[1]/scale)
        record["lambda_drop_max_leverage"] = dropped
        record["leave_one_out_relative_change"] = abs(dropped-slope)/max(abs(slope), 1e-12)
    flags = []
    if len(d) < 100:
        flags.append("fewer_than_100_intervals")
    if record["single_trade_interval_share"] > 0.5:
        flags.append("majority_single_trade_buckets")
    if record["ambiguous_endpoint_share"] > 0:
        flags.append("timestamp_ties_with_ambiguous_endpoint_prices")
    if record["max_leverage"] > 0.2:
        flags.append("high_leverage_bucket")
    if record["leave_one_out_relative_change"] is not None and record["leave_one_out_relative_change"] > 0.5:
        flags.append("large_leave_one_out_change")
    if record["ci95_low_calendar_hac"] <= 0 <= record["ci95_high_calendar_hac"]:
        flags.append("calendar_hac_ci_includes_zero")
    if slope < 0:
        flags.append("negative_lambda")
    record["diagnostic_flags"] = "|".join(flags)
    if response == "logodds":
        record["illustrative_p50_move_per_100_contracts_pp"] = float(100*(expit(100*slope)-0.5))
    return record


def apply_fdr(estimates: pd.DataFrame) -> pd.DataFrame:
    """Separate testing families by response/variant; train adjustment uses train only."""
    d = estimates.copy()
    d["q_value_calendar_hac_all"] = np.nan
    d["q_value_calendar_hac_train"] = np.nan
    for _, rows in d.groupby(["variant", "response"]):
        valid = rows[rows.fit_status.eq("ok") & rows.p_value_calendar_hac.notna()]
        if len(valid):
            d.loc[valid.index, "q_value_calendar_hac_all"] = multipletests(valid.p_value_calendar_hac, method="fdr_bh")[1]
        train = valid[valid.split.eq("train")]
        if len(train):
            d.loc[train.index, "q_value_calendar_hac_train"] = multipletests(train.p_value_calendar_hac, method="fdr_bh")[1]
    return d


def group_summaries(estimates: pd.DataFrame, bootstrap_draws: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Equal-weight contract summaries, with whole-event-cluster bootstrap intervals.

    Bootstrap CIs describe cross-event sampling variability in estimated slopes.
    They do not add each slope's estimation uncertainty or correct selection bias.
    No CI or group-comparison inference is made from a single event.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for scope in ["all_exploratory", "train_only"]:
        scoped = estimates if scope == "all_exploratory" else estimates[estimates.split.eq("train")]
        for (variant, response, group), attempted in scoped.groupby(["variant", "response", "information_class"], sort=True):
            d = attempted[attempted.fit_status.eq("ok")]
            values = d["lambda"].to_numpy(dtype=float)
            record = {"scope": scope, "variant": variant, "response": response, "information_class": group,
                      "attempted_contracts": len(attempted), "estimated_contracts": len(d),
                      "event_clusters": int(d.event_cluster.nunique()),
                      "lambda_units": attempted.lambda_units.iloc[0], "lambda_mean": None,
                      "lambda_median": None, "lambda_p25": None, "lambda_p75": None,
                      "lambda_min": None, "lambda_max": None, "positive_lambda_contracts": 0,
                      "positive_calendar_hac_p05": 0, "positive_calendar_hac_fdr05": 0,
                      "mean_ci95_low_event_bootstrap": None, "mean_ci95_high_event_bootstrap": None,
                      "median_ci95_low_event_bootstrap": None, "median_ci95_high_event_bootstrap": None}
            if len(d):
                q_col = "q_value_calendar_hac_train" if scope == "train_only" else "q_value_calendar_hac_all"
                record.update(lambda_mean=float(values.mean()), lambda_median=float(np.median(values)),
                              lambda_p25=float(np.quantile(values, 0.25)), lambda_p75=float(np.quantile(values, 0.75)),
                              lambda_min=float(values.min()), lambda_max=float(values.max()),
                              positive_lambda_contracts=int(d["lambda"].gt(0).sum()),
                              positive_calendar_hac_p05=int((d["lambda"].gt(0) & d.p_value_calendar_hac.lt(0.05)).sum()),
                              positive_calendar_hac_fdr05=int((d["lambda"].gt(0) & d[q_col].lt(0.05)).sum()))
            clusters = [g["lambda"].to_numpy(dtype=float) for _, g in d.groupby("event_cluster")]
            if len(clusters) >= 2 and bootstrap_draws:
                means, medians = [], []
                for _ in range(bootstrap_draws):
                    sampled = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
                    means.append(float(sampled.mean()))
                    medians.append(float(np.median(sampled)))
                record.update(mean_ci95_low_event_bootstrap=float(np.quantile(means, 0.025)),
                              mean_ci95_high_event_bootstrap=float(np.quantile(means, 0.975)),
                              median_ci95_low_event_bootstrap=float(np.quantile(medians, 0.025)),
                              median_ci95_high_event_bootstrap=float(np.quantile(medians, 0.975)))
            rows.append(record)
    return pd.DataFrame(rows)


def make_figure(primary: pd.DataFrame, output_dir: Path, manifest: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/capstone-lambda-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = primary[primary.fit_status.eq("ok")].sort_values(["information_class", "lambda"])
    if d.empty:
        return
    colors = {"broad_based": "#225ea8", "sports": "#666666", "single_name": "#b35806", "needs_review": "#777777"}
    fig, ax = plt.subplots(figsize=(12, max(7, len(d)*0.24)))
    for i, (_, row) in enumerate(d.iterrows()):
        slope, low, high = (1000*row[col] for col in ["lambda", "ci95_low_calendar_hac", "ci95_high_calendar_hac"])
        ax.errorbar(slope, i, xerr=np.array([[max(slope-low, 0)], [max(high-slope, 0)]]),
                    fmt="o", markersize=3, linewidth=0.8, color=colors.get(row.information_class, "#777777"))
    ax.set_yticks(np.arange(len(d)), d.ticker, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Kyle lambda: log-odds per 1,000 net contracts (symmetric log scale)")
    ax.set_ylabel("Contract ticker")
    ax.set_title("Five-minute Kyle lambda: full-life OLS and 95% calendar HAC intervals", loc="left", fontsize=12)
    from matplotlib.lines import Line2D
    labels = sorted(d.information_class.unique())
    fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=colors.get(g, "#777777"), label=g)
                        for g in labels], loc="lower center", bbox_to_anchor=(0.62, 0.025),
               ncol=len(labels), frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.012, "Source: saved Kalshi Paper 2 sample; observed trade buckets only. "
             f"Trades {manifest['trade_start_utc'][:10]} to {manifest['trade_end_utc'][:10]} UTC. "
             "HAC: Bartlett weights, six actual five-minute calendar lags.", fontsize=7)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    (output_dir / "figures").mkdir(exist_ok=True)
    fig.savefig(output_dir / "figures" / "lambda_logodds_by_contract.png", dpi=170)
    plt.close(fig)


def run(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT,
        min_intervals: int = 50, hac_lags: int = 6, bootstrap_draws: int = 2000, seed: int = 42) -> dict:
    if min_intervals < 3 or hac_lags < 0 or bootstrap_draws < 0:
        raise ValueError("Invalid minimum intervals, HAC lags, or bootstrap count")
    buckets, manifest = build_dataset(input_dir, output_dir)
    records = []
    for ticker, market_buckets in buckets.groupby("ticker", sort=True):
        metadata = {key: market_buckets[key].iloc[0] for key in
                    ["event_ticker", "event_cluster", "series_ticker", "category", "information_class", "split"]}
        for variant in VARIANTS:
            for response in RESPONSES:
                records.append({"ticker": ticker, **metadata,
                                **fit_lambda(market_buckets, variant, response, min_intervals, hac_lags)})
    estimates = apply_fdr(pd.DataFrame(records))
    estimates.to_csv(output_dir / "lambda_estimates.csv", index=False)
    estimates.to_parquet(output_dir / "lambda_estimates.parquet", index=False)
    primary = estimates[estimates.variant.eq("full") & estimates.response.eq("logodds")].copy()
    primary.to_csv(output_dir / "lambda_by_contract.csv", index=False)
    groups = group_summaries(estimates, bootstrap_draws, seed)
    groups.to_csv(output_dir / "lambda_group_summary.csv", index=False)
    make_figure(primary, output_dir, manifest)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               **{k: v for k, v in manifest.items() if k != "input_sha256"},
               "min_intervals_per_fit": min_intervals, "hac_lags_5min": hac_lags,
               "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
               "runtime_versions": {"python": sys.version.split()[0], **{name: version(name) for name in
                    ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pyarrow"]}},
               "source_sha256": {**manifest["source_sha256"],
                    str(Path(__file__).relative_to(PROJECT_ROOT)): file_hash(Path(__file__))},
               "primary_fit_status": {k: int(v) for k, v in primary.fit_status.value_counts().items()},
               "full_logodds_positive_slopes": int(primary["lambda"].gt(0).sum()),
               "full_logodds_positive_hac_p05": int((primary["lambda"].gt(0) & primary.p_value_calendar_hac.lt(0.05)).sum()),
               "full_logodds_positive_hac_fdr05": int((primary["lambda"].gt(0) & primary.q_value_calendar_hac_all.lt(0.05)).sum()),
               "high_leverage_primary_fits": int(primary.max_leverage.gt(0.2).sum()),
               "primary_fits_large_leave_one_out_change": int(primary.leave_one_out_relative_change.gt(0.5).sum()),
               "primary_fits_hac_ci_including_zero": int((primary.ci95_low_calendar_hac.le(0) & primary.ci95_high_calendar_hac.ge(0)).sum()),
               "sensitivity_fit_counts": {variant: int(estimates[(estimates.variant==variant) &
                    estimates.response.eq("logodds")].fit_status.eq("ok").sum()) for variant in VARIANTS},
               "limitations": ["Contemporaneous transaction-price/flow association does not identify private-information frequency.",
                    "Only one single-name market: no single-name versus broad-based replication or group inference.",
                    "Convenience sample; selected by activity/volume, not exchange-wide random sampling.",
                    "Single-trade buckets have zero within-bucket price changes and are retained as in the specified estimator.",
                    "Exact timestamp ties can obscure endpoint ordering; an unambiguous-endpoint sensitivity is reported.",
                    "All-event estimates are exploratory; later simulator calibration must use train-only summaries.",
                    "Event bootstrap CIs do not propagate within-market estimation uncertainty or remove cross-event dependence."]}
    (output_dir / "lambda_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-intervals", type=int, default=50)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output_dir, args.min_intervals,
                         args.hac_lags, args.bootstrap_draws, args.seed), indent=2))
