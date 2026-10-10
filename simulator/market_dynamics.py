"""Load training-calibrated exchange trade activity and background innovations.

Counts operate on independent five-minute windows. They are exchange trade
executions; a later quote engine must decide which trades fill the maker.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .price_impact import ImpactParameters, ImpactPriceProcess

DEFAULT_CONFIG = (Path(__file__).resolve().parents[1]/"data_pipeline"/"output"/"paper2"/
                  "calibration"/"simulator"/"market_dynamics"/"market_dynamics_config.json")


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sample_counts(rate_per_second: float, alpha: float, rng, draws: int = 1, duration_seconds: float = 300) -> np.ndarray:
    """Gamma-Poisson: E[N]=300*rate; Var[N]=mean+alpha*mean^2."""
    if (not np.isfinite(rate_per_second) or rate_per_second < 0 or not np.isfinite(alpha) or alpha < 0 or draws < 1 or
            not np.isfinite(duration_seconds) or not 0 < duration_seconds <= 300):
        raise ValueError("Invalid arrival rate, dispersion or draw count")
    mu = duration_seconds*rate_per_second
    if mu == 0:
        return np.zeros(draws, dtype=np.int64)
    multipliers = rng.gamma(shape=1/alpha, scale=alpha, size=draws) if alpha > 1e-10 else np.ones(draws)
    return rng.poisson(mu*multipliers)


class MarketDynamics:
    """One empirical training-event template per episode; all time is seconds.

    `sample_window` produces all market executions. Noise/impact update the
    reference in `advance_window`; own fills must not add the same flow again.
    """
    def __init__(self, group: str, rng, ticker: str | None = None,
                 impact_scenario: str = "base", arrival_scenario: str = "base", noise_scenario: str = "base",
                 noise_distribution: str = "empirical", config_path: Path = DEFAULT_CONFIG):
        path = Path(config_path)
        config = json.loads(path.read_text())
        if config.get("clock_units") != "seconds" or config.get("count_window_seconds") != 300 or config.get("sigma_units") != "log_odds_per_sqrt_second":
            raise ValueError("Unsupported dynamics time or volatility units")
        profile = config["groups"].get(group)
        if profile is None or profile["status"] != "calibrated":
            raise ValueError(f"No supported market dynamics for {group}")
        vol = profile["volatility"].get(impact_scenario)
        if vol is None or vol["status"] != "calibrated":
            raise ValueError("No supported residual volatility for selected impact scenario")
        if arrival_scenario not in profile["arrival"]["rate_scales"] or noise_scenario not in vol["scenarios"]:
            raise ValueError("Unknown arrival/noise scenario")
        if noise_distribution not in {"empirical", "gaussian"}:
            raise ValueError("Unknown innovation distribution")
        chosen = ticker or str(rng.choice(profile["event_tickers"]))
        if chosen not in profile["event_tickers"]:
            raise ValueError("Event template does not belong to the requested training group")
        self.template = config["event_templates"][chosen]
        self.ticker, self.group = chosen, group
        self.rate_scale = float(profile["arrival"]["rate_scales"][arrival_scenario])
        impact_path = (path.parent/config["impact_config_file"]).resolve()
        if file_hash(impact_path) != config["upstream_sha256"]["price_impact_config.json"]:
            raise ValueError("Price-impact config changed; recalibrate market dynamics")
        self.impact = ImpactParameters.load(group, impact_scenario, impact_path)
        bank_path = path.parent/config["mark_bank_file"]
        if file_hash(bank_path) != config["mark_bank_sha256"]:
            raise ValueError("Empirical mark bank changed")
        with np.load(bank_path, allow_pickle=False) as bank:
            key = self.template["mark_bank_key"]
            self.quantities = bank[f"{key}_quantity"].copy()
            self.signs = bank[f"{key}_sign"].copy()
            self.phases = bank[f"{key}_phase"].copy()
            # Missing quote support is explicitly replaced with another
            # supported training event in the same group, not held-out quotes.
            noise_ticker = chosen if "noise" in self.template else str(rng.choice(vol["supported_tickers"]))
            self.noise_template = config["event_templates"][noise_ticker]
            self.noise_samples = bank[self.noise_template["noise"]["standardized_residual_key"]].copy()
        self.noise_template_ticker = noise_ticker
        base_sigma = float(vol["scenarios"]["base"])
        sigma_scale = float(vol["scenarios"][noise_scenario])/base_sigma if base_sigma else 1.0
        self.sigma = float(self.noise_template["noise"]["sigma_logodds_per_sqrt_second"])*sigma_scale if base_sigma else float(vol["scenarios"][noise_scenario])
        self.degenerate_empirical_noise_fallback = bool(self.sigma > 0 and np.std(self.noise_samples) <= 1e-12)
        self.noise_distribution = noise_distribution
        self.n_phases = int(config["n_lifetime_phases"])
        if not np.isfinite(self.quantities).all() or not (self.quantities>0).all() or not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("Invalid empirical trade marks")

    def sample_window(self, life_fraction: float, rng) -> dict:
        if not np.isfinite(life_fraction) or not 0 <= life_fraction < 1:
            raise ValueError("Life fraction must be in [0,1)")
        phase = min(int(life_fraction*self.n_phases), self.n_phases-1)
        record = self.template["phases"][phase]
        duration = min(300.0, (1-life_fraction)*self.template["lifetime_seconds"])
        n = int(sample_counts(float(record["arrival_rate_per_second"])*self.rate_scale,
                              float(record["gamma_poisson_alpha"]), rng, duration_seconds=duration)[0])
        if n > 2_000_000:
            raise RuntimeError("Sampled execution count exceeds this driver's allocation limit; no count was clipped")
        eligible = np.flatnonzero(self.phases == phase)
        fallback = len(eligible) == 0
        if fallback:
            eligible = np.arange(len(self.quantities))
        ids = rng.choice(eligible, size=n, replace=True)
        quantities, signs = self.quantities[ids], self.signs[ids]
        return {"duration_seconds": duration, "phase": phase, "n_trades": n,
                "event_times_seconds": np.sort(rng.uniform(0, duration, size=n)),
                "quantities": quantities, "signs": signs,
                "signed_flow_contracts": float(np.dot(quantities, signs)),
                "total_contracts": float(quantities.sum()), "phase_mark_fallback": fallback}

    def sample_innovation(self, rng, draws: int = 1, duration_seconds: float = 300) -> np.ndarray:
        if draws < 1 or not np.isfinite(duration_seconds) or not 0 < duration_seconds <= 300:
            raise ValueError("Require a positive draw count")
        z = rng.normal(size=draws) if self.noise_distribution == "gaussian" or self.degenerate_empirical_noise_fallback else rng.choice(self.noise_samples, size=draws)
        return self.sigma*np.sqrt(duration_seconds)*z

    def advance_window(self, price: ImpactPriceProcess, life_fraction: float, rng) -> dict:
        if price.parameters != self.impact:
            raise ValueError("Price process and dynamics use different impact parameters")
        window = self.sample_window(life_fraction, rng)
        innovation = float(self.sample_innovation(rng, duration_seconds=window["duration_seconds"])[0])
        update = price.advance(window["signed_flow_contracts"], innovation)
        # Quote support is a trailing window. An earlier-life use is a labelled
        # scenario extrapolation; it is not evidence of measured early-life noise.
        import pandas as pd
        ts = pd.to_datetime(self.template["open_time"], utc=True)+pd.Timedelta(seconds=life_fraction*self.template["lifetime_seconds"])
        support = self.noise_template["noise"]
        extrapolated = (self.noise_template_ticker != self.ticker or
                        ts < pd.to_datetime(support["pair_start_utc"], utc=True) or
                        ts+pd.Timedelta(seconds=window["duration_seconds"]) > pd.to_datetime(support["pair_end_utc"], utc=True))
        return {**window, **update, "noise_sigma_logodds_per_sqrt_second": self.sigma,
                "noise_template_ticker": self.noise_template_ticker, "volatility_extrapolated": bool(extrapolated),
                "degenerate_empirical_noise_fallback": self.degenerate_empirical_noise_fallback}
