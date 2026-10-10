"""Sample exchange trade executions and flow/noise reference updates."""
import argparse
import json
from pathlib import Path

import numpy as np

from .market_dynamics import DEFAULT_CONFIG, MarketDynamics
from .price_impact import ImpactPriceProcess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", default="sports")
    parser.add_argument("--ticker")
    parser.add_argument("--impact-scenario", default="base")
    parser.add_argument("--arrival-scenario", default="base")
    parser.add_argument("--noise-scenario", default="base")
    parser.add_argument("--noise-distribution", choices=["empirical", "gaussian"], default="empirical")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--life-fraction", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.windows < 1:
        raise ValueError("Require at least one window")
    rng = np.random.default_rng(args.seed)
    dynamics = MarketDynamics(args.group, rng, args.ticker, args.impact_scenario, args.arrival_scenario,
                              args.noise_scenario, args.noise_distribution, args.config)
    price = ImpactPriceProcess(.5, dynamics.impact)
    duration = dynamics.template["lifetime_seconds"]
    fraction = args.life_fraction if args.life_fraction is not None else max(0, 1-86400/duration)
    rows = []
    for i in range(args.windows):
        if fraction >= 1:
            break
        result = dynamics.advance_window(price, fraction, rng)
        rows.append({"window": i, "life_fraction": fraction, **{k: v for k, v in result.items()
                     if k not in {"quantities", "signs", "event_times_seconds"}}})
        fraction += result["duration_seconds"]/duration
    print(json.dumps({"group": args.group, "training_event_template": dynamics.ticker,
                      "noise_template": dynamics.noise_template_ticker, "noise_distribution": args.noise_distribution,
                      "sigma_logodds_per_sqrt_second": dynamics.sigma, "windows": rows}, indent=2))


if __name__ == "__main__":
    main()
