"""Run a small reference-price path using a saved training-only impact config."""
import argparse
import json
from pathlib import Path

from .price_impact import DEFAULT_CONFIG, ImpactParameters, ImpactPriceProcess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", default="broad_based")
    parser.add_argument("--scenario", default="base")
    parser.add_argument("--initial-probability", type=float, default=0.5)
    parser.add_argument("--signed-flows", type=float, nargs="+", default=[100, -40, 20])
    args = parser.parse_args()
    params = ImpactParameters.load(args.group, args.scenario, args.config)
    process = ImpactPriceProcess(args.initial_probability, params)
    rows = [{"signed_market_contracts": flow, **process.advance(flow)} for flow in args.signed_flows]
    print(json.dumps({"information_class": args.group, "scenario": args.scenario,
                      "beta_logodds_per_contract": params.beta,
                      "initial_probability": args.initial_probability, "steps": rows}, indent=2))


if __name__ == "__main__":
    main()
