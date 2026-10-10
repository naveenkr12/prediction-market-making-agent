"""A signed-market-flow jump in latent log odds, calibrated on training events.

The reference is observed after each flow update. This is a modelling proxy
for transaction prices, not a calibrated execution/quote model. The process
can accept independent log-odds innovations from a separate price driver.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

DEFAULT_CONFIG = (Path(__file__).resolve().parents[1] / "data_pipeline" / "output" /
                  "paper2" / "calibration" / "simulator" / "price_impact_config.json")


@dataclass(frozen=True)
class ImpactParameters:
    beta: float
    information_class: str = "manual"
    scenario: str = "manual"

    def __post_init__(self):
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Impact beta must be finite and nonnegative")

    @classmethod
    def load(cls, information_class: str, scenario: str = "base", path: Path = DEFAULT_CONFIG):
        config = json.loads(Path(path).read_text())
        if config.get("beta_units") != "log_odds_per_signed_market_contract" or config.get("reference_observation") != "post_trade":
            raise ValueError("Unsupported calibration units or reference observation convention")
        profile = config["groups"].get(information_class)
        if profile is None or profile["status"] != "calibrated":
            raise ValueError(f"No supported calibration for {information_class}; choose an explicitly supported group")
        if scenario not in profile["scenarios"]:
            raise ValueError(f"Unknown impact scenario: {scenario}")
        return cls(float(profile["scenarios"][scenario]), information_class, scenario)


def displayed_probability(log_odds):
    """Keep float representations interior; latent log odds remain unmodified."""
    p = expit(log_odds)
    return np.clip(p, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))


class ImpactPriceProcess:
    """Reference-price updates; market flow is +YES taker / -NO taker.

    `innovation_logodds` is an optional already-generated base innovation.
    Flow is a quantity, so its jump is not multiplied by elapsed time.
    """
    def __init__(self, initial_probability: float, parameters: ImpactParameters):
        if not np.isfinite(initial_probability) or not 0 < initial_probability < 1:
            raise ValueError("Initial probability must be strictly between zero and one")
        self.log_odds = float(logit(initial_probability))
        self.parameters = parameters

    @property
    def probability(self) -> float:
        return float(displayed_probability(self.log_odds))

    def advance(self, signed_market_contracts: float, innovation_logodds: float = 0.0) -> dict:
        if not np.isfinite(signed_market_contracts) or not np.isfinite(innovation_logodds):
            raise ValueError("Flow and innovation must be finite")
        before = self.log_odds
        impact = self.parameters.beta * signed_market_contracts
        updated = before + innovation_logodds + impact
        if not np.isfinite(updated):
            raise ValueError("Log-odds update overflow")
        self.log_odds = float(updated)
        return {"probability_before": float(displayed_probability(before)), "probability": self.probability,
                "log_odds": self.log_odds, "impact_logodds": float(impact),
                "innovation_logodds": float(innovation_logodds),
                "float_probability_saturated": bool(expit(updated) in (0.0, 1.0))}

    def advance_maker_fill(self, yes_inventory_change: float, innovation_logodds: float = 0.0) -> dict:
        """Maker sells YES -> positive market flow; maker buys YES -> negative.

        Use this for an own-fill-only experiment. When supplied aggregate market
        flow already contains that fill, call `advance` once to avoid duplication.
        """
        return self.advance(-yes_inventory_change, innovation_logodds)


def observe_trade_flows(initial_probability: float, signed_quantities: np.ndarray,
                        parameters: ImpactParameters) -> tuple[np.ndarray, np.ndarray]:
    """Post-trade reference observations for a zero-innovation replay segment."""
    flow = np.asarray(signed_quantities, dtype=float)
    if flow.ndim != 1 or not np.isfinite(flow).all():
        raise ValueError("Trade flow must be a finite one-dimensional array")
    if not np.isfinite(initial_probability) or not 0 < initial_probability < 1:
        raise ValueError("Initial probability must be strictly between zero and one")
    latent = logit(initial_probability) + parameters.beta*np.cumsum(flow)
    if not np.isfinite(latent).all():
        raise ValueError("Replay log-odds overflow")
    return latent, displayed_probability(latent)
