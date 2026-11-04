"""Load and verify the Paper 3 information-arrival calibration (hash-checked).

All values come from `calibration/calibrate_information_arrival.py` outputs in
`data_pipeline/output/paper3/calibration/`. Loading verifies the sha256 of every
artifact recorded in `information_config.json` and of the upstream Paper 2
market-dynamics inputs it was calibrated from (training-only by construction).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, file_hash
from . import kernels as K

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT/"data_pipeline"/"output"/"paper3"/"calibration"
CLASSES = ("broad_based", "sports")


@dataclass
class InformationCalibration:
    config: dict
    directory: Path
    _bank: dict = field(default_factory=dict, repr=False)
    _rates: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, directory: Path = DEFAULT_DIR, verify_upstream: bool = True):
        directory = Path(directory)
        config = json.loads((directory/"information_config.json").read_text())
        if config.get("schema_version") != 1 or config["units"]["time"] != "seconds" or \
                config["units"]["sigma_c"] != "log_odds_per_sqrt_second":
            raise ValueError("Unsupported information calibration schema or units")
        for name, digest in config["file_sha256"].items():
            if file_hash(directory/name) != digest:
                raise ValueError(f"Information calibration artifact changed: {name}")
        if verify_upstream:
            dyn = Path(DYNAMICS_CONFIG)
            upstream = {"paper2/dynamics": dyn, "paper2/mark_bank": dyn.parent/"trade_mark_bank.npz",
                        "paper2/noise_intervals": dyn.parent/"training_noise_intervals.parquet",
                        "paper2/impact": dyn.parent.parent/"price_impact_config.json"}
            for key, path in upstream.items():
                if file_hash(path) != config["input_sha256"][key]:
                    raise ValueError(f"Upstream Paper 2 input changed since information calibration: {key}")
        out = cls(config, directory)
        with np.load(directory/"jump_magnitude_bank.npz", allow_pickle=False) as bank:
            out._bank = {k: bank[k].copy() for k in bank.files}
        with np.load(directory/"uninformed_sign_rates.npz", allow_pickle=False) as rates:
            out._rates = {k: rates[k].copy() for k in rates.files}
        return out

    # ------------------------------------------------------------- templates
    def template(self, ticker: str) -> dict:
        record = self.config["templates"].get(ticker)
        if record is None:
            raise KeyError(f"Not a calibrated training template: {ticker}")
        return record

    def information_class(self, ticker: str) -> str:
        return self.template(ticker)["information_class"]

    def release_time(self, ticker: str) -> pd.Timestamp:
        return pd.Timestamp(self.template(ticker)["t_r_utc"])

    def close_time(self, ticker: str) -> pd.Timestamp:
        return pd.Timestamp(self.template(ticker)["t_c_utc"])

    def ttr_at_close(self, ticker: str) -> float:
        return float((self.release_time(ticker)-self.close_time(ticker)).total_seconds())

    def early_revelation(self, ticker: str) -> bool:
        return bool(self.template(ticker)["early_revelation"])

    def candidate_roster(self) -> list[str]:
        """Train broad_based then sports templates without early revelation (pre-matching)."""
        return list(self.config["candidate_roster"])

    def sigma_c(self, noise_ticker: str) -> float:
        """Continuous background sigma_c of a noise-supported (donor) template; zero is not floored."""
        record = self.template(noise_ticker)
        if not record.get("noise_supported"):
            raise ValueError(f"{noise_ticker} has no own quote support; use its legacy donor's sigma_c")
        return float(record["sigma_c"])

    def p0_emp(self, ticker: str) -> float:
        """Clipped empirical simulation start probability."""
        return float(self.template(ticker)["p0_sim"])

    # ----------------------------------------------------------------- jumps
    def jump_rates(self, information_class: str, scale: float = 1.0) -> dict:
        buckets = self.config["classes"][information_class]["jumps"]["buckets"]
        return {b: scale*float(buckets[b]["lambda_per_second"]) for b in K.TTR_BUCKETS}

    def magnitude_bank(self, information_class: str) -> np.ndarray:
        key = self.config["classes"][information_class]["jumps"]["magnitude_bank_key"]
        return self._bank[key].copy()

    # ------------------------------------------------------------------ flow
    def eta(self, information_class: str) -> float:
        return float(self.config["classes"][information_class]["flow"]["eta_per_contract"])

    def beta(self, information_class: str) -> float:
        return float(self.config["classes"][information_class]["impact_beta_base"])

    def constant_alpha(self, information_class: str) -> float:
        record = self.config["classes"][information_class]["flow"]["constant_alpha"]
        if record.get("status") != "calibrated":
            raise ValueError("Constant-alpha variant was not calibrated")
        return float(record["alpha0"])

    def size_bin(self, information_class: str, size) -> np.ndarray:
        edges = self._rates[f"{information_class}_size_bin_edges"]
        return np.searchsorted(edges, np.asarray(size, dtype=float), side="right")

    def uninformed_rate(self, ticker: str, size, phase) -> np.ndarray:
        """u[tpl, size bin, lifetime phase] in [0.02, 0.98]."""
        key = self.template(ticker)["mark_bank_key"]
        table = self._rates[f"{key}_u_bin_phase"]
        return table[self.size_bin(self.information_class(ticker), size), np.asarray(phase, dtype=int)]

    def uninformed_rate_by_bin(self, ticker: str, size) -> np.ndarray:
        key = self.template(ticker)["mark_bank_key"]
        return self._rates[f"{key}_u_bin"][self.size_bin(self.information_class(ticker), size)]
