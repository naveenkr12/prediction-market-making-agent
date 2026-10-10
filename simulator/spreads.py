"""Training-only external book spread scenarios around the latent reference."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .market_dynamics import file_hash
from .fees import decimal, floor_grid
from .execution import BookSnapshot

DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/"data_pipeline/output/paper2/calibration/simulator/execution/execution_config.json"


class SpreadSampler:
    def __init__(self, group, ticker, rng, dynamics_path, config_path=DEFAULT_CONFIG):
        path = Path(config_path)
        config = json.loads(path.read_text())
        if config["spread_units"]!="log_odds_ask_minus_bid" or file_hash(dynamics_path)!=config["dynamics_config_sha256"]:
            raise ValueError("Spread units or upstream dynamics changed")
        supported = config["groups"][group]["supported_tickers"]
        self.ticker = ticker if ticker in supported else str(rng.choice(supported))
        self.template = config["templates"][self.ticker]
        bank = path.parent/config["bank_file"]
        if file_hash(bank)!=config["bank_sha256"]:
            raise ValueError("Spread bank changed")
        with np.load(bank, allow_pickle=False) as arrays:
            self.gaps = arrays[self.template["bank_key"]].copy()
        if not np.isfinite(self.gaps).all() or not (self.gaps>0).all():
            raise ValueError("Invalid spread bank")

    def sample(self, log_odds, rng, tick):
        gap = float(rng.choice(self.gaps))
        # The latent reference is the log-odds center, not necessarily the
        # arithmetic midpoint after nonlinear transformation/tick rounding.
        bid = floor_grid(decimal(float(expit(log_odds-gap/2))), tick)
        ask = -floor_grid(-decimal(float(expit(log_odds+gap/2))), tick)
        # At floating-point saturation retain an executable, one-tick book.
        # This changes displayed quotes only; the latent reference is retained.
        bid = min(1-tick, max(decimal(0), bid))
        ask = min(decimal(1), max(tick, bid+tick, ask))
        return BookSnapshot(float(bid), float(ask))
