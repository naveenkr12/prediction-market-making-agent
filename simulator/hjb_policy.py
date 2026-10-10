"""Adapt the frozen published-model HJB policy to the calibrated environment.

Reuse the original Monte Carlo quote extraction without modifying the HJB
reproduction or exposing its bare-module imports to the rest of the project.
The policy is a reference benchmark, not an optimum in the new environment.
"""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .execution import QuoteAction
from .fees import decimal, floor_grid
from .market_dynamics import file_hash

ROOT = Path(__file__).resolve().parents[1]
HJB_ROOT = ROOT/"hjb_baseline"
DEFAULT_GRID = HJB_ROOT/"output/data/grid_production.npz"
DEFAULT_VALUE = HJB_ROOT/"output/data/V_production.npy"


@lru_cache(maxsize=1)
def legacy_modules():
    """Load legacy dependencies under private names; restore all bare aliases."""
    names = ("params", "intensities", "price_process", "hjb_solver", "quotes", "monte_carlo")
    missing = object()
    previous = {name:sys.modules.get(name, missing) for name in names}
    prefix = "simulator._legacy_hjb"
    package = ModuleType(prefix)
    package.__path__ = [str(HJB_ROOT)]
    sys.modules[prefix] = package
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(prefix+"."+name, HJB_ROOT/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
            sys.modules[name] = module
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded


def executable_action(bid, ask, bid_size, ask_size, observation, tick="0.01", lot="0.01"):
    """Outward rounding and post-only projection; preserve disabled sides."""
    tick, lot = decimal(tick), decimal(lot)
    prices, sizes = {}, {}
    projected = {}
    for side, raw, size in (("bid",bid,bid_size),("ask",ask,ask_size)):
        prices[side], sizes[side] = None, 0.
        projected[side+"_post_only_adjusted"] = False
        if not np.isfinite(raw) or size<=0:
            continue
        rounded = floor_grid(decimal(raw),tick) if side=="bid" else -floor_grid(-decimal(raw),tick)
        price = min(rounded,-floor_grid(-decimal(observation["ask"]),tick)-tick) if side=="bid" else max(
            rounded,floor_grid(decimal(observation["bid"]),tick)+tick)
        projected[side+"_post_only_adjusted"] = price!=rounded
        qty = max(decimal(0),floor_grid(decimal(size),lot))
        if 0<price<1 and qty>0:
            prices[side], sizes[side] = float(price), float(qty)
    crossing = prices["bid"] is not None and prices["ask"] is not None and prices["bid"]>=prices["ask"]
    if crossing:
        prices, sizes = {"bid":None,"ask":None}, {"bid":0.,"ask":0.}
    action = QuoteAction(prices["bid"],prices["ask"],sizes["bid"],sizes["ask"])
    return action, {**projected,"crossed_quotes_disabled":bool(crossing),
        "bid_disabled":action.bid_size==0,"ask_disabled":action.ask_size==0}


class HJBReferencePolicy:
    def __init__(self, grid_path=DEFAULT_GRID, value_path=DEFAULT_VALUE, time_mapping="episode"):
        if time_mapping not in {"episode","lifetime"}:
            raise ValueError("Unknown HJB time mapping")
        self.modules = legacy_modules()
        self.parameters = self.modules["params"].DEFAULT
        self.grid_path, self.value_path = Path(grid_path), Path(value_path)
        with np.load(self.grid_path,allow_pickle=False) as data:
            self.grid = self.modules["hjb_solver"].Grid(**{name:data[name].copy() for name in ("t","p","q","pi")})
        self.value = np.load(self.value_path,allow_pickle=False,mmap_mode="r")
        g, par = self.grid, self.parameters
        if (self.value.shape!=(len(g.t),len(g.p),len(g.q)) or not np.isfinite(self.value).all() or
                any(not np.isfinite(x).all() or np.any(np.diff(x)<=0) for x in (g.t,g.p,g.q,g.pi)) or
                not np.isclose(g.t[0],0) or not np.isclose(g.t[-1],par.T) or
                not np.allclose(g.q, np.arange(-par.Q,par.Q+par.delta/2,par.delta)) or
                not np.allclose(self.value[-1],self.modules["hjb_solver"].terminal_condition(par,g.p,g.q),atol=1e-10)):
            raise ValueError("Saved grid/value is incompatible with the frozen paper parameters")
        # Refuse accidental extrapolation. Original batch extraction clips
        # neighboring inventories at Q, including off-lattice partial fills.
        self.interpolator = RegularGridInterpolator((g.t,g.p,g.q),self.value,bounds_error=True)
        self.time_mapping = time_mapping

    def raw_quotes(self, t, probability, inventory):
        par = self.parameters
        if not np.isfinite([t,probability,inventory]).all() or not 0<=t<=par.T or not 0<=probability<=1 or abs(inventory)>par.Q+1e-9:
            raise ValueError("HJB state outside supported time/probability/inventory domain")
        p = float(np.clip(probability,self.grid.p[0],self.grid.p[-1]))
        q = float(np.clip(inventory,-par.Q,par.Q))
        bid, ask = self.modules["monte_carlo"]._batch_optimal_quotes(t,np.array([p]),np.array([q]),
            par,self.grid,self.interpolator)
        return float(bid[0]),float(ask[0]),p

    def action(self, observation, *, start_elapsed, lifetime, inventory_limit=100, tick="0.01", lot="0.01"):
        par = self.parameters
        if not 0<inventory_limit<=par.Q or not 0<=start_elapsed<lifetime:
            raise ValueError("Unsupported inventory limit or episode horizon")
        elapsed = float(observation["elapsed_seconds"])
        fraction = (elapsed-start_elapsed)/(lifetime-start_elapsed) if self.time_mapping=="episode" else elapsed/lifetime
        if not -1e-10<=fraction<=1+1e-10:
            raise ValueError("Observation outside evaluation horizon")
        t = par.T*float(np.clip(fraction,0,1))
        q, p = float(observation["inventory"]),float(observation["probability"])
        bid, ask, clipped = self.raw_quotes(t,p,q)
        action, projections = executable_action(bid,ask,min(par.delta,max(0,inventory_limit-q)),
            min(par.delta,max(0,inventory_limit+q)),observation,tick,lot)
        return action, {"hjb_time":t,"hjb_probability":clipped,"hjb_probability_clipped":clipped!=p,
            "hjb_raw_bid":bid if np.isfinite(bid) else None,"hjb_raw_ask":ask if np.isfinite(ask) else None,
            **projections}

    def provenance(self):
        return {"parameters":asdict(self.parameters),"time_mapping":self.time_mapping,
            "quote_extractor":"Original monte_carlo._batch_optimal_quotes; neighboring q clipped exactly as in reproduction",
            "fractional_inventory":"Linear value interpolation; same contracts as original; no inventory rescaling",
            "grid_file":str(self.grid_path),"grid_sha256":file_hash(self.grid_path),
            "value_file":str(self.value_path),"value_sha256":file_hash(self.value_path),
            "grid_shape":list(self.value.shape),"value_at_initial_mid_flat":float(self.interpolator([[0,.5,0]])[0]),
            "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in HJB_ROOT.glob("*.py")},
            "optimality_claim_in_impacted_environment":False}
