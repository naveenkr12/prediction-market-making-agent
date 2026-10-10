"""Event-driven calibrated maker environment with reset/step and audit ledgers.

Research simulator: aggregate market executions are exogenous. Maker fills
capture a subset of their volume, so aggregate impact is applied exactly once.
The core has no Gym dependency; it implements the modern five-return step API.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd

from .accounting import BinaryAccount
from .execution import QuoteAction, RestingOrder
from .fees import FeeSchedule, decimal
from .market_dynamics import MarketDynamics, DEFAULT_CONFIG as DYNAMICS_CONFIG
from .price_impact import ImpactPriceProcess
from .spreads import SpreadSampler, DEFAULT_CONFIG as EXECUTION_CONFIG


@dataclass(frozen=True)
class EnvironmentConfig:
    group: str = "broad_based"
    ticker: str | None = None
    impact_scenario: str = "base"
    arrival_scenario: str = "base"
    noise_scenario: str = "base"
    initial_probability: float = .5
    initial_cash: float = 1000
    inventory_limit: float = 100
    price_tick: str = "0.01"
    quantity_lot: str = "0.01"
    participation: float = .1
    queue_ahead_contracts: float = 50
    maker_fee_multiplier: float = 1
    balance_precision: str = "0.0001"
    start_life_fraction: float = .9
    max_steps: int = 10000
    inventory_penalty_per_contract2_second: float = 0
    terminal_penalty_per_contract2: float = 0

    def __post_init__(self):
        numbers = [self.participation, self.queue_ahead_contracts, self.start_life_fraction,
                   self.inventory_penalty_per_contract2_second, self.terminal_penalty_per_contract2]
        tick = decimal(self.price_tick)
        if (not np.isfinite(numbers).all() or not 0<self.participation<=1 or self.queue_ahead_contracts<0 or
                not 0<=self.start_life_fraction<1 or self.max_steps<1 or int(self.max_steps)!=self.max_steps or
                min(numbers[-2:])<0 or not 0<tick<.5 or decimal(1)%tick!=0):
            raise ValueError("Invalid environment scenario or price tick")


def bridged_innovations(times, duration, terminal_innovation, sigma, rng):
    """Preserve the calibrated window innovation; assume Brownian motion inside.

    Gaussian bridge fluctuations vanish at the window endpoint, even when the
    terminal residual is empirical/heavy-tailed. Subwindow dynamics are an
    assumption: the calibration only measures five-minute endpoints.
    """
    times = np.asarray(times, dtype=float)
    if (not np.isfinite(times).all() or not 0<duration<=300 or np.any(np.diff(times)<0) or
            np.any(times<0) or np.any(times>=duration) or not np.isfinite([terminal_innovation, sigma]).all() or sigma<0):
        raise ValueError("Invalid bridge times or innovation")
    endpoints = np.r_[times, duration]
    increments = np.diff(np.r_[0, endpoints])
    brownian = np.cumsum(sigma*np.sqrt(increments)*rng.normal(size=len(endpoints)))
    values = endpoints/duration*terminal_innovation + brownian-endpoints/duration*brownian[-1]
    values[-1] = terminal_innovation
    return np.diff(np.r_[0, values])


class TradingEnvironment:
    """Quotes persist until replacement/cancel; one action covers up to 300 s.

    step(QuoteAction(...)) cancels/replaces both sides, resetting queue priority
    and fee accumulators. step(None) retains orders. QuoteAction() cancels both.
    Finite but unaffordable sizes are clipped at admission, with diagnostics;
    malformed, crossed or marketable actions raise before advancing the clock.
    """
    def __init__(self, config=None, dynamics_path=DYNAMICS_CONFIG, execution_path=EXECUTION_CONFIG,
                 dynamics_factory=None, spread_factory=None):
        self.config = config or EnvironmentConfig()
        self.dynamics_path, self.execution_path = Path(dynamics_path), Path(execution_path)
        self._dynamics_factory, self._spread_factory = dynamics_factory, spread_factory
        self._reset_done = False
        self._seed_sequence = np.random.SeedSequence()

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options)-{"seconds_to_close", "initial_probability"}:
            raise ValueError("Unsupported reset options")
        if seed is not None:
            self._seed_sequence = np.random.SeedSequence(seed)
        # Keep selection, market marks/noise, spreads, intrawindow bridges and
        # settlement independent of how many maker fills a policy obtains.
        streams = [np.random.default_rng(s) for s in self._seed_sequence.spawn(5)]
        selection, self.market_rng, self.spread_rng, self.bridge_rng, self.settlement_rng = streams
        cfg = self.config
        self.dynamics = self._dynamics_factory(selection) if self._dynamics_factory else MarketDynamics(
            cfg.group, selection, cfg.ticker, cfg.impact_scenario, cfg.arrival_scenario, cfg.noise_scenario,
            config_path=self.dynamics_path)
        self.spreads = self._spread_factory(self.dynamics, self.spread_rng) if self._spread_factory else SpreadSampler(
            cfg.group, self.dynamics.ticker, self.spread_rng, self.dynamics_path, self.execution_path)
        self.tick = decimal(cfg.price_tick)
        fees = FeeSchedule(cfg.maker_fee_multiplier, cfg.balance_precision)
        self.account = BinaryAccount(cfg.initial_cash, cfg.inventory_limit, cfg.quantity_lot, fees)
        self.price = ImpactPriceProcess(options.get("initial_probability", cfg.initial_probability), self.dynamics.impact)
        self.lifetime = float(self.dynamics.template["lifetime_seconds"])
        start = cfg.start_life_fraction*self.lifetime
        if "seconds_to_close" in options:
            seconds = float(options["seconds_to_close"])
            if not np.isfinite(seconds) or not 0<seconds<=self.lifetime:
                raise ValueError("seconds_to_close must be within the market lifetime")
            start = self.lifetime-seconds
        self.elapsed, self.start_elapsed = start, start
        self.book = self.spreads.sample(self.price.log_odds, self.spread_rng, self.tick)
        self.orders, self.ledger, self.steps = {}, [], []
        self.step_number, self.order_number = 0, 0
        self.done, self.last_flow, self.total_reward, self.total_risk_penalty = False, 0., 0., 0.
        self._reset_done = True
        return self.observation(), self._info()

    @property
    def reserved_cash(self):
        lot = self.account.lot
        return sum((o.remaining*((o.price if o.side=="bid" else 1-o.price)+
                    self.account.fees.reserve_bound(o.price, Decimal(1), lot)) for o in self.orders.values()), Decimal(0))

    def observation(self):
        a = self.account
        result = {"probability": self.price.probability, "log_odds": self.price.log_odds,
            "bid": self.book.bid, "ask": self.book.ask, "cash": float(a.cash), "inventory": float(a.inventory),
            "collateral": float(a.collateral), "reserved_cash": float(self.reserved_cash),
            "available_cash": float(a.available_cash-self.reserved_cash), "equity": float(a.equity(self.price.probability)),
            "elapsed_seconds": self.elapsed, "seconds_to_close": max(0., self.lifetime-self.elapsed),
            "life_fraction": self.elapsed/self.lifetime, "last_signed_flow": self.last_flow,
            "beta": self.dynamics.impact.beta, "sigma": self.dynamics.sigma, "fees": float(a.total_fees)}
        for side in ("bid", "ask"):
            order = self.orders.get(side)
            result[f"own_{side}"] = float(order.price) if order else 0.
            result[f"own_{side}_remaining"] = float(order.remaining) if order else 0.
            # Synthetic queue state is observable in this research setting;
            # it must be masked/estimated for a live-data policy experiment.
            result[f"own_{side}_queue"] = float(order.queue_ahead) if order else 0.
        return result

    def _info(self, window_start=None):
        ts = pd.Timestamp(self.dynamics.template["open_time"])+pd.Timedelta(seconds=self.elapsed)
        first = pd.Timestamp(self.dynamics.template["open_time"])+pd.Timedelta(
            seconds=self.elapsed if window_start is None else window_start)
        noise = self.dynamics.noise_template["noise"]
        spread = self.spreads.template
        return {"ticker": self.dynamics.ticker, "group": self.config.group,
            "noise_template_ticker": self.dynamics.noise_template_ticker, "spread_template_ticker": self.spreads.ticker,
            "timestamp_utc": ts.isoformat(), "fill_model": "queue_and_volume_participation_scenario",
            "fee_profile": self.account.fees.label, "maker_fee_multiplier": self.config.maker_fee_multiplier,
            "balance_precision": self.config.balance_precision,
            "volatility_extrapolated": bool(self.dynamics.noise_template_ticker!=self.dynamics.ticker or
                first<pd.Timestamp(noise["pair_start_utc"]) or ts>pd.Timestamp(noise["pair_end_utc"])),
            "spread_extrapolated": bool(self.spreads.ticker!=self.dynamics.ticker or
                first<pd.Timestamp(spread["start_utc"]) or ts>pd.Timestamp(spread["end_utc"])),
            "settled": self.account.settled}

    def _record(self, kind, **values):
        row = {"kind": kind, "step": self.step_number, "elapsed_seconds": self.elapsed,
            "probability": self.price.probability, "log_odds": self.price.log_odds,
            "cash": self.account.cash, "inventory": self.account.inventory,
            "collateral": self.account.collateral, **values}
        self.ledger.append({k:(str(v) if isinstance(v, Decimal) else v) for k,v in row.items()})

    def _replace(self, action):
        if isinstance(action, dict):
            action = QuoteAction(**action)
        if not isinstance(action, QuoteAction):
            raise TypeError("Action must be QuoteAction, dict or None")
        action.validate(self.book, self.tick, self.account.lot)
        requests = {s:(getattr(action, s) if getattr(action,s) is not None else .5,
                       getattr(action,s+"_size")) for s in ("bid", "ask")}
        admitted, _ = self.account.admit_quotes(requests)
        for order in self.orders.values():
            self._record("cancel", order_id=order.order_id, side=order.side, remaining=order.remaining)
        self.orders = {}
        diagnostics = {}
        for side in ("bid", "ask"):
            diagnostics[side+"_requested"] = float(requests[side][1])
            diagnostics[side+"_admitted"] = float(admitted[side])
            if admitted[side]:
                self.order_number += 1
                price = decimal(requests[side][0])
                order = RestingOrder(self.order_number, side, price, admitted[side],
                    self.book.queue_ahead(side, price, self.config.queue_ahead_contracts))
                self.orders[side] = order
                self._record("quote", order_id=order.order_id, side=side, price=price,
                             quantity=order.remaining, queue_ahead=order.queue_ahead)
        return diagnostics

    def step(self, action):
        if not self._reset_done or self.done:
            raise RuntimeError("Call reset before stepping, and after episode end")
        before = float(self.account.equity(self.price.probability))
        diagnostics = self._replace(action) if action is not None else {}
        self.step_number += 1
        start = self.elapsed
        window = self.dynamics.sample_window(start/self.lifetime, self.market_rng)
        duration = float(window["duration_seconds"])
        if not 0<duration<=300 or duration>self.lifetime-start+1e-7:
            raise ValueError("Market driver returned invalid window duration")
        innovation = float(self.dynamics.sample_innovation(self.market_rng, duration_seconds=duration)[0])
        pieces = bridged_innovations(window["event_times_seconds"], duration, innovation, self.dynamics.sigma, self.bridge_rng)
        n = len(window["quantities"])
        if n!=len(window["signs"]) or n!=len(window["event_times_seconds"]):
            raise ValueError("Market driver returned mismatched execution arrays")
        risk_integral, previous_offset, fills = 0., 0., 0
        for i, (offset, quantity, sign) in enumerate(zip(window["event_times_seconds"], window["quantities"], window["signs"])):
            if sign not in (-1, 1) or not np.isfinite(quantity) or quantity<=0:
                raise ValueError("Invalid market execution")
            risk_integral += float(self.account.inventory)**2*(offset-previous_offset)
            previous_offset = offset
            self.elapsed = start+float(offset)
            self.price.advance(0, pieces[i])
            self._record("noise", innovation_logodds=float(pieces[i]))
            self.book = self.spreads.sample(self.price.log_odds, self.spread_rng, self.tick)
            side = "ask" if sign==1 else "bid"
            execution_limit = self.book.ask if sign==1 else self.book.bid
            order = self.orders.get(side)
            own_qty, queue_before = Decimal(0), order.queue_ahead if order else Decimal(0)
            if order:
                own_qty = order.match(sign, quantity, execution_limit, self.config.participation, self.account.lot)
            if own_qty:
                accounting = self.account.fill(side, order.price, own_qty, order.rounding_accumulator)
                order.remaining -= own_qty
                order.rounding_accumulator = accounting["accumulator"]
                fills += 1
                self._record("fill", order_id=order.order_id, side=side, price=order.price, quantity=own_qty,
                    market_quantity=float(quantity), market_sign=int(sign), execution_limit=execution_limit,
                    queue_before=queue_before, queue_after=order.queue_ahead, **accounting)
            p_before, l_before = self.price.probability, self.price.log_odds
            update = self.price.advance(float(sign*quantity))
            self._record("market", market_sign=int(sign), market_quantity=float(quantity), maker_quantity=own_qty,
                probability_before=p_before, log_odds_before=l_before, impact_logodds=update["impact_logodds"],
                float_probability_saturated=update["float_probability_saturated"])
        risk_integral += float(self.account.inventory)**2*(duration-previous_offset)
        self.elapsed = min(self.lifetime, start+duration)
        self.price.advance(0, pieces[-1])
        self._record("noise", innovation_logodds=float(pieces[-1]))
        self.book = self.spreads.sample(self.price.log_odds, self.spread_rng, self.tick)
        self.last_flow = float(window["signed_flow_contracts"])
        terminated = self.lifetime-self.elapsed<1e-7
        truncated = self.step_number>=self.config.max_steps and not terminated
        terminal_inventory = float(self.account.inventory)
        if terminated or truncated:
            for order in self.orders.values():
                self._record("cancel", order_id=order.order_id, side=order.side, remaining=order.remaining)
            self.orders = {}
        settlement = {}
        if terminated:
            # Synthetic scenario, not an historical result or a calibrated
            # resolution model. Settlement risk is Bernoulli at closing p.
            outcome = int(self.settlement_rng.random()<self.price.probability)
            settlement = self.account.settle(outcome)
            self._record("settlement", **settlement)
        after = float(self.account.equity(self.price.probability))
        penalty = self.config.inventory_penalty_per_contract2_second*risk_integral
        if terminated:
            penalty += self.config.terminal_penalty_per_contract2*terminal_inventory**2
        reward = after-before-penalty
        self.total_reward += reward
        self.total_risk_penalty += penalty
        self.done = terminated or truncated
        info = {**self._info(start), **diagnostics, "market_trades": n, "maker_fills": fills,
            "signed_flow_contracts": self.last_flow, "window_innovation_logodds": innovation,
            "equity_change": after-before, "risk_penalty": penalty, "reward": reward,
            "phase_mark_fallback": bool(window["phase_mark_fallback"]),
            **{k:float(v) if isinstance(v, Decimal) else v for k,v in settlement.items()}}
        self.steps.append({"step": self.step_number, "elapsed_seconds": self.elapsed,
                           "terminated": terminated, "truncated": truncated, **info, **self.observation()})
        return self.observation(), reward, terminated, truncated, info

    def render(self):
        o = self.observation()
        return f"p={o['probability']:.4f} q={o['inventory']:.2f} cash=${o['cash']:.4f} equity=${o['equity']:.4f} remaining={o['seconds_to_close']:.1f}s"

    def close(self):
        pass
