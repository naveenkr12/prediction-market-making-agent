"""InformationTradingEnvironment: walks a precomputed (episode, arm) path (spec Sections 6.2-6.4).

The reset/step interface is identical to `TradingEnvironment` (five-return
step; legacy observation keys plus `arm`, `family`, `snipe_level`). Quotes,
admission, queue/participation matching and maker-fee accounting are the
legacy code (inherited `_replace`, `RestingOrder.match`, `BinaryAccount.fill`).
Fills are booked at the standing quote at execution nodes BEFORE that
execution's flow update. Prices come only from the path: fills and snipes
never move them. Observations contain the public state only (no Theta, no
informed flags, no future nodes or signs).

Sniping (S1, Section 6.4): at each proposal node of side X, an alive side-X
order with remaining Q that is crossed at the node's exact unclipped p fills
fully at its price (maker fee via `BinaryAccount.fill`), bypassing queue and
participation, with no price update and ledger kind `snipe`; it is not market
flow. Under S0 proposal nodes are inert.

Decision interval (Section 9.3 hook): 300 (default) = one step per legacy
market window. With 60, `step` also stops at the skeleton's 60-s decision
nodes strictly inside windows; the observed book is then the most recently
sampled one (no extra spread draw) and market windows are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json

import numpy as np
from scipy.special import expit

from ..benchmark_metrics import episode_diagnostics
from ..demo_trading_env import FILL_SCENARIOS
from ..execution import QuoteAction
from ..fees import FeeSchedule, decimal
from ..accounting import BinaryAccount
from ..policies import quote_policy
from ..price_impact import ImpactPriceProcess
from ..trading_env import EnvironmentConfig, TradingEnvironment
from ..validate_trading_env import audit_ledger
from .paths import FLOW_LAW, EpisodePath
from .skeleton import DECISION, EXECUTION, JUMP, PROPOSAL_ASK, PROPOSAL_BID, WINDOW_END
from .taker_fees import TakerFeeSchedule, snipe_profitable

SNIPE_LEVELS = ("S0", "S1")
EXTRA_OBSERVATION_KEYS = ("arm", "family", "snipe_level")
EXTRA_INFO_KEYS = ("arm", "snipe_fills", "jumps_applied", "decision_stop")
POLICIES = ("hjb", "fixed", "skew", "none")


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class InformationTradingEnvironment(TradingEnvironment):
    def __init__(self, path: EpisodePath, config: EnvironmentConfig | None = None, *, snipe_level: str = "S0",
                 decision_interval: int = 300, taker_fees: TakerFeeSchedule = TakerFeeSchedule(),
                 record_proposals: bool = False):
        sk = path.skeleton
        config = config or replace(sk.config, initial_probability=path.arm.p0)
        for name in ("group", "ticker", "impact_scenario", "arrival_scenario", "noise_scenario", "price_tick"):
            if getattr(config, name) != getattr(sk.config, name):
                raise ValueError(f"Environment config {name} differs from the episode skeleton")
        if config.initial_probability != path.arm.p0:
            raise ValueError("config.initial_probability must equal the arm's p0")
        if snipe_level not in SNIPE_LEVELS:
            raise ValueError("Unknown snipe level")
        if decision_interval not in (300, 60):
            raise ValueError("decision_interval must be 300 (legacy) or 60")
        if path.arm.legacy_nodes and (snipe_level != "S0" or decision_interval != 300):
            raise ValueError("Legacy-node paths (golden, L0b) have no proposal or decision nodes")
        if decision_interval == 60 and sk.decision_spacing != 60:
            raise ValueError("60-s decisions require 60-s skeleton decision nodes")
        super().__init__(config, sk.dynamics_path, sk.execution_path)
        self.path = path
        self.snipe_level = snipe_level
        self.decision_interval = decision_interval
        self.taker_fees = taker_fees
        self.record_proposals = record_proposals

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options)-{"seconds_to_close", "initial_probability"}:
            raise ValueError("Unsupported reset options")
        path, sk, cfg = self.path, self.path.skeleton, self.config
        if seed is not None and seed != sk.path_seed:
            raise ValueError("Reset seed differs from the precomputed path seed")
        if "seconds_to_close" in options and float(options["seconds_to_close"]) != sk.seconds:
            raise ValueError("seconds_to_close differs from the precomputed episode")
        if "initial_probability" in options and options["initial_probability"] != path.arm.p0:
            raise ValueError("initial_probability differs from the arm's p0")
        self.dynamics, self.spreads = sk.dynamics, sk.spreads
        self.tick = decimal(cfg.price_tick)
        self._tick_float = float(self.tick)
        fees = FeeSchedule(cfg.maker_fee_multiplier, cfg.balance_precision)
        self.account = BinaryAccount(cfg.initial_cash, cfg.inventory_limit, cfg.quantity_lot, fees)
        self.price = ImpactPriceProcess(path.arm.p0, self.dynamics.impact)
        if self.price.log_odds != path.initial_log_odds:
            raise AssertionError("Initial log odds differ from the precomputed path")
        self.lifetime = sk.lifetime
        self.elapsed, self.start_elapsed = sk.start, sk.start
        self.book = path.initial_book
        self.orders, self.ledger, self.steps = {}, [], []
        self.step_number, self.order_number = 0, 0
        self.done, self.last_flow, self.total_reward, self.total_risk_penalty = False, 0., 0., 0.
        self._window, self._ptr = 0, path.window_node_slices[0].start
        self._offset = 0.0
        self.trace, self.proposal_log = [], []
        self.snipe_count, self.jump_count = 0, 0
        self._reset_done = True
        return self.observation(), self._info()

    def observation(self):
        result = super().observation()
        result.update(arm=self.path.arm.name, family=self.path.arm.family, snipe_level=self.snipe_level)
        return result

    # ------------------------------------------------------------------- step
    def step(self, action):
        if not self._reset_done or self.done:
            raise RuntimeError("Call reset before stepping, and after episode end")
        path, sk, cfg = self.path, self.path.skeleton, self.config
        before = float(self.account.equity(self.price.probability))
        diagnostics = self._replace(action) if action is not None else {}
        self.step_number += 1
        start = self.elapsed
        window = sk.windows[self._window]
        stop = path.window_node_slices[self._window].stop
        kinds, refs, elapsed, offsets = path.kinds, path.refs, path.elapsed, path.offsets
        x, l_bg, l_after, p_bg = path.x, path.l_bg, path.l_after, path.p_bg
        stop_at_decisions = self.decision_interval == 60
        jump_active = path.jump_active        # per-jump mask: thinned or jump-free arms leave inert jump nodes
        sniping = self.snipe_level == "S1"
        lot, participation = self.account.lot, cfg.participation
        risk_integral, previous_offset = 0., self._offset
        fills = snipes = jumps = trades = 0
        executed = []
        first_node = self._ptr
        pending = None
        reached_end = decision_stop = False
        while self._ptr < stop:
            i = self._ptr
            self._ptr += 1
            kind = kinds[i]
            if kind == PROPOSAL_BID or kind == PROPOSAL_ASK:
                side = "bid" if kind == PROPOSAL_BID else "ask"
                order = self.orders.get(side) if sniping else None
                if order is not None and order.remaining > 0:
                    p = p_bg[i]
                    m = float(order.price)
                    # Necessary condition (taker fee >= 0): edge per contract > one tick.
                    gap = p-m if side == "ask" else m-p
                    crossed = gap > .5*self._tick_float and snipe_profitable(
                        side, order.remaining, order.price, p, self.tick, self.taker_fees)
                    if self.record_proposals:
                        self.proposal_log.append({"node": i, "side": side, "order_id": order.order_id,
                            "price": str(order.price), "remaining": str(order.remaining), "latent_probability": p,
                            "crossed": bool(crossed)})
                    if crossed:
                        offset = offsets[i]
                        risk_integral += float(self.account.inventory)**2*(offset-previous_offset)
                        previous_offset = offset
                        self.elapsed = elapsed[i]
                        piece = x[i] if pending is None else pending+x[i]
                        pending = None
                        self.price.log_odds = l_bg[i]
                        self._record("noise", innovation_logodds=float(piece))
                        quantity = order.remaining
                        fee = self.taker_fees.fee(quantity, order.price)
                        accounting = self.account.fill(side, order.price, quantity, order.rounding_accumulator)
                        order.remaining -= quantity
                        order.rounding_accumulator = accounting["accumulator"]
                        snipes += 1
                        self._record("snipe", order_id=order.order_id, side=side, price=order.price, quantity=quantity,
                                     latent_probability=p, taker_fee=fee, taker_fee_label=self.taker_fees.label,
                                     mean_latency_seconds=sk.mean_latency, **accounting)
                        continue
                pending = x[i] if pending is None else pending+x[i]
                continue
            if kind == DECISION and not stop_at_decisions or kind == JUMP and not jump_active[refs[i]]:
                pending = x[i] if pending is None else pending+x[i]
                continue
            offset = offsets[i]
            risk_integral += float(self.account.inventory)**2*(offset-previous_offset)
            previous_offset = offset
            self.elapsed = elapsed[i]
            piece = x[i] if pending is None else pending+x[i]
            pending = None
            self.price.log_odds = l_bg[i]
            self._record("noise", innovation_logodds=float(piece))
            if kind == EXECUTION:
                j = refs[i]
                sign, quantity = path.exec_sign[j], sk.exec_quantity[j]
                if sign not in (-1, 1) or not np.isfinite(quantity) or quantity <= 0:
                    raise ValueError("Invalid market execution")
                self.book = path.books[i]
                side = "ask" if sign == 1 else "bid"
                execution_limit = self.book.ask if sign == 1 else self.book.bid
                order = self.orders.get(side)
                own_qty, queue_before = Decimal(0), order.queue_ahead if order else Decimal(0)
                if order:
                    own_qty = order.match(sign, quantity, execution_limit, participation, lot)
                if own_qty:
                    accounting = self.account.fill(side, order.price, own_qty, order.rounding_accumulator)
                    order.remaining -= own_qty
                    order.rounding_accumulator = accounting["accumulator"]
                    fills += 1
                    self._record("fill", order_id=order.order_id, side=side, price=order.price, quantity=own_qty,
                        market_quantity=float(quantity), market_sign=int(sign), execution_limit=execution_limit,
                        queue_before=queue_before, queue_after=order.queue_ahead, **accounting)
                p_before, l_before = self.price.probability, self.price.log_odds
                self.price.log_odds = l_after[i]
                extra = {} if path.arm.family == "legacy" else self._flow_fields(j)
                self._record("market", market_sign=int(sign), market_quantity=float(quantity), maker_quantity=own_qty,
                    probability_before=p_before, log_odds_before=l_before,
                    impact_logodds=float(path.exec_increment[j]),
                    float_probability_saturated=_saturated(l_after[i]), **extra)
                trades += 1
                executed.append(j)
                self.trace.append((i, l_bg[i], l_after[i], self.book.bid, self.book.ask))
            elif kind == JUMP:
                j = refs[i]
                p_before, l_before = self.price.probability, self.price.log_odds
                self.price.log_odds = l_after[i]
                jumps += 1
                self._record("jump", jump_sign=int(path.jump_sign[j]), jump_magnitude=float(sk.jump_magnitude[j]),
                    jump_logodds=float(path.jump_step[j]), jump_law=path.arm.jumps,
                    probability_before=p_before, log_odds_before=l_before)
                self.trace.append((i, l_bg[i], l_after[i]))
            elif kind == DECISION:
                decision_stop = True
                break
            elif kind == WINDOW_END:
                self.book = path.books[i]
                self.trace.append((i, l_bg[i], l_after[i], self.book.bid, self.book.ask))
                reached_end = True
                break
        if pending is not None:
            raise AssertionError("Background pieces were left unrecorded")
        whole_window = reached_end and first_node == path.window_node_slices[self._window].start
        if path.arm.family == "legacy":
            self.last_flow, innovation = path.window_signed_flow[self._window], window.innovation
        elif whole_window:
            self.last_flow, innovation = path.window_signed_flow[self._window], path.window_background[self._window]
        else:
            self.last_flow = self._signed_flow(executed)
            innovation = float(np.sum(np.asarray(x[first_node:self._ptr])))
        if reached_end:
            self._window += 1
            self._offset = 0.0
        else:
            self._offset = previous_offset
        self.snipe_count += snipes
        self.jump_count += jumps
        terminated = reached_end and self.lifetime-self.elapsed < 1e-7
        truncated = self.step_number >= cfg.max_steps and not terminated
        terminal_inventory = float(self.account.inventory)
        if terminated or truncated:
            for order in self.orders.values():
                self._record("cancel", order_id=order.order_id, side=order.side, remaining=order.remaining)
            self.orders = {}
        settlement = {}
        if terminated:
            outcome = int(path.settlement["outcome"])
            settlement = self.account.settle(outcome)
            self._record("settlement", **settlement)
        after = float(self.account.equity(self.price.probability))
        penalty = cfg.inventory_penalty_per_contract2_second*risk_integral
        if terminated:
            penalty += cfg.terminal_penalty_per_contract2*terminal_inventory**2
        reward = after-before-penalty
        self.total_reward += reward
        self.total_risk_penalty += penalty
        self.done = terminated or truncated
        info = {**self._info(start), **diagnostics, "market_trades": trades, "maker_fills": fills,
            "signed_flow_contracts": self.last_flow, "window_innovation_logodds": innovation,
            "equity_change": after-before, "risk_penalty": penalty, "reward": reward,
            "phase_mark_fallback": bool(window.window["phase_mark_fallback"]),
            **{k: float(v) if isinstance(v, Decimal) else v for k, v in settlement.items()},
            "arm": path.arm.name, "snipe_fills": snipes, "jumps_applied": jumps, "decision_stop": decision_stop}
        self.steps.append({"step": self.step_number, "elapsed_seconds": self.elapsed,
                           "terminated": terminated, "truncated": truncated, **info, **self.observation()})
        return self.observation(), reward, terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _flow_fields(self, j):
        path = self.path
        law = FLOW_LAW[path.arm.flow]
        fields = {"flow_law": law, "flow_u": float(path.exec_u[j])}
        if law in {"fg", "constant", "fg_half"}:
            fields.update(flow_alpha=float(path.exec_alpha[j]), flow_one_minus_alpha=float(path.exec_one_minus_alpha[j]))
        fields.update({f"flow_{k}": v for k, v in path.flow_parameters.items()})
        return fields

    def _signed_flow(self, executed):
        if not executed:
            return 0.
        idx = np.asarray(executed)
        return float(np.dot(self.path.skeleton.exec_quantity[idx], self.path.exec_sign[idx]))

    def walk_hashes(self) -> dict:
        """V8 hashes of what this walk saw: public-node L trace, books, settlement."""
        signal = [t[:3] for t in self.trace]
        books = [(self.path.initial_book.bid, self.path.initial_book.ask)]+[t[3:] for t in self.trace if len(t) == 5]
        settled = [r for r in self.ledger if r["kind"] == "settlement"]
        if len(settled) != 1 or int(settled[0]["settlement_outcome"]) != self.path.settlement["outcome"]:
            raise AssertionError("Walk did not reach the path's settlement")
        return {"signal": _sha(signal), "book": _sha(books), "settlement": _sha(self.path.settlement)}


def _saturated(log_odds: float) -> bool:
    """Exactly `ImpactPriceProcess.advance`'s float_probability_saturated flag."""
    return bool(expit(log_odds) in (0.0, 1.0))


# ------------------------------------------------------------------ runners
def run_policy(env, policy: str, hjb=None, *, seed=None, options=None):
    """Run one episode exactly as `evaluate_hjb.run` calls each policy; returns per-step records."""
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    cfg = env.config
    obs, initial_info = env.reset(seed=seed, options=options)
    records = [{"observation": obs, "info": initial_info}]
    actions = []
    while not env.done:
        if policy == "hjb":
            action, diagnostics = hjb.action(obs, start_elapsed=env.start_elapsed, lifetime=env.lifetime,
                inventory_limit=cfg.inventory_limit, tick=cfg.price_tick, lot=cfg.quantity_lot)
            actions.append({"elapsed_seconds": env.elapsed, **diagnostics})
        else:
            action = QuoteAction() if policy == "none" else quote_policy(obs, size=hjb.parameters.delta,
                inventory_skew=.0005 if policy == "skew" else 0., tick=cfg.price_tick)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append({"action": action, "observation": obs, "reward": reward, "terminated": terminated,
                        "truncated": truncated, "info": info})
    if not env.account.settled:
        raise AssertionError("Episode must reach genuine settlement")
    return records, actions


def _strip(mapping, keys):
    return {k: v for k, v in mapping.items() if k not in keys}


def golden_episode_check(skeleton, policy: str, hjb, *, fill: str = "base") -> dict:
    """V7 for one (template, seed, policy): golden walk == TradingEnvironment, byte for byte (repr)."""
    from .paths import build_path, make_arm
    queue, participation = FILL_SCENARIOS[fill]
    sk = skeleton
    cfg = EnvironmentConfig(group=sk.group, ticker=sk.ticker, impact_scenario=sk.config.impact_scenario,
                            queue_ahead_contracts=queue, participation=participation)
    legacy = TradingEnvironment(cfg, sk.dynamics_path, sk.execution_path)
    options = {"seconds_to_close": sk.seconds}
    legacy_records, legacy_actions = run_policy(legacy, policy, hjb, seed=sk.path_seed, options=options)
    path = build_path(sk, make_arm("golden", p0=cfg.initial_probability))
    env = InformationTradingEnvironment(path, cfg)
    records, actions = run_policy(env, policy, hjb, seed=sk.path_seed, options=options)
    if len(records) != len(legacy_records):
        raise AssertionError("Different number of steps")
    for new, old in zip(records, legacy_records):
        if repr(_strip(new["observation"], EXTRA_OBSERVATION_KEYS)) != repr(old["observation"]):
            raise AssertionError("Observation differs")
        if repr(_strip(new["info"], EXTRA_INFO_KEYS)) != repr(old["info"]):
            raise AssertionError("Info differs")
        for key in ("action", "reward", "terminated", "truncated"):
            if repr(new.get(key)) != repr(old.get(key)):
                raise AssertionError(f"{key} differs")
    if repr(env.ledger) != repr(legacy.ledger):
        raise AssertionError("Ledger differs")
    stripped = [_strip(s, EXTRA_OBSERVATION_KEYS+EXTRA_INFO_KEYS) for s in env.steps]
    if repr(stripped) != repr(legacy.steps):
        raise AssertionError("Step records differ")
    if repr(actions) != repr(legacy_actions):
        raise AssertionError("HJB action diagnostics differ")
    if (env.account.cash, env.account.inventory, env.total_reward) != (
            legacy.account.cash, legacy.account.inventory, legacy.total_reward):
        raise AssertionError("Terminal account differs")
    return {"ticker": sk.ticker, "path_seed": sk.path_seed, "policy": policy, "steps": len(records)-1,
            "ledger_rows": len(env.ledger), "realized_pnl": float(env.account.cash-env.account.initial_cash)}


def golden_equivalence_sweep(calibration, *, seeds=(0, 1), policies=POLICIES, tickers=None, hjb=None) -> list[dict]:
    """Full V7 sweep: all 35 templates x seeds (replicate indices) x 4 policies."""
    from ..hjb_policy import HJBReferencePolicy
    from .skeleton import build_skeleton, event_roster
    hjb = hjb or HJBReferencePolicy()
    results = []
    for group, ticker, _ in event_roster():
        if tickers is not None and ticker not in tickers:
            continue
        for replicate in seeds:
            sk = build_skeleton(ticker, replicate, calibration=calibration)
            for policy in policies:
                results.append(golden_episode_check(sk, policy, hjb))
    return results


ANCHOR_COLUMNS = ("seed", "realized_pnl", "cash", "terminal_inventory", "settlement_outcome", "closing_probability",
                  "market_trades", "maker_fills", "maker_contracts", "fees", "spread_capture", "background_price_pnl",
                  "flow_price_pnl", "settlement_surprise_pnl", "max_drawdown", "hjb_decisions")


def l0_anchor_episode(group: str, ticker: str, event_index: int, replicate: int, policy: str, hjb, *,
                      seed: int = 20261009, impact: str = "base", fill: str = "base", hours_to_close: float = 1.) -> dict:
    """One L0 episode with `TradingEnvironment` itself and Section 5 seeding (episode_results.csv columns)."""
    from .skeleton import path_seed
    from ..market_dynamics import DEFAULT_CONFIG
    queue, participation = FILL_SCENARIOS[fill]
    pseed = path_seed(event_index, replicate, seed)
    lifetime = json.loads(DEFAULT_CONFIG.read_text())["event_templates"][ticker]["lifetime_seconds"]
    seconds = min(hours_to_close*3600, lifetime)
    cfg = EnvironmentConfig(group=group, ticker=ticker, impact_scenario=impact, queue_ahead_contracts=queue,
                            participation=participation)
    env = TradingEnvironment(cfg)
    records, actions = run_policy(env, policy, hjb, seed=pseed, options={"seconds_to_close": seconds})
    obs = records[0]["observation"]
    audit_ledger(env.ledger, initial_cash=cfg.initial_cash, initial_log_odds=obs["log_odds"],
                 beta=env.dynamics.impact.beta, inventory_limit=cfg.inventory_limit,
                 maker_fee_multiplier=cfg.maker_fee_multiplier, balance_precision=cfg.balance_precision,
                 quantity_lot=cfg.quantity_lot, participation=cfg.participation)
    attribution = episode_diagnostics(env.ledger, cfg.initial_cash, obs["probability"], env.start_elapsed, env.elapsed)
    info = records[-1]["info"]
    return {"seed": pseed, "realized_pnl": float(env.account.cash-env.account.initial_cash),
            "cash": float(env.account.cash), "terminal_inventory": float(info["terminal_inventory"]),
            "settlement_outcome": info["settlement_outcome"], "closing_probability": env.price.probability,
            "market_trades": sum(r["kind"] == "market" for r in env.ledger), "hjb_decisions": len(actions),
            **{k: attribution[k] for k in ("maker_fills", "maker_contracts", "fees", "spread_capture",
                                            "background_price_pnl", "flow_price_pnl", "settlement_surprise_pnl",
                                            "max_drawdown")}}


def l0_anchor_check(saved_results, *, hjb=None, tickers=None, replicates=None, policies=POLICIES,
                    tolerance: float = 1e-9) -> dict:
    """Reproduce saved hjb_benchmark episode_results rows (impact base, fill base) to `tolerance`."""
    import pandas as pd
    from ..hjb_policy import HJBReferencePolicy
    from .skeleton import event_roster
    hjb = hjb or HJBReferencePolicy()
    saved = pd.read_csv(saved_results) if not isinstance(saved_results, pd.DataFrame) else saved_results
    saved = saved[saved.impact_scenario.eq("base") & saved.fill_scenario.eq("base")].set_index(
        ["ticker", "replicate", "policy"])
    checked, worst = 0, 0.
    for group, ticker, index in event_roster():
        if tickers is not None and ticker not in tickers:
            continue
        reps = replicates if replicates is not None else sorted(saved.loc[ticker].index.get_level_values(0).unique())
        for replicate in reps:
            for policy in policies:
                row = saved.loc[(ticker, replicate, policy)]
                got = l0_anchor_episode(group, ticker, index, replicate, policy, hjb)
                for column in ANCHOR_COLUMNS:
                    diff = abs(float(got[column])-float(row[column]))
                    worst = max(worst, diff)
                    if diff > tolerance:
                        raise AssertionError(f"L0 anchor mismatch {ticker} r{replicate} {policy} {column}: "
                                             f"{got[column]} vs {row[column]}")
                checked += 1
    return {"episodes": checked, "max_abs_difference": worst, "tolerance": tolerance}
