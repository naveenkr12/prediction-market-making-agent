"""Additive PnL attribution, nonadditive diagnostics and the independent ledger auditor (Section 6.5).

Additive identity (reconciles within 1e-8 for every episode):

    PnL = sum_{fills incl. snipes} dq (p - P) - fees + sum_{price updates} q_- dp + q_c (Y - p_c)

with the price-update term split by channel: background (`noise` rows), flow
(`market` rows) and jump (`jump` rows). p is the ledger's displayed
probability, so the identity telescopes exactly; any price change on another
row kind is an error.

The auditor reconstructs cash, inventory, fees and collateral from the ledger
with Decimal arithmetic, recomputes every flow update once from its logged law
(LLRs via the independent `reference.py` calculator, not `kernels.py`), every
jump once, and checks that snipes have no price update, were crossed at the
logged unclipped p under the taker-fee rule, filled the full remaining order
and are not counted as market flow.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np

from . import reference as R

CHANNELS = {"noise": "background", "market": "flow", "jump": "jump"}
PRICE_NEUTRAL = {"quote", "cancel", "fill", "snipe", "settlement"}
TTR_EDGES = (3600.0, 10800.0, 86400.0)


def attribution(ledger, initial_cash, initial_probability, tolerance: float = 1e-8) -> dict:
    """Additive identity with channel split; raises if it does not reconcile within `tolerance`."""
    if not ledger:
        raise ValueError("Empty ledger")
    prev_p, prev_q = float(initial_probability), 0.
    capture = {"fill": 0., "snipe": 0.}
    price = {"background": 0., "flow": 0., "jump": 0.}
    fees = settlement = 0.
    for row in ledger:
        kind = row["kind"]
        p, q = float(row["probability"]), float(row["inventory"])
        if kind in CHANNELS:
            price[CHANNELS[kind]] += prev_q*(p-prev_p)
        elif kind in PRICE_NEUTRAL:
            if p != prev_p:
                raise AssertionError(f"Price moved on a {kind} row")
        else:
            raise AssertionError(f"Unknown ledger kind {kind}")
        if kind in ("fill", "snipe"):
            capture[kind] += float(row["inventory_change"])*(p-float(row["price"]))
            fees += float(row["fee"])
        elif kind == "settlement":
            settlement += float(row["terminal_inventory"])*(int(row["settlement_outcome"])-p)
        prev_p, prev_q = p, q
    pnl = float(ledger[-1]["cash"])+prev_q*prev_p-float(initial_cash)
    attributed = capture["fill"]+capture["snipe"]-fees+sum(price.values())+settlement
    error = pnl-attributed
    if abs(error) > tolerance:
        raise AssertionError(f"Additive attribution does not reconcile: {error}")
    return {"pnl": pnl, "spread_capture_fills": capture["fill"], "spread_capture_snipes": capture["snipe"],
            "fees": fees, "background_price_pnl": price["background"], "flow_price_pnl": price["flow"],
            "jump_price_pnl": price["jump"], "settlement_surprise_pnl": settlement, "attribution_error": error}


def _ttr_bucket(ttr: float) -> str:
    return "B"+str(int(np.searchsorted(TTR_EDGES, ttr, side="right")))


def diagnostics(ledger, *, initial_probability, lifetime, release_minus_close_seconds,
                markout_seconds: float = 300., jump_window_seconds: float = 60.) -> dict:
    """Nonadditive diagnostics (reported separately from the additive identity)."""
    rows = list(ledger)
    prev_p = float(initial_probability)
    rv = {c: 0. for c in CHANNELS.values()}
    rv_bucket: dict = {}
    flow_markout = jump_markout = markout_5m = 0.
    snipe_count, snipe_volume, snipe_pnl = 0, 0., 0.
    exogenous_volume = maker_exogenous_volume = 0.
    fills = []                      # (elapsed, dq, p_at_fill)
    last_fill_dq = 0.
    times, probs = [], []
    p_close = q_close = None
    for row in rows:
        kind, p, t = row["kind"], float(row["probability"]), float(row["elapsed_seconds"])
        dp = p-prev_p
        if kind in CHANNELS:
            channel = CHANNELS[kind]
            rv[channel] += dp*dp
            bucket = _ttr_bucket(release_minus_close_seconds+lifetime-t)
            rv_bucket[f"{channel}_{bucket}"] = rv_bucket.get(f"{channel}_{bucket}", 0.)+dp*dp
        if kind == "fill":
            dq = float(row["inventory_change"])
            fills.append((t, dq, p))
            last_fill_dq = dq
            maker_exogenous_volume += abs(dq)
        elif kind == "market":
            flow_markout += last_fill_dq*dp
            last_fill_dq = 0.
            exogenous_volume += float(row["market_quantity"])
        elif kind == "snipe":
            dq = float(row["inventory_change"])
            fills.append((t, dq, p))
            snipe_count += 1
            snipe_volume += abs(dq)
            snipe_pnl += dq*(p-float(row["price"]))
        elif kind == "jump":
            jump_markout += sum(dq for (s, dq, _) in fills if t-jump_window_seconds <= s < t)*dp
        elif kind == "settlement":
            p_close, q_close = p, float(row["terminal_inventory"])
        if kind not in ("fill", "market"):
            last_fill_dq = 0.
        if kind != "settlement":
            times.append(t)
            probs.append(p)
        prev_p = p
    times_a, probs_a = np.asarray(times), np.asarray(probs)
    for t, dq, p in fills:
        k = int(np.searchsorted(times_a, t+markout_seconds, side="right"))-1
        markout_5m += dq*(float(probs_a[k])-p)
    result = {"flow_markout": flow_markout, "jump_markout": jump_markout, "markout_5m": markout_5m,
              "snipe_count": snipe_count, "snipe_volume": snipe_volume, "snipe_pnl": snipe_pnl,
              "exogenous_volume": exogenous_volume, "maker_exogenous_fill_volume": maker_exogenous_volume,
              "sniper_volume": snipe_volume, **{f"realized_variance_{k}": v for k, v in rv.items()},
              **{f"realized_variance_{k}": v for k, v in sorted(rv_bucket.items())}}
    if p_close is not None:
        result.update(p_close_variance=p_close*(1-p_close), terminal_inventory_risk=q_close**2*p_close*(1-p_close))
    return result


# ----------------------------------------------------------------- auditor
def _maker_fee(price, quantity, side, multiplier, grid, accumulator):
    revenue = quantity*price*(-1 if side == "bid" else 1)
    model = Decimal(".0175")*Decimal(str(multiplier))*quantity*price*(1-price)
    trade = model.quantize(Decimal(".000001"), rounding=ROUND_CEILING)
    aligned = ((revenue-trade)/grid).to_integral_value(rounding=ROUND_FLOOR)*grid
    rounding = revenue-trade-aligned
    accumulator += rounding
    rebate = min((accumulator/grid).to_integral_value(rounding=ROUND_FLOOR)*grid,
                 ((trade+rounding)/grid).to_integral_value(rounding=ROUND_FLOOR)*grid)
    accumulator -= rebate
    return revenue, trade, rounding, rebate, trade+rounding-rebate, accumulator


def _expected_flow_increment(row, beta):
    law = row.get("flow_law", "fl")
    sign, size = int(Decimal(str(row["market_sign"]))), float(row["market_quantity"])
    if law == "fl":
        b = float(row["flow_beta"]) if "flow_beta" in row else beta
        if b is None:
            raise ValueError("Fixed-impact (FL) rows need beta")
        return b*(sign*size), 1e-10
    signal = {"kind": "flow", "law": law, "sign": sign, "size": size}
    if law == "fg":
        signal.update(u=float(row["flow_u"]), eta=float(row["flow_eta"]))
    elif law == "constant":
        signal.update(u=float(row["flow_u"]), alpha=float(row["flow_alpha0"]))
    elif law == "fg_half":
        signal.update(beta=float(row["flow_beta"]))
    one, zero = R.class_log_likelihoods(signal)
    return one-zero, 1e-12


def audit_information_ledger(rows, *, initial_cash, initial_log_odds, inventory_limit, maker_fee_multiplier,
                             balance_precision, quantity_lot, participation, tick, beta=None,
                             taker_fee_coefficient: float = 0.07, expected_executions: int | None = None,
                             expected_jumps: int | None = None) -> dict:
    cash, q, latent = Decimal(str(initial_cash)), Decimal(0), float(initial_log_odds)
    grid, lot, tick_d = Decimal(balance_precision), Decimal(quantity_lot), Decimal(str(tick))
    orders, accumulators = {}, {}
    fills = snipes = trades = jumps = settlements = 0
    total_fees = Decimal(0)
    previous = None
    for index, row in enumerate(rows):
        kind = row["kind"]
        if kind == "quote":
            key = str(row["order_id"])
            orders[key] = Decimal(str(row["quantity"]))
            accumulators[key] = Decimal(0)
        elif kind == "cancel":
            orders.pop(str(row["order_id"]), None)
        elif kind == "noise":
            latent += float(row["innovation_logodds"])
        elif kind == "market":
            trades += 1
            volume, own = float(row["market_quantity"]), float(Decimal(str(row["maker_quantity"])))
            sign = int(Decimal(str(row["market_sign"])))
            assert sign in (-1, 1) and 0 <= own <= volume*participation+1e-10
            if own:
                assert previous["kind"] == "fill" and float(previous["quantity"]) == own
                assert previous["side"] == ("ask" if sign == 1 else "bid")
            expected, tol = _expected_flow_increment(row, beta)
            assert abs(float(row["impact_logodds"])-expected) < tol, f"Flow update mismatch at row {index}"
            assert abs(float(row["log_odds"])-float(row["log_odds_before"])-float(row["impact_logodds"])) < 1e-12
            latent += float(row["impact_logodds"])
        elif kind == "jump":
            jumps += 1
            sign, a = int(row["jump_sign"]), float(row["jump_magnitude"])
            assert sign in (-1, 1) and a >= 0 and float(row["jump_logodds"]) == sign*a
            assert abs(float(row["log_odds"])-float(row["log_odds_before"])-sign*a) < 1e-12
            latent += sign*a
        elif kind in ("fill", "snipe"):
            key = str(row["order_id"])
            quantity, price = Decimal(str(row["quantity"])), Decimal(str(row["price"]))
            assert 0 < quantity <= orders[key] and quantity % lot == 0
            if kind == "snipe":
                snipes += 1
                assert quantity == orders[key], "A snipe must fill the full remaining order"
                assert previous is not None and float(previous["log_odds"]) == float(row["log_odds"]), \
                    "Snipes must not update the price"
                assert float(previous["elapsed_seconds"]) == float(row["elapsed_seconds"])
                p = float(row["latent_probability"])
                log_odds = float(row["log_odds"])
                logistic = 1/(1+math.exp(-log_odds)) if log_odds >= 0 else math.exp(log_odds)/(1+math.exp(log_odds))
                assert abs(p-logistic) < 1e-15, "Snipe crossing must use the node's unclipped p"
                raw = Decimal(str(taker_fee_coefficient))*quantity*price*(1-price)
                fee_cents = (raw/Decimal("0.01")).to_integral_value(rounding=ROUND_CEILING)*Decimal("0.01")
                assert Decimal(str(row["taker_fee"])) == fee_cents
                qf = float(quantity)
                edge = qf*(p-float(price)) if row["side"] == "ask" else qf*(float(price)-p)
                assert edge > float(fee_cents)+qf*float(tick_d), f"Snipe at row {index} was not crossed"
            else:
                fills += 1
                limit = Decimal(str(row["execution_limit"]))
                assert price >= limit if row["side"] == "bid" else price <= limit
            orders[key] -= quantity
            revenue, trade, rounding, rebate, fee, accumulators[key] = _maker_fee(
                price, quantity, row["side"], maker_fee_multiplier, grid, accumulators[key])
            assert Decimal(str(row["fee"])) == fee and fee >= 0
            assert Decimal(str(row["trade_fee"])) == trade and Decimal(str(row["rebate"])) == rebate
            assert Decimal(str(row["rounding_fee"])) == rounding
            assert Decimal(str(row["cash_change"])) == revenue-fee
            assert Decimal(str(row["cash_before"])) == cash and Decimal(str(row["inventory_before"])) == q
            total_fees += fee
            cash += revenue-fee
            q += quantity*(1 if row["side"] == "bid" else -1)
        elif kind == "settlement":
            settlements += 1
            outcome = int(Decimal(str(row["settlement_outcome"])))
            assert settlements == 1 and outcome in (0, 1)
            assert Decimal(str(row["settlement_adjustment"])) == q*outcome
            assert Decimal(str(row["terminal_inventory"])) == q
            cash += q*outcome
            q = Decimal(0)
        else:
            raise AssertionError(f"Unknown ledger event: {kind}")
        assert Decimal(str(row["cash"])) == cash, f"Cash mismatch at row {index}"
        assert Decimal(str(row["inventory"])) == q, f"Inventory mismatch at row {index}"
        collateral = max(-q, Decimal(0)) if settlements == 0 else Decimal(0)
        assert Decimal(str(row["collateral"])) == collateral, f"Collateral mismatch at row {index}"
        assert cash >= max(-q, Decimal(0)) and abs(q) <= Decimal(str(inventory_limit))
        assert abs(float(row["log_odds"])-latent) < 1e-8, f"Latent price mismatch at row {index}"
        previous = row
    if expected_executions is not None:
        assert trades == expected_executions, "Each execution's flow update must appear exactly once"
    if expected_jumps is not None:
        assert jumps == expected_jumps, "Each jump must appear exactly once"
    return {"passed": True, "ledger_rows": len(rows), "market_trades": trades, "maker_fills": fills,
            "snipes": snipes, "jumps": jumps, "settlements": settlements, "terminal_cash": str(cash),
            "terminal_inventory": str(q), "fees": str(total_fees), "terminal_log_odds": latent}


# ------------------------------------------- per-arm path diagnostics (P5)
from scipy.special import expit  # noqa: E402  (P5 additions; earlier code is unchanged)


def flow_moments(path) -> dict:
    """Sign/size moments of one (episode, arm) path (policy-independent).

    `informed_fraction` is a hidden diagnostic (M FG arms only) and is never
    observed by policies.
    """
    sk = path.skeleton
    sign, size = np.asarray(path.exec_sign, dtype=float), np.asarray(sk.exec_quantity, dtype=float)
    n = len(sign)
    window_flow = np.abs(np.asarray(path.window_signed_flow, dtype=float))
    active = np.asarray(path.jump_active, dtype=bool)
    jumps = path.jump_applied and bool(active.any())
    if jumps and not active.all():      # thinned dose: statistics over the applied jumps only
        jump_sign, jump_magnitude, jump_step = path.jump_sign[active], sk.jump_magnitude[active], path.jump_step[active]
    else:
        jump_sign, jump_magnitude, jump_step = path.jump_sign, sk.jump_magnitude, path.jump_step
    return {"trades": n, "buy_fraction": float(np.mean(sign == 1)) if n else np.nan,
            "mean_size": float(size.mean()) if n else np.nan,
            "mean_signed_contracts_per_trade": float(np.mean(sign*size)) if n else np.nan,
            "signed_contracts": float(np.sum(sign*size)),
            "informed_fraction": float(np.mean(path.exec_informed)) if n and path.arm.flow in {"fg", "fg_constant"} else np.nan,
            "mean_abs_window_flow": float(window_flow.mean()) if len(window_flow) else np.nan,
            "jumps_applied": int(len(jump_sign)) if jumps else 0,
            "mean_jump_magnitude": float(np.mean(jump_magnitude)) if jumps else np.nan,
            "up_jump_fraction": float(np.mean(jump_sign == 1)) if jumps else np.nan,
            "flow_llr_sum": float(np.sum(path.exec_increment)), "jump_llr_sum": float(np.sum(jump_step)) if jumps else 0.0}


def _pair_variance(p, up_prob, l_up, l_down):
    """E[(Delta p)^2] of a two-point move from p (unclipped)."""
    p_up, p_down = expit(l_up), expit(l_down)
    return up_prob*(p_up-p)**2+(1-up_prob)*(p_down-p)**2


def information_variance(path, calibration=None) -> dict:
    """Current-state (predictive) information variance per channel, summed over the episode.

    flow: per execution E[(Delta p)^2 | p before the trade] under the arm's public sign law
    (M FG: P(buy | p) = (1-alpha)u + alpha p with the GM LLRs; F0: 0; FL: P(buy) = u, +-beta c;
    FG-half: P(buy) = (1-alpha)/2 + alpha p, +-beta c); exact two-point formula.
    jump: sum over node intervals of h * lambda_J(TTR) * E_a[E[(Delta p)^2 | p]] (bank average;
    M and Theta-free signs: q(p, a); legacy-law: 1/2) — only for arms with jumps on.
    background: Ito approximation sigma^2 h p^2 (1-p)^2 at the interval's left value.
    """
    from . import kernels as K
    from .skeleton import EXECUTION
    sk, arm = path.skeleton, path.arm
    l_bg, l_after = np.asarray(path.l_bg), np.asarray(path.l_after)
    l_prev = np.r_[path.initial_log_odds, l_after[:-1]]
    p_prev = expit(l_prev)
    flow = 0.0
    exec_nodes = np.flatnonzero(np.asarray(path.kinds) == EXECUTION)
    if len(exec_nodes) and arm.flow not in {"f0", "bank_fl"}:
        j = np.asarray(path.refs)[exec_nodes]
        L = l_bg[exec_nodes]
        p = expit(L)
        c = sk.exec_quantity[j]
        if arm.flow in {"fg", "fg_constant"}:
            a, om, u = path.exec_alpha[j], path.exec_one_minus_alpha[j], path.exec_u[j]
            up = om*u+a*p
            inc_up = K.gm_llr(np.ones(len(j), np.int8), a, om, u)
            inc_down = K.gm_llr(-np.ones(len(j), np.int8), a, om, u)
        elif arm.flow == "su_fl":
            up = path.exec_u[j]
            inc_up, inc_down = path.beta*c, -path.beta*c
        else:   # fg_half_theta_free
            a, om = path.exec_alpha[j], path.exec_one_minus_alpha[j]
            up = om*0.5+a*p
            inc_up, inc_down = path.beta*c, -path.beta*c
        flow = float(np.sum(_pair_variance(p, up, L+inc_up, L+inc_down)))
    elif arm.flow == "bank_fl":
        flow = np.nan
    h = sk.node_h if not arm.legacy_nodes else np.diff(np.r_[sk.start, np.asarray(path.elapsed)])
    sigma2 = path.sigma2
    background = float(np.sum(sigma2*h*(p_prev*(1-p_prev))**2))
    jump = 0.0
    if path.jump_applied:
        bank = calibration.magnitude_bank(sk.information_class) if calibration is not None else sk.jump_magnitude
        if len(bank):
            elapsed = np.asarray(path.elapsed)
            ttr = sk.ttr_at_close+sk.lifetime-(elapsed-h/2)
            bucket = K.ttr_bucket_index(ttr)
            # Thinned dose: intensity = keep probability x skeleton rates (1.0 on every other path).
            rates = np.array([sk.jump_rates[b] for b in K.TTR_BUCKETS])*path.jump_keep_probability
            lam = np.where(bucket >= 0, rates[np.maximum(bucket, 0)], 0.0)
            a = np.asarray(bank, dtype=float)[None, :]
            pp = p_prev[:, None]
            if arm.jumps == "legacy":
                up = np.full_like(pp*a, 0.5)
            else:
                up = pp*expit(a)+(1-pp)*expit(-a)
            per = _pair_variance(pp, up, l_prev[:, None]+a, l_prev[:, None]-a).mean(axis=1)
            jump = float(np.sum(h*lam*per))
    return {"info_variance_flow": flow, "info_variance_jump": jump, "info_variance_background": background}


def kyle_windows(path) -> tuple[np.ndarray, np.ndarray]:
    """Per legacy 300-s window with >= 1 execution: F = sum sign*c and Delta L = L after the last
    trade - L after the first trade (the post-trade observation convention of the flow replay and
    of `calibrate_price_impact`'s first-to-last transaction endpoints)."""
    from .skeleton import EXECUTION
    sk = path.skeleton
    kinds, refs, l_after = path.kinds, path.refs, path.l_after
    F, dL = [], []
    for k, w in enumerate(sk.windows):
        sl = path.window_node_slices[k]
        nodes = [i for i in range(sl.start, sl.stop) if kinds[i] == EXECUTION]
        if not nodes:
            continue
        j = np.asarray([refs[i] for i in nodes])
        F.append(float(np.dot(path.exec_sign[j], sk.exec_quantity[j])))
        dL.append(float(l_after[nodes[-1]]-l_after[nodes[0]]))
    return np.asarray(F), np.asarray(dL)


def ols_slope(F, dL) -> float:
    """OLS slope of Delta L on F with intercept; degenerate design (Sxx <= 1e-24) -> 0 (replay convention)."""
    F, dL = np.asarray(F, dtype=float), np.asarray(dL, dtype=float)
    if len(F) < 2:
        return 0.0
    fc = F-F.mean()
    sxx = float(np.dot(fc, fc))
    return 0.0 if sxx <= 1e-24 else float(np.dot(fc, dL-dL.mean())/sxx)
