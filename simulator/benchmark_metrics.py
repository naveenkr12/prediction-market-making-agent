"""Episode accounting attribution and dependence-aware scenario statistics."""
from __future__ import annotations
import numpy as np


def expected_shortfall(losses, tail_probability=.05, axis=-1):
    """Mean of exactly the worst tail mass, fractional boundary weight included.

    Including all observations tied at the VaR threshold would over-count the
    tail when many quiet episodes have zero PnL. Do not clip signed losses.
    """
    data = np.asarray(losses,dtype=float)
    if data.size==0 or not np.isfinite(data).all() or not 0<tail_probability<=1:
        raise ValueError("Invalid tail distribution")
    data = np.sort(data,axis=axis)
    data = np.flip(data,axis=axis)
    n = data.shape[axis]
    mass = tail_probability*n
    full = int(np.floor(mass+1e-12))
    fraction = max(0.,mass-full)
    result = np.take(data,np.arange(full),axis=axis).sum(axis=axis)
    if fraction>1e-12:
        result += fraction*np.take(data,full,axis=axis)
    return result/mass


def pnl_statistics(pnl):
    pnl = np.asarray(pnl,dtype=float)
    if pnl.ndim!=1 or len(pnl)==0 or not np.isfinite(pnl).all():
        raise ValueError("Need a finite nonempty episode PnL vector")
    return {"episodes":len(pnl),"mean_pnl":float(pnl.mean()),"median_pnl":float(np.median(pnl)),
        "std_pnl":float(pnl.std(ddof=1)) if len(pnl)>1 else 0.,
        "pnl_p05":float(np.quantile(pnl,.05,method="linear")),
        "pnl_p95":float(np.quantile(pnl,.95,method="linear")),
        "var95_loss":float(np.quantile(-pnl,.95,method="linear")),
        "es95_loss":float(expected_shortfall(-pnl)),"tail_mass_episodes":.05*len(pnl),
        "loss_probability":float(np.mean(pnl<0)),"zero_pnl_probability":float(np.mean(pnl==0)),
        "worst_pnl":float(pnl.min()),"best_pnl":float(pnl.max())}


def episode_diagnostics(ledger, initial_cash, initial_probability, start_elapsed, end_elapsed):
    if not ledger or end_elapsed<=start_elapsed:
        raise ValueError("Need an episode ledger and a positive time horizon")
    previous_p, previous_q = float(initial_probability),0.
    previous_time, peak = float(start_elapsed),float(initial_cash)
    spread_capture = background_pnl = flow_pnl = settlement_pnl = 0.
    markout_cost = fees = q2_time = drawdown = max_q = 0.
    fills = contracts = 0
    maker_delta_at_trade = 0.
    for row in ledger:
        elapsed, q, price = float(row["elapsed_seconds"]),float(row["inventory"]),float(row["probability"])
        if elapsed<previous_time-1e-7:
            raise ValueError("Ledger time goes backward")
        q2_time += previous_q**2*max(0.,elapsed-previous_time)
        kind = row["kind"]
        if kind=="fill":
            delta = float(row["inventory_change"])
            spread_capture += delta*(price-float(row["price"]))
            fees += float(row["fee"])
            fills += 1
            contracts += float(row["quantity"])
            maker_delta_at_trade = delta
        elif kind=="noise":
            background_pnl += previous_q*(price-previous_p)
            maker_delta_at_trade = 0.
        elif kind=="market":
            flow_pnl += previous_q*(price-previous_p)
            markout_cost -= maker_delta_at_trade*(price-previous_p)
            maker_delta_at_trade = 0.
        elif kind=="settlement":
            settlement_pnl += float(row["terminal_inventory"])*(int(row["settlement_outcome"])-price)
        equity = float(row["cash"])+q*price
        peak = max(peak,equity)
        drawdown = max(drawdown,peak-equity)
        max_q = max(max_q,abs(q))
        previous_p,previous_q,previous_time = price,q,elapsed
    final_pnl = float(ledger[-1]["cash"])+previous_q*previous_p-initial_cash
    attributed = spread_capture-fees+background_pnl+flow_pnl+settlement_pnl
    if abs(final_pnl-attributed)>1e-8:
        raise AssertionError("Episode PnL attribution does not reconcile")
    return {"maker_fills":fills,"maker_contracts":contracts,"fees":fees,
        "spread_capture":spread_capture,"background_price_pnl":background_pnl,"flow_price_pnl":flow_pnl,
        "settlement_surprise_pnl":settlement_pnl,"immediate_market_impact_markout_cost":markout_cost,
        "max_drawdown":drawdown,"max_abs_inventory":max_q,
        "inventory_rms":float(np.sqrt(q2_time/(end_elapsed-start_elapsed))),
        "pnl_attribution_error":final_pnl-attributed}


def balanced_cube(frame, column):
    pivot = frame.pivot(index="ticker",columns="replicate",values=column).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Event bootstrap requires a balanced, paired replicate panel")
    return pivot.to_numpy(dtype=float)


def hierarchical_samples(cube, rng, draws):
    """Resample event templates, then paired replicates within each event."""
    cube = np.asarray(cube,dtype=float)
    if cube.ndim!=2 or not np.isfinite(cube).all() or draws<20:
        raise ValueError("Invalid bootstrap panel or draw count")
    k,r = cube.shape
    events = rng.integers(0,k,size=(draws,k))
    replicates = rng.integers(0,r,size=(draws,k,r))
    return cube[events[:,:,None],replicates].reshape(draws,k*r)


def summarize_cell(frame, rng, draws=1000):
    pnl = frame.realized_pnl.to_numpy()
    result = pnl_statistics(pnl)
    cube = balanced_cube(frame,"realized_pnl")
    boot = hierarchical_samples(cube,rng,draws)
    for name,values in (("mean_pnl",boot.mean(axis=1)),("var95_loss",np.quantile(-boot,.95,axis=1,method="linear")),
                        ("es95_loss",expected_shortfall(-boot,axis=1))):
        lower,upper = np.quantile(values,[.025,.975],method="linear")
        result[name+"_bootstrap95_low"],result[name+"_bootstrap95_high"] = float(lower),float(upper)
    k,r = cube.shape
    result["training_events"],result["replicates_per_event"] = k,r
    # MC error conditional on these fixed event templates, distinct from the
    # resampled-event range and from calibration/model uncertainty.
    result["mean_pnl_conditional_mc_se"] = float(np.sqrt(np.sum(cube.var(axis=1,ddof=1)/r)/k**2)) if r>1 else 0.
    for column in ("fees","maker_fills","maker_contracts","max_drawdown","inventory_rms",
                   "max_abs_inventory","abs_terminal_inventory","spread_capture",
                   "background_price_pnl","flow_price_pnl","settlement_surprise_pnl","immediate_market_impact_markout_cost"):
        result["mean_"+column] = float(frame[column].mean())
    result["p95_max_drawdown"] = float(np.quantile(frame.max_drawdown,.95))
    result["no_fill_probability"] = float(frame.maker_fills.eq(0).mean())
    result["inventory_limit_probability"] = float(frame.inventory_limit_reached.mean())
    result["flat_at_settlement_probability"] = float(frame.abs_terminal_inventory.lt(.005).mean())
    result["mean_return_on_initial_cash_pct"] = 100*float((frame.realized_pnl/frame.initial_cash).mean())
    result["hjb_post_only_projection_rate"] = float(frame.hjb_post_only_projections.sum()/max(1,frame.hjb_decisions.sum()))
    result["hjb_probability_clip_rate"] = float(frame.hjb_probability_clips.sum()/max(1,frame.hjb_decisions.sum()))
    result["noise_extrapolated_episode_fraction"] = float(frame.volatility_extrapolated_any.mean())
    result["spread_extrapolated_episode_fraction"] = float(frame.spread_extrapolated_any.mean())
    return result
