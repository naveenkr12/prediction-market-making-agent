import sys
from types import ModuleType
import numpy as np
import pandas as pd
import pytest

from simulator.benchmark_metrics import expected_shortfall,pnl_statistics,episode_diagnostics,hierarchical_samples
from simulator.hjb_policy import HJBReferencePolicy,legacy_modules,executable_action
from simulator.trading_env import EnvironmentConfig,TradingEnvironment
from simulator.evaluate_hjb import cached_environment


@pytest.fixture(scope="module")
def hjb(): return HJBReferencePolicy()


def test_private_legacy_loader_restores_conflicting_bare_modules():
    # Force a fresh loader in a subprocess-equivalent module state, including
    # preexisting unrelated modules which must not be replaced permanently.
    names=["params","intensities","price_process","hjb_solver","quotes","monte_carlo"]
    saved={name:sys.modules.get(name) for name in names}
    sentinels={name:ModuleType(name) for name in names}
    try:
        sys.modules.update(sentinels)
        legacy_modules.cache_clear()
        loaded=legacy_modules()
        assert loaded["params"].DEFAULT.T==1
        assert all(sys.modules[name] is sentinels[name] for name in names)
    finally:
        for name,module in saved.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module


@pytest.mark.parametrize("t,p,q",[(0,.5,0),(.5,.2,-50),(.99,.8,60),(.5,.5,95.25),(.9,.001,-99.5)])
def test_adapter_matches_frozen_original_monte_carlo_quotes(hjb,t,p,q):
    legacy=hjb.modules["monte_carlo"]
    bid,ask=legacy._batch_optimal_quotes(t,np.array([p]),np.array([q]),hjb.parameters,hjb.grid,hjb.interpolator)
    actual=hjb.raw_quotes(t,p,q)
    assert actual[0]==bid[0] and actual[1]==ask[0]
    if abs(q)<=90:
        scalar=hjb.modules["quotes"].optimal_quotes(t,p,q,hjb.parameters,hjb.grid,hjb.interpolator)
        assert actual[:2]==scalar


def test_partial_inventory_boundary_never_extrapolates_value(hjb):
    assert np.isnan(hjb.raw_quotes(.5,.5,100)[0])
    assert np.isnan(hjb.raw_quotes(.5,.5,-100)[1])
    for q in (-99.99,99.99):assert np.isfinite(hjb.raw_quotes(.5,.5,q)[:2]).all()
    for p in (0,1):
        _,_,used=hjb.raw_quotes(.5,p,0)
        assert hjb.grid.p[0]<=used<=hjb.grid.p[-1]
    with pytest.raises(ValueError):hjb.raw_quotes(.5,.5,101)


def test_normalized_episode_and_lifetime_clocks_and_partial_size(hjb):
    obs={"elapsed_seconds":900.,"inventory":99.25,"probability":.5,"bid":.45,"ask":.55}
    action,diag=hjb.action(obs,start_elapsed=600,lifetime=1200)
    assert diag["hjb_time"]==.5 and action.bid_size==.75 and action.ask_size==10
    lifetime=HJBReferencePolicy(time_mapping="lifetime")
    _,diag=lifetime.action(obs,start_elapsed=600,lifetime=1200)
    assert diag["hjb_time"]==.75
    with pytest.raises(ValueError):hjb.action(obs,start_elapsed=600,lifetime=1200,inventory_limit=200)


def test_tick_rounding_and_post_only_projection_preserve_disabled_sides():
    obs={"bid":.49,"ask":.51}
    action,info=executable_action(.55,.45,10,10,obs)
    # Simultaneous projections crossing inside a narrow book cancel both.
    assert info["bid_post_only_adjusted"] and info["ask_post_only_adjusted"]
    assert info["crossed_quotes_disabled"] and action.bid is None and action.ask is None
    assert action.bid_size==0 and action.ask_size==0
    action,info=executable_action(.60,.60,10,10,{"bid":.45,"ask":.55})
    assert action.bid==.54 and action.ask==.60 and info["bid_post_only_adjusted"]


def test_outward_rounding_and_nonfinite_side_disabled():
    obs={"bid":.45,"ask":.55}
    action,diag=executable_action(.473,.526,10,10,obs)
    assert action.bid==.47 and action.ask==.53
    action,diag=executable_action(np.nan,.526,0,10,obs)
    assert action.bid is None and action.bid_size==0 and action.ask==.53


def test_es_uses_exact_tail_mass_when_var_has_tied_zero_losses():
    pnl=np.array([-10]+[0]*99)
    stats=pnl_statistics(pnl)
    assert stats["var95_loss"]==0 and stats["es95_loss"]==2
    assert expected_shortfall([10,5,0,0],.375)==pytest.approx((10+.5*5)/1.5)
    assert expected_shortfall([-2,-1],.05)==-1
    with pytest.raises(ValueError):pnl_statistics([np.nan])


def test_drawdown_and_attribution_include_intrawindow_and_settlement():
    # Buy at .4 with mark .5 (+1), fall to .2 (-3), recover .5 (+3),
    # settle NO (-5). Intermediate drawdown is not visible in final-window p.
    ledger=[{"kind":"fill","elapsed_seconds":10,"inventory":10,"inventory_change":10,
        "probability":.5,"price":.4,"quantity":10,"fee":0,"cash":96},
        {"kind":"noise","elapsed_seconds":20,"inventory":10,"probability":.2,"cash":96},
        {"kind":"noise","elapsed_seconds":30,"inventory":10,"probability":.5,"cash":96},
        {"kind":"settlement","elapsed_seconds":40,"inventory":0,"probability":.5,"cash":96,
         "terminal_inventory":10,"settlement_outcome":0}]
    result=episode_diagnostics(ledger,100,.5,0,40)
    assert result["max_drawdown"]==5 and result["spread_capture"]==pytest.approx(1)
    assert result["background_price_pnl"]==pytest.approx(0) and result["settlement_surprise_pnl"]==-5
    assert result["inventory_rms"]==pytest.approx(np.sqrt(75))


def test_hierarchical_event_sampling_preserves_paired_replicate_values():
    cube=np.array([[1,2],[10,20]])
    sample=hierarchical_samples(cube,np.random.default_rng(1),1000)
    assert sample.shape==(1000,4) and set(np.unique(sample))=={1,2,10,20}
    assert sample.mean()==pytest.approx(cube.mean(),abs=.7)


@pytest.mark.parametrize("ticker,group",[("KXCPI-26AUG-T0.5","broad_based"),("KXARMOMINF-26AUG13-T1.8","broad_based"),
    ("KXMLSGAME-26AUG22CLTDCU-TIE","sports")])
def test_cached_sources_reproduce_uncached_environment_exactly(hjb,ticker,group):
    import json
    from simulator.spreads import DEFAULT_CONFIG
    cfg=EnvironmentConfig(group=group,ticker=ticker)
    original=TradingEnvironment(cfg)
    obs,_=original.reset(seed=123,options={"seconds_to_close":600})
    support=json.loads(DEFAULT_CONFIG.read_text())["groups"][group]["supported_tickers"]
    cached=cached_environment(cfg,original,support)
    copy_obs,_=cached.reset(seed=123,options={"seconds_to_close":600})
    assert copy_obs==obs
    while not original.done:
        action,_=hjb.action(obs,start_elapsed=original.start_elapsed,lifetime=original.lifetime)
        obs,*_=original.step(action)
        copy_obs,*_=cached.step(action)
        assert obs==copy_obs
    assert original.ledger==cached.ledger


def test_persisted_small_hjb_benchmark_and_independent_validator(tmp_path):
    from simulator.evaluate_hjb import run
    from simulator.validate_hjb_benchmark import validate
    result=run(tmp_path,replicates=2,bootstrap_draws=20,groups=["sports"],impacts=["zero","base"],fills=["base"],
        tickers=["KXMLSGAME-26AUG22CLTDCU-TIE"])
    assert result["passed"] and result["episodes"]==16
    assert validate(tmp_path)["passed"]
    summary=pd.read_csv(tmp_path/"risk_summary.csv")
    assert len(summary)==8 and summary[summary.policy.eq("none")].mean_pnl.eq(0).all()
