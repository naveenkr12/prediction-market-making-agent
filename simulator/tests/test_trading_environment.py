from decimal import Decimal
import numpy as np
import pandas as pd
import pytest

from simulator.accounting import BinaryAccount
from simulator.execution import BookSnapshot, QuoteAction, RestingOrder
from simulator.fees import FeeSchedule, decimal
from simulator.price_impact import ImpactParameters
from simulator.trading_env import EnvironmentConfig, TradingEnvironment, bridged_innovations
from simulator.policies import quote_policy
from calibrate_execution import build_spread_bank


@pytest.mark.parametrize("side,p,qty,expected", [("bid", .55, .1, "-0.06"), ("ask", .55, .1, "0.05")])
def test_signed_revenue_rounding(side, p, qty, expected):
    schedule = FeeSchedule(1, "0.01")
    result = schedule.charge(p, qty, side)
    assert result["trade_fee"]==Decimal("0.000434")
    assert result["cash_change"]==Decimal(expected)
    assert result["fee"]>=0


def test_official_fee_rounding_example_and_per_order_rebate():
    # Exact published rounding example: revenue=-.055, model fee=.00363825.
    # Pick a multiplier which produces that model fee at p=.55, qty=.1.
    multiplier = Decimal("0.00363825")/(Decimal("0.0175")*Decimal(".1")*Decimal(".55")*Decimal(".45"))
    result = FeeSchedule(multiplier, "0.01").charge(".55", ".1", "bid")
    assert result["trade_fee"]==Decimal(".003639")
    assert result["rounding_fee"]==Decimal(".001361")
    assert result["fee"]==Decimal(".005")
    schedule = FeeSchedule(1, "0.01")
    carried, total, rebates = Decimal(0), Decimal(0), []
    for _ in range(10):
        result = schedule.charge(".5", 1, "bid", carried)
        total += result["fee"]
        carried = result["accumulator"]
        rebates.append(result["rebate"])
    assert total==Decimal(".05")  # .04375 maker model fee, cent-aligned account
    assert sum(rebates)>0
    assert carried<Decimal(".01")


def test_zero_multiplier_still_rounds_fractional_revenue():
    result = FeeSchedule(0, "0.01").charge(".55", ".1", "bid")
    assert result["trade_fee"]==0 and result["fee"]==Decimal(".005")


@pytest.mark.parametrize("side,outcome,expected", [("bid",0,"94.50"),("bid",1,"104.50"),("ask",0,"105.50"),("ask",1,"95.50")])
def test_binary_settlement_long_short_both_outcomes(side, outcome, expected):
    account = BinaryAccount(100, fees=FeeSchedule(0))
    account.fill(side, ".55", 10)
    assert account.collateral==(10 if side=="ask" else 0)
    settlement = account.settle(outcome)
    assert settlement["cash"]==Decimal(expected)
    assert account.inventory==0 and account.collateral==0
    with pytest.raises(RuntimeError): account.settle(outcome)
    with pytest.raises(RuntimeError): account.fill("bid", ".5", 1)


def test_cash_inventory_limits_and_reservations_fractional():
    account = BinaryAccount(1, inventory_limit=1.25, fees=FeeSchedule(1))
    admitted, reserved = account.admit_quotes({"bid":(".49",100),"ask":(".51",100)})
    assert admitted["bid"]==Decimal("1.25")
    assert admitted["ask"]<Decimal("1.25")
    assert reserved<=account.available_cash
    # Either order of all admitted fills is affordable.
    for order in [("bid","ask"),("ask","bid")]:
        a = BinaryAccount(1, inventory_limit=1.25, fees=FeeSchedule(1))
        for side in order:
            if admitted[side]:
                a.fill(side, ".49" if side=="bid" else ".51", admitted[side])
        assert a.available_cash>=0 and abs(a.inventory)<=Decimal("1.25")
    with pytest.raises(ValueError): account.fill("ask", ".5", 100)
    assert account.cash==1 and account.inventory==0
    with pytest.raises(ValueError): account.fill("bid", ".5", ".005")


def test_l2_complementary_ask_and_price_priority():
    book = BookSnapshot.from_kalshi({"orderbook_fp":{"yes_dollars":[[".40","2.50"],[".45","10"]],
        "no_dollars":[[".44","7.25"],[".46","12.50"]]}})
    assert book.bid==.45 and book.ask==.54
    assert book.queue_ahead("ask", decimal(".56"), 999)==Decimal("19.75")
    assert book.queue_ahead("bid", decimal(".46"), 999)==0
    with pytest.raises(ValueError): BookSnapshot.from_kalshi({"orderbook_fp":{"yes_dollars":[],"no_dollars":[]}})


def test_queue_consumption_partial_fills_and_signs():
    order = RestingOrder(1,"ask",decimal(".55"),decimal(10),decimal(5))
    assert order.match(-1,20,.55,1,decimal(".01"))==0
    assert order.match(1,20,.54,1,decimal(".01"))==0
    assert order.queue_ahead==5
    assert order.match(1,3,.55,.5,decimal(".01"))==0
    assert order.queue_ahead==2
    assert order.match(1,7.25,.55,.5,decimal(".01"))==Decimal("2.62")
    assert order.queue_ahead==0


@pytest.mark.parametrize("action", [QuoteAction(.5,.5,1,1),QuoteAction(.501,.6,1,1),QuoteAction(.6,None,1,0),
    QuoteAction(None,.4,0,1),QuoteAction(.4,None,-1,0),QuoteAction(.4,None,.005,0),QuoteAction(None,None,1,0)])
def test_bad_crossed_or_marketable_quotes_rejected(action):
    with pytest.raises(ValueError): action.validate(BookSnapshot(.45,.55),decimal(".01"),decimal(".01"))


class FixedDynamics:
    ticker="TEST"
    impact=ImpactParameters(.001)
    sigma=0.
    noise_template_ticker="TEST"
    noise_template={"noise":{"pair_start_utc":"2026-01-01T00:00:00Z","pair_end_utc":"2026-01-02T00:00:00Z"}}
    template={"lifetime_seconds":600.,"open_time":"2026-01-01T00:00:00Z"}

    def sample_window(self, fraction, rng):
        duration=min(300.,600*(1-fraction))
        return {"duration_seconds":duration,"event_times_seconds":np.array([.2,.6])*duration,
            "quantities":np.array([25.,40.]),"signs":np.array([1,-1]),"signed_flow_contracts":-15.,"phase_mark_fallback":False}

    def sample_innovation(self, rng, duration_seconds): return np.array([.2])


class FixedSpread:
    ticker="TEST"
    template={"start_utc":"2026-01-01T00:00:00Z","end_utc":"2026-01-02T00:00:00Z"}
    def sample(self, log_odds, rng, tick): return BookSnapshot(.45,.55)


def fixture_env(**kwargs):
    return TradingEnvironment(EnvironmentConfig(start_life_fraction=0, maker_fee_multiplier=0,participation=1,
        **kwargs), dynamics_factory=lambda rng:FixedDynamics(), spread_factory=lambda dynamics,rng:FixedSpread())


def test_flow_applied_once_fill_before_impact_and_reward_telescopes():
    env=fixture_env()
    obs,_=env.reset(seed=1)
    total=0
    for i in range(2):
        obs,reward,terminated,truncated,info=env.step(QuoteAction(None,.51,0,7))
        total+=reward
        assert terminated==(i==1) and not truncated
    assert env.price.log_odds==pytest.approx(2*(.2-.001*15))
    markets=[r for r in env.ledger if r["kind"]=="market"]
    fills=[r for r in env.ledger if r["kind"]=="fill"]
    assert len(markets)==4 and len(fills)==2
    assert float(fills[0]["log_odds"])==pytest.approx(.04)
    assert float(markets[0]["log_odds"])==pytest.approx(.04+.025)
    assert float(fills[0]["price"])==.51
    assert float(fills[0]["inventory_change"])==-7
    assert total==pytest.approx(obs["cash"]-1000)
    assert obs["inventory"]==0 and obs["collateral"]==0 and obs["reserved_cash"]==0
    assert "settlement_outcome" not in obs
    with pytest.raises(RuntimeError):env.step(None)


def test_order_retention_cancel_and_truncation_does_not_settle():
    env=fixture_env(max_steps=1)
    env.reset(seed=1)
    obs,reward,terminated,truncated,info=env.step(QuoteAction(None,.51,0,100))
    assert not terminated and truncated and not env.account.settled
    assert "settlement_outcome" not in info and obs["inventory"]==-25
    assert obs["reserved_cash"]==0
    assert reward==pytest.approx(obs["equity"]-1000)
    env=fixture_env()
    env.reset(seed=1)
    env.step(QuoteAction(None,.51,0,100))
    order=env.orders["ask"]
    env.step(None)
    assert len([r for r in env.ledger if r["kind"]=="quote"])==1
    assert order.remaining==50
    env=fixture_env()
    env.reset(seed=1)
    env.step(QuoteAction(None,.51,0,100))
    q=env.account.inventory
    env.step(QuoteAction())
    assert len([r for r in env.ledger if r["kind"]=="fill"])==1
    assert q==-25


def test_inventory_penalties_use_seconds_and_pre_settlement_inventory():
    env=fixture_env(inventory_penalty_per_contract2_second=.001,terminal_penalty_per_contract2=.1)
    env.reset(seed=1)
    obs,r,_,_,info=env.step(QuoteAction(None,.51,0,7))
    assert info["risk_penalty"]==pytest.approx(.001*49*240)
    obs,r,_,_,info=env.step(QuoteAction())
    assert info["risk_penalty"]==pytest.approx(.001*49*300+.1*49)
    assert env.total_reward+env.total_risk_penalty==pytest.approx(obs["cash"]-1000)


def test_seed_and_policy_independent_market_paths():
    envs=[fixture_env(),fixture_env()]
    for env in envs:env.reset(seed=9)
    for _ in range(2):
        envs[0].step(QuoteAction(None,.51,0,7))
        envs[1].step(QuoteAction())
    assert [r for r in envs[0].ledger if r["kind"]=="market"][0]["probability"]==[r for r in envs[1].ledger if r["kind"]=="market"][0]["probability"]
    assert envs[0].price.log_odds==envs[1].price.log_odds
    assert envs[0].steps[-1]["settlement_outcome"]==envs[1].steps[-1]["settlement_outcome"]
    old=envs[0].ledger.copy()
    envs[0].reset(seed=9)
    for _ in range(2):envs[0].step(QuoteAction(None,.51,0,7))
    assert envs[0].ledger==old


def test_invalid_action_is_atomic_and_short_last_window():
    env=fixture_env()
    env.reset(seed=1,options={"seconds_to_close":125})
    with pytest.raises(ValueError):env.step(QuoteAction(.6,None,10,0))
    assert env.elapsed==475 and not env.ledger
    obs,_,done,trunc,info=env.step(QuoteAction())
    assert done and not trunc and env.elapsed==600 and obs["seconds_to_close"]==0


def test_coverage_flag_includes_start_of_window():
    env=fixture_env()
    env.reset(seed=1)
    env.dynamics.noise_template={"noise":{"pair_start_utc":"2026-01-01T00:03:20Z","pair_end_utc":"2026-01-02T00:00:00Z"}}
    env.spreads.template={"start_utc":"2026-01-01T00:03:20Z","end_utc":"2026-01-02T00:00:00Z"}
    _,_,_,_,info=env.step(QuoteAction())
    assert info["volatility_extrapolated"] and info["spread_extrapolated"]
    _,_,_,_,info=env.step(QuoteAction())
    assert not info["volatility_extrapolated"] and not info["spread_extrapolated"]


def test_bridge_endpoint_and_gaussian_intrawindow_variance():
    rng=np.random.default_rng(2)
    values=np.array([bridged_innovations([150],300,.3,.02,rng) for _ in range(10000)])
    assert np.max(abs(values.sum(axis=1)-.3))<1e-15
    assert values[:,0].mean()==pytest.approx(.15,abs=.006)
    assert values[:,0].var()==pytest.approx(.02**2*75,rel=.05)
    assert bridged_innovations([],10,.1,0,rng)[0]==.1


def test_saturated_reference_keeps_finite_executable_book():
    from simulator.spreads import SpreadSampler
    sampler=SpreadSampler.__new__(SpreadSampler)
    sampler.gaps=np.array([.1])
    for latent in (-1000, 1000):
        book=sampler.sample(latent,np.random.default_rng(1),decimal(".01"))
        assert 0<=book.bid<book.ask<=1
        obs={"probability":1. if latent>0 else 0.,"inventory":0,"bid":book.bid,"ask":book.ask}
        quote_policy(obs).validate(book,decimal(".01"),decimal(".01"))


def test_training_spreads_do_not_load_holdout_and_keep_boundaries():
    tickers=[f"T{i}" for i in range(5)]
    config={"training":{"tickers":tickers},"groups":{"g":{"status":"calibrated","event_tickers":tickers}}}
    markets=pd.DataFrame([{"ticker":t,"open_time":"2026-01-01T00:00Z","close_time":"2026-01-01T00:10Z"} for t in tickers+["HOLDOUT"]])
    read=[]
    def loader(ticker):
        assert ticker!="HOLDOUT"
        read.append(ticker)
        return pd.DataFrame({"end_period_ts":[1767225660,1767225720,1767225780],
            "yes_bid_close_dollars":[.45,0,.5],"yes_ask_close_dollars":[.55,.1,.5]})
    result,bank,audit=build_spread_bank(config,markets,loader,min_quotes=1)
    assert set(read)==set(tickers) and len(result["templates"])==5
    assert all(len(v)==1 for v in bank.values())
    assert audit.interior_positive_spread_rows.sum()==5


@pytest.mark.parametrize("group", ["broad_based","sports"])
def test_saved_training_environment_full_episode_and_common_random_numbers(group):
    envs=[TradingEnvironment(EnvironmentConfig(group=group)),TradingEnvironment(EnvironmentConfig(group=group))]
    observations=[]
    for env in envs:
        obs,_=env.reset(seed=42,options={"seconds_to_close":3600})
        observations.append(obs)
    while not envs[0].done:
        for i,env in enumerate(envs):
            action=quote_policy(observations[i],inventory_skew=.0005) if i==0 else QuoteAction()
            observations[i],_,_,_,_=env.step(action)
            assert observations[i]["available_cash"]>=-1e-9
        assert envs[0].price.log_odds==envs[1].price.log_odds
        assert envs[0].book==envs[1].book
    assert envs[0].account.settled and envs[1].account.settled
    assert envs[1].account.cash==1000
    assert envs[0].total_reward==pytest.approx(float(envs[0].account.cash)-1000)
    assert any(r["kind"]=="fill" for r in envs[0].ledger)


def test_independent_audit_detects_corrupt_accounting_and_double_impact():
    from simulator.validate_trading_env import audit_ledger
    env=fixture_env()
    env.reset(seed=1)
    for _ in range(2):env.step(QuoteAction(None,.51,0,7))
    args=dict(initial_cash=1000,initial_log_odds=0,beta=.001,inventory_limit=100,
        maker_fee_multiplier=0,balance_precision=".0001",quantity_lot=".01",participation=1)
    assert audit_ledger(env.ledger,**args)["passed"]
    rows=[r.copy() for r in env.ledger]
    fill=next(r for r in rows if r["kind"]=="fill")
    fill["cash"]="100000"
    with pytest.raises(AssertionError):audit_ledger(rows,**args)
    rows=[r.copy() for r in env.ledger]
    market=next(r for r in rows if r["kind"]=="market")
    market["impact_logodds"]*=2
    with pytest.raises(AssertionError):audit_ledger(rows,**args)


def test_disk_ledger_roundtrip_and_zero_quote_pnl(tmp_path):
    from simulator.demo_trading_env import run
    from simulator.validate_trading_env import validate_run
    report=run(tmp_path,episodes=1,groups=["sports"],impacts=["base"],fills=["base"],policies=["none","skew"])
    assert report["passed"] and report["episodes"]==2 and report["maker_fills"]>0
    assert validate_run(tmp_path)["passed"]
    summary=pd.read_csv(tmp_path/"episode_summary.csv")
    assert summary.loc[summary.policy.eq("none"),"realized_pnl"].iloc[0]==0


def test_public_depth_capture_is_bounded_and_records_availability(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"data_pipeline"))
    import json
    from capture_orderbook import capture
    class FakeClient:
        def _get(self,path,params):
            assert path=="/markets/TEST-YES/orderbook" and params=={"depth":0}
            return {"orderbook_fp":{"yes_dollars":[[".4","1"]],"no_dollars":[[".5","1.25"]]}}
    result=capture("TEST-YES",tmp_path,client=FakeClient())
    saved=json.loads(Path(result["file"]).read_text())
    assert saved["available_at_utc"]>=saved["request_started_utc"]
    assert saved["sequence_number"] is None
    assert not result["historical_depth_recovered"]
    BookSnapshot.from_kalshi(saved["payload"])
