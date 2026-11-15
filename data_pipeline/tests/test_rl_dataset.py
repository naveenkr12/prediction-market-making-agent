"""RL dataset: venue conversion, bar construction and paper-grid alignment."""
import json

import numpy as np
import pandas as pd
import pytest

import rl_config as R
from calibration_quality import normalize_trades
from rl_collect_polymarket import to_kalshi_schema, binary_market

YES, NO = "111", "222"


def pm_trade(asset, side, price, size=5, ts=1_790_000_000, tx="0xa"):
    return {"asset": asset, "side": side, "price": price, "size": size, "timestamp": ts,
            "transactionHash": tx, "outcome": "x", "outcomeIndex": 0 if asset == YES else 1}


@pytest.mark.parametrize("asset,side,price,yes_price,sign", [
    (YES, "BUY", 0.62, 0.62, 1),     # buy YES        -> YES taker at 0.62
    (YES, "SELL", 0.62, 0.62, -1),   # sell YES       -> NO taker
    (NO, "BUY", 0.38, 0.62, -1),     # buy NO at .38  -> NO taker, YES price .62
    (NO, "SELL", 0.38, 0.62, 1),     # sell NO        -> YES taker
])
def test_polymarket_trades_map_to_yes_price_and_taker_sign(asset, side, price, yes_price, sign):
    raw = to_kalshi_schema([pm_trade(asset, side, price)], "0xcid", YES)
    norm, _ = normalize_trades(raw, "0xcid")
    assert len(norm) == 1
    assert norm.price.iloc[0] == pytest.approx(yes_price)
    assert norm.sign.iloc[0] == sign


def test_polymarket_conversion_drops_identity_fields():
    t = dict(pm_trade(YES, "BUY", .5), proxyWallet="0xabc", name="n", pseudonym="p")
    raw = to_kalshi_schema([t], "0xcid", YES)
    assert not {"proxyWallet", "name", "pseudonym"} & set(raw.columns)


def test_binary_market_requires_clean_resolution():
    base = {"outcomes": json.dumps(["Yes", "No"]), "clobTokenIds": json.dumps([YES, NO]),
            "enableOrderBook": True, "closedTime": "2026-09-01 00:00:00+00",
            "startDate": "2026-08-30T00:00:00Z", "volumeNum": 1e5}
    assert binary_market(dict(base, outcomePrices=json.dumps(["0", "1"])))["y"] == 0
    assert binary_market(dict(base, outcomePrices=json.dumps(["1", "0"])))["y"] == 1
    assert binary_market(dict(base, outcomePrices=json.dumps(["0.5", "0.5"]))) is None


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """One Kalshi contract: 10-minute episode, sparse candles, three trades."""
    monkeypatch.setattr(R, "DATA_DIR", tmp_path)
    t0 = 1_790_000_000 - 1_790_000_000 % 60
    root = tmp_path / "kalshi"
    for sub in ["markets", "trades", "candlesticks"]:
        (root / sub).mkdir(parents=True)
    trades = pd.DataFrame([
        {"trade_id": str(i), "ticker": "KXT-1", "created_time": pd.Timestamp(t0 + s, unit="s", tz="UTC").isoformat(),
         "yes_price_dollars": p, "no_price_dollars": f"{1 - float(p):.4f}", "count_fp": q,
         "taker_outcome_side": side, "is_block_trade": False}
        for i, (s, p, q, side) in enumerate([(-30, "0.4000", "1.00", "yes"), (65, "0.5000", "2.00", "yes"),
                                             (70, "0.5200", "3.00", "no"), (590, "0.5500", "4.00", "yes")])])
    trades.to_parquet(root / "trades" / "KXT-1.parquet")
    # Candles only where the book changed: minutes 0 and 5 (end_period_ts = bar end).
    pd.DataFrame({"end_period_ts": [t0 + 60, t0 + 360], "yes_bid_close_dollars": ["0.4800", "0.5000"],
                  "yes_ask_close_dollars": ["0.5200", "0.5400"], "volume_fp": ["0", "0"],
                  "open_interest_fp": ["10", "10"]}).to_parquet(root / "candlesticks" / "KXT-1.parquet")
    pd.DataFrame([{"venue": "kalshi", "market_id": "KXT-1", "ticker": "KXT-1", "category": "Sports",
                   "series_ticker": "KXT", "event_ticker": "KXT-EV", "title": "t", "y_outcome": 1,
                   "close_time": pd.Timestamp(t0 + 600, unit="s", tz="UTC").isoformat(),
                   "episode_start_utc": pd.Timestamp(t0, unit="s", tz="UTC").isoformat(),
                   "episode_end_utc": pd.Timestamp(t0 + 600, unit="s", tz="UTC").isoformat(),
                   "episode_hours": 600 / 3600, "episode_bars": 10, "episode_trades": 3,
                   "episode_contracts": 9.0, "bars_with_trade": 2, "bars_with_quote_update": 2,
                   "active_bar_share": .4}]).to_parquet(root / "markets" / "contracts.parquet")
    return t0


def test_bars_flows_quotes_and_tau(tiny_dataset):
    import rl_build_bars
    rl_build_bars.run()
    b = pd.read_parquet(R.DATA_DIR / "dataset" / "bars.parquet")
    assert len(b) == 10 and b.tau_start.iloc[0] == 0 and b.tau_end.iloc[-1] == 1
    m1 = b.iloc[1]
    assert (m1.n_buy, m1.n_sell, m1.signed_flow) == (1, 1, -1.0)
    assert m1.vwap == pytest.approx((0.5 * 2 + 0.52 * 3) / 5)
    assert b.n_trades.sum() == 3                       # pre-window trade excluded from flows
    assert b.last_price.iloc[0] == pytest.approx(0.40)  # but sets the opening last price
    # Minute 0 quote observed; minutes 1-4 carried forward and flagged; minute 5 updated.
    assert b.quote_observed.tolist()[:6] == [True, False, False, False, False, True]
    assert b.yes_bid.iloc[3] == pytest.approx(0.48) and b.quote_age_s.iloc[3] == 180
    assert b.yes_mid.iloc[5] == pytest.approx(0.52)
    assert (b.split == "train").all()


def test_paper_grid_conserves_trades_and_maps_time(tiny_dataset):
    import importlib
    import rl_build_bars, rl_loader
    rl_build_bars.run()
    importlib.reload(rl_loader)
    g = rl_loader.paper_grid("KXT-1", n_steps=100)
    assert len(g) == 100 and g.t.iloc[0] == 0 and g.dt.iloc[0] == pytest.approx(0.01)
    assert g.n_trades.sum() == 3 and g.signed_flow.sum() == pytest.approx(2 + 4 - 3)
    assert g.attrs["seconds_per_step"] == pytest.approx(6.0)
    # Quote at step start: no bar has ended at t=0, so the first quote appears at 60 s.
    assert np.isnan(g.yes_bid.iloc[0]) and g.yes_bid.iloc[10] == pytest.approx(0.48)


@pytest.mark.parametrize("series,question,keep", [
    ("economy", "First to 5k: Gold or ETH?", False),
    ("economy", "Will Tesla (TSLA) beat quarterly earnings?", False),
    ("economy", 'Will Powell say "Good afternoon" during January press conference?', False),
    ("fed-rates", "Will Donald Trump announce a new nominee for Chair of the Federal Reserve?", False),
    ("fed-rates", "Fed rate cut by October 2025 meeting?", True),
    ("fomc", "Will there be no change in Fed interest rates after the April 2026 meeting?", True),
    ("economy", "Will US CPI exceed 3% in September?", True),
])
def test_polymarket_economics_is_macro_only(series, question, keep):
    from rl_collect_polymarket import topical
    assert topical(R.CATEGORIES["Economics"], series, question) is keep
    assert topical(R.CATEGORIES["Sports"], series, question)   # no filter elsewhere
