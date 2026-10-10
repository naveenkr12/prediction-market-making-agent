"""Meaningful failure cases: archive gaps, bad trades, and correlated sampling."""
from argparse import Namespace
from datetime import datetime, timezone

import pandas as pd
import pytest

from calibration_quality import normalize_trades, profile_contract, event_cluster, classify_market, audit_existing
from kalshi_client import KalshiClient, KalshiAPIError
from collect_calibration import run


def market(ticker="KXCPI-26JUL-T0.3", event="KXCPI-26JUL", category="Economics", series="KXCPI"):
    return {"ticker": ticker, "event_ticker": event, "series_ticker": series,
            "series_title": "CPI", "category": category, "market_type": "binary", "result": "yes",
            "open_time": "2026-07-01T00:00:00Z", "close_time": "2026-07-02T00:00:00Z",
            "rules_primary": "CPI exceeds threshold", "volume_fp": "100.00", "status": "finalized"}


def trades(ticker, intervals=50):
    return pd.DataFrame([{"ticker": ticker, "trade_id": str(i),
                          "created_time": ts.isoformat(), "yes_price_dollars": "0.5000",
                          "no_price_dollars": "0.5000", "count_fp": "2.00", "taker_outcome_side": "yes",
                          "is_block_trade": False} for i, ts in enumerate(
                              pd.date_range("2026-07-01", periods=intervals, freq="5min", tz="UTC"))])


def test_interval_threshold_is_not_trade_count():
    m = market()
    d = trades(m["ticker"])
    assert profile_contract(m, d, tape_complete=True)["calibration_eligible"]
    d["created_time"] = "2026-07-01T00:00:01Z"
    r = profile_contract(m, d, tape_complete=True)
    assert r["active_5min_intervals"] == 1
    assert not r["calibration_eligible"]


def test_unknown_history_and_missing_metadata_do_not_pass():
    m = market()
    r = profile_contract(m, trades(m["ticker"]), tape_complete=None)
    assert r["paper_interval_threshold_met"] and not r["calibration_eligible"]
    assert "not_verified" in r["exclusion_reasons"]
    m["event_ticker"] = ""
    assert not profile_contract(m, trades(m["ticker"]), tape_complete=True)["metadata_complete"]


def test_fractional_quantities_dedup_blocks_and_bad_fields():
    m = market()
    d = trades(m["ticker"], 4)
    d["count_fp"] = ["1.25", "2.50", "-1", "3.00"]
    d.loc[1, "taker_outcome_side"] = "no"
    d.loc[3, "is_block_trade"] = True
    d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    clean, stats = normalize_trades(d, m["ticker"])
    assert clean.quantity.tolist() == [1.25, 2.5]
    assert clean.sign.tolist() == [1, -1]
    assert stats["duplicate_trade_ids"] == 1 and stats["invalid_trade_rows"] == 1
    assert stats["block_trade_rows"] == 1


def test_logodds_boundary_and_volume_reconciliation():
    m = market()
    d = trades(m["ticker"])
    d.loc[0, ["yes_price_dollars", "no_price_dollars"]] = ["1.0000", "0.0000"]
    r = profile_contract(m, d, tape_complete=True)
    assert r["active_5min_intervals"] == 50 and r["usable_logodds_intervals"] == 49
    m["volume_fp"] = "10000"
    assert "observed_volume_below" in profile_contract(m, d, tape_complete=True)["exclusion_reasons"]
    m["volume_fp"] = "1"
    assert "observed_volume_above" in profile_contract(m, d, tape_complete=True)["exclusion_reasons"]


def test_economics_not_automatically_broad_based_and_release_clusters():
    assert classify_market({"category": "Economics", "series_ticker": "KXTOPBUSINESS"})[0] == "needs_review"
    assert classify_market({"category": "Sports", "series_ticker": "KXNFLGAME"})[0] == "sports"
    assert classify_market({"category": "Economics", "series_ticker": "KXH200CHINA"})[0] == "single_name"
    assert classify_market({"category": "Economics", "series_ticker": "KXARMOMINF"})[0] == "broad_based"
    assert event_cluster(market()) == event_cluster(market(event="KXCPIYOY-26JUL", series="KXCPIYOY"))


def test_archive_trade_routing_overlap_and_dedup():
    client = KalshiClient()
    client._cutoffs = {"trades_created_ts": "2026-07-01T00:00:00Z"}
    calls = []
    t = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
    def paginated(path, params, items_key, max_pages=None):
        calls.append((path, params.copy()))
        yield {"trade_id": "overlap", "created_time": "2026-07-01T00:00:00Z"}
        yield {"trade_id": path, "created_time": "2026-07-01T00:00:01Z"}
    client._paginate = paginated
    tape = client.get_trades("TEST", t-100, t+100)
    assert len(tape) == 3
    assert [x[0] for x in calls] == ["/markets/trades", "/historical/trades"]
    assert calls[0][1]["min_ts"] == t-1 and calls[1][1]["max_ts"] == t


def test_pagination_refuses_truncated_tape():
    client = KalshiClient()
    client._get = lambda *a: {"trades": [{"trade_id": "1"}], "cursor": "next"}
    with pytest.raises(KalshiAPIError, match="incomplete"):
        list(client._paginate("/markets/trades", {}, "trades", max_pages=1))


def test_historical_listing_filters_locally():
    client = KalshiClient()
    seen = []
    def paginated(path, params, items_key, max_pages=None):
        seen.append((path, params))
        yield {"ticker": "old", "close_time": "2025-01-01T00:00:00Z"}
        yield {"ticker": "recent", "close_time": "2026-07-01T00:00:00Z"}
    client._paginate = paginated
    start = int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp())
    result = client.list_markets(series_ticker="TEST", min_close_ts=start, historical=True, mve_filter="exclude")
    assert [r["ticker"] for r in result] == ["recent"]
    assert "min_close_ts" not in seen[0][1]
    assert "mve_filter" not in seen[0][1]  # incompatible with historical series filter


def test_collector_refills_after_rejection_and_resumes(monkeypatch, tmp_path):
    import collect_calibration as collector
    rejected = market(ticker="A", event="EVENT-A")
    accepted = market(ticker="B", event="EVENT-A")
    same_release = market(ticker="C", event="EVENT-A")
    sports = market(ticker="D", event="GAME-D", category="Sports", series="KXNFLGAME")
    monkeypatch.setattr(collector, "discover_calibration_candidates",
                        lambda *a: ([rejected, accepted, same_release, sports], []))
    class Client:
        request_count = 0
        def get_cutoffs(self): return {"trades_created_ts": "2026-08-10T00:00:00Z"}
        def get_trades(self, ticker, *a, **kw):
            self.request_count += 1
            return trades(ticker, 1 if ticker == "A" else 50).to_dict("records")
    args = Namespace(target_total=2, end_date="2026-10-09", lookback_days=90, output_dir=tmp_path,
                     min_intervals=50, min_volume=1000, max_per_event=1, max_per_series=10,
                     max_series_per_category=2, max_trade_pages=10, max_candidates=10,
                     seed=42, candle_lookback_days=0, retry_failed=False)
    client = Client()
    result = run(args, client)
    assert result["target_reached"] and result["selected_events"] == 2
    assert result["selected_by_category"] == {"Economics": 1, "Sports": 1}
    assert client.request_count == 3  # rejection does not consume EVENT-A's place
    again = Client()
    assert run(args, again)["target_reached"] and again.request_count == 0
    # Offline re-audit must recognize complete saved histories and keep rejects.
    audited = audit_existing(tmp_path, tmp_path / "reports")
    assert audited["contracts_audited"] == 3 and audited["calibration_eligible"] == 2
    assert audited["target_reached"] and audited["selected_but_ineligible"] == 0


def test_unbalanced_fallback_is_explicit_and_keeps_event_caps(monkeypatch, tmp_path):
    import collect_calibration as collector
    econ = market(ticker="E", event="ECON-E")
    sports = [market(ticker=f"S{i}", event=f"GAME-{i}", category="Sports", series="KXNFLGAME")
              for i in range(4)]
    monkeypatch.setattr(collector, "discover_calibration_candidates", lambda *a: ([econ]+sports, []))
    class Client:
        request_count = 0
        def get_cutoffs(self): return {}
        def get_trades(self, ticker, *a, **kw): return trades(ticker).to_dict("records")
    args = Namespace(target_total=4, end_date="2026-10-09", lookback_days=90, output_dir=tmp_path,
                     min_intervals=50, min_volume=1000, max_per_event=1, max_per_series=10,
                     max_series_per_category=2, max_trade_pages=10, max_candidates=10,
                     seed=42, candle_lookback_days=0, retry_failed=False, allow_unbalanced=True)
    result = run(args, Client())
    assert result["selected_by_category"] == {"Economics": 1, "Sports": 3}
    assert result["target_reached"] and not result["balanced_target_reached"]
    assert result["selected_event_clusters"] == 4
