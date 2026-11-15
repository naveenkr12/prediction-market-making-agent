"""
Integrity checks for the RL dataset. Exits non-zero if any hard check fails.

Run from data_pipeline/:   python3 rl_validate.py
Writes DATA_DIR/dataset/validation_report.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import rl_config as R
from rl_collect_kalshi import save_json


def validate() -> dict:
    d = R.DATA_DIR / "dataset"
    c = pd.read_parquet(d / "contracts.parquet")
    b = pd.read_parquet(d / "bars.parquet")
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}

    checks["market_ids_unique"] = c.market_id.is_unique
    checks["labels_complete"] = bool(c[["venue", "category", "series_ticker", "event_ticker", "title",
                                        "y_outcome", "split", "episode_start_utc", "episode_end_utc"]]
                                     .notna().all().all())
    checks["y_outcome_binary"] = bool(c.y_outcome.isin([0, 1]).all())
    split_sets = c.groupby("split").event_ticker.apply(set)
    checks["events_do_not_straddle_splits"] = all(
        not (split_sets[a] & split_sets[b2]) for a in split_sets.index for b2 in split_sets.index if a < b2)
    checks["density_gates_met"] = bool((c.episode_trades >= R.MIN_EPISODE_TRADES).all()
                                       and (c.active_bar_share >= R.MIN_ACTIVE_BAR_SHARE).all())

    g = b.groupby("market_id")
    checks["bars_contiguous"] = bool((g.bar_index.agg(lambda s: (np.diff(np.sort(s)) == 1).all())).all())
    checks["bar_counts_match"] = bool((g.size() == g.n_bars.first()).all())
    checks["tau_in_unit_interval"] = bool(b.tau_start.between(0, 1).all() and b.tau_end.between(0, 1).all())
    checks["tau_monotone"] = bool(g.tau_start.apply(lambda s: s.is_monotonic_increasing).all())
    checks["prices_in_unit_interval"] = bool(all(
        b[col].dropna().between(0, 1).all() for col in ["p_ref", "yes_bid", "yes_ask", "yes_mid", "vwap", "last_price"]))
    two = b[b.two_sided]
    checks["two_sided_books_uncrossed"] = bool((two.yes_bid < two.yes_ask).all())
    checks["flow_identity"] = bool(np.allclose(b.signed_flow, b.buy_contracts - b.sell_contracts))
    checks["trade_counts_identity"] = bool((b.n_trades == b.n_buy + b.n_sell).all())
    bar_trades = g.n_trades.sum()
    checks["bar_trades_match_contracts"] = bool(
        (bar_trades.reindex(c.market_id).to_numpy() == c.episode_trades.to_numpy()).all())
    checks["split_labels_consistent"] = bool(
        (b.groupby("market_id").split.first().reindex(c.market_id).to_numpy() == c.split.to_numpy()).all())

    notes["p_ref_missing_share"] = float(b.p_ref.isna().mean())
    notes["kalshi_two_sided_share"] = float(b[b.venue == "kalshi"].two_sided.mean()) if (b.venue == "kalshi").any() else None
    notes["kalshi_quote_observed_share"] = (float(b[b.venue == "kalshi"].quote_observed.mean())
                                            if (b.venue == "kalshi").any() else None)
    notes["contracts_by_venue_category_split"] = (c.groupby(["venue", "category", "split"]).size()
                                                  .rename("n").reset_index().to_dict("records"))
    report = {"checks": checks, "all_passed": all(checks.values()), "notes": notes}
    save_json(d / "validation_report.json", report)
    return report


if __name__ == "__main__":
    r = validate()
    for k, v in r["checks"].items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps({k: v for k, v in r["notes"].items() if k != "contracts_by_venue_category_split"}, indent=2))
    sys.exit(0 if r["all_passed"] else 1)
