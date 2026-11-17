"""
Build the unified, labelled RL dataset from the raw Kalshi + Polymarket pulls.

Outputs under DATA_DIR/dataset/:
  contracts.parquet          one row per accepted contract: venue, category,
                             series/event, question text, outcome label,
                             episode window, density stats, train/val/test split
  bars.parquet               one row per contract-minute (all contracts)
  trades/<venue>/<id>.parquet
                             cleaned trade events with tau, for finer
                             resampling (rl_loader.paper_grid)
  manifest.json              settings, row counts, per-category statistics

Bar conventions (both venues):
  bar i covers [bar_start, bar_start + 60 s); tau = (t - episode_start) /
  (episode_end - episode_start), so the HJB policy's t = T * tau holds.
  Quotes are as of the bar's end. Kalshi candles are only emitted in minutes
  where something changed, so missing minutes are forward-filled and flagged
  quote_observed=False with quote_age_s. Polymarket has no historical bid/ask:
  yes_bid/yes_ask are NaN and p_ref is the published midpoint.

  sign convention: +1 = YES taker (buys YES), -1 = NO taker; signed_flow =
  buy_contracts - sell_contracts, exactly as in calibration/build_intervals.

y_outcome is the settlement label. It is retrospective; never feed it to a
policy as an observation.

Run from data_pipeline/:   python3 rl_build_bars.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import rl_config as R
from calibration_quality import normalize_trades
from rl_collect_kalshi import save_json
from utils import safe_filename

log = logging.getLogger("rl_build_bars")


def assign_splits(contracts: pd.DataFrame) -> pd.Series:
    """Chronological by close time within venue x category; events never straddle splits."""
    split = pd.Series("train", index=contracts.index)
    for _, g in contracts.groupby(["venue", "category"]):
        events = (g.assign(close=pd.to_datetime(g.close_time, utc=True))
                  .groupby("event_ticker").close.max().sort_values())
        n = len(events)
        # Earliest events always train, so tiny groups never end up test-only.
        n_train = max(1, round(n * R.SPLIT_FRACTIONS["train"]))
        n_val = min(n - n_train, round(n * R.SPLIT_FRACTIONS["validation"]))
        labels = {}
        for i, ev in enumerate(events.index):
            labels[ev] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
        split[g.index] = g.event_ticker.map(labels)
    return split


def kalshi_quotes(candles: pd.DataFrame) -> pd.DataFrame:
    """Normalize live (`*_dollars`) and archived (no suffix) candle columns."""
    def col(name):
        for c in (f"{name}_dollars", name):
            if c in candles.columns:
                return pd.to_numeric(candles[c], errors="coerce")
        return pd.Series(np.nan, index=candles.index)
    q = pd.DataFrame({"obs_ts": candles.end_period_ts.astype("int64"),
                      "yes_bid": col("yes_bid_close"), "yes_ask": col("yes_ask_close"),
                      "candle_volume": pd.to_numeric(candles.get("volume_fp", candles.get("volume")), errors="coerce"),
                      "open_interest": pd.to_numeric(candles.get("open_interest_fp", candles.get("open_interest")),
                                                     errors="coerce")})
    return q.sort_values("obs_ts").drop_duplicates("obs_ts", keep="last").reset_index(drop=True)


def build_contract(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    venue, mid = row.venue, row.market_id
    fname = safe_filename(mid)
    root = R.DATA_DIR / venue
    ep_start = int(pd.Timestamp(row.episode_start_utc).timestamp())
    ep_end = int(pd.Timestamp(row.episode_end_utc).timestamp())
    length = ep_end - ep_start
    g0 = ep_start - ep_start % R.BAR_SECONDS
    n_bars = math.ceil((ep_end - g0) / R.BAR_SECONDS)
    bar_start = g0 + R.BAR_SECONDS * np.arange(n_bars)
    bar_end = np.minimum(bar_start + R.BAR_SECONDS, ep_end)

    raw = pd.read_parquet(root / "trades" / f"{fname}.parquet")
    tr, _ = normalize_trades(raw, mid)
    tr["ts"] = (tr.timestamp - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    # Whole-second window, as in the collectors: fills inside the closing
    # second (close_time has whole-second resolution) belong to the episode.
    sec = np.floor(tr.ts)
    tr = tr[(sec >= ep_start) & (sec <= ep_end)].copy()
    tr["tau"] = ((tr.ts - ep_start) / length).clip(upper=1.0)
    tr["bar_index"] = np.minimum(((tr.ts - g0) // R.BAR_SECONDS).astype(int), n_bars - 1)
    tr["buy_q"] = np.where(tr.sign > 0, tr.quantity, 0.0)
    tr["sell_q"] = np.where(tr.sign < 0, tr.quantity, 0.0)
    tr["pq"] = tr.price * tr.quantity
    agg = tr.groupby("bar_index").agg(
        n_trades=("price", "size"), n_buy=("sign", lambda s: int((s > 0).sum())),
        n_sell=("sign", lambda s: int((s < 0).sum())), buy_contracts=("buy_q", "sum"),
        sell_contracts=("sell_q", "sum"), volume_contracts=("quantity", "sum"), pq=("pq", "sum"),
        first_trade_price=("price", "first"), last_trade_price=("price", "last"),
        high_trade_price=("price", "max"), low_trade_price=("price", "min"))
    bars = pd.DataFrame({"bar_index": np.arange(n_bars), "bar_start_ts": bar_start, "bar_end_ts": bar_end})
    bars = bars.merge(agg, left_on="bar_index", right_index=True, how="left")
    for c in ["n_trades", "n_buy", "n_sell", "buy_contracts", "sell_contracts", "volume_contracts", "pq"]:
        bars[c] = bars[c].fillna(0)
    bars[["n_trades", "n_buy", "n_sell"]] = bars[["n_trades", "n_buy", "n_sell"]].astype(int)
    bars["signed_flow"] = bars.buy_contracts - bars.sell_contracts
    bars["vwap"] = np.where(bars.volume_contracts > 0, bars.pq / bars.volume_contracts.replace(0, np.nan), np.nan)
    bars = bars.drop(columns="pq")
    # Last trade price as of bar end, including trades before the window.
    pre = raw_last_price(raw, mid, ep_start)
    bars["last_price"] = bars.last_trade_price.ffill().fillna(pre if pre is not None else np.nan)

    if venue == "kalshi":
        q = kalshi_quotes(pd.read_parquet(root / "candlesticks" / f"{fname}.parquet"))
        asof = pd.merge_asof(bars[["bar_end_ts"]].astype("int64"), q, left_on="bar_end_ts", right_on="obs_ts",
                             direction="backward")
        observed = q.obs_ts - R.BAR_SECONDS
        bars["quote_observed"] = bars.bar_start_ts.isin(observed).to_numpy()
        bars["quote_age_s"] = (bars.bar_end_ts - asof.obs_ts).to_numpy()
        bars["yes_bid"], bars["yes_ask"] = asof.yes_bid.to_numpy(), asof.yes_ask.to_numpy()
        bars["open_interest"] = asof.open_interest.to_numpy()
        bars["two_sided"] = (bars.yes_bid > 0) & (bars.yes_ask < 1) & (bars.yes_bid < bars.yes_ask)
        bars["yes_mid"] = np.where(bars.two_sided, (bars.yes_bid + bars.yes_ask) / 2, np.nan)
        bars["spread"] = np.where(bars.two_sided, bars.yes_ask - bars.yes_bid, np.nan)
        bars["quote_source"] = "kalshi_minute_candle"
    else:
        ph = pd.read_parquet(root / "price_history" / f"{fname}.parquet")
        asof = pd.merge_asof(bars[["bar_end_ts"]].astype("int64"), ph.astype({"ts": "int64"}),
                             left_on="bar_end_ts", right_on="ts", direction="backward")
        bars["quote_observed"] = ((asof.ts > bars.bar_start_ts) & (asof.ts <= bars.bar_end_ts)).to_numpy()
        bars["quote_age_s"] = (bars.bar_end_ts - asof.ts).to_numpy()
        bars["yes_bid"] = bars["yes_ask"] = bars["spread"] = np.nan
        bars["open_interest"] = np.nan
        bars["two_sided"] = False
        bars["yes_mid"] = asof.yes_mid.to_numpy()
        bars["quote_source"] = "polymarket_price_history_mid"
    # One reference probability for the agent: quote mid, else last trade.
    bars["p_ref"] = bars.yes_mid.fillna(bars.last_price)
    bars["p_ref_source"] = np.where(bars.yes_mid.notna(), "mid", np.where(bars.last_price.notna(), "last_trade", "none"))

    bars["tau_start"] = ((bars.bar_start_ts - ep_start) / length).clip(0, 1)
    bars["tau_end"] = ((bars.bar_end_ts - ep_start) / length).clip(0, 1)
    bars["seconds_to_close"] = ep_end - bars.bar_end_ts
    bars["bar_start_utc"] = pd.to_datetime(bars.bar_start_ts, unit="s", utc=True)
    for k in ["venue", "category", "series_ticker", "event_ticker", "market_id", "split", "y_outcome"]:
        bars[k] = row[k]
    bars["n_bars"] = n_bars
    lead = ["venue", "category", "series_ticker", "event_ticker", "market_id", "split", "bar_index", "n_bars",
            "bar_start_utc", "bar_start_ts", "bar_end_ts", "tau_start", "tau_end", "seconds_to_close",
            "p_ref", "p_ref_source", "yes_bid", "yes_ask", "yes_mid", "spread", "two_sided",
            "quote_observed", "quote_age_s", "quote_source"]
    bars = bars[lead + [c for c in bars.columns if c not in lead]]
    tr_out = tr[["trade_id", "timestamp", "ts", "tau", "bar_index", "price", "quantity", "sign"]].assign(
        venue=venue, market_id=mid)
    return bars, tr_out


def raw_last_price(raw: pd.DataFrame, mid: str, before_ts: int) -> float | None:
    tr, _ = normalize_trades(raw, mid)
    tr = tr[tr.timestamp < pd.Timestamp(before_ts, unit="s", tz="UTC")]
    return float(tr.price.iloc[-1]) if len(tr) else None


def paper_reference() -> dict:
    """Expected taker arrivals per episode in Paper 1 at p=0.5 and zero spread (B=1)."""
    xi, A0, A1 = 2.0, 100.0, 150.0
    integral = A0 + A1 * ((math.exp(xi) - 1) / xi - 1) / (math.exp(xi) - 1)
    per_side = integral * math.sqrt(0.25)
    return {"arrivals_per_side_per_episode_at_p0.5_zero_spread": per_side,
            "arrivals_total_per_episode": 2 * per_side, "trade_size_contracts": 10.0,
            "mc_steps": R.PAPER_N_STEPS,
            "note": "Paper 1 parameters are illustrative, not calibrated; compare orders of magnitude only."}


def run() -> dict:
    out = R.DATA_DIR / "dataset"
    (out / "trades").mkdir(parents=True, exist_ok=True)
    frames = []
    for venue in ["kalshi", "polymarket"]:
        p = R.DATA_DIR / venue / "markets" / "contracts.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("No contracts.parquet found; run the collectors first.")
    contracts = pd.concat(frames, ignore_index=True)
    contracts["event_ticker"] = contracts.event_ticker.replace("", np.nan).fillna(contracts.market_id)
    contracts["split"] = assign_splits(contracts)
    all_bars = []
    for _, row in contracts.iterrows():
        bars, tr = build_contract(row)
        (out / "trades" / row.venue).mkdir(parents=True, exist_ok=True)
        tr.to_parquet(out / "trades" / row.venue / f"{safe_filename(row.market_id)}.parquet", index=False)
        all_bars.append(bars)
        log.info("%s %s %s: %d bars, %d trades", row.venue, row.category, row.market_id[:40], len(bars), len(tr))
    bars = pd.concat(all_bars, ignore_index=True)
    stats = bars.groupby("market_id").agg(
        two_sided_share=("two_sided", "mean"), median_spread=("spread", "median"),
        quote_observed_share=("quote_observed", "mean"), trades_per_bar=("n_trades", "mean"),
        p_ref_missing_share=("p_ref", lambda s: s.isna().mean()))
    contracts = contracts.merge(stats, left_on="market_id", right_index=True, how="left")
    contracts.to_parquet(out / "contracts.parquet", index=False)
    bars.to_parquet(out / "bars.parquet", index=False)
    per_cat = {}
    for (venue, cat), g in contracts.groupby(["venue", "category"]):
        per_cat[f"{venue}/{cat}"] = {
            "contracts": int(len(g)), "events": int(g.event_ticker.nunique()),
            "series": sorted(g.series_ticker.unique().tolist()),
            "split_counts": g.split.value_counts().to_dict(),
            "episode_hours": float(g.episode_hours.median()),
            "median_bars_per_episode": float(g.episode_bars.median()),
            "median_trades_per_episode": float(g.episode_trades.median()),
            "median_trades_per_minute": float(g.trades_per_bar.median()),
            "median_contracts_per_episode": float(g.episode_contracts.median()),
            "median_spread": None if g.median_spread.isna().all() else float(g.median_spread.median()),
            "median_two_sided_share": float(g.two_sided_share.median()),
            "yes_share": float(g.y_outcome.mean()),
        }
    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(), "data_dir": str(R.DATA_DIR),
                "bar_seconds": R.BAR_SECONDS, "split_fractions": R.SPLIT_FRACTIONS,
                "episode_hours": {k: v["episode_hours"] for k, v in R.CATEGORIES.items()},
                "density_gates": {"min_episode_trades": R.MIN_EPISODE_TRADES,
                                  "min_active_bar_share": R.MIN_ACTIVE_BAR_SHARE},
                "rows": {"contracts": int(len(contracts)), "bars": int(len(bars))},
                "per_venue_category": per_cat, "paper1_reference": paper_reference()}
    save_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    m = run()
    print(json.dumps({"rows": m["rows"], "per_venue_category": m["per_venue_category"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
