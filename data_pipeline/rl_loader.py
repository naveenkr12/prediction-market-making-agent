"""
Read the RL dataset built by rl_build_bars.py / rl_record_orderbooks.py.

    import rl_loader as L
    train = L.contracts(split="train", category="Sports")
    bars  = L.bars(train.market_id.iloc[0])                # 1-minute bars
    grid  = L.paper_grid(train.market_id.iloc[0])          # Paper 1 grid, 1000 steps
    for ts, book in L.book_snapshots("kalshi", ticker):    # recorded live L2
        ...

paper_grid puts a real episode on the same clock as hjb_baseline's Monte
Carlo: step n covers t in [n/N, (n+1)/N) * T with T = 1. Quotes and p_ref are
as of the step start (what a policy could observe when quoting); trade counts
and flows are those executed during the step (what the paper samples as
Poisson arrivals). Steps shorter than one minute repeat the minute quote,
flagged by quote_age_s.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import rl_config as R
from utils import safe_filename

DATASET = R.DATA_DIR / "dataset"


def contracts(split: str | None = None, category: str | None = None, venue: str | None = None) -> pd.DataFrame:
    c = pd.read_parquet(DATASET / "contracts.parquet")
    for col, val in (("split", split), ("category", category), ("venue", venue)):
        if val is not None:
            c = c[c[col] == val]
    return c.reset_index(drop=True)


def bars(market_id: str | None = None, **filters) -> pd.DataFrame:
    flt = [("market_id", "==", market_id)] if market_id else None
    b = pd.read_parquet(DATASET / "bars.parquet", filters=flt)
    for col, val in filters.items():
        b = b[b[col] == val]
    return b.sort_values(["market_id", "bar_index"]).reset_index(drop=True)


def trades(market_id: str, venue: str | None = None) -> pd.DataFrame:
    venue = venue or contracts().set_index("market_id").venue[market_id]
    return pd.read_parquet(DATASET / "trades" / venue / f"{safe_filename(market_id)}.parquet")


def paper_grid(market_id: str, n_steps: int = R.PAPER_N_STEPS, T: float = 1.0) -> pd.DataFrame:
    meta = contracts().set_index("market_id").loc[market_id]
    b, tr = bars(market_id), trades(market_id, meta.venue)
    t0 = pd.Timestamp(meta.episode_start_utc).timestamp()
    t1 = pd.Timestamp(meta.episode_end_utc).timestamp()
    tau = np.arange(n_steps) / n_steps
    step_ts = t0 + tau * (t1 - t0)
    # A bar's quote is known at its end; use the latest bar ended by step start.
    q = b[["bar_end_ts", "p_ref", "yes_bid", "yes_ask", "yes_mid", "spread", "two_sided", "quote_age_s"]]
    g = pd.merge_asof(pd.DataFrame({"step_ts": step_ts}), q.astype({"bar_end_ts": "float64"}),
                      left_on="step_ts", right_on="bar_end_ts", direction="backward")
    g["quote_age_s"] = g.quote_age_s + (g.step_ts - g.bar_end_ts)
    idx = np.minimum((tr.tau.to_numpy() * n_steps).astype(int), n_steps - 1)
    def per_step(values):
        return np.bincount(idx, weights=values, minlength=n_steps)
    buy = tr.sign.to_numpy() > 0
    out = pd.DataFrame({
        "step": np.arange(n_steps), "t": T * tau, "dt": T / n_steps, "tau": tau,
        "step_start_utc": pd.to_datetime(step_ts, unit="s", utc=True),
        "seconds_to_close": t1 - step_ts,
        "p_ref": g.p_ref, "yes_bid": g.yes_bid, "yes_ask": g.yes_ask, "yes_mid": g.yes_mid,
        "spread": g.spread, "two_sided": g.two_sided.astype("boolean").fillna(False).astype(bool), "quote_age_s": g.quote_age_s,
        "n_buy": per_step(buy.astype(float)).astype(int), "n_sell": per_step((~buy).astype(float)).astype(int),
        "buy_contracts": per_step(np.where(buy, tr.quantity, 0.0)),
        "sell_contracts": per_step(np.where(~buy, tr.quantity, 0.0)),
    })
    out["n_trades"] = out.n_buy + out.n_sell
    out["signed_flow"] = out.buy_contracts - out.sell_contracts
    out.attrs.update(market_id=market_id, venue=meta.venue, category=meta.category,
                     y_outcome=int(meta.y_outcome), episode_seconds=t1 - t0,
                     seconds_per_step=(t1 - t0) / n_steps)
    return out


def book_snapshots(venue: str, market_id: str, include_unchanged: bool = False) -> Iterator[tuple[str, object]]:
    """Yield (available_at_utc, BookSnapshot) from recorded live L2, in time order.

    One-sided books (BookSnapshot requires bid and ask) are skipped. With
    include_unchanged=True, an unchanged observation re-yields the last book.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from simulator.execution import BookSnapshot
    files = sorted((R.DATA_DIR / "live_orderbooks" / venue).glob(f"*/{safe_filename(market_id)}.jsonl.gz"))
    last = None
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                row = json.loads(line)
                if row["unchanged"]:
                    if include_unchanged and last is not None:
                        yield row["available_at_utc"], last
                    continue
                try:
                    last = BookSnapshot.from_kalshi(row["payload"], row["available_at_utc"])
                except ValueError:
                    last = None
                    continue
                yield row["available_at_utc"], last
