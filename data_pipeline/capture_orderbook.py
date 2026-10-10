"""Bounded, read-only collection of current public L2 snapshots.

This cannot recover historical depth for the already settled calibration
contracts. For future execution studies, collect active tickers alongside
ordered trades/book deltas. REST snapshots alone do not recover queue priority.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from kalshi_client import KalshiClient

DEFAULT_OUTPUT = Path(__file__).resolve().parent/"output"/"paper2"/"orderbook_snapshots"


def capture(ticker, output_dir=DEFAULT_OUTPUT, samples=1, interval_seconds=5., client=None):
    if (not re.fullmatch(r"[A-Z0-9_.-]+",ticker) or not isinstance(samples,int) or not 1<=samples<=10000 or
            not math.isfinite(interval_seconds) or interval_seconds<1):
        raise ValueError("Invalid ticker, bounded sample count or interval")
    client = client or KalshiClient()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    # Append timestamped observations rather than overwriting earlier capture.
    path = output_dir/(ticker+".jsonl")
    for index in range(samples):
        started = datetime.now(timezone.utc).isoformat()
        payload = client._get(f"/markets/{ticker}/orderbook", {"depth":0})
        received = datetime.now(timezone.utc).isoformat()
        if "orderbook_fp" not in payload:
            raise ValueError("Expected current fixed-point orderbook schema")
        row = {"ticker":ticker,"request_started_utc":started,"available_at_utc":received,
            "source":"public_rest_orderbook", "sequence_number":None, "payload":payload}
        with path.open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
        if index+1<samples:
            time.sleep(interval_seconds)
    return {"file":str(path),"snapshots_written":samples,"historical_depth_recovered":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker",required=True)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--samples",type=int,default=1)
    parser.add_argument("--interval-seconds",type=float,default=5.)
    args=parser.parse_args()
    print(json.dumps(capture(args.ticker,args.output_dir,args.samples,args.interval_seconds),indent=2))
