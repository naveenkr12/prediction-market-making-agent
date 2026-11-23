# Kalshi live paper trading: instructions for an LLM agent

Purpose: run this repository's ready-made **market-making** strategies
(frozen HJB, fixed spread, inventory skew, PPO agent from PR #3) against
**live Kalshi production market data**. We never send orders. We keep our own
ledger and simulate whether our resting quotes would have been filled by the
real takers we observe.

Everything below was checked against Kalshi's docs and live endpoints on
**3 November 2026**. Re-verify anything marked *verify* before relying on it.

---

## 1. The one thing to understand first: demo data is NOT production data

Kalshi's demo exchange (`demo.kalshi.co`) lists the **same market tickers** as
production. Its **order books and trades are its own**: thin, and made by demo
users. Kalshi's docs say demo prices and behaviour "may not be reflective of
those in real markets". We measured it on the three busiest open NFL markets:

| | Production | Demo |
|---|---|---|
| Book depth (contracts, both sides) | 8.1M to 20.6M | 4,300 to 9,500 |
| Price levels in the book | 81 to 97 | 9 to 13 |
| Trades in the last 24 h | 413 to 1,000+ | 5 to 25 |
| Best bid / ask | e.g. 0.18 / 0.19 | e.g. 0.19 / 0.20 |

Demo top-of-book roughly tracks production, but there is almost no volume, so
**fills simulated on demo data are meaningless**. Other markets are worse
(for example a Fed market quoted 0.25 / 0.28 in production and 0.32 / 1.00 in
demo).

**Therefore:**

| Need | Use |
|---|---|
| Market data (books, trades, prices, settlement) | **Production**, read only |
| Fills for our quotes | **Simulated** from production trades (section 6.5) |
| Any order placement, demo or production | Not used. See section 5. |

---

## 2. Endpoints

| | Production | Demo |
|---|---|---|
| REST | `https://external-api.kalshi.com/trade-api/v2` | `https://external-api.demo.kalshi.co/trade-api/v2` |
| REST (also supported) | `https://api.elections.kalshi.com/trade-api/v2` | `https://demo-api.kalshi.co/trade-api/v2` |
| WebSocket | `wss://external-api-ws.kalshi.com/trade-api/ws/v2` | `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2` |
| WebSocket (also supported) | `wss://api.elections.kalshi.com/trade-api/ws/v2` | `wss://demo-api.kalshi.co/trade-api/ws/v2` |

Credentials are **not shared** between environments. Demo keys only work on
demo hosts, and production keys only on production hosts.

Docs: https://docs.kalshi.com/getting_started/api_environments ,
machine-readable specs at `https://docs.kalshi.com/openapi.yaml` (REST) and
`https://docs.kalshi.com/asyncapi.yaml` (WebSocket).

---

## 3. Two ways to get live production data

### Mode A: no account, REST polling (default, works today)

These production endpoints need **no API key** (verified live):

| Data | Request |
|---|---|
| Full order books, many markets at once | `GET /markets/orderbooks?tickers=T1&tickers=T2...` (repeat `tickers`; 25 per call tested fine) |
| One book | `GET /markets/{ticker}/orderbook` |
| Recent trades for a market | `GET /markets/trades?ticker=T&min_ts=<unix seconds>&limit=1000` |
| Market metadata, status, result | `GET /markets/{ticker}`; `GET /markets?series_ticker=S&status=open` |

This repository already does this: `data_pipeline/rl_record_orderbooks.py`
polls batched books every 10 s and writes them in the format that
`simulator.execution.BookSnapshot.from_kalshi` reads.

Limits: Kalshi does not document unauthenticated rate limits. Bursts of 25
requests were fine in testing. Stay at or below about 5 requests per second,
and back off on HTTP 429.

Suggested cadence: books every 5 to 10 s (batched), and trades every 5 s per
market, for up to about 10 markets. The strategies decide every 5 minutes, so
this is plenty.

Weakness: trades between polls are fetched by time, so nothing is lost. Book
changes between polls are not seen, so queue estimates are coarser.

### Mode B: production API key, WebSocket (better fidelity)

The WebSocket **rejects unauthenticated connections with HTTP 401** in both
environments (verified). For production data you need a **production** API
key from a real, verified Kalshi account (kalshi.com, Profile Settings, API
Keys). A demo key does **not** unlock production data. Using the key only to
read market data and never calling order endpoints keeps this read only.

Channels:

| Channel | Gives | Market filter |
|---|---|---|
| `orderbook_delta` | `orderbook_snapshot`, then `orderbook_delta` messages | required: `market_ticker` or `market_tickers` |
| `trade` | every public trade | optional |
| `ticker` | best bid/ask, last price, volume | optional |
| `market_lifecycle_v2` | open / close / settle events | optional |

Subscribe:

```json
{"id": 1, "cmd": "subscribe",
 "params": {"channels": ["orderbook_delta", "trade"], "market_tickers": ["TICKER1", "TICKER2"]}}
```

Message shapes (from `asyncapi.yaml`):

```json
{"type": "orderbook_snapshot", "sid": 2, "seq": 2,
 "msg": {"market_ticker": "FED-23DEC-T3.00", "clock": 4182057,
         "yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "333.00"]],
         "no_dollars_fp":  [["0.5400", "20.00"],  ["0.5600", "146.00"]]}}

{"type": "orderbook_delta", "sid": 2, "seq": 3,
 "msg": {"market_ticker": "FED-23DEC-T3.00", "clock": 4182061,
         "price_dollars": "0.9600", "delta_fp": "-54.00", "side": "yes",
         "ts_ms": 1669149841000}}

{"type": "trade", "sid": 11, "seq": 2,
 "msg": {"trade_id": "d91b...", "market_ticker": "HIGHNY-22DEC23-B53.5",
         "yes_price_dollars": "0.3600", "no_price_dollars": "0.6400", "count_fp": "136.00",
         "taker_outcome_side": "no", "taker_book_side": "ask", "is_block_trade": false,
         "ts_ms": 1669149841000}}
```

Rules:

- The book lists **bids only**. `yes_dollars_fp` holds YES bids, and
  `no_dollars_fp` holds NO bids. A NO bid at price x is a YES ask at 1 − x.
- `seq` must increase by exactly 1 within a subscription (`sid`). **On a gap,
  the book is wrong**: send `update_subscription` with `get_snapshot` for all
  markets, or unsubscribe and subscribe again. A repeated `subscribe` does not
  resync.
- Ignore message types and fields you do not recognise. Kalshi adds fields.
- Reconnect with exponential backoff. The Python `websockets` library handles
  ping/pong itself.
- Note the field names: WebSocket uses `yes_dollars_fp`/`no_dollars_fp`, while
  REST uses `orderbook_fp.yes_dollars`/`no_dollars`. Convert before calling
  `BookSnapshot.from_kalshi` (code in section 6).

---

## 4. Authentication (only needed for Mode B)

Three headers on every authenticated request and on the WebSocket handshake:

| Header | Value |
|---|---|
| `KALSHI-ACCESS-KEY` | key id |
| `KALSHI-ACCESS-TIMESTAMP` | Unix time in **milliseconds** |
| `KALSHI-ACCESS-SIGNATURE` | base64 signature of `timestamp + METHOD + path` |

`path` is the full path **without the query string**, for example
`/trade-api/v2/portfolio/balance`. For the WebSocket, sign `GET` and
`/trade-api/ws/v2`. Keys are Ed25519 (the default) or RSA. RSA uses PSS with
SHA-256, MGF1 SHA-256 and salt length equal to the digest length. Tested
helper:

```python
import base64, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, ed25519

def load_key(pem_path):
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def kalshi_headers(key_id, private_key, method, path):
    ts = str(int(time.time() * 1000))
    message = (ts + method.upper() + path.split("?")[0]).encode()
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        sig = private_key.sign(message)
    else:  # RSA
        sig = private_key.sign(message,
                               padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                           salt_length=padding.PSS.DIGEST_LENGTH),
                               hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts}

# WebSocket: websockets.connect(url, additional_headers=kalshi_headers(key_id, key, "GET", "/trade-api/ws/v2"))
```

Secrets: read the key id and PEM path from environment variables
(`KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`). Never print them, log them,
commit them, or paste them into chat. Kalshi shows the private key only once.

---

## 5. Why we do not use the demo account or the order endpoints

We are a **market maker**: we post resting bid/ask quotes and earn the spread
when takers trade against them. Our results depend entirely on real takers
reaching our quotes. That is why we simulate fills from live **production**
trades (section 6.5) instead of placing orders anywhere.

- **Demo** is Kalshi's paper-trading sandbox, but it is a separate market with
  almost no takers (5 to 25 trades a day on the busiest NFL markets). Quotes
  posted there would almost never fill and would say nothing about the real
  market.
- **Order endpoints** (Create Order V2, `POST /portfolio/events/orders`, and
  cancel) are not needed. We never send orders. Only the market-data parts of
  the Trade API (`/trade-api/v2` books, trades, markets and the WebSocket) are
  used.

Revisit this only if the team later wants to test a real order loop (order
acknowledgements, cancels, latency, Kalshi's own queue position), which would
be done on demo with mock money and is out of scope here.

## 6. The paper-trading loop

### 6.1 Market selection and episode clock

- Choose open binary markets in our categories (see `data_pipeline/rl_config.py`
  for the series lists) with real volume (`volume_24h_fp`).
- Our strategies trade the **last hour before close** and decide **every 300
  s** (12 decisions per episode), as in `simulator.trading_env` and
  `rl_agent/calibrated_env.py`.
- Close time: Kalshi `close_time` is often a late backstop. Sports markets
  close early when the game ends (`can_close_early: true`). For the time
  feature use `expected_expiration_time` if present, else `close_time`. Log
  which one you used, and record the real close when it happens.
- Read the price grid from the market's `price_ranges` / `price_level_structure`.
  Most markets use a 0.01 tick; some use finer ticks.

### 6.2 Live book (works for WebSocket snapshots and deltas)

```python
from decimal import Decimal
from simulator.execution import BookSnapshot

class LiveBook:
    def __init__(self):
        self.levels = {"yes": {}, "no": {}}   # price string -> Decimal size
        self.last_seq = None

    def on_message(self, m):
        seq = m.get("seq")
        if self.last_seq is not None and seq is not None and seq != self.last_seq + 1:
            raise RuntimeError("seq gap: request a fresh snapshot")
        self.last_seq = seq
        msg = m["msg"]
        if m["type"] == "orderbook_snapshot":
            self.levels = {s: {p: Decimal(q) for p, q in msg.get(f"{s}_dollars_fp") or []} for s in ("yes", "no")}
        elif m["type"] == "orderbook_delta":
            book = self.levels[msg["side"]]
            size = book.get(msg["price_dollars"], Decimal(0)) + Decimal(msg["delta_fp"])
            if size > 0: book[msg["price_dollars"]] = size
            else: book.pop(msg["price_dollars"], None)

    def snapshot(self, timestamp=None):
        payload = {"orderbook_fp": {f"{s}_dollars": [[p, str(q)] for p, q in sorted(self.levels[s].items())]
                                    for s in ("yes", "no")}}
        return BookSnapshot.from_kalshi(payload, timestamp)   # raises if one side is empty
```

In Mode A, skip `LiveBook` and pass the REST response straight to
`BookSnapshot.from_kalshi(response)`. In the WebSocket case, keep one seq
counter per subscription; the class above assumes one market per sid.

### 6.3 Observation the strategies expect

```python
book = live_book.snapshot()
p = (book.bid + book.ask) / 2          # reference probability (live proxy for the simulator's latent price)
observation = {"probability": p, "bid": book.bid, "ask": book.ask,
               "inventory": ledger_inventory,
               "elapsed_seconds": now - open_ts}   # for HJBReferencePolicy
```

### 6.4 Strategy adapters (all from this repo)

**Frozen HJB**, using `simulator.hjb_policy.HJBReferencePolicy`:

```python
from simulator.hjb_policy import HJBReferencePolicy
hjb = HJBReferencePolicy()                      # time_mapping="episode"
lifetime = close_ts - open_ts
action, info = hjb.action(observation, start_elapsed=lifetime - 3600, lifetime=lifetime,
                          inventory_limit=100, tick="0.01", lot="0.01")
```

**Fixed spread / inventory skew**, using `simulator.policies.quote_policy(observation,
size=10, inventory_skew=0.0 or 0.0005)`.

**PPO agent (PR #3, branch `naveenkr12:rl-agent/calibrated-simulator`)**:
`rl_agent/calibrated_env.py` defines the contract, which must match exactly.
Install `rl_agent/requirements-rl.txt` (gymnasium, stable-baselines3, torch).

```python
import numpy as np
from stable_baselines3 import PPO
from simulator.hjb_policy import executable_action

model = PPO.load("rl_agent/models/ppo_capstone_broad_v2")     # the trained agent in PR #3; never the *_buggy_reward_DO_NOT_USE files
frac = float(np.clip((now - (close_ts - 3600)) / 3600, 0, 1))
vec = np.array([2 * frac - 1, 2 * p - 1, q / 100], dtype=np.float32)
a, _ = model.predict(vec, deterministic=True)
a = np.clip(a, -1, 1)
bid = p - (0.03 + 0.17 * a[0]); ask = p + (0.03 + 0.17 * a[1])
action, diag = executable_action(bid, ask, min(10, max(0, 100 - q)), min(10, max(0, 100 + q)),
                                 {"bid": book.bid, "ask": book.ask}, tick="0.01", lot="0.01")
```

Every adapter returns a `simulator.execution.QuoteAction(bid, ask, bid_size, ask_size)`.
A side with size 0 or price `None` is not quoted.

### 6.5 Simulated fills (conservative, same rule as the simulator)

When a quote is placed or replaced:

- The quote becomes live after a latency delay (use 1 s; measure the real
  round trip and log it).
- Initial `queue_ahead` = `book.queue_ahead(side, Decimal(price), proxy)`
  (pass the price as a `Decimal`). This counts the resting size at our price
  **and better**. A price that improves the best quote starts with 0 ahead.
- Replacing a quote loses queue priority (new queue). Keeping it unchanged
  keeps the queue.

On each public trade after the quote is live, reuse the simulator's matcher:

```python
from decimal import Decimal
from simulator.execution import RestingOrder

def on_trade(order: RestingOrder, trade: dict, participation=Decimal("0.1"), lot=Decimal("0.01")):
    sign = 1 if trade["taker_outcome_side"] == "yes" else -1   # YES taker lifts asks, NO taker hits bids
    filled = order.match(sign, Decimal(trade["count_fp"]), Decimal(trade["yes_price_dollars"]), participation, lot)
    order.remaining -= filled          # match() does not decrement remaining; the caller must
    return filled
```

What `match` does: it ignores trades on the wrong side or at a worse price
(our ask must be at or below the trade price; our bid at or above it). It
first uses the trade volume to clear `queue_ahead`. Then we get at most
`participation` (base 10%) of the rest, capped by our remaining size. Tested:
an ask at 0.44 with 50 ahead got 0 from a 40-lot trade at 0.44, then 7 from
an 80-lot trade at 0.45.

Do **not** count fills from book deltas alone. A shrinking level may be a
cancel, not a trade. Skip block trades (`is_block_trade: true`).

### 6.6 Ledger and settlement

- Book every fill with `simulator.accounting.BinaryAccount.fill(side, price,
  quantity)`, which applies `simulator.fees.FeeSchedule` maker fees,
  collateral and the 100-contract inventory limit.
- Mark to market every minute at `p`. At settlement, poll `GET /markets/{ticker}`
  until `status` is `settled`/`finalized` and `result` is `yes`/`no`, then
  `BinaryAccount.settle(outcome)`.
- Write an append-only JSONL ledger: every quote change, every public trade
  seen, every simulated fill with the queue ahead before and after, every
  mark and the settlement. Include the observation and the strategy name, so
  any number can be recomputed.
- Report per episode the same metrics as `simulator.benchmark_metrics`:
  realized PnL, fills, mean absolute inventory, VaR95/ES95 across episodes.
  Run all strategies on the **same** markets and episodes so comparisons are
  paired.

---

## 7. Hard safety rules for the agent

1. **Never send orders.** The paper engine uses GET requests and the
   WebSocket only. Block every `/portfolio/` request in code (an allowlist of
   GET market-data paths), not by convention.
2. Use production hosts for data only; never point the engine at demo data.
3. Never commit, print or upload API keys or PEM files. Add `*.pem` and
   `.env` to `.gitignore`.
4. Respect rate limits: back off exponentially on 429/5xx. Never poll faster
   than the strategy needs.
5. Market data stays inside the team (exchange data terms).
6. If the book is stale (no update for 60 s), a `seq` gap is not yet
   repaired, or one side of the book is empty, pull our simulated quotes
   (no fills) until it is healthy again, and log why.
7. Stop and ask a human before changing strategy parameters, using real
   money, or anything not described here.

---

## 8. Known limitations

- Simulated fills ignore our own price impact. We quote 10 contracts against
  books with millions, so this is minor, but not zero for thin markets.
- Queue position is estimated from the aggregated book, not from Kalshi's
  real price-time queue (which needs our own resting order). Cancellations
  ahead of us are not credited, so fills are understated, which is
  conservative.
- Mode A sees book changes only at poll times.
- Queues on liquid production books are deep. On 3 November 2026, joining the
  best bid of the busiest NFL market meant about 96,000 contracts ahead, so
  passive quotes at the touch will rarely fill. That is real, not a bug.
- The PPO agent was trained on the `broad_based` group with one seed. It
  quotes rarely (0.10 fills per episode in PR #3's evaluation), so live
  results will be sparse. Run many episodes before comparing.
- Live `p` is the book mid. The simulator's `probability` is a latent
  reference price, so the strategies see a slightly different input live.

## 9. Acceptance checks before trusting results

- [ ] Book from `LiveBook` equals a fresh REST `GET /markets/{ticker}/orderbook` taken at the same moment.
- [ ] Trades from the WebSocket equal `GET /markets/trades` for the same window (by `trade_id`).
- [ ] Replaying a recorded hour through the paper engine twice gives identical ledgers.
- [ ] The order guard raises if any `/portfolio/` request is attempted.
- [ ] HJB on recorded data gives fills and PnL in the same range as `simulator/evaluate_hjb.py`.
