# Live quoting dashboard (paper only)

Runs the repository's market-making strategies on **live Kalshi production
data** and shows their quotes next to the real market in a browser.
No orders are ever sent. Fills are simulated from real trades.

Strategies: HJB (Paper 1, `simulator/hjb_policy.py`), fixed spread and
inventory skew (`simulator/policies.py`), and the PPO agent from
`rl_agent/models/ppo_capstone_broad_v2.zip` (Naveen's calibrated-simulator
agent).

## Run

From the repository root:

```bash
pip install -r data_pipeline/requirements.txt -r calibration/requirements.txt
pip install -r rl_agent/requirements-rl.txt      # optional: enables the PPO agent
python -m dashboard.server
```

Open http://127.0.0.1:8050. Useful options:

```bash
python -m dashboard.server --decision-seconds 60            # faster updates (training used 300)
python -m dashboard.server --tickers KXNFLGAME-26NOV15NEDET-DET
python -m dashboard.server --close-within-hours 48 --max-markets 4
python -m dashboard.server --no-ppo
```

With no tickers given it picks the most-traded open binary markets whose
expected close is within the next 12 hours, one per event, across sports,
crypto, financials and economics series (`config.py`). Stop with Ctrl+C.
Every quote, fill, cancel and settlement is written to
`dashboard/runs/<start time>/ledger.jsonl` (not committed).

## What happens

| Every | Step |
|---|---|
| 5 s | Fetch all tracked order books in one request (`GET /markets/orderbooks`) |
| 5 s | Fetch new trades per market (`GET /markets/trades`) and simulate fills |
| 300 s (configurable) | Each strategy decides new quotes from the current book |
| 60 s near close | Check market status; settle accounts when the result is published |

Data comes from Kalshi's public endpoints, which need no API key.
`kalshi_feed.py` refuses every non-GET request, every non-market path and
every host other than production, so it cannot place an order.

## Fill rule

The same as the simulator (`RestingOrder.match`, `BinaryAccount`), with the
real trade price as the limit:

1. A new quote becomes live after 1 s.
2. Its queue ahead is the resting size at its price and better when placed.
   Improving the best price means nothing is ahead. Keeping the same price at
   the next decision keeps the queue position; a new price resets it.
3. A YES-taker trade at price P can fill our ask if the ask is at or below P.
   A NO-taker trade can fill our bid if the bid is at or above P.
4. Trade volume first clears the queue ahead. We then get at most 10% of
   the rest, up to our remaining size and inventory limit (100).
5. Book changes alone never create fills. Block trades are ignored.
6. Quotes are pulled when the book is one-sided or stale (older than 60 s).

Fees use `simulator.fees.FeeSchedule` (maker fee multiplier 1). PnL is marked
at the book mid until settlement.

## Episode clock

The strategies were trained on the last hour before close. The clock uses
`expected_expiration_time` when Kalshi provides it (sports markets list a
late backstop `close_time` and close early when the game ends). More than an
hour before close the clock is held at the episode start (t = 0); the
dashboard shows each strategy's clock as "Episode t".

## Limitations

- Fills are estimates. Kalshi's real queue, cancellations ahead of us and
  our own effect on other traders are not observable. Not crediting
  cancellations makes fills conservative.
- Books are sampled every 5 s, so the queue estimate is coarse.
- Strategies see the book mid as the probability; in the simulator they saw
  a latent reference price.
- On liquid markets the queue at the best price can be tens of thousands of
  contracts, so quotes at the touch rarely fill. That is real behaviour.

## Tests

```bash
python -m pytest dashboard/tests -q
```

The tests cover latency, queue clearing, keeping and losing queue priority,
wrong-side and block trades, one-sided and stale books, settlement, the
read-only guard, and that every strategy returns post-only quotes (PPO only
when stable-baselines3 is installed).
