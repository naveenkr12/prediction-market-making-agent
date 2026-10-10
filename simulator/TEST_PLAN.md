# Trading environment test plan

Target: every financial state transition has an independent numerical check,
and every full episode has a ledger reconstruction. No code-coverage percentage
is claimed. Keep the existing collection/calibration regression suite passing.

| Area | Test type | Target and example |
|---|---|---|
| Binary account | Unit | Long/short × YES/NO outcomes; fully collateralized settlement, repeat settlement rejected |
| Money and fees | Unit + independent audit | Exact Decimal balances, buy/sell rounding, published rounding example, multi-fill rebate accumulator |
| Admission and risk | Unit + integration | Both fill orderings stay funded, fractional lots, no position overshoot, free cash after both quote reserves nonnegative |
| Quote actions | Unit | Reject own crossing, off-grid and marketable quotes before clock/orders change |
| Queue matcher | Unit | Wrong-side/worse-limit trades cannot fill, queue consumed first, partial volume conserved |
| L2 parsing | Contract test | Fixed-point YES/NO bids map to complementary YES asks and price-priority depth; empty book rejected |
| Event impact | Integration + audit | Maker sell sign opposite inventory change; execution before impact; entire market flow applied once, including unfilled flow |
| Background process | Statistical unit | Bridge endpoint exactly calibrated innovation; conditional intrawindow variance in correct seconds |
| Episode lifecycle | Integration | Retain/cancel/replace orders, short final window, end-state settlement and truncation without fictitious settlement |
| Rewards | Integration + audit | Rewards plus penalties telescope to PnL; integral uses seconds and pre-settlement inventory |
| Random streams | Integration | Identical seed reproduces complete ledger; different policies see identical market path/outcome |
| Training isolation | Pipeline unit | Quote loader refuses held-out ticker; invalid/locked/boundary quotes excluded; same frozen inputs required |
| Real saved calibration | End-to-end | Broad-based and sports complete episodes load hashed configs, produce fills, settle and retain collateral invariants |
| Artifact integrity | End-to-end | CSV money round trip, independent ledger reconstruction, corrupted cash/impact rejected, hashes checked |
| Public capture | Mocked contract test | Bounded read-only endpoint, raw schema retained with local availability timestamp |

Run `python -m pytest simulator/tests calibration/tests data_pipeline/tests -q`.
Verified 2026-10-09: **83 tests pass**, including 32 environment checks and
17 HJB adapter/benchmark checks. The HJB checks cover original-policy
equivalence, partial inventory, normalized time, import isolation, post-only
rounding, exact ES tail mass with ties, intrawindow drawdown, cached source
equivalence and independent persisted evaluation.
Run the default demo plus standalone validator to verify the recorded 162
episodes independently of unit fixtures.

Unmeasured gaps: historical maker fill probabilities/queue rank, ordered book
deltas, cancellations/replenishment, live latency, dated series fee profiles,
real resolution delay, serial event dependence, counterfactual execution-price
lambda, HJB parameter recalibration and RL performance. The public collector test is mocked;
no active-market snapshot was fetched. No live order submission is implemented.
