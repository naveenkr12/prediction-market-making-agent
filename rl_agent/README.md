# RL Quoting Agent (not yet started)

This will hold the reinforcement-learning market-making agent. The complete
calibrated environment is now in `../simulator/`. Future RL and adapted HJB
reference policies must share the same scenarios and random market paths for
PnL / VaR / ES comparisons. The published HJB reproduction remains its own
benchmark; its optimality does not extend to the modified impact/queue model.

Planned steps:
1. Add a Gymnasium wrapper, normalized observations and action encoding for
   the implemented `../simulator/trading_env.py` reset/step environment.
   The frozen HJB reference adapter with explicit seconds/normalized-time
   units is complete; see `../simulator/HJB_BENCHMARK.md` for its paired
   profitability/risk evaluation and assumptions.
2. Train an RL quoting agent (algorithm TBD) against it.
3. Calibrate a second, data-driven parameter set from `../data_pipeline/`'s
   collected Kalshi data, and evaluate both the HJB benchmark and the RL
   agent under it as a robustness check.
   The training-only price-impact coefficient and uncertainty scenarios are
   now available in `../simulator/`, along with calibrated market trade
   activity, empirical marks and background innovations. The complete quoting
   environment now includes quotes, partial fills, cash/collateral, fractional
   fees and settlement; see `../simulator/TRADING_ENVIRONMENT.md`. Queue and
   participation remain explicit scenarios pending richer orderbook data.
   The original HJB
   policy is a reference strategy under impact, not its proven optimum.
4. Compare RL vs. the HJB reference in the calibrated environment on PnL
   mean/std, VaR, ES, and terminal
   inventory control.
