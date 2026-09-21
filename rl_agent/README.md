# RL Quoting Agent (not yet started)

This will hold the reinforcement-learning market-making agent, trained and
evaluated in the same environment (price process, order intensities,
inventory mechanics) as `../hjb_baseline/`, so its performance can be
compared directly against the published HJB-optimal benchmark on the same
PnL / VaR / ES statistics.

Planned steps:
1. Wrap the `hjb_baseline` environment as a standard RL env (Gym-style).
2. Train an RL quoting agent (algorithm TBD) against it.
3. Calibrate a second, data-driven parameter set from `../data_pipeline/`'s
   collected Kalshi data, and evaluate both the HJB benchmark and the RL
   agent under it as a robustness check.
4. Compare RL vs. HJB-optimal on PnL mean/std, VaR, ES, and terminal
   inventory control.
