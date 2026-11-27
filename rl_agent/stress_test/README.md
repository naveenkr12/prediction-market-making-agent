# Stress test: RL v3 vs HJB under informed traders, news jumps and snipers

**Question (the proposal's core question):** when the assumptions behind the Feil & Nendel HJB strategy break
(informed traders, discrete news jumps, fast snipers picking off stale quotes), does the RL v3 agent hold up better?

**Setup.** RL v3 (`rl_agent/v3/models/ppo_v3.zip`, 300 s decisions) is added as a fifth player to Rithvik's
Paper 3 information simulator (`simulator/information/`). Same contracts (roster R\*), same 20 replicates, same seeds,
same prices and settlement as HJB, fixed and skew, so every comparison is paired. Worlds: M0 base, M1 informed
traders, M2 news jumps, M3 both; each with snipers off (S0) and on (S1). v3 never saw informed traders or snipers in training.

## Result

Paired difference, RL v3 minus HJB, mean profit per one-hour episode, hierarchical bootstrap 95% range
(contracts, then replicates):

| Market | Snipers | Base | Informed | Jumps | Both |
|---|---|---|---|---|---|
| Sports | **on** | **+3.30** [+0.99, +5.69] | **+3.60** [+1.24, +5.75] | **+2.62** [+0.76, +4.34] | **+3.09** [+1.62, +4.64] |
| Sports | off | +0.09 [-1.14, +1.23] | +0.36 [-0.84, +1.38] | +0.27 [-0.90, +1.26] | +0.67 [-0.29, +1.59] |
| Economics | on | -0.03 [-1.18, +1.04] | +0.11 [-0.89, +1.14] | +0.02 [-0.99, +0.92] | -0.06 [-1.11, +0.96] |
| Economics | off | +0.05 [-0.33, +0.48] | -0.01 [-0.38, +0.41] | -0.01 [-0.40, +0.38] | -0.01 [-0.38, +0.35] |

- **Where HJB breaks (sports with snipers, HJB loses $3-4 per market-hour), v3 loses $2.6-3.6 less, clearly in all four worlds.**
- Everywhere else the two are statistically tied: RL gives up nothing.
- Tail risk: v3's ES95 loss is roughly 30-45% lower than HJB's in most cells (`risk_summary.csv`).

Charts: `results/stress_test_pnl.png` (profit by world) and `results/v3_minus_hjb.png` (paired difference).

## Caveats

- One trained v3 model (one training seed), 300 s decisions; p0 fixed at 0.5 in this simulator.
- The sniper level (S1) is an uncalibrated execution scenario, as stated in Rithvik's spec.
- v3 wins mostly by quoting less (about 1-2 fills per episode vs HJB's 4-13): it learned when not to quote.
- v3's flow features do not see snipes (logged as `snipe`, not `market`) and only see a jump at the next trade.

## Reproduce

```bash
STRESS_V3=1 python -m simulator.information.evaluate --output-dir rl_agent/stress_test/results \
    --roster data_pipeline/output/paper3/experiment/roster.json --workers 8
python rl_agent/stress_test/analyze.py
```

`STRESS_V3=1` is opt-in: without it, Rithvik's runner and checks are unchanged (v3 is not in `POLICIES`, and the
L0 anchor loop never runs v3). Code: `v3_policy.py` (v3 as a player in the information simulator), `analyze.py`
(paired comparison and charts), and small hooks in `simulator/information/environment.py` (`run_policy`) and
`simulator/information/evaluate.py` (`EXTRA_POLICIES`).

Next: more v3 training seeds, the 60 s model with `decision_interval=60`, and training v3 with snipers present.
