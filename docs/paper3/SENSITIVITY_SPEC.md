# Paper 3 sensitivity experiments: quote refresh and jump intensity

Status: requested by the user on 2026-11-01, after Tier 1. Lane: STANDARD. This adds to validated infrastructure, and the Tier 1 results must reproduce byte for byte. The base specification is `IMPLEMENTATION_SPEC.md`; everything here is in addition to it.

Shared settings for both experiments:
- **Templates and seeds:** roster R\* (32 templates) from `data_pipeline/output/paper3/experiment/roster.json`, 20 replicates, Tier 1 seeds and template order (§5).
- **Policies:** hjb, fixed, skew 0.0005, none.
- **Other assumptions as in Tier 1:** base fills, latency 2 s, taker fee 0.07, p0 = 0.5.
- **Bootstrap:** common-index hierarchical bootstrap with 1,000 draws (§9.4).
- **Scopes:** all, broad_based, sports.

## Constraints

- No tracked file is modified: neither Naveen's HJB nor the Paper 2 code or outputs. The only edits allowed are new or untracked Paper 3 files.
- After all code changes, Tier 1 must regenerate exactly: `episode_results.csv` sha256 = `7a273dd206b40a90a4a7c42462340ab1be07757fa58b0f42520bb0bfb1746f25`. Every existing test must still pass.
- Outputs go to `data_pipeline/output/paper3/sensitivity/{quote_refresh,jump_intensity}/`. Each output has its own `manifest.json` recording sha256 for every artifact, `checkpoints/` (not committed) and `episode_results.csv` (not committed; it must be regenerable to its recorded sha256).

## A. Quote refresh: 300 s against 60 s

- **What changes:** only the decision interval (`decision_interval` ∈ {300, 60}).
  - The policies are unchanged. HJB uses the same frozen value function and the same `time_mapping="episode"`, and it reads time from `elapsed_seconds`, so it is the same quoting rule evaluated 5× as often.
  - Market windows, executions, books (spread draws only at legacy points), jumps, proposals, background path and settlement are identical between intervals (shared decision nodes; already tested in P3).
- **Cells:**
  - matched arms M0, M1, M2 (matched s) and M3 (s = 1);
  - equal-s arms M0-s1, M2-s1 and M1-native;
  - each × S0/S1 × {300, 60}.
  - The 300-s cells must equal the Tier 1 rows exactly (assert).
- **Implementation detail to document:** what happens when a policy re-submits an unchanged quote at a 60-s stop (queue priority kept or reset, under the legacy `_replace` semantics). Report quote updates per episode.
- **Contrasts** (mean PnL, ES95, VaR95, snipe PnL, snipe count, fills, |q|):
  - per arm and S: dec60 − dec300;
  - refresh × sniping: (60,S1) − (60,S0) − (300,S1) + (300,S0);
  - at 60 s, the jump contrasts M2−M0, M3−M1, M2-s1−M0-s1 and M3−M1-native, together with their difference from the 300-s contrasts (refresh × jump);
  - HJB − fixed and HJB − skew at 60 s.

## B. Jump intensity: 0×, 0.5×, 1×, 2× at fixed background (s = 1)

- **Families:**
  - F0 flow: 0× = M0-s1, then M2-s1 at 0.5×, 1× and 2×.
  - FG flow: 0× = M1-native, then M3 at 0.5×, 1× and 2×.
- **Settings:** s = 1 exactly in every cell, so background volatility is fixed. Each cell runs × S0/S1. The 0× and 1× cells must equal the Tier 1 rows exactly (assert).
- **Coupled jump construction** (common random numbers across doses; the Tier 1 streams are unchanged):
  - 0.5×: independently thin the 1× jumps. Keep a jump iff a thinning uniform from a new stream (episode spawn child 12) is < 0.5. Thinned jump nodes stay in the node set as inert nodes, so W, proposals and all other inputs are identical to 1×.
  - 2×: the 1× jumps plus an independent Poisson process with the same time-varying λ_J (TTR buckets). Its times, magnitudes from the class bank, and sign uniforms V_J come from a new stream (child 13). Its nodes are inserted by Brownian-bridge refinement of W from the same child 13: W at every original node is unchanged, so the background increments summed between original nodes are identical to 1×.
  - The resulting jump processes have the right laws: Poisson(0.5·λ_J) and Poisson(2·λ_J), with i.i.d. bank magnitudes and sign-tilt.
  - Tests required: nesting (0.5× ⊂ 1× ⊂ 2× jump sets); background values at original nodes identical across doses; bounded rate-law checks; V2 posterior equality at 0.5× and 2×.
  - Children 12 and 13 must not change the existing children 0–11 (spawn keys are positional).
- **Information revealed changes with dose.** Report it per class and dose:
  - U = E[p_c(1−p_c)] by the Rao–Blackwellized estimator (`matching.py`, s = 1, N = 200,000, with standard errors);
  - information revealed I = p0(1−p0) − U;
  - the jump share of I;
  - the empirical mean p_c(1−p_c) from the simulated paths, as a cross-check.
- **Contrasts:**
  - each dose against 0× and against 1×;
  - sniping × dose: (d,S1) − (d,S0) − (0,S1) + (0,S0);
  - a bootstrap dose-response slope of mean PnL and snipe PnL on dose (OLS on dose levels 0, 0.5, 1, 2 within each bootstrap draw);
  - snipe PnL per unit of information revealed (descriptive).

## Reporting

For both experiments, add sections §6a and §6b to `simulator/INFORMATION_ARRIVAL.md`. For HJB, and briefly for fixed, give mean PnL with 95% intervals, ES95, snipe PnL and snipe counts, by class and pooled. State that experiment B changes total information revealed. Avoid causal over-claiming.
