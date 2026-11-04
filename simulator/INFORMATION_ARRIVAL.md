# Paper 3: information arrival, jumps and stale-quote sniping

This document describes the Paper 3 extension of the Paper 2 trading simulator: the model, its calibration,
the experiment arms and contrasts, variance matching, how to run everything, the validation record, and the
limitations. The single source of truth for the design is `docs/paper3/IMPLEMENTATION_SPEC.md` (revision 2
plus amendments A-1 to A-3). All numbers below come from the artifacts in
`data_pipeline/output/paper3/calibration/` and `data_pipeline/output/paper3/experiment/`.

Everything here is a **training-template synthetic Monte Carlo**. It is not evidence about real-market
profitability, and none of the contrasts below should be read as causal effects in real markets.

## 1. Model

**Hidden state.** Each episode has a hidden binary outcome Θ ∈ {0, 1}. The public price is p = expit(L), where L
is the log odds. L is held unclipped in float64; only the displayed probability (observations, quotes, HJB
inputs, legacy L0 settlement) is clipped. An episode is the last 3,600 s before the template close t_c, with
policy decisions every 300 s (one legacy market window).

**Three information channels** (kernels in `simulator/information/kernels.py`):

1. **Background (continuous).** Over a node interval h with σ² = s·σ_c²:
   X = (σ²h/2)(2Θ − 1) + σ ΔW, and L += X. This Gaussian experiment's LLR is exactly X.
   σ_c is the template's continuous (non-jump) residual volatility, taken from the legacy noise donor
   (§2). s is the background scale (s = 1 is native; matched arms use the s from §4).
2. **Order flow (Glosten–Milgrom).** An execution of size c is informed with probability
   α(c; η) = min(1 − exp(−ηc), 1 − 1e−12). An informed trader trades in the direction of Θ; an uninformed trader buys
   with probability u (u depends on template, size bin and lifetime phase). Signs are +1 for a YES taker and −1 for a NO taker.
   The update is the exact LLR:
   buy: log[(α + (1−α)u) / ((1−α)u)], sell: log[((1−α)(1−u)) / (α + (1−α)(1−u))].
   - FG-half variant: u = ½ and α = tanh(βc/2), so the LLR is ±βc.
   - F0: uninformed signs, no update.
   - FL (legacy): L += β·sign·c with the Paper 2 fixed impact β.
3. **Jumps (public news proxy).** Jump times are Poisson with intensity λ_J(class, time-to-release bucket). Magnitudes
   a are drawn from the class magnitude bank. Under Θ the sign is + with probability expit(±a) (the sign-tilt
   experiment), and L += sign·a, whose LLR is exactly ±a.

**M family (Θ-known generators).** Θ = 1{U_M < p0}. Every signal is drawn conditional on Θ, every update is the
exact Bayesian LLR (p-independent), p is a martingale, and the contract settles at Θ.

**L family (Θ-free generators).** Signals are drawn from their exact marginal law given the current p, which
gives the same LLR updates without a hidden Θ:
- background: Θ̃ ~ Bernoulli(p) on the L-aux stream;
- flow: buy iff V_F < (1−α)u + αp;
- jumps: + with probability p·expit(a) + (1−p)·expit(−a).

The contract settles at 1{U_L < p_close}, where U_L is the legacy settlement draw (amendment A-3).

Legacy-law components (L0c) are not Bayesian experiments. They use a driftless Gaussian background in log odds,
FL flow, and fair-coin jumps.

**Sniping (S1).** Bid-side and ask-side snipe proposals are independent Poisson processes of rate 1/ℓ̄ (ℓ̄ = 2 s). At a
side-X proposal, an alive side-X maker order with remaining Q is filled in full at its own price (maker fee
charged, no price update, not counted as market flow) if it is crossed at the node's exact unclipped p:
- ask: Q(p − a) > taker_fee(Q, a) + Q·tick
- bid: Q(b − p) > taker_fee(Q, b) + Q·tick

where taker_fee = ceil_cent(0.07·Q·P·(1−P)). Under S0 the proposals are inert.

**Common random numbers.** All arms, policies and S levels of an episode share one base-input skeleton (seeding
as in `evaluate_hjb`, 12 spawned streams). Prices never react to the policy's fills or snipes, so each
(episode, arm) price path is precomputed once. The tests in `simulator/tests/test_information_statistics.py`
check the sharing directly:
- M0–M3 use the same W increments, Θ, S_u signs and jump draws;
- M1 and M3 have identical flow signs;
- L0 and L0b differ only in execution signs;
- the Θ-free arms share background noise and L-aux uniforms.

## 2. Calibration (`calibration/calibrate_information_arrival.py`, train split only)

**Data access.** The calibration report records `heldout_files_loaded = 0`. Per-ticker tapes and quotes are
loaded through a training guard that refuses held-out tickers. Three shared tables are not per-ticker files:
the market metadata table (`markets/kalshi_selected_contracts.parquet`), `five_minute_intervals.parquet` and
`lambda_estimates.parquet`. Each is read in full and filtered to train-split tickers in memory before use; no
held-out row enters any statistic. The experiment runner records the same kind of check in
`manifest.json["training_isolation"]`: all 35 simulated templates are in the train split of `event_splits.csv`,
and no raw tape or quote file is read.

**Residual decomposition.** The calibration finds 24 templates with their own quote support. For each it refits
the legacy residual regression and confirms σ_legacy² = σ_c² + jump variance; the largest identity error is
1.1e−19. Templates without their own support use the legacy donor law: one supported template of the same
class, drawn uniformly.

**Jumps** (|η̃| > 4σ̂ bipower rule; magnitudes in log odds):

| Class | Jumps B0/B1/B2/B3 | λ_J per hour B0/B1/B2/B3 | Bank size | E[a²] (bootstrap 95%) |
|---|---|---|---|---|
| broad_based | 0 / 2 / 13 / 33 | 0.35 / 0.35 / 0.31 / 0.22 | 48 | 0.45 (0.25–0.76) |
| sports | 9 / 7 / 3 / 7 | 1.54 / 0.33 / 0.12 / 0.11 | 26 | 1.35 (0.81–1.96) |

Broad B0 and B1 are pooled B0+B1+B2 (fewer than 5 jumps on their own), and sports B2 is pooled B2+B1.

**Informed flow.** η is fitted by bisection so that the flow-only post-trade replay's median Kyle λ matches the
train-median Kyle λ target. Amendments A-1 (bisection tolerance 0.1%) and A-2 (validation CI combines the
calibration and validation MC errors) apply.

| Class | η per contract | Target λ (log-odds/contract) | Validation rel. error [95% CI] | α at size quantiles 10/50/90/99% |
|---|---|---|---|---|
| broad_based | 3.256e−5 | 4.33e−5 | +0.12% [−0.56%, +0.80%] (validated) | 3.3e−5 / 3.3e−4 / 6.5e−3 / 0.113 |
| sports | 1.819e−5 | 1.25e−5 | +0.26% [−0.28%, +0.80%] (validated) | 2.3e−5 / 2.9e−4 / 3.3e−3 / 0.036 |

- FG-half (u = ½, α = tanh(βc/2) with the base β) reproduces 93.6% (broad) and 94.9% (sports) of the target λ.
- The constant-α variant (Tier 2): broad α0 = 0.0338, validated. **Sports failed**, because the MC SE tolerance
  was not reached within 51,200 replications. The sports constant-α Tier 2 arms therefore cannot be built.

**Release times (`release_time_verification.csv`).**
- Broad templates use `occurrence_datetime`.
- Sports templates use the template close, flagged `provisional_sports_close`; official game-end times are deferred.
- Two broad templates release before the close (early revelation) and are excluded by rule:
  KXFEDDECISION-26JUL-C26 (−1,503 s) and KXPCECORE-26JUL-T0.2 (−111 s).
- **Ten broad templates** carry the flag `occurrence_after_settlement_ts`, meaning the recorded settlement
  timestamp precedes `occurrence_datetime`:
  - KXAAAGASM-26AUG31-4.08
  - KXAAAGASW-26AUG03-4.110, KXAAAGASW-26AUG10-4.010, KXAAAGASW-26AUG24-4.080, KXAAAGASW-26JUL13-3.960,
    KXAAAGASW-26JUL20-4.060, KXAAAGASW-26JUL27-4.080
  - KXCPIYOY-26JUL-T4.2
  - KXJOBLESSCLAIMS-26JUL23-200000
  - KXU3-26AUG-T4.1

  For these, the time-to-release buckets (and hence λ_J) depend on a metadata time that the settlement record
  contradicts. Twelve of the fifteen sports templates carry the same flag on top of the provisional close.

**Starting probability.** p0_emp is the last train trade before t_c − 3,600 s, clipped to [0.01, 0.99]. Several
templates sit at the clip (0.01 or 0.99).

## 3. Arms and what each contrast identifies

All arms use: base fill scenario (queue 50, participation 0.1); base β; calibrated η; p0 = 0.5 unless marked
"emp"; 20 replicates; four policies (frozen HJB with episode time mapping, fixed, skew 0.0005, none).

| Arm | Construction | Rosters | S |
|---|---|---|---|
| L0 | `TradingEnvironment` itself; reproduces the Paper 2 benchmark rows | all 35 (anchor), R* | S0 |
| L0b | L0 with bank signs replaced by S_u signs | R* | S0 |
| L0c | σ_c Gaussian legacy-law background + legacy-law (fair-coin) jumps at native λ_J + FL flow | R* | S0 |
| L1J | Θ-free martingale background + Θ-free tilted jumps + FL flow | R* | S0 |
| L1 | L1J without jumps | R* | S0 |
| FGH-L / FGH-LJ | Θ-free background + Θ-free FG-half flow, without / with jumps | R* | S0 |
| M0 | F0 flow, no jumps, matched s | R* | S0, S1 |
| M1 | FG flow, no jumps, matched s | R* | S0, S1 |
| M2 | F0 flow, jumps, matched s | R* | S0, S1 |
| M3 | FG flow, jumps, s = 1 (the matching target) | R* | S0, S1 |
| M1-native | FG flow, no jumps, s = 1 (also serves as the secondary M1 at s = 1) | R* | S0, S1 |
| M0-s1, M2-s1 | secondary native arms at s = 1 (M3 serves as the secondary M3) | R* | S0, S1 |
| M0-emp … M3-emp | M 2×2 at p0_emp with s re-matched at that p0 | R*_emp | S0, S1 |

Contrasts are differences of marginal statistics (mean PnL, VaR95, ES95, mean |q| before settlement, drawdown,
fills, snipe PnL). Bootstrap ranges come from a hierarchical bootstrap: 1,000 draws, resampling templates and
then replicates, with one index set shared by every cell on a roster and scope. The ES95 of the paired PnL
difference is reported as a secondary statistic.

**M-family contrasts.**
- **Flow, M1 − M0.** Both arms are matched to the same expected terminal uncertainty. The contrast replaces
  Gaussian background information with informed-flow information.
- **Jump, M2 − M0 and M3 − M1.** At equal expected terminal uncertainty, these replace part of the Gaussian
  background with discrete jumps. They do **not** compare "jumps vs no jumps at equal everything else".
  - M2 − M0 compares two arms with different matched s. For example, sports M0 has a median s of 8.4 while M2
    has 1.03, so M0's background is far more volatile.
  - M3 − M1 compares the s = 1 target against M1 at its own matched s.

  This is by design: matching equalizes expected terminal uncertainty, not volatility paths.
- **Interaction, M3 − M2 − M1 + M0.** Combines the four arms; the same matched-s caveat applies.
- **Sniping × jump.** (M2,S1) − (M2,S0) − (M0,S1) + (M0,S0), and the M3/M1 analogue. These measure how much the
  sniping effect differs between the jump and no-jump arms.
- **Sniping main effects.** (A, S1) − (A, S0) for each M arm.

**Legacy chain.**
- **L0b − L0 isolates one change:** flow signs from the bank versus S_u. Amendment A-3 makes the settlement
  uniform shared. The only other difference is that L0b compares U_L with the unclipped p_close while L0
  compares it with the displayed p, and these differ only when p_close is outside the display clip.
- **L0c − L0b bundles several changes:**
  - a Gaussian σ_c background on the full skeleton node set, replacing the bridged empirical window
    innovation at σ_legacy (which drops the jump share of variance from the background);
  - legacy-law jumps at the native λ_J.
- **L1J − L0c also bundles several changes:**
  - martingale (Θ̃-drifted) background instead of the driftless log-odds background;
  - tilted Θ-free jump signs instead of fair coins.
- **L1J − L1** is the jump effect within the L family.

**Contextual contrasts.** M1-native − L1 and M3 − L1J are bundled mechanism replacements. They switch to the
Θ-known generators, Bayesian FG flow instead of FL fixed impact, and settlement at Θ, all at once.

**FG-half contrasts.** FGH-L − L1 and FGH-LJ − L1J replace FL flow with Θ-free FG-half flow under identical
execution times, sizes, background, jumps and settlement convention.

**Policy contrasts.** HJB − fixed and HJB − skew are computed within every (arm, S) cell. For L0 they are reported
on both R* and the all-35 anchor roster.

## 4. Variance matching (`simulator/information/matching.py`)

M-arm LLRs are p-independent. Given Θ, the closing log odds are L_c = L0 + Λ_B + Λ_N, where
Λ_B ~ N((2Θ−1)V/2, V) and V = s·σ_c(donor)²·3600.

**Estimator.** For each draw of (donor, Θ, Λ_N) from the vectorized sampler, the Rao–Blackwellized expected
terminal uncertainty is h_i(V) = q_i·E_Z[expit(−(ℓ_i + V/2 + √V Z))], with ℓ_i = L0 + Λ_N,i and q_i = expit(ℓ_i).
- The sampler uses the skeleton's generators, and its law is tested against `paths.build_path`.
- h_i is decreasing in V, so the average Û_A(s) is monotone in s.

**Quadrature.**
- √V ≤ 1: Gauss–Hermite in Z.
- √V > 1: Gauss–Legendre in the logistic variable Λ = logit u over [−40, 40], using E = E_Λ[Φ((Λ − m)/√V)].

Plain Gauss–Legendre on u ∈ (0,1) converges only at about n⁻², so it was not used. Two orders must agree within
1e−9 at every evaluation; the largest disagreement observed was 8.8e−15. On 1,000 random cases the maximum error
against `scipy.integrate.quad` was 8.9e−16.

**Procedure** (per template, arm and p0):
1. The target U* = Û_M3(1) is computed on an independent draw set of N = 200,000.
2. s_hi is doubled from 4 (cap 1,024), then s is found by bisection.
3. The match is validated on independent draws. The gap's CI uses the delta method with the validation and
   target variances. A match is accepted if |gap| < 2% and the CI covers 0; otherwise N is doubled, at most 3 times.

**Results** (`roster.json`, `matching_report.csv`; frozen before any policy run; 1,780 s on 10 workers):
- **R* = 32 templates** (17 broad, 15 sports). **R*_emp** contains the same 32 templates.
- **Exclusions:**
  - KXAAAGASW-26JUL27-4.080: its own σ_c is exactly 0, so Û does not depend on s and M0–M2 cannot be matched.
  - KXFEDDECISION-26JUL-C26 and KXPCECORE-26JUL-T0.2: early revelation, excluded by rule; they are analysed
    only in Tier 2, which is not implemented.
- All 192 matched (template, arm, p0) rows were accepted: 183 at N = 200,000 and 9 after one doubling.
- Largest |gap|: 0.22% at p0 = 0.5 and 0.57% at p0_emp.

Matched s (median, with min–max):

| p0 | Class | M0 | M1 | M2 |
|---|---|---|---|---|
| 0.5 | broad | 1.67 (1.06–33.8) | 1.68 (1.06–33.6) | 1.00 (0.95–1.51) |
| 0.5 | sports | 8.43 (4.45–26.5) | 8.43 (4.44–26.5) | 1.03 (0.98–3.35) |
| emp | broad | 1.45 (1.06–36.9) | 1.44 (1.06–36.7) | 0.98 (0.76–1.53) |
| emp | sports | 8.74 (4.86–28.2) | 8.75 (4.85–28.2) | 1.00 (0.07–3.75) |

How to read the table:
- s_M1 ≈ s_M0 because calibrated FG flow carries very little terminal information (α at the median size is about 3e−4).
- Sports M0/M1 need s ≈ 8 because jumps carry most of M3's information beyond the background.
- **Weakly identified p0_emp M2 scales.** Examples: s = 0.072 (KXEPLGAME-26AUG23BRIAVL-TIE, p0 = 0.01),
  0.76 (KXAAAGASW-26JUL20-4.060, p0 = 0.01) and 0.87 (KXMLSGAME-26SEP05CLBCOL-TIE, p0 = 0.20). These pass the
  rule, but at extreme p0 the background contributes almost nothing to U, so s barely moves Û within MC error.
  Treat these s values as poorly determined, even though U is matched.
- Matching equalizes expected terminal uncertainty only, not the variance time profile. The time profiles at
  900, 1,800 and 2,700 s are in the V9 cross-check.

## 5. Running

All commands run from the repository root with `/opt/anaconda3/bin/python` and `PYTHONDONTWRITEBYTECODE=1`.
Workers are given one BLAS thread each, set automatically by evaluate and validate. For matching, export
`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` (the recorded matching run oversubscribed BLAS threads, which
only affects speed).

| Step | Command | Runtime (10 cores) |
|---|---|---|
| Calibration | `python calibration/calibrate_information_arrival.py` | not timed by the script; dominated by the flow-replay bisection |
| Matching | `python -m simulator.information.matching --workers 10` | 1,780 s |
| Tier 1 | `python -m simulator.information.evaluate --workers 10 --replicates 20` (`--tier2` adds the implemented Tier 2 cells) | 164 s in `run()`: 141 s simulation for 74,480 episodes, then aggregation (see note) |
| Validation | `python -m simulator.information.validate --workers 10` (`--scale reduced` for a smoke check; `--c2` validates a temporary copy of the Paper 2 benchmark) | 608 s at full scale |

Runtime note: one evaluate invocation took 1,063 s of wall-clock time, but only 164 s were measured inside
`run()`. The gap fell before the run started and was not reproduced or explained.

Evaluate checkpoints each (template, arm) under `experiment/checkpoints/` and resumes only when the plan,
roster hash and code hash all match.

Outputs (`data_pipeline/output/paper3/experiment/`):
- `episode_results.csv` (74,480 rows, 48 MB): **not committed**. Regenerate it with the Tier 1 command above,
  which takes about 2–3 min from fixed seeds. The result must match the sha256 in `manifest.json` (`7a273dd2…f25`).
  The sensitivity runner and validation item V11 read this file, so run Tier 1 first on a fresh checkout.
- `risk_summary.csv`, `paired_contrasts.csv`
- `arm_diagnostics.csv`, `path_diagnostics.csv`, `fidelity_report.json`
- `ledgers/`, `mean_pnl_by_arm.png`
- `manifest.json`: seeds, configuration, calibration, roster and code hashes, the preservation manifest,
  runtimes, per-episode check summary and training isolation
- `validation_report.json`
- `v12_preregistration.md`, `v12_confirmation.json`

### Sensitivity experiments (§6a, §6b; `docs/paper3/SENSITIVITY_SPEC.md`)

| Step | Command | Runtime (10 cores) |
|---|---|---|
| A. Quote refresh | `python -m simulator.information.sensitivity --experiment quote_refresh --workers 10` | 247 s in `run()`: 211 s simulation for 71,680 episodes |
| B. Jump intensity | `python -m simulator.information.sensitivity --experiment jump_intensity --workers 10` | 115 s in `run()`: 93 s simulation for 40,960 episodes, plus 256 RB estimates of U (270 worker-seconds) |

- Run with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- The runner uses the frozen Tier 1 `roster.json` and compares against the Tier 1 `episode_results.csv`. It
  checkpoints per (template, cell) under `sensitivity/<experiment>/checkpoints/` (not committed).
- Outputs (`data_pipeline/output/paper3/sensitivity/{quote_refresh,jump_intensity}/`):
  - `episode_results.csv` (not committed; regenerable to the sha256 in `manifest.json`)
  - `risk_summary.csv`, `paired_contrasts.csv` (including a `mean_snipe_count` statistic on the same bootstrap
    indices), `cell_summary.csv` (decisions and quote updates per episode), `path_diagnostics.csv`
  - `tier1_equality.json`
  - B only: `information_revealed.csv` and `information_revealed_by_template.csv`
  - `manifest.json`: design, seeds and streams, code, calibration and roster hashes, preservation manifest,
    checks, runtimes, and the sha256 of every artifact
- Both experiments were regenerated from scratch a second time. Every artifact had the same sha256.
- After the code changes these experiments need (dose streams in `skeleton.py`, a per-jump mask in `paths.py`
  and `environment.py`, an optional prebuilt path in `evaluate.run_arm_replicate`, masked jump statistics in
  `metrics.py`), Tier 1 still regenerates exactly. `episode_results.csv` sha256 is `7a273dd2…f25`, and
  `path_diagnostics`, `risk_summary`, `paired_contrasts`, `arm_diagnostics` and `fidelity_report` are
  byte-identical.

## 6. Tier 1 results (training-template simulation; hierarchical bootstrap 95% ranges)

**Checks.** All 74,480 episodes passed:
- the ledger audit and additive attribution (largest error 1.2e−13);
- none-policy PnL = 0;
- shared signal, book and settlement hashes within each (template, replicate, arm) across policies and S levels;
- the L0 anchor, where all 2,800 episodes reproduce the saved Paper 2 rows (largest difference 2.8e−14).

**Marginal statistics, HJB policy, pooled over R*** (dollars per one-hour episode; ES95 is the average loss in the worst 5%):

| Arm | S | Mean PnL [95%] | ES95 loss [95%] | Snipe PnL |
|---|---|---|---|---|
| L0 (all 35) | S0 | +0.13 [−0.46, +0.71] | 15.7 [11.0, 19.2] | 0 |
| L0 (R*) | S0 | +0.07 [−0.55, +0.77] | 16.1 [11.7, 19.7] | 0 |
| L1J | S0 | −0.02 [−0.59, +0.66] | 13.2 [10.1, 15.9] | 0 |
| M0 | S0 | +0.07 [−0.63, +0.79] | 11.9 [9.2, 14.3] | 0 |
| M0 | S1 | −1.70 [−2.99, −0.38] | 19.9 [17.1, 21.6] | −1.84 |
| M3 | S0 | −0.20 [−0.73, +0.31] | 11.9 [9.1, 14.0] | 0 |
| M3 | S1 | −1.43 [−2.54, −0.48] | 14.8 [13.1, 16.7] | −1.54 |
| M1-native | S1 | −0.10 [−1.00, +0.79] | 15.1 [12.9, 17.0] | −0.57 |

**Main contrasts, pooled over R*** (mean PnL and ES95 loss differences; negative ES95 difference = smaller tail loss):

| Contrast | S | HJB mean PnL | HJB ES95 | Fixed mean PnL | Fixed ES95 |
|---|---|---|---|---|---|
| flow M1−M0 | S0 | −0.20 [−0.51, +0.01] | +0.19 [−0.70, +1.17] | −0.13 [−0.33, +0.02] | +0.26 [−0.90, +1.08] |
| jump M2−M0 | S0 | −0.08 [−0.42, +0.18] | −0.25 [−1.72, +1.05] | +0.14 [−0.17, +0.44] | +1.01 [−0.95, +2.74] |
| jump M2−M0 | S1 | +0.46 [−0.52, +1.40] | −4.21 [−6.70, −1.32] | +0.80 [−0.27, +1.81] | −1.96 [−5.51, +1.64] |
| jump M3−M1 | S0 | −0.08 [−0.46, +0.23] | −0.25 [−1.89, +1.21] | +0.17 [−0.09, +0.43] | +2.08 [+0.09, +4.18] |
| jump M3−M1 | S1 | +0.52 [−0.47, +1.45] | −4.97 [−7.36, −2.33] | +0.81 [−0.33, +1.85] | −1.79 [−5.16, +1.56] |
| interaction | S0 | +0.00 [−0.27, +0.21] | +0.01 [−1.22, +1.02] | +0.03 [−0.17, +0.25] | +1.07 [−0.62, +2.82] |
| sniping×jump (M2/M0) | S1−S0 | +0.55 [−0.37, +1.47] | −3.96 [−6.49, −1.11] | +0.66 [−0.38, +1.65] | −2.98 [−6.76, +0.85] |
| sniping×jump (M3/M1) | S1−S0 | +0.60 [−0.35, +1.56] | −4.73 [−7.02, −2.11] | +0.65 [−0.46, +1.66] | −3.87 [−7.36, −0.30] |
| L0b−L0 | S0 | +0.08 [−0.32, +0.44] | −1.15 [−3.69, +1.21] | +0.10 [−0.42, +0.61] | −0.73 [−4.05, +2.72] |
| L0c−L0b | S0 | +0.05 [−0.63, +0.75] | −1.52 [−4.51, +1.20] | −0.04 [−0.76, +0.63] | +0.35 [−2.74, +3.88] |
| L1J−L0c | S0 | −0.22 [−0.66, +0.09] | −0.31 [−1.53, +0.85] | −0.14 [−0.57, +0.20] | −0.63 [−1.85, +0.45] |
| L1J−L1 | S0 | −0.35 [−0.86, +0.09] | −0.39 [−1.75, +0.83] | −0.42 [−1.05, +0.12] | −0.27 [−1.69, +1.35] |
| M1-native−L1 | S0 | −0.36 [−1.29, +0.50] | −1.15 [−3.85, +1.25] | −0.15 [−1.25, +0.99] | −0.76 [−4.86, +2.82] |
| M3−L1J | S0 | −0.18 [−0.96, +0.52] | −1.27 [−4.07, +1.22] | +0.07 [−0.97, +1.14] | −0.03 [−5.04, +3.88] |

By-class values (broad and sports) are in `paired_contrasts.csv` (columns `scope` and `roster`).

**Jump contrasts at equal background volatility (secondary arms, s = 1, pooled over R*):**

| Contrast | S | HJB mean PnL | HJB ES95 | HJB snipe PnL | Fixed mean PnL |
|---|---|---|---|---|---|
| jump M2-s1 − M0-s1 | S0 | −0.17 [−0.45, +0.06] | −0.54 [−1.98, +0.87] | 0 | −0.18 [−0.47, +0.07] |
| jump M2-s1 − M0-s1 | S1 | −1.38 [−2.11, −0.75] | +0.35 [−0.51, +1.58] | −0.99 [−1.30, −0.69] | −1.22 [−1.97, −0.53] |
| jump M3 − M1-native | S0 | −0.17 [−0.54, +0.11] | −0.51 [−1.85, +0.69] | 0 | −0.20 [−0.52, +0.06] |
| jump M3 − M1-native | S1 | −1.34 [−1.98, −0.74] | −0.25 [−1.32, +1.18] | −0.97 [−1.29, −0.68] | −1.24 [−1.94, −0.59] |
| sniping×jump (M2-s1/M0-s1) | S1−S0 | −1.21 [−1.82, −0.65] | +0.89 [−0.49, +2.52] | −0.99 [−1.30, −0.69] | −1.03 [−1.68, −0.37] |
| sniping×jump (M3/M1-native) | S1−S0 | −1.17 [−1.72, −0.64] | +0.27 [−0.90, +2.08] | −0.97 [−1.29, −0.68] | −1.04 [−1.65, −0.43] |

By class, the S1 jump effect on HJB mean PnL is concentrated in sports. For M3 − M1-native it is −2.56 [−3.40, −1.63] in sports and −0.25 [−0.68, +0.18] in broad.

**Reading.**
- **Without sniping, jumps have no clear effect.** Under S0, no mean-PnL contrast in either table has a 95% range that excludes zero.
- **With stale-quote sniping, adding jump-borne information costs the market maker money.** This contrast holds the background fixed and adds jumps on top, so the jump arms also reveal more total information over the hour. It therefore measures "more information, part of it discontinuous", not discontinuity alone. At equal background volatility (s = 1), adding jumps lowers S1 mean PnL by about $1.3–1.4 per episode for HJB (−1.38 [−2.11, −0.75] for M2-s1 − M0-s1; −1.34 [−1.98, −0.74] for M3 − M1-native) and by about $1.2 for fixed. The sniping×jump interaction is negative with a 95% range excluding zero for HJB, fixed and skew. About $1.0 of the loss is snipe PnL itself; the rest comes through other channels and is not decomposed here. The effect is concentrated in sports, where jumps are more frequent and larger per episode (1.53 against 0.33 simulated jumps per episode; E[a²] 1.35 against 0.45). Per hour, sports λ_J is lower than broad λ_J in buckets B1–B3.
- **Jumps produce fewer stale-quote crossings than the matched no-jump arms, but deeper ones.** At s = 1 they produce slightly more (2.3 against 1.85). HJB, S1, per episode (`risk_summary.csv`):

  | Arm | Snipes | Snipe PnL | Per snipe |
  |---|---|---|---|
  | M0-s1 | 1.85 | −0.55 | about −0.30 |
  | M2-s1, M3 | 2.3 | −1.54 | about −0.67 |
  | matched M0, M1 (s ≈ 1.7 broad, 8.4 sports median) | 6.1 | −1.84 | about −0.30 |

  The per-snipe depth is a ratio of measured means. The snipe counts measure crossed exposure, because the hazard was validated in V12.
- **The matched-arm S1 tail contrasts are a comparator effect, not evidence that jumps reduce risk.** Under S1 the matched jump contrasts show smaller ES95 for the jump arms (HJB M3−M1: −4.97 [−7.36, −2.33]). This is attributable to the matched comparators: to reach the same terminal uncertainty, M0 and M1 need a much more volatile Gaussian background, which roughly triples their snipe count (ES95 19.9 for matched M0 against 14.7 for M0-s1). Two pieces of evidence support this reading. First, the ES95 contrast disappears at equal s (M2-s1 − M0-s1: +0.35 [−0.51, +1.58]; M3 − M1-native: −0.25 [−1.32, +1.18]). Second, it is concentrated in sports, where matched s is largest (M3−M1 HJB S1 ES95: sports −5.91 [−9.48, −2.34], broad −0.59 [−2.38, +1.23]). The variance-matched design therefore compares "jumps" with "a much more volatile diffusion carrying the same terminal information". It does not compare jumps with no jumps.
- **Synthesis: how much information arrives matters more than its form for mean PnL; form changes how the losses arrive.** The two designs answer different questions. Equal s asks what happens when news arrives as jumps on top of the existing background, and the answer is a loss of about $1.3–1.4 per episode under S1 (HJB). The matched design holds the information revealed fixed and asks whether its form matters. Under S1 the mean-PnL contrasts are small and their ranges include zero (HJB M2−M0: +0.46 [−0.52, +1.40]; M3−M1: +0.52 [−0.47, +1.45]). Snipe losses track the amount of information that arrives while quotes are resting. They are −0.55 for M0-s1, −1.54 for the jump arms, and −1.84 for matched M0, which carries the same information as M3 entirely through the diffusion. Jumps deliver that loss as a few deep crossings and the diffusion as many shallow ones. Neither design isolates a pure "discontinuity" effect. The jump-intensity sensitivity (§6b) varies the amount of jump information at a fixed background.
- **Policy contrasts.** HJB − fixed for L0 is +0.13 [−0.16, +0.45] on R* (+0.12 [−0.15, +0.40] on all 35).
  It is +0.93 [+0.19, +1.70] in M0 under S1.
- Every number here is conditional on the 32 training templates, synthetic flow, and the assumed fill and
  sniping scenarios.

## 6a. Sensitivity A: quote refresh, 300 s against 60 s (`sensitivity/quote_refresh/`)

**Design.**
- Cells: M0, M1, M2 (matched s), M3, M0-s1, M2-s1 and M1-native, each × S0/S1 × decision interval {300, 60}.
- R* (32 templates), 20 replicates, four policies, Tier 1 seeds; 71,680 episodes.
- Only `decision_interval` changes. Each (episode, arm) path is the same precomputed object at both intervals,
  so the node set, W, executions, jumps, books, proposals and settlement are shared.
- **Checks.** All passed:
  - Every 300-s cell reproduces its Tier 1 rows exactly: 35,840 rows, every shared outcome and hash column.
  - For all 4,480 (arm, template, replicate) paths, the signal, book, settlement and base-input hashes, trade
    counts, jump counts, p_close and outcome are identical between 300 s and 60 s.
  - Per-episode checks as in Tier 1: ledger audit, attribution error ≤ 1.2e−13, none-policy PnL = 0, hash
    sharing across policies and S levels.

**What a 60-s decision means for each policy.**
- **HJB evaluates the same frozen rule five times as often.**
  - `HJBReferencePolicy.action` reads time from the observation's `elapsed_seconds` as
    t = T·(elapsed − start)/(lifetime − start), with time_mapping "episode" and T = 1. It then evaluates the
    same frozen value function at (t, displayed p, q).
  - It is called at 60 stops per episode instead of 12, and the first decision is identical (tested).
  - The post-only projection uses the observation's book. At a 60-s stop that is the most recently sampled book;
    no extra spread draw is made.
  - No policy reads `last_signed_flow`, which at a 60-s stop covers only the executions since the previous stop.
- **Queue priority is reset at every stop.**
  - Every step calls the legacy `_replace` that `InformationTradingEnvironment` inherits from
    `TradingEnvironment`. It cancels every resting order and posts new ones, each with a new order id and the
    full requested size.
  - Each new order gets a fresh queue position from `BookSnapshot.queue_ahead`. Simulated books carry no depth,
    so the position is 0 if the price improves the same-side top of the book and the 50-contract proxy otherwise.
  - An unchanged re-submitted quote therefore loses its queue progress and any partial-fill reduction. There is
    no keep-the-resting-order path (tested).
- **Quote updates per episode** (pooled; `cell_summary.csv`):

  | Policy | Decisions (300 / 60) | Quote rows (300 / 60) | Same-price re-submissions (300 / 60) |
  |---|---|---|---|
  | HJB | 12 / 60 | 24.0 / 119.7 | 4.0 / 42.4 |
  | fixed | 12 / 60 | 24.0 / 119.6 | 3.9 / 44.0 |
  | skew | 12 / 60 | 24.0 / 119.8 | 3.8 / 44.0 |

  At 60 s about 35–37% of posted quotes repeat the price of the order they replace and lose its queue position
  (300 s: about 16%).
  The 60-s arm therefore bundles **more frequent re-pricing with more frequent queue resets** (and full-size
  refills). This sensitivity does not separate the two.

**HJB marginals, pooled over R\*** (mean PnL [95%]; ES95 loss, snipe PnL and snipe count shown as 300 → 60):

| Arm | S | Mean PnL, 300 s | Mean PnL, 60 s | ES95 | Snipe PnL | Snipes |
|---|---|---|---|---|---|---|
| M0 | S0 | +0.07 [−0.63, +0.79] | +0.49 [−0.33, +1.40] | 11.9 → 17.7 | 0 | 0 |
| M0 | S1 | −1.70 [−2.99, −0.38] | −1.41 [−2.67, −0.22] | 19.9 → 23.0 | −1.84 → −1.63 | 6.14 → 5.23 |
| M1 | S0 | −0.12 [−0.73, +0.48] | +0.50 [−0.24, +1.32] | 12.1 → 17.8 | 0 | 0 |
| M1 | S1 | −1.95 [−3.22, −0.67] | −1.23 [−2.50, −0.04] | 19.8 → 22.3 | −1.82 → −1.60 | 6.05 → 5.15 |
| M2 | S0 | −0.01 [−0.59, +0.60] | +0.18 [−0.47, +0.86] | 11.7 → 16.0 | 0 | 0 |
| M2 | S1 | −1.23 [−2.39, −0.21] | −1.19 [−2.17, −0.20] | 15.7 → 18.8 | −1.54 → −1.33 | 2.42 → 1.45 |
| M3 | S0 | −0.20 [−0.73, +0.31] | +0.04 [−0.58, +0.67] | 11.9 → 16.8 | 0 | 0 |
| M3 | S1 | −1.43 [−2.54, −0.48] | −1.31 [−2.24, −0.41] | 14.8 → 18.7 | −1.54 → −1.35 | 2.31 → 1.50 |
| M0-s1 | S0 | +0.08 [−0.58, +0.75] | +0.50 [−0.18, +1.22] | 12.2 → 16.5 | 0 | 0 |
| M0-s1 | S1 | +0.11 [−0.78, +1.08] | +0.35 [−0.43, +1.16] | 14.7 → 18.0 | −0.55 → −0.24 | 1.85 → 0.75 |
| M2-s1 | S0 | −0.08 [−0.63, +0.49] | +0.11 [−0.50, +0.74] | 11.6 → 15.7 | 0 | 0 |
| M2-s1 | S1 | −1.27 [−2.42, −0.25] | −1.25 [−2.24, −0.29] | 15.1 → 18.5 | −1.54 → −1.33 | 2.28 → 1.44 |
| M1-native | S0 | −0.03 [−0.65, +0.60] | +0.39 [−0.23, +1.06] | 12.4 → 17.3 | 0 | 0 |
| M1-native | S1 | −0.10 [−1.00, +0.79] | +0.29 [−0.47, +1.02] | 15.1 → 18.8 | −0.57 → −0.27 | 1.88 → 0.82 |

**Refresh contrast dec60 − dec300, HJB** (pooled unless marked; snipe count = mean snipes per episode):

| Arm | S | Mean PnL | Broad | Sports | ES95 | Snipe PnL | Snipe count |
|---|---|---|---|---|---|---|---|
| M0 | S0 | +0.41 [−0.11, +0.96] | +0.23 [−0.23, +0.71] | +0.63 [−0.34, +1.74] | +5.82 [+1.85, +9.90] | 0 | 0 |
| M0 | S1 | +0.29 [−0.72, +1.28] | +0.06 [−1.05, +1.10] | +0.55 [−1.14, +2.46] | +3.12 [−0.60, +6.87] | +0.21 [+0.03, +0.39] | −0.91 [−1.49, −0.33] |
| M1 | S0 | +0.63 [+0.10, +1.23] | +0.27 [−0.18, +0.74] | +1.03 [+0.10, +2.10] | +5.70 [+1.79, +10.05] | 0 | 0 |
| M1 | S1 | +0.72 [−0.32, +1.76] | +0.36 [−0.66, +1.41] | +1.13 [−0.60, +3.02] | +2.46 [−1.19, +6.60] | +0.22 [+0.04, +0.40] | −0.91 [−1.50, −0.35] |
| M2 | S0 | +0.19 [−0.27, +0.66] | +0.13 [−0.34, +0.59] | +0.25 [−0.50, +1.06] | +4.33 [−0.38, +9.72] | 0 | 0 |
| M2 | S1 | +0.05 [−0.71, +0.76] | −0.29 [−1.35, +0.66] | +0.43 [−0.71, +1.63] | +3.16 [−0.90, +8.21] | +0.21 [+0.09, +0.35] | −0.97 [−1.38, −0.57] |
| M3 | S0 | +0.25 [−0.19, +0.71] | +0.10 [−0.34, +0.51] | +0.41 [−0.35, +1.27] | +4.97 [+0.66, +9.57] | 0 | 0 |
| M3 | S1 | +0.13 [−0.60, +0.89] | −0.22 [−1.21, +0.63] | +0.53 [−0.50, +1.80] | +3.81 [−0.28, +8.20] | +0.19 [+0.06, +0.33] | −0.81 [−1.23, −0.38] |
| M0-s1 | S0 | +0.42 [−0.10, +0.97] | +0.18 [−0.36, +0.67] | +0.69 [−0.29, +1.69] | +4.29 [−0.12, +9.79] | 0 | 0 |
| M0-s1 | S1 | +0.25 [−0.55, +1.09] | −0.26 [−1.35, +0.73] | +0.81 [−0.48, +2.11] | +3.31 [−1.61, +8.47] | +0.31 [+0.17, +0.46] | −1.10 [−1.56, −0.64] |
| M2-s1 | S0 | +0.20 [−0.24, +0.65] | +0.12 [−0.37, +0.56] | +0.28 [−0.45, +1.08] | +4.05 [−0.45, +9.64] | 0 | 0 |
| M2-s1 | S1 | +0.02 [−0.72, +0.79] | −0.28 [−1.36, +0.66] | +0.37 [−0.71, +1.55] | +3.42 [−0.95, +8.33] | +0.21 [+0.08, +0.34] | −0.84 [−1.26, −0.43] |
| M1-native | S0 | +0.42 [−0.10, +0.96] | +0.14 [−0.31, +0.57] | +0.73 [−0.21, +1.74] | +4.93 [+0.94, +9.50] | 0 | 0 |
| M1-native | S1 | +0.39 [−0.41, +1.19] | −0.23 [−1.23, +0.66] | +1.09 [−0.17, +2.40] | +3.68 [−1.19, +8.36] | +0.30 [+0.15, +0.45] | −1.06 [−1.54, −0.58] |

**Fixed policy, briefly.** Pooled mean PnL changes are small, and every range includes zero (−0.28 to +0.79).
Risk rises sharply instead:
- ES95: +16.7 to +20.4 under S0 and +14.2 to +16.7 under S1, every range excluding zero. Pooled fixed ES95 goes
  from 14–17 to 31–37 under S0, and from 20–24 to 36–38 under S1.
- Mean |q|: +8.1 to +10.1 under S0. Fills: +3.9 to +4.7 per episode.
- In the matched no-jump arms M0 and M1 under S1, fixed is sniped **more** at 60 s (+2.03 [+0.43, +3.78] and
  +1.89 [+0.38, +3.58] snipes; snipe PnL −0.70 and −0.65). In the jump and s = 1 arms it is sniped less
  (−1.14 to −1.62 snipes).

The fixed rule has no inventory control. Re-posting full size at a fresh queue position five times as often is
consistent with these larger inventories and tails; the mechanism was not decomposed.

**Refresh × sniping**, (60,S1) − (60,S0) − (300,S1) + (300,S0):

| Arm | HJB mean PnL | HJB broad | HJB sports | HJB snipe PnL | HJB snipes | Fixed mean PnL | Fixed sports |
|---|---|---|---|---|---|---|---|
| M0 | −0.13 [−1.08, +0.79] | −0.17 [−1.20, +0.84] | −0.08 [−1.53, +1.36] | +0.21 [+0.03, +0.39] | −0.91 [−1.49, −0.33] | −0.82 [−2.41, +0.63] | −1.79 [−3.83, +0.51] |
| M1 | +0.09 [−0.84, +1.00] | +0.09 [−0.87, +1.10] | +0.09 [−1.45, +1.56] | +0.22 [+0.04, +0.40] | −0.91 [−1.50, −0.35] | −0.68 [−2.08, +0.72] | −1.60 [−3.44, +0.37] |
| M2 | −0.14 [−0.83, +0.53] | −0.42 [−1.46, +0.49] | +0.18 [−0.74, +1.23] | +0.21 [+0.09, +0.35] | −0.97 [−1.38, −0.57] | +0.03 [−0.97, +0.91] | +0.65 [−0.39, +1.69] |
| M3 | −0.12 [−0.78, +0.54] | −0.32 [−1.30, +0.54] | +0.12 [−0.88, +1.15] | +0.19 [+0.06, +0.33] | −0.81 [−1.23, −0.38] | −0.02 [−0.97, +0.83] | +0.40 [−0.48, +1.35] |
| M0-s1 | −0.17 [−0.80, +0.52] | −0.43 [−1.54, +0.48] | +0.13 [−0.75, +1.07] | +0.31 [+0.17, +0.46] | −1.10 [−1.56, −0.64] | +0.37 [−0.69, +1.33] | +1.36 [+0.26, +2.48] |
| M2-s1 | −0.17 [−0.87, +0.49] | −0.40 [−1.48, +0.51] | +0.08 [−0.82, +1.10] | +0.21 [+0.08, +0.34] | −0.84 [−1.26, −0.43] | −0.01 [−1.00, +0.88] | +0.57 [−0.34, +1.51] |
| M1-native | −0.03 [−0.71, +0.67] | −0.38 [−1.37, +0.50] | +0.36 [−0.51, +1.28] | +0.30 [+0.15, +0.45] | −1.06 [−1.54, −0.58] | +0.32 [−0.65, +1.26] | +1.14 [+0.06, +2.29] |

**Jump contrasts at 60 s and refresh × jump**, (A,60) − (B,60) − (A,300) + (B,300), HJB:

| Contrast | S | Jump effect at 300 s | Jump effect at 60 s | Refresh × jump | Broad | Sports | Snipe PnL |
|---|---|---|---|---|---|---|---|
| M2 − M0 | S0 | −0.08 [−0.42, +0.18] | −0.31 [−0.96, +0.22] | −0.23 [−0.70, +0.18] | −0.09 [−0.26, +0.04] | −0.38 [−1.36, +0.45] | 0 |
| M2 − M0 | S1 | +0.46 [−0.52, +1.40] | +0.23 [−0.63, +1.08] | −0.24 [−1.25, +0.80] | −0.35 [−1.07, +0.38] | −0.11 [−2.26, +1.90] | +0.00 [−0.14, +0.16] |
| M3 − M1 | S0 | −0.08 [−0.46, +0.23] | −0.46 [−1.12, +0.04] | −0.38 [−0.82, +0.02] | −0.17 [−0.40, −0.01] | −0.62 [−1.52, +0.20] | 0 |
| M3 − M1 | S1 | +0.52 [−0.47, +1.45] | −0.07 [−0.90, +0.74] | −0.59 [−1.69, +0.50] | −0.58 [−1.39, +0.19] | −0.60 [−2.82, +1.37] | −0.03 [−0.16, +0.12] |
| M2-s1 − M0-s1 | S0 | −0.17 [−0.45, +0.06] | −0.39 [−0.76, −0.08] | −0.22 [−0.52, +0.04] | −0.06 [−0.20, +0.03] | −0.41 [−1.01, +0.15] | 0 |
| M2-s1 − M0-s1 | S1 | −1.38 [−2.11, −0.75] | −1.60 [−2.34, −0.93] | −0.22 [−0.66, +0.19] | −0.02 [−0.25, +0.17] | −0.45 [−1.28, +0.42] | −0.10 [−0.19, −0.04] |
| M3 − M1-native | S0 | −0.17 [−0.54, +0.11] | −0.34 [−0.68, −0.06] | −0.17 [−0.43, +0.14] | −0.04 [−0.16, +0.03] | −0.32 [−0.91, +0.25] | 0 |
| M3 − M1-native | S1 | −1.34 [−1.98, −0.74] | −1.59 [−2.27, −0.95] | −0.26 [−0.68, +0.18] | +0.01 [−0.21, +0.22] | −0.56 [−1.46, +0.29] | −0.11 [−0.19, −0.04] |

For fixed, the S1 jump effect at s = 1 is −1.70 [−2.53, −0.89] (M2-s1 − M0-s1) and −1.66 [−2.55, −0.82]
(M3 − M1-native) at 60 s. The refresh × jump differences are −0.48 [−1.16, +0.10] and −0.42 [−1.09, +0.26].

**HJB minus baselines at 60 s.**
- HJB − fixed:
  - Mean PnL: ranges include zero in every cell except M0 under S1, +1.50 [+0.07, +2.77].
  - ES95: −13.4 to −19.3 in every cell, all ranges excluding zero. Both tails grow at 60 s. Against 300 s, the
    fixed ES95 rises by +16.7 to +20.4 (S0) and +14.2 to +16.7 (S1); HJB's rises by +4.1 to +5.8 (S0) and +2.5 to
    +3.8 (S1). HJB − fixed ES95 therefore stays at −13.4 to −19.3.

**HJB inventory at 60 s** (pooled; mean |q| before settlement with the dec60 − dec300 contrast [95%], and mean
of the per-episode maximum |inventory|, 300 → 60):

| Arm | S | Mean \|q\| 300 → 60 | Δ mean \|q\| | Mean max \|inventory\| 300 → 60 |
|---|---|---|---|---|
| M0 | S0 | 7.17 → 10.10 | +2.93 [+1.23, +4.87] | 8.52 → 11.75 |
| M0 | S1 | 17.69 → 16.58 | −1.11 [−3.88, +1.62] | 24.06 → 23.39 |
| M1 | S0 | 7.03 → 9.69 | +2.66 [+1.00, +4.53] | 8.44 → 11.46 |
| M1 | S1 | 17.63 → 16.39 | −1.24 [−3.93, +1.35] | 24.04 → 23.16 |
| M2 | S0 | 6.14 → 7.02 | +0.87 [−0.32, +2.36] | 7.32 → 8.27 |
| M2 | S1 | 13.76 → 12.29 | −1.47 [−3.70, +1.06] | 16.98 → 14.76 |
| M3 | S0 | 6.09 → 7.09 | +1.00 [−0.23, +2.48] | 7.43 → 8.45 |
| M3 | S1 | 13.61 → 12.23 | −1.38 [−3.46, +0.92] | 16.88 → 14.78 |
| M0-s1 | S0 | 6.50 → 7.39 | +0.89 [−0.31, +2.32] | 7.79 → 8.76 |
| M0-s1 | S1 | 11.49 → 8.89 | −2.59 [−4.77, −0.18] | 14.61 → 11.00 |
| M2-s1 | S0 | 6.08 → 6.59 | +0.52 [−0.73, +1.99] | 7.22 → 7.71 |
| M2-s1 | S1 | 13.64 → 11.84 | −1.80 [−3.90, +0.52] | 16.78 → 14.22 |
| M1-native | S0 | 6.52 → 7.73 | +1.21 [−0.07, +2.62] | 8.03 → 9.33 |
| M1-native | S1 | 11.43 → 9.12 | −2.31 [−4.56, +0.17] | 14.78 → 11.56 |

Under S0, HJB inventories rise at 60 s; only the matched no-jump arms M0 and M1 have ranges that exclude zero.
Under S1 they fall in every arm, but only M0-s1 has a range that excludes zero. This is consistent with fewer
snipes, since snipes fill the full remaining order. The change is much smaller than fixed's (+8.1 to +10.1
under S0).
- HJB − skew mean PnL: +2.01 [+1.10, +3.05] (M0, S1) and +2.18 [+1.21, +3.24] (M1, S1). The other cells are
  small (−0.10 to +0.60); one of them, M1 S0 (+0.39 [+0.03, +0.84]), excludes zero.

**Reading (HJB unless stated).**
- **Mean PnL barely depends on the decision interval.** Only one of 14 dec60 − dec300 mean-PnL ranges excludes
  zero (M1, S0: +0.63 [+0.10, +1.23]). The point estimates are mostly positive and larger in sports.
- **60-s decisions are sniped less.** In every arm under S1:
  - snipe count falls by 0.8–1.1 per episode;
  - snipe PnL improves by +0.19 to +0.31;
  - all 95% ranges exclude zero.

  The refresh × sniping mean-PnL contrasts are near zero (−0.17 to +0.09, all including zero). The smaller snipe
  loss does not show up as a measurable mean-PnL gain.
- **Tail risk rises with refresh.** HJB ES95 rises by +4.0 to +5.8 under S0 (4 of 7 ranges exclude zero) and by
  +2.5 to +3.8 under S1 (all ranges include zero). For fixed the rise is much larger.
- **The S1 jump effect at equal background survives faster refresh.**
  - At s = 1 it is −1.60 (M2-s1 − M0-s1) and −1.59 (M3 − M1-native) at 60 s, against −1.38 and −1.34 at 300 s.
    The refresh × jump differences (about −0.2) include zero.
  - **At s = 1 only**, its snipe-PnL component is slightly negative: −0.10 [−0.19, −0.04] (M2-s1 − M0-s1) and
    −0.11 [−0.19, −0.04] (M3 − M1-native). There, faster refresh trims snipe losses less in the jump arms than in
    the no-jump arms. This is consistent with jump crossings being harder to avoid by re-pricing, but it was not
    isolated.
  - The matched contrasts show no such difference: +0.00 [−0.14, +0.16] (M2 − M0) and −0.03 [−0.16, +0.12]
    (M3 − M1).
- **The 60-s arm is not a pure "refresh" treatment.** It also resets queue priority at every stop (see above),
  and the HJB rule was solved for a different model. These are comparisons under the stated execution
  conventions, not optimal-refresh results.

## 6b. Sensitivity B: jump intensity 0×, 0.5×, 1×, 2× at s = 1 (`sensitivity/jump_intensity/`)

**Design.**
- Two families, each × S0/S1:
  - F0 flow: 0× = M0-s1, then M2-s1 at 0.5×, 1× and 2× (`M2-s1-x0.5`, `M2-s1`, `M2-s1-x2`).
  - FG flow: 0× = M1-native, then M3 at 0.5×, 1× and 2× (`M3-x0.5`, `M3`, `M3-x2`).
- s = 1 in every cell, so the background is fixed.
- R*, 20 replicates, four policies; 40,960 episodes.
- **Coupled jump construction** (`skeleton.thinning_draws`, `skeleton.superpose_jumps`). Two new episode
  streams are spawned positionally after the twelve Tier 1 children, so children 0–11 are unchanged (tested).
  - 0×: the Tier 1 jump-free arm on the Tier 1 skeleton.
  - 0.5×: each native jump is kept iff a child-12 uniform is < 0.5. Dropped jumps stay in the node set as inert
    nodes, so W, executions, proposals, Θ and every other input equal 1× (shared base-input hash).
  - 2×: the native jumps plus an independent Poisson(λ_J) process with the same TTR-bucket rates. Its times,
    bank magnitudes, V_J and the Brownian-bridge normals all come from child 13. The new nodes are inserted by
    sequential Brownian-bridge refinement of each native W increment, so W at every native node is unchanged up
    to float rounding of the telescoping sum (≤ 1e−12, tested).
  - Thinned and superposed processes are Poisson(0.5·λ_J) and Poisson(2·λ_J) with i.i.d. bank marks and the
    sign tilt. The jump sets are nested: 0.5× ⊂ 1× ⊂ 2×, with identical times, magnitudes, V_J and signs (tested).
  - V2 holds at 0.5× and 2×: the independent posterior equals the path L at every node (< 1e−12, tested).
- **Checks.** All passed:
  - 0× and 1× reproduce the Tier 1 rows exactly (20,480 rows).
  - 0×, 0.5× and 1× share the base-input hash in every episode.
  - Thinned jumps are a subset of 1×, and 2× = 1× + the inserted jumps, in every episode.
  - Mean applied jumps per episode: 0 / 0.47 / 0.90 / 1.77, against 0 / 0.45 / 0.90 / 1.80 expected.
  - Per-episode checks as in Tier 1 (attribution error ≤ 1.1e−13, none-policy PnL = 0, hash sharing).

**Experiment B changes the total information revealed.** Adding jumps on top of a fixed background adds
information, so every dose contrast below is "more jump-borne information". It is not "the same information
arriving as jumps".

The table reports:
- U = E[p_c(1 − p_c)] by the Rao–Blackwellized estimator of `matching.py`: s = 1, p0 = 0.5, N = 200,000 per
  template, family and dose (law = Poisson(dose·λ_J)), averaged over R* templates with equal weights, ± SE;
- information revealed I = p0(1 − p0) − U = 0.25 − U;
- ΔI = I − I(0×), and the jump share ΔI/I;
- the empirical mean p_c(1 − p_c) of the simulated paths (± i.i.d. SE over the 20 × K episodes), as a cross-check.

| Family | Scope | Dose | U (RB ± SE) | I | ΔI vs 0× | Jump share | Empirical p_c(1−p_c) | Jumps/episode (expected / simulated) |
|---|---|---|---|---|---|---|---|---|
| F0 | pooled | 0× | 0.2360 ± 0.00000 | 0.0140 | 0 | 0 | 0.2361 ± 0.0011 | 0 / 0 |
| F0 | pooled | 0.5× | 0.2177 ± 0.00002 | 0.0323 | 0.0183 | 0.57 | 0.2145 ± 0.0022 | 0.45 / 0.47 |
| F0 | pooled | 1× | 0.2025 ± 0.00002 | 0.0475 | 0.0336 | 0.71 | 0.1971 ± 0.0029 | 0.90 / 0.90 |
| F0 | pooled | 2× | 0.1786 ± 0.00002 | 0.0714 | 0.0574 | 0.80 | 0.1759 ± 0.0034 | 1.80 / 1.77 |
| F0 | broad | 0× | 0.2348 ± 0.00001 | 0.0152 | 0 | 0 | 0.2360 ± 0.0018 | 0 / 0 |
| F0 | broad | 0.5× | 0.2314 ± 0.00001 | 0.0186 | 0.0034 | 0.18 | 0.2325 ± 0.0020 | 0.17 / 0.17 |
| F0 | broad | 1× | 0.2281 ± 0.00001 | 0.0219 | 0.0068 | 0.31 | 0.2298 ± 0.0021 | 0.33 / 0.32 |
| F0 | broad | 2× | 0.2216 ± 0.00002 | 0.0284 | 0.0132 | 0.46 | 0.2241 ± 0.0024 | 0.67 / 0.64 |
| F0 | sports | 0× | 0.2374 ± 0.00000 | 0.0126 | 0 | 0 | 0.2361 ± 0.0012 | 0 / 0 |
| F0 | sports | 0.5× | 0.2022 ± 0.00003 | 0.0478 | 0.0352 | 0.74 | 0.1941 ± 0.0039 | 0.77 / 0.81 |
| F0 | sports | 1× | 0.1734 ± 0.00004 | 0.0766 | 0.0640 | 0.84 | 0.1600 ± 0.0048 | 1.54 / 1.54 |
| F0 | sports | 2× | 0.1299 ± 0.00005 | 0.1201 | 0.1075 | 0.90 | 0.1213 ± 0.0052 | 3.09 / 3.06 |
| FG | pooled | 0× | 0.2348 ± 0.00001 | 0.0152 | 0 | 0 | 0.2349 ± 0.0011 | 0 / 0 |
| FG | pooled | 0.5× | 0.2167 ± 0.00002 | 0.0333 | 0.0181 | 0.54 | 0.2137 ± 0.0022 | 0.45 / 0.47 |
| FG | pooled | 1× | 0.2017 ± 0.00002 | 0.0483 | 0.0332 | 0.69 | 0.1962 ± 0.0029 | 0.90 / 0.90 |
| FG | pooled | 2× | 0.1781 ± 0.00002 | 0.0719 | 0.0568 | 0.79 | 0.1752 ± 0.0034 | 1.80 / 1.77 |
| FG | broad | 0× | 0.2346 ± 0.00001 | 0.0154 | 0 | 0 | 0.2359 ± 0.0018 | 0 / 0 |
| FG | broad | 0.5× | 0.2312 ± 0.00001 | 0.0188 | 0.0034 | 0.18 | 0.2322 ± 0.0020 | 0.17 / 0.17 |
| FG | broad | 1× | 0.2279 ± 0.00001 | 0.0221 | 0.0067 | 0.30 | 0.2296 ± 0.0022 | 0.33 / 0.32 |
| FG | broad | 2× | 0.2215 ± 0.00002 | 0.0285 | 0.0132 | 0.46 | 0.2238 ± 0.0024 | 0.67 / 0.64 |
| FG | sports | 0× | 0.2351 ± 0.00001 | 0.0149 | 0 | 0 | 0.2338 ± 0.0013 | 0 / 0 |
| FG | sports | 0.5× | 0.2003 ± 0.00003 | 0.0497 | 0.0348 | 0.70 | 0.1927 ± 0.0039 | 0.77 / 0.81 |
| FG | sports | 1× | 0.1719 ± 0.00004 | 0.0781 | 0.0631 | 0.81 | 0.1584 ± 0.0048 | 1.54 / 1.54 |
| FG | sports | 2× | 0.1289 ± 0.00005 | 0.1211 | 0.1062 | 0.88 | 0.1202 ± 0.0052 | 3.09 / 3.06 |

- The empirical cross-check agrees within about one i.i.d. SE in broad and within 1.9 SE pooled.
- In sports the simulated values sit 1.7–2.8 i.i.d. SE below the RB values at every jump dose. The doses share
  the same 300 episodes, so these gaps are not independent.
- **Open item: the sports gap is not resolved here.** It has the same sign (simulated below RB) at every jump
  dose and in both families. It may be sampling variation or a small systematic discrepancy between the
  simulated paths and the RB law.
  - An ad hoc path-only run of the F0 family on 4,500 fresh sports episodes per dose found the same sign with a
    gap of about 1.3 SE. **That run was not archived and is not evidence of record.**
  - The V9 full-environment cross-check in §7 covers matching at native intensity only, not these doses.
- FG flow adds little information over F0 (I at 0×: 0.0152 against 0.0140).

**HJB marginals** (mean PnL [95%]; ES95 shown as S0 / S1; snipe PnL and snipes are S1, since S0 has no snipes):

| Family | Dose | Pooled S0 | Pooled S1 | ES95 S0 / S1 | Snipe PnL | Snipes | Broad S1 | Sports S1 |
|---|---|---|---|---|---|---|---|---|
| F0 | 0× | +0.08 [−0.58, +0.75] | +0.11 [−0.78, +1.08] | 12.2 / 14.7 | −0.55 | 1.85 | +0.52 [−0.40, +1.53] | −0.36 [−1.80, +1.26] |
| F0 | 0.5× | −0.03 [−0.65, +0.62] | −0.77 [−1.81, +0.20] | 11.6 / 15.0 | −1.12 | 2.08 | +0.41 [−0.55, +1.50] | −2.12 [−3.52, −0.65] |
| F0 | 1× | −0.08 [−0.63, +0.49] | −1.27 [−2.42, −0.25] | 11.6 / 15.1 | −1.54 | 2.28 | +0.28 [−0.75, +1.42] | −3.03 [−4.54, −1.47] |
| F0 | 2× | −0.11 [−0.63, +0.45] | −2.14 [−3.35, −1.00] | 11.5 / 16.3 | −2.18 | 2.67 | +0.01 [−1.04, +1.19] | −4.57 [−5.85, −3.15] |
| FG | 0× | −0.03 [−0.65, +0.60] | −0.10 [−1.00, +0.79] | 12.4 / 15.1 | −0.57 | 1.88 | +0.51 [−0.46, +1.51] | −0.79 [−2.19, +0.59] |
| FG | 0.5× | −0.09 [−0.64, +0.47] | −0.92 [−1.95, −0.01] | 11.9 / 15.2 | −1.13 | 2.10 | +0.41 [−0.60, +1.48] | −2.44 [−3.71, −1.17] |
| FG | 1× | −0.20 [−0.73, +0.31] | −1.43 [−2.54, −0.48] | 11.9 / 14.8 | −1.54 | 2.31 | +0.26 [−0.79, +1.43] | −3.35 [−4.67, −2.02] |
| FG | 2× | −0.27 [−0.81, +0.26] | −2.29 [−3.48, −1.20] | 11.6 / 16.0 | −2.18 | 2.69 | −0.06 [−1.10, +1.08] | −4.83 [−5.89, −3.72] |

**Fixed marginals under S1** (pooled mean PnL; snipe PnL; snipes):
- F0: −0.46 / −1.04 / −1.68 / −2.46 at 0× / 0.5× / 1× / 2×; snipe PnL −1.02 / −1.52 / −1.91 / −2.54; snipes
  3.45 / 3.55 / 3.65 / 3.95.
- FG: −0.49 / −1.16 / −1.73 / −2.61.

**Dose contrasts, S1, HJB** (pooled mean PnL unless marked):

| Family | Dose | vs 0× | vs 0×, broad | vs 0×, sports | ES95 vs 0× | vs 1× | Sniping × dose | Sniping × dose, sports |
|---|---|---|---|---|---|---|---|---|
| F0 | 0× | — | — | — | — | +1.38 [+0.75, +2.11] | — | — |
| F0 | 0.5× | −0.88 [−1.38, −0.43] | −0.11 [−0.41, +0.22] | −1.76 [−2.45, −1.08] | +0.33 [−0.42, +1.46] | +0.50 [+0.15, +0.88] | −0.77 [−1.21, −0.39] | −1.53 [−2.12, −0.96] |
| F0 | 1× | −1.38 [−2.11, −0.75] | −0.25 [−0.70, +0.21] | −2.66 [−3.54, −1.71] | +0.35 [−0.51, +1.58] | — | −1.21 [−1.82, −0.65] | −2.30 [−3.15, −1.46] |
| F0 | 2× | −2.24 [−3.25, −1.36] | −0.51 [−1.12, +0.11] | −4.20 [−5.40, −3.03] | +1.62 [+0.35, +2.99] | −0.86 [−1.44, −0.37] | −2.05 [−2.85, −1.30] | −3.73 [−4.67, −2.81] |
| FG | 0× | — | — | — | — | +1.34 [+0.74, +1.98] | — | — |
| FG | 0.5× | −0.82 [−1.32, −0.39] | −0.10 [−0.38, +0.21] | −1.65 [−2.36, −0.91] | +0.10 [−0.67, +1.22] | +0.51 [+0.19, +0.90] | −0.77 [−1.20, −0.38] | −1.51 [−2.17, −0.92] |
| FG | 1× | −1.34 [−1.98, −0.74] | −0.25 [−0.68, +0.18] | −2.56 [−3.40, −1.63] | −0.25 [−1.32, +1.18] | — | −1.17 [−1.72, −0.64] | −2.17 [−2.97, −1.33] |
| FG | 2× | −2.20 [−3.13, −1.35] | −0.57 [−1.17, +0.04] | −4.04 [−5.22, −2.82] | +0.87 [−0.19, +2.16] | −0.86 [−1.44, −0.39] | −1.96 [−2.70, −1.28] | −3.44 [−4.32, −2.52] |

Sniping × dose is (d,S1) − (d,S0) − (0,S1) + (0,S0). Under S0 no dose contrast of HJB mean PnL has a range
excluding zero (pooled −0.06 to −0.24).

**Dose-response slope** (OLS of the statistic on dose levels 0, 0.5, 1, 2 within each bootstrap draw; dollars
per unit of λ_J multiplier):

| Family | Policy | S | Mean PnL, pooled | Broad | Sports | Snipe PnL, pooled | Broad | Sports |
|---|---|---|---|---|---|---|---|---|
| F0 | HJB | S0 | −0.09 [−0.31, +0.09] | +0.03 [−0.03, +0.12] | −0.22 [−0.66, +0.14] | 0 | 0 | 0 |
| F0 | HJB | S1 | −1.08 [−1.56, −0.66] | −0.26 [−0.57, +0.06] | −2.01 [−2.61, −1.43] | −0.80 [−1.05, −0.57] | −0.22 [−0.29, −0.16] | −1.45 [−1.60, −1.29] |
| F0 | HJB | S1−S0 | −0.99 [−1.37, −0.63] | −0.29 [−0.59, +0.02] | −1.79 [−2.24, −1.34] | −0.80 [−1.05, −0.57] | −0.22 [−0.29, −0.16] | −1.45 [−1.60, −1.29] |
| FG | HJB | S0 | −0.12 [−0.37, +0.09] | +0.04 [−0.01, +0.12] | −0.31 [−0.81, +0.14] | 0 | 0 | 0 |
| FG | HJB | S1 | −1.06 [−1.52, −0.66] | −0.29 [−0.60, +0.02] | −1.94 [−2.52, −1.38] | −0.79 [−1.04, −0.57] | −0.23 [−0.29, −0.16] | −1.43 [−1.57, −1.27] |
| FG | HJB | S1−S0 | −0.94 [−1.29, −0.61] | −0.33 [−0.64, −0.04] | −1.63 [−2.07, −1.20] | −0.79 [−1.04, −0.57] | −0.23 [−0.29, −0.16] | −1.43 [−1.57, −1.27] |
| F0 | fixed | S1 | −1.00 [−1.50, −0.55] | −0.22 [−0.52, +0.11] | −1.88 [−2.51, −1.25] | −0.75 [−0.97, −0.54] | −0.23 [−0.30, −0.16] | −1.33 [−1.51, −1.16] |
| FG | fixed | S1 | −1.04 [−1.54, −0.58] | −0.26 [−0.51, +0.05] | −1.94 [−2.58, −1.32] | −0.73 [−0.96, −0.53] | −0.23 [−0.30, −0.16] | −1.31 [−1.49, −1.13] |

**Snipe PnL per unit of information revealed.** This is the dose response restated per unit of expected
p(1 − p) reduction. I is a deterministic, increasing function of dose, so the ratio adds no independent evidence
that snipe losses track information. It is a descriptive ratio of measured means (S1; `information_revealed.csv`)
and **carries no bootstrap range**. Incremental snipe PnL per unit of incremental information,
(snipe(d) − snipe(0×))/ΔI, at 0.5× / 1× / 2×:

| Family | Policy | Pooled | Broad | Sports |
|---|---|---|---|---|
| F0 | HJB | −31.2 / −29.5 / −28.5 | −41.7 / −38.5 / −34.5 | −30.1 / −28.4 / −27.6 |
| FG | HJB | −30.9 / −29.3 / −28.4 | −41.1 / −38.5 / −34.8 | −29.7 / −28.2 / −27.5 |
| F0 | fixed | −27.0 / −26.6 / −26.4 | −41.8 / −39.0 / −35.4 | −25.4 / −25.1 / −25.2 |

Read as dollars of snipe PnL per episode per unit of expected p(1 − p) reduction. For example, 0.01 of
additional information revealed goes with about −$0.29 of HJB snipe PnL per episode, pooled. The ratio is not
constant: for HJB F0 pooled it drifts by about 9% across doses (−31.2 / −29.5 / −28.5), and by more in broad
(−41.7 / −38.5 / −34.5).

**Reading.**
- **Without sniping, jump intensity has no clear effect on mean PnL** (S0 slopes −0.09 and −0.12, ranges
  including zero).
- **With sniping, mean PnL falls with jump intensity, monotonically across the four doses.**
  - The HJB S1 slope is about −$1.1 per unit of λ_J multiplier pooled, −$2.0 in sports, and about −$0.3 in broad
    (range includes zero for S1).
  - The snipe PnL slope (−0.80) accounts for roughly three quarters of the mean-PnL slope. The rest goes through
    other channels and is not decomposed.
  - Per unit dose the S1 effect against 0× shrinks with dose: −1.76, −1.38 and −1.12 per unit at 0.5×, 1× and
    2× (F0, HJB). The bootstrap ranges overlap, and no curvature test was run.
- **Per unit of information revealed (descriptive).** Restated per unit of incremental information, HJB
  incremental snipe PnL is roughly stable across doses: −31.2 / −29.5 / −28.5 pooled (F0), a drift of about 9%,
  with no bootstrap range. This restates the dose response, because I is a deterministic function of dose; it is
  not separate evidence. The ratio is larger in broad, where little jump information arrives per episode.
- **Tail.** The only HJB S1 ES95 contrast against 0× whose range excludes zero is F0 at 2×: +1.62 [+0.35, +2.99].
  FG at 2× is +0.87 [−0.19, +2.16], and at 0.5× and 1× all the ES95 contrasts include zero.
- **Sports carries the effect**, as in Tier 1: more jumps per episode (1.54 against 0.33 at 1×) and larger
  E[a²].
- **The families agree.** F0 and FG are nearly identical because calibrated FG flow carries little information.
- **These are not "discontinuity" effects.** The dose changes the amount of information revealed (from 0.014 to
  0.071 pooled, and from 0.013 to 0.120 in sports). The contrasts measure "more jump-borne information at a
  fixed background", conditional on the training templates and the assumed sniping scenario.

## 7. Validation (`validation_report.json`, full scale, 10 workers, 608 s; all V-items passed)

- **V1 kernels.** Closed forms hold to 1e−12: GM Δp, zero mean and variance; FG-half LLR = ±βc; sign-tilt LLR;
  jump martingale; Θ-free sampler laws; finite L at extreme LLRs.
- **V2.** An independent posterior calculator matches the path L at every node: 500 episodes per M arm
  (including the p0_emp arms), largest difference 1.2e−13.
- **V3.** Permutation invariance of the terminal posterior holds, largest difference 6.8e−14 (900 episodes).
- **V4.** Martingale moments: 20,000 price-path episodes per M arm per class, QR-reduced instruments, joint χ²,
  Holm across M arms. Smallest unadjusted p = 0.45; no rejection.
  - FG-half L arms do not reject (p = 0.40–0.78).
  - L0b, L0c, L1 and L1J reject strongly (p ≈ 0). This is expected and reported as a diagnostic: FL
    fixed-impact flow with S_u signs, and legacy-law components, are not Bayesian updates, so p is not a
    martingale in those arms.
- **V5.** Outcome calibration: smallest unadjusted p = 0.029 (broad M0), Holm-adjusted 0.147; no rejection.
- **V6.** The budget identity is exact (≤ 1.7e−16), and the weighted-drift aggregate Σ(2p₋ − 1)Δp has mean
  zero (p ≥ 0.29).
- **V7.** The golden sweep reproduces `TradingEnvironment` byte-for-byte over 35 templates × 2 seeds ×
  4 policies. The L0 anchor reproduces 2,800 episodes (difference ≤ 2.8e−14).
- **V8.** Base-input, signal, book and settlement hashes behave as specified.
- **V9.** All R* gaps are within tolerance. In the full-environment no-quote cross-check (3 templates × M0–M2 ×
  2,000 episodes), |z| ≤ 1.26 against the RB estimate.
- **V10 fidelity (report only):**
  - Jump counts match the calibrated λ_J: sports M3 1.53 per episode vs 1.54 expected; broad 0.33 vs 0.33.
    Mean jump size matches the bank mean.
  - Simulated sports Kyle λ is 2.0–2.9e−5, against a target of 1.25e−5. The flow-only replay used for
    calibration hits the target; the full arms add background and jumps between the first and last trade.
  - **Broad replicate-averaged Kyle λ medians are about 0**, because most broad windows contain 0–1 trades and
    the slope is then defined as 0. The pooled broad median is 2.8–3.0e−5.
  - **Excess kurtosis shortfall.** 300-s ΔL excess kurtosis in the simulation is far below train data. The
    per-template median is about 0 for the no-jump arms and 16–22 for the jump arms, against train medians of 10
    (broad) and 32 (sports); pooled values are 18 vs 58 (broad) and 16 vs 80 (sports). The top-3 share of |ΔL| is
    also lower (sports 0.66 vs 0.89).
- **V11.** Every one of the 74,480 experiment episodes was audited.
- **V12 sniping hazard.** The check compares snipes with opposite-side proposal "probes" (an unbiased PASTA
  estimate of crossed exposure).
  - The first full-scale run gave a snipe/probe ratio of 0.979. A pooled binomial CI rejected 1 (p = 0.017),
    but that test treats arms and policies that share skeletons as independent.
  - After seeing that result, the primary test was changed to a cluster-robust ratio CI, clustered on
    (template, replicate): p = 0.107, not rejected. **This was a post-hoc change.**
  - A **pre-registered confirmation** (`v12_preregistration.md`) then ran on fresh seeds (replicate offset
    6,000,000; 60 replicates × R* × M0–M3 × {fixed, hjb}; 15,360 episodes; 1,920 clusters). It gave ratio
    0.9993 (SE 0.0074, p = 0.93; `v12_confirmation.json`):
    - by arm: M0 1.003, M1 1.006, M2 0.985, M3 0.989;
    - by policy: fixed 1.001, hjb 0.996.
  - An independent code reviewer's ad-hoc run on another disjoint replicate range gave 1.0045 (SE 0.0126). That run was not archived, so it is corroboration only; the pre-registered run above is the evidence of record.
  - The unit tests show the check detects a sniper that ignores 10% of crossed proposals. Tier 2 grid
    convergence was not run.
- **V13.**
  - Training isolation: calibration held-out files = 0, all templates are train.
  - C-1: only new files.
  - C-4: preserved paths unchanged.
  - C-2 is checked separately (`--c2` runs the Paper 2 validator on a temporary copy).

## 8. Amendments (recorded before the calibration rerun; none tunes a result)

- **A-1.** The flow-replay bisection tolerance was tightened from 0.5% to 0.1% of the target, so that bisection
  error is below the required MC precision.
- **A-2.** The validation CI for the replay relative error uses √(SE_cal² + SE_val²), because the fitted
  parameter carries calibration-sample MC error.
- **A-3.** U_L is the legacy settlement draw (stream child 4), not stream 10. As a result, L0 → L0b changes
  only the flow signs, and the whole L chain shares one settlement uniform.

## 9. Limitations

- **Reduced-form jumps.** Jumps are large 300-s quote-midpoint residuals (|η̃| > 4σ̂), not identified news
  events. Intensities come from a trailing support window of about 3 days, with pooled buckets where counts are low.
- **Provisional sports release time.** Sports t_r is the template close; official game-end times are deferred.
  Ten broad templates (and twelve sports templates) have settlement timestamps earlier than their recorded
  occurrence time.
- **S1 is an uncalibrated execution scenario.** The latency ℓ̄ = 2 s, the taker fee rule and the
  full-remaining fills are assumptions, not estimates of real sniper behaviour.
- **Kyle-λ matching convention.** η matches the train-median Kyle λ in a flow-only post-trade replay. It does
  not identify informed traders, and full arms show a different λ (V10).
- **p0 = 0.5 is a counterfactual.** Real final-hour prices are often near 0 or 1, as p0_emp shows. The p0_emp
  arms are the sensitivity check.
- **Legacy final-hour arrival understatement.** The legacy arrival model understates final-hour trade arrival by
  about 4.5× (sports) and 2.4× (broad); the spec reports this and it was not recomputed here. Flow information
  and fills in the final hour are therefore understated.
- **Excluded templates.**
  - KXAAAGASW-26JUL27-4.080 (σ_c = 0);
  - KXFEDDECISION-26JUL-C26 and KXPCECORE-26JUL-T0.2 (early revelation; Tier 2 not implemented). The L0 anchor
    still uses all 35 templates.
- **HJB is not optimal under these dynamics.** The frozen Paper 2 HJB reference was solved for a different model
  (no informed flow, no jumps, no sniping). HJB-minus-baseline contrasts are comparisons, not optimality tests.
- **Volatility shape.** Simulated returns are much less fat-tailed than train data (V10 excess kurtosis and
  top-3 share), particularly in the no-jump arms.
- **Broad Kyle λ.** Broad Kyle λ medians are 0 under the replicate-averaged convention, because broad windows
  are mostly too thin to estimate a slope.
- **Sports constant-α calibration failed.** The MC SE tolerance was not reached, so the constant-α Tier 2
  sensitivity is unavailable for sports; broad is validated.
- **Weakly identified M2 scales.** Some p0_emp M2 matched s values are poorly determined (§4), although the
  matched U itself is accurate.
- **Tier 2 status.**
  - **Run as dedicated sensitivity experiments:** 60-s quote refresh (§6a) and the coupled jump-intensity dose
    response at s = 1 (§6b), via `simulator/information/sensitivity.py`.
  - Implemented behind `--tier2`, not yet run: constant-α, other fill scenarios and ℓ̄ = 30 s. The
    `--tier2` jump-rate cells (`-jx`) redraw jump times without coupling and do not hold background noise
    fixed across doses, so use §6b rather than these cells for dose response.
  - **60-s decisions also reset queue priority.** The legacy `_replace` cancels and re-posts every quote, so §6a
    cannot separate faster re-pricing from queue resets. A variant that keeps unchanged resting orders is not
    implemented.
  - **Not implemented:** the zero-latency grid (and its convergence check), S1-thinned executions, TTR-clock
    arrivals, and early-revelation arms.
- **Scope of the bootstrap.** The bootstrap resamples templates and replicates. It does not cover calibration
  uncertainty (for example, the bootstrap ranges of λ_J and E[a²] in §2) or cross-market dependence.
