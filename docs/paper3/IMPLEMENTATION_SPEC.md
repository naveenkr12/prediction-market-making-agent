# Paper 3 core extension — ELEVATED implementation specification (revision 2)

Status: research plan approved by the user (2026-10-31) after three Sol plan-review rounds (round 3 APPROVE).
Revision 2 incorporates Sol's specification review (13 findings). This document is the single source of truth.
Scope: **core Bayesian information-jump extension and its four martingale-family arms**, the legacy
comparison chain, stale-quote execution, required sensitivities, validation and experiment.
**Deferred:** deadline-resolution (Beta) volatility, GARCH multiplier, news-linked trading activity,
DR-AS heuristic policy, official sports game-end timestamps. **Tier 2** items (Section 9.3) are specified and
run only after Tier 1 completes and validates; any not run is reported as not run.

Notation: Θ ∈ {0,1} hidden outcome; L log odds; p = expit(L); LLR = log f1/f0. "Shared stream" = random
stream spawned from the episode SeedSequence and consumed identically in every arm/policy/S level using it.

---------------------------------------------------------------------------------------------------
## 0. Objective and hard constraints
Objective: measure how the frozen HJB reference and simple policies perform when (a) trade flow carries
outcome information (Bayesian Glosten–Milgrom flow), (b) public information arrives in discrete jumps,
(c) both, in a fair martingale, variance-matched comparison, with stale-quote sniping as a primary factor;
and connect to the existing Paper 2 simulator through a single-change legacy chain.

- C-1 **Zero modifications** to any existing tracked file. New files only. `git diff --name-status main`
  must show only `A` entries.
- C-2 Existing suite (83 tests) and `python -m simulator.validate_hjb_benchmark data_pipeline/output/paper2/calibration/simulator/hjb_benchmark` pass.
- C-3 Training-only data access (train split from `event_splits.csv`); held-out ticker access raises.
- C-4 Preservation manifest: sha256 of `hjb_baseline/*.py`, `hjb_baseline/output/data/{V_production.npy,grid_production.npz}`,
  `simulator/*.py`, `calibration/*.py`, all files under `data_pipeline/output/paper2/calibration/simulator/` identical before/after.
- C-5 No commits, pushes, PRs. Work only in the designated worktree.
- C-6 `/opt/anaconda3/bin/python` (3.13.x), no new dependencies.

## 1. New file layout
```
simulator/information/__init__.py
simulator/information/kernels.py        # pure update kernels and samplers (no I/O)
simulator/information/config.py         # load/verify Paper 3 calibration (hashes), roster
simulator/information/skeleton.py       # policy-independent base random inputs per episode
simulator/information/paths.py          # per-(episode, arm) precomputed latent/price/book paths
simulator/information/environment.py    # InformationTradingEnvironment (walks precomputed paths; fills, snipes)
simulator/information/taker_fees.py
simulator/information/matching.py       # terminal-LLR variance matching (Rao–Blackwellized)
simulator/information/reference.py      # independent sequential posterior calculator
simulator/information/metrics.py        # attribution identity, diagnostics
simulator/information/bootstrap.py      # common-index paired hierarchical bootstrap
simulator/information/evaluate.py       # experiment CLI (checkpointing)
simulator/information/validate.py       # offline validation CLI (full-scale checks)
simulator/INFORMATION_ARRIVAL.md
calibration/calibrate_information_arrival.py
simulator/tests/test_information_kernels.py
simulator/tests/test_information_environment.py
simulator/tests/test_information_matching.py
simulator/tests/test_information_statistics.py   # bounded samples
calibration/tests/test_information_calibration.py
data_pipeline/output/paper3/calibration/...
data_pipeline/output/paper3/experiment/...
```
Reuse by import only: `BinaryAccount`, `FeeSchedule`, `QuoteAction`, `RestingOrder`, `BookSnapshot`,
`SpreadSampler`, `MarketDynamics`, `ImpactParameters`, `ImpactPriceProcess`, `bridged_innovations`,
`TradingEnvironment`, `EnvironmentConfig`, `HJBReferencePolicy`, `quote_policy`, `audit_ledger`,
`episode_diagnostics`, `benchmark_metrics.{expected_shortfall,pnl_statistics}`, `evaluate_hjb.cached_environment`,
`demo_trading_env.FILL_SCENARIOS`, calibration helpers (`residual_noise_fit`, `fit_lambda` etc.).

## 2. Times, clocks and roster (D5)
Per template: t_c = template close (open_time + lifetime_seconds); episodes = last 3600 s before t_c.
t_r: broad_based = `occurrence_datetime` (metadata) if present, else t_c with flag `close_fallback`;
sports = t_c, flag `provisional_sports_close`. t_s recorded only (no discounting).
**Release-time verification (broad):** table per broad template with t_r, expected_expiration_time,
settlement_ts, series and rules excerpt; flag inconsistencies (e.g. occurrence after expected_expiration).
Persist in calibration outputs and docs.
TTR = t_r − t; buckets B0 [0,3600), B1 [3600,10800), B2 [10800,86400), B3 [86400,∞) seconds.
**Early-revelation rule:** a template with t_r < t_c is "early revelation" (known from data:
KXFEDDECISION-26JUL-C26 −1,503 s, KXPCECORE-26JUL-T0.2 −111 s; implementer recomputes by rule).
**Primary roster R\***: train templates of broad_based and sports that are (i) not early-revelation and
(ii) matched for every M arm (Section 8; e.g. a template with σ_c = 0 such as KXAAAGASW-26JUL27-4.080 cannot
match M0 and is excluded). R\* is computed by the matching stage and **frozen before any policy run**;
the same R\* is used for every matched M contrast and every L/contextual contrast. The **L0 anchor** uses
all 35 templates (Section 9). Early-revelation templates are analysed only in Tier 2 (M family, absorbing
revelation, excluded from matching).
Regime pre-release close (t_r > t_c): no information event represents the release; settlement reveals Θ
(M) / uses U_L (L). p_close = p at t_c−.

## 3. Calibration (`calibration/calibrate_information_arrival.py`), train only
Inputs and hashes as in the existing calibrations; record all input hashes; refuse held-out tickers.

### 3.1 Residuals and normalization
Source rows: `training_noise_intervals.parquet` filtered to the contract's own class label (rows labelled
`pooled` are duplicates — drop them; assert after filtering that (ticker, left_time) is unique), full
300-s consecutive observed pairs only. Per supported contract fit exactly the legacy `residual_noise_fit`
regression (import it) and obtain η_k and degrees-of-freedom d (2 with a flow slope, 1 intercept-only).
Assert recomputed σ_legacy² = Σ η² / ((n − d)·300) equals the stored template sigma² (relative 1e-10).
**Effective innovations** η̃_k = η_k·sqrt(n/(n − d)), so Σ η̃² / (n·300) = σ_legacy² exactly. All jump
detection, magnitudes, continuous variance and intensities use η̃.

### 3.2 Jump detection (predeclared)
Adjacent products |η̃_i||η̃_{i−1}| only for consecutive observed pairs. Local σ̂_k² = (π/2)·mean of products
within ±12 observed pairs of k excluding products involving k−1, k, k+1; ≥ 12 products else contract-level
(π/2)·mean(all products). σ̂_k² is per-300-s variance. Jump iff |η̃_k| > 4σ̂_k. Contract with all η̃ = 0 has no
jumps. Record per contract: n, d, jumps (time, TTR bucket, η̃).

### 3.3 Continuous background σ_c (per supported template)
σ_c² = Σ_{non-jump} η̃_k² / (n·300) (log-odds² per s); identity σ_c² + Σ_{jump} η̃_k²/(n·300) = σ_legacy² asserted.
**Noise donors:** legacy `MarketDynamics` selects a donor noise template at reset (selection stream) for
templates without own support. The new arms keep **the same donor law and the same donor draw** (the
skeleton reuses `MarketDynamics` construction on the selection stream), and σ_c is the donor's σ_c.
Matching integrates over the donor distribution (Section 8). Zero σ_c is not floored.

### 3.4 Jump intensity and magnitudes (per class)
λ_J(class, bucket) = jumps / exposure seconds over that class's supported contracts (exposure = 300 s per
pair, assigned to the bucket of the pair's **left endpoint** TTR = t_r − left_time). If jumps < 5 in bucket b,
pool counts **and** exposure with buckets added in the deterministic neighbour order
B0: B0→B1→B2→B3; B1: B1→B0→B2→B3; B2: B2→B1→B3→B0; B3: B3→B2→B1→B0, stopping at the first prefix with ≥ 5
jumps; λ_J(b) = pooled count / pooled exposure; record contributing buckets. Class total < 5 → pooled-class
profile (flag). (Known from data: broad B0 = 0, B1 = 2, B2 = 13, B3 = 33 jumps.) Magnitude bank M_class = {|η̃_k|} of flagged jumps.
Event bootstrap (2,000 draws, seed 42, contracts within class) for λ_J and E[a²]: low/base/high recorded;
primary = full-sample point estimates. Report probability-scale conditional variance at p = .5 by bucket and
implied vs detected counts; label as reduced-form jump proxies.

### 3.5 Uninformed sign rates u
Size bins: 5 class-level quantile bins of mark size (pooled class marks). k = legacy lifetime phase.
u_{tpl,b,k} = (n_yes + 20·u_{class,b})/(n + 20), empty → u_{class,b}; clip [0.02, 0.98]; also u_{tpl,b} per
template for fallback. Recorded assumption: α small ⇒ YES rate ≈ u.

### 3.6 Informed flow FG: α(c; η) = min(1 − exp(−η c), 1 − 1e−12)
Flow-only replay, post-trade observation convention, per class:
- Frozen inputs: train buckets from `five_minute_intervals.parquet` (train split; exclude buckets with
  invalid log-odds endpoints exactly as Paper 2), each bucket's trade sequence (sizes, Paper 2 normalized
  order), and frozen uniforms per (replication, bucket, trade): V_Θ, V_I, V_u. These are held fixed across
  all candidate η (common random numbers).
- Segment per bucket: p_start = bucket open price; Θ = 1{V_Θ < p_start}; I = 1{V_I < α(c)};
  S_u = 2·1{V_u < u} − 1 ∈ {−1,+1}; sign = 2Θ−1 if I else S_u; L += GM LLR (Section 4.3). Observed open = L after first
  trade; close = L after last; F = Σ sign·c.
- Contract slope = OLS slope of ΔL on F with intercept (implement directly; at η = 0, ΔL ≡ 0 and the slope is
  defined as 0, not a fit failure); class statistic = median over contracts of the replication-averaged slope.
- Objective g(η) = class statistic − target (train-median Kyle λ from `price_impact_config.json`).
  Bracket: η_lo = 0 (g < 0), η_hi doubled from 1e−6 until g > 0 (max 60 doublings, else fail loudly).
  Bisection to |g|/target < 0.5%.
- Replications: R starting 200, doubled until the batch-means MC SE of the class statistic < 0.25% of target.
- Validation on independent uniforms (R_val = R): relative error < 2% with 95% CI covering 0; else report
  as a calibration limitation (do not silently re-tune).
- Also report FG-half replay statistic (u = ½, α = tanh(βc/2), base β) and a constant-α variant
  α ≡ α0 (Tier 2 sensitivity) calibrated by the same procedure.
Report η, α at size quantiles (10/50/90/99%), sign/size moments, information variance.

### 3.7 Empirical starting probabilities
p0_emp = last train-tape trade price strictly before t_c − 3600 s (else first after; record source);
simulation start clipped to [0.01, 0.99] (record).

### 3.8 Outputs `data_pipeline/output/paper3/calibration/`
`information_config.json`, `jump_magnitude_bank.npz`, `uninformed_sign_rates.npz`, `jump_detection.csv`,
`decomposition_check.csv`, `release_time_verification.csv`, `flow_replay_calibration.csv`,
`calibration_report.json` (heldout_files_loaded = 0), sha256 of all.

## 4. Kernels (`kernels.py`)
Latent L in float64. Display p (clipped) only for observations/quotes/HJB inputs; filtering, LLRs,
sniping crossing and validation use unclipped expit(L). Saturation counted.
4.1 Background M: interval h, σ² = s·σ_c²: X = (σ²h/2)(2Θ−1) + σΔW; L += X.
4.2 Background L (Θ-free exact marginal transition): Θ̃ ~ Bernoulli(expit(L)) from the L-aux stream;
L += (σ²h/2)(2Θ̃−1) + σΔW (same ΔW). Legacy-law (L0c): L += σΔW.
4.3 Flow. **Global sign convention:** every execution sign is in {−1,+1} (+1 YES taker); the uninformed sign is
S_u = 2·1{V_u < u} − 1 everywhere (replay, F0, FL from L0b, FG uninformed branch); assert all realized signs ∈ {−1,+1}.
LLR (α = α(c) or tanh(βc/2) for FG-half; u = u_{tpl,b,k} or ½):
buy: log[(α+(1−α)u)/((1−α)u)]; sell: log[((1−α)(1−u))/(α+(1−α)(1−u))].
M sampler: I = 1{V_I < α}; sign = 2Θ−1 if I else S_u. Θ-free sampler: buy iff V_F < (1−α)u + α·expit(L) (single
uniform; no other branch). F0: sign = S_u, no update. FL: sign = S_u (L0b+) or bank sign (L0), L += β·sign·c
using `ImpactPriceProcess.advance` arithmetic order (before + innovation + impact).
4.4 Jumps: magnitude a from M_class (shared), times Poisson λ_J(class, TTR bucket) (shared; generated in
every arm, applied when jumps on). M: sign + iff V_J < expit(a) if Θ=1 else expit(−a). Θ-free: + iff
V_J < p·expit(a) + (1−p)·expit(−a). Legacy-law: + iff V_J < ½. L += sign·a.
4.5 Settlement: M pays Θ = 1{U_M < p0}; L pays 1{U_L < p_close} (U_L never used by generators); L0 uses
`TradingEnvironment`'s settlement stream unchanged.

## 5. Skeleton (`skeleton.py`) — base random inputs, policy/arm/S-independent
Seeding: path_seed = `int(np.random.SeedSequence([seed, event_index, replicate]).generate_state(1)[0])`
with seed 20261009, groups ordered broad_based then sports, tickers sorted within group, **global event_index**
over that full ordering (preserved when running subsets), exactly as `evaluate_hjb.run`.
Episode SeedSequence(path_seed).spawn(N): children 0–4 exactly as `TradingEnvironment.reset` (selection,
market, spread, bridge, settlement); then in fixed order: 5 jump, 6 flow-sign (V_u, V_I, V_F), 7 background
generator W, 8 L-aux Θ̃, 9 U_M, 10 U_L, 11 snipe proposals.
Skeleton contents (generated before any policy act):
- Legacy driver draws on legacy streams in legacy order: `MarketDynamics(...)` built on the selection stream
  exactly as reset (template, donor); per window `sample_window` then `sample_innovation` (market stream).
- Jump events (times, magnitudes, V_J) for the whole episode.
- Snipe proposal processes: two independent homogeneous Poisson processes (bid side, ask side) of rate 1/ℓ̄
  over the episode (stream 11), ℓ̄ = 2 s primary.
- Decision nodes: deterministic times every 60 s from episode start that fall strictly inside a 300-s market
  window (always included in every non-golden run, inert unless the 60-s decision sensitivity is active), so
  the node set — and therefore W, all M price paths and all exogenous executions — is identical between
  300-s and 60-s decision runs.
- Node set per window: execution times, jump times, proposal times, decision nodes, window end.
- W increments with exact variances between consecutive nodes (stream 7). For zero-latency (Tier 2) a
  0.25-s / 0.0625-s grid is added by Brownian-bridge refinement of the same path (coarse-node values unchanged).
- Shared uniforms V_u, V_I, V_F per execution and V_J per jump; U_M, U_L.
**Spread draws are not precomputed**: `SpreadSampler.sample(log_odds, rng, tick)` is called online at the
same points and order as legacy — reset (after any donor-selection draw consumed by `SpreadSampler.__init__`),
then at each execution **after** the background piece and **before** matching, and at window end — never at
jump or proposal nodes. Spread draws therefore consume the spread stream identically in every arm.

## 6. Paths (`paths.py`) and environment (`environment.py`)
6.1 Precomputation: since fills and snipes never move prices and all kernels are policy-independent, for
each (episode, arm) compute before the policy loop the complete latent path L at every node (all nodes,
regardless of S level, so price paths are identical across S0/S1 and policies), the external books (online
SpreadSampler in legacy order during precomputation), realized signs and jump signs, and the settlement.
The environment walks this path.
6.2 Golden branch (L0-equivalent): uses only legacy nodes (executions, window end), the empirical window
innovation with legacy `bridged_innovations` pieces on the legacy bridge stream, bank signs, legacy reset
draw order, legacy ledger rows and arithmetic (via `ImpactPriceProcess.advance` calls in legacy order). It
must reproduce `TradingEnvironment` ledgers/observations/rewards/settlement byte-for-byte (all 35 templates ×
2 seeds × 4 policies). L0 in the experiment runs `TradingEnvironment` itself.
L0b uses the golden branch with bank signs replaced by S_u (and recomputed signed-flow fields); L0c onward
use Gaussian W at the skeleton nodes.
6.3 Step/policy interface identical to `TradingEnvironment` (reset/step 5-tuple, observation keys plus
`arm`, `family`, `snipe_level`); observations never contain Θ, I, future nodes or future signs. Fills follow
legacy matching (queue, participation, admission) at execution nodes, booked at standing quote with maker fee
before the flow update.
6.4 Sniping (S1): at each proposal node of side X, if the maker has an alive side-X order with remaining Q
and it is crossed at that node's exact p — ask a_m: Q(p − a_m) > taker_fee(Q, a_m) + Q·tick; bid b_m:
Q(b_m − p) > taker_fee(Q, b_m) + Q·tick — it fills fully at its price (maker fee via `BinaryAccount.fill`),
bypassing queue/participation, no price update, ledger kind `snipe`, not added to market flow. Under S0
proposal nodes are inert. This is the exact occupation-time hazard (rate 1/ℓ̄ while alive∧crossed).
`taker_fee(Q,P) = ceil_cent(0.07·Q·P·(1−P))` (coefficient configurable; scenario label recorded).
6.5 Attribution (`metrics.py`), additive identity (must reconcile within 1e-8):
PnL = Σ_fills(incl. snipes) Δq·(p − P) − fees + Σ_price updates q_−·Δp + q_c·(Y − p_c),
with price-update terms split by channel (background, flow, jump). Nonadditive diagnostics reported
separately: flow markout (new fill Δq × next flow Δp), jump markout (fills within 60 s before a jump × jump Δp),
snipe count/volume/loss (Δq(p−P) of snipes, negative = loss), 5-minute markouts, p_c(1−p_c),
q_c²p_c(1−p_c), finite-grid realized variance by channel and TTR bucket, exogenous vs sniper volume.
Independent auditor in `validate.py` reconstructs cash/inventory/fees/collateral from the ledger, checks
each execution's flow update once, each jump once, snipes have no price update.

## 7. Arms (all on roster R\* unless stated; base fill (50, 0.1); base β; calibrated η; p0 = 0.5)
M family (shared Θ, settle Θ), each × S{S0,S1}:
| M0 F0, no jumps, s_M0 | M1 FG, no jumps, s_M1 | M2 F0, jumps, s_M2 | M3 FG, jumps, s = 1 (target) |
L family (Θ-free, settle U_L), S0:
L0 `TradingEnvironment` (all 35 templates for anchor; R\* for contrasts) → L0b (S_u signs) → L0c (σ_c Gaussian
legacy-law + legacy-law jumps at native λ_J) → L1J (Θ-free martingale background + tilted jumps; FL flow) →
L1 (L1J without jumps). s = 1.
Contextual: M1-native (FG, J0, s = 1) × S{0,1}; contrasts M1-native vs L1 and M3 vs L1J (bundled mechanism
replacement). FG-half in L family: FGH-L (J0) vs L1, FGH-LJ (J1) vs L1J, identical execution rules,
times, sizes, background, jumps, settlement convention.
Secondary (volatility-increasing): M0, M1, M2 at s = 1 × S{0,1} (+ M3).
Required sensitivity: M0–M3 at p0_emp with s re-matched at that p0 × S{0,1}; its own roster R\*_emp (same
rule) reported.

## 8. Variance matching (`matching.py`)
For M arms the experiments are fixed and p-independent, so given Θ, L_c = L0 + Λ_B + Λ_N with Λ_N = flow + jump
LLRs and Λ_B ~ N((2Θ−1)V/2, V), V = s·σ_c(donor)²·T_ep, independent of Λ_N given Θ.
Rao–Blackwellized per draw i: sample donor (legacy donor law), Θ, Λ_N from the skeleton generators
(vectorized); q_i = expit(L0 + Λ_N,i); h_i(V) = q_i·E_Z[expit(−(ℓ_i + V/2 + √V Z))], ℓ_i = L0 + Λ_N,i
(identity E[p'(1−p')] = q·E_{Θ=1}[1 − p']). h_i is decreasing in V for every i (Blackwell, per draw), so
Û_A(s) = mean_i h_i(s·σ_c(donor_i)²T_ep) is monotone. Quadrature with error control: evaluate the logistic–
normal integral with a method accurate for all V (e.g. integrate over Z when √V ≤ 1, over the logistic
variable via Gauss–Legendre on u ∈ (0,1) with Lg = logit u when √V > 1); accept only if two orders agree
within 1e−9 absolute at every grid point; verify against `scipy.integrate.quad` on 1,000 random cases.
Procedure per (template, arm, p0): N = 200,000 draws; target U\* = Û_{M3}(1) on an **independent** draw set;
grid s ∈ [0, s_hi] with s_hi doubled from 4 until Û_A(s_hi) < U\* (cap 1,024 → unmatched); solve s by
bisection on the monotone curve. Validation on independent draws: gap = Û_A(s)/Û\* − 1 with a delta-method
95% CI including target uncertainty; accept if |gap| < 2% and CI covers 0, else double N (max 3 times) then
mark unmatched. Unmatched or σ_c ≡ 0 templates leave R\*. Cross-check: full-environment no-quote runs
(bounded: 3 templates × M0–M2 × 2,000 episodes) agree with Û within MC error. Report s, U\*, gaps, and time
profiles of variance; matching equalizes expected terminal uncertainty only (martingale arms).

## 9. Experiment (`evaluate.py`)
9.1 Runner: templates/order/seeds per Section 5; 20 replicates; policies hjb (frozen, time_mapping
"episode"), fixed, skew (0.0005), none, called exactly as `evaluate_hjb`. Multiprocessing over templates;
deterministic seeds; checkpoint per (template, arm) to resume; runtime logged.
Per episode: ledger audit; attribution reconciles; policies within (template, replicate, arm, S) share the
signal-path, book and settlement hashes; settlement genuine; none-policy PnL = 0.
9.2 Tier 1 (required): L0 anchor (all 35 templates, impact base, fill base, 4 policies, 20 replicates; exact
per-episode equality with saved `hjb_benchmark/episode_results.csv`, tolerance 1e−9); M 2×2 × S{0,1}; L chain
L0b–L1 (R\*); M1-native × S{0,1}; FGH-L, FGH-LJ; secondary native M0–M2 × S{0,1}; p0_emp M 2×2 × S{0,1};
per-arm diagnostics (sign/size moments, current-state information variance, 5-min markouts, simulated Kyle λ).
9.3 Tier 2 (if Tier 1 validated): jump-intensity dose-response ×{0.5, 2}; latency ℓ̄ ∈ {0 (grid, with
0.25 vs 0.0625 s convergence: paired 95% CI of Δ mean PnL within ±$0.01 and Δ fills within ±1%), 30 s};
conservative/optimistic fills; 60-s decision interval (market windows unchanged: one `sample_window`/
`sample_innovation` draw per original 300-s window and spread draws only at legacy points; with
`decision_interval=60`, `step(action)` advances to the next decision node or window end, whichever comes first,
and returns there; the observation's external book is the most recently sampled book (no extra spread draw);
post-only validation uses that book; verified: identical exogenous executions and identical M price paths at
common times between 300-s and 60-s runs); constant-α flow; S1-thinned (each exogenous execution within 60 s after a jump — union of
overlapping windows — removed independently with probability π_thin = min(1, E[sniper contracts per jump]/
E[exogenous contracts in post-jump windows]) estimated per class **and policy** from a pilot S1 run; reported
as a scenario, not a bound); TTR-clock arrival sensitivity; early revelation (M only, absorbing p = Θ at t_r,
excluded from matching, own summary).
9.4 Estimands (`bootstrap.py`): common resampling indices (templates, then replicates within template)
shared by all arms/policies/S levels in a contrast; each statistic recomputed per arm on the same resample
and differenced. Primary risk contrasts = differences of marginal statistics (mean, VaR95, ES95, mean |q| before
settlement, drawdown, fills, snipe loss); ES of paired PnL differences reported as secondary. 1,000 draws.
Contrasts: flow M1−M0; jump M2−M0, M3−M1; interaction M3−M2−M1+M0; sniping–jump (M2,S1)−(M2,S0)−(M0,S1)+(M0,S0)
and (M3,·)−(M1,·) analogue; HJB minus fixed/skew per arm; legacy chain successive differences and L1J−L1;
contextual M1-native−L1, M3−L1J; FGH-L−L1, FGH-LJ−L1J; secondary and p0_emp analogues.
9.5 Outputs `data_pipeline/output/paper3/experiment/`: `episode_results.csv`, `risk_summary.csv`,
`paired_contrasts.csv`, `matching_report.csv`, `roster.json`, `fidelity_report.json`, representative ledgers,
`manifest.json` (seeds, configs, hashes, preservation manifest, runtimes), figures.

## 10. Validation
Bounded automated tests (suite total ≤ 5 min) and full offline validation (`python -m simulator.information.validate`):
- V1 kernels: closed forms (GM Δp_B = αp(1−p)/q, Δp_S = −αp(1−p)/(1−q), zero mean, variance
  α²p²(1−p)²/(q(1−q))); FG-half = ±βc (1e−14); sign-tilt LLR ±a; q(p,a) martingale; Θ-free samplers' laws;
  Gaussian LLR = X; finite L at extreme LLR.
- V2 reference posterior (independent code from logged signals) equals path L at every node (1e−12), all M
  arms, bounded sample in tests, ≥ 500 episodes per arm offline.
- V3 restricted terminal permutation invariance (fixed p-independent experiments).
- V4 martingale moments E[φ(G−)Δp] = 0, φ ∈ {1, p, p², TTR dummies, size bin, phase, last flow sign, time since
  last jump} × channel; collinear columns dropped by QR (rank tolerance 1e−10, reported); joint χ², α = 0.01,
  Holm across M arms; offline sample ≥ 20,000 price-path episodes per M arm per class (price paths are
  policy-independent; no policy loop needed); L arms reported as diagnostics.
- V5 outcome calibration E[φ(G_t)(Θ − p_t)] = 0 at t ∈ {900, 1800, 2700, close} s and reliability by decile.
- V6 budget identity (labelled one weighted-drift aggregate).
- V7 golden equivalence and L0 anchor.
- V8 hashes: (a) base-input hash (skeleton inputs: times, sizes, uniforms, W increments, jump times/
  magnitudes, proposal times, donor) equal across arms, policies and S levels; (b) public signal-path hash
  (realized signs, jump signs and times, L at all nodes), (c) book hash, (d) settlement hash — each equal
  across policies and S levels within an arm.
- V9 matching gaps/CIs and full-env cross-check.
- V10 fidelity (report): simulated Kyle λ (flow-only replay and full M1/M3) vs target; jump counts/sizes vs
  calibration; top-3 share of |ΔL| and excess kurtosis vs train data (recomputed).
- V11 ledger audit and additive attribution for every episode.
- V12 sniping: proposal-hazard check (empirical snipe rate while crossed ≈ 1/ℓ̄ with CI); Tier 2 grid convergence.
- V13 training isolation; C-1, C-2, C-4.

## 11. Phases
P1 kernels, taker fees, reference + V1–V3 tests. P2 calibration script + tests + artifacts. P3 skeleton, paths,
environment, sniping, golden/anchor tests (V7, V8). P4 matching + tests (V9) + roster. P5 evaluate, bootstrap,
metrics, validate + bounded statistical tests; smoke run (2 templates × 2 replicates, all Tier 1 arms).
P6 (orchestrator) full Tier 1, offline validation, Tier 2 as time permits, docs.

## 12. Acceptance
All Section 10 checks pass (or failures diagnosed and reported as limitations for user decision); C-1…C-6
hold; full suite passes; Tier 1 complete with per-episode checks; validator passes; documentation states:
reduced-form jumps; provisional sports t_r; S1 uncalibrated execution scenario; Kyle-λ matching convention;
p0 = 0.5 counterfactual; legacy final-hour arrival understatement (~4.5× sports, ~2.4× broad); excluded
templates and why; HJB not optimal under these dynamics.

## Amendments after implementation start

Both were recorded before the calibration was rerun. They correct how the acceptance test is built. They do not tune any result.

- **A-1 (§3.6 bisection tolerance).** Changed from 0.5% to 0.1% of target. At 0.5%, bisection alone could leave an error larger than the MC precision the test requires (SE < 0.25%). The validation CI could then reject a correctly fitted parameter: the first run gave sports +0.46% calibration gap and a +0.65% validation error, CI [+0.33%, +0.97%].
- **A-2 (§3.6 validation CI).** The 95% CI for the validation relative error is now ±1.96·√(SE_cal² + SE_val²)/target, not ±1.96·SE_val/target. The fitted parameter carries MC error from the calibration sample, so even with exact bisection the validation-only CI would reject a correct fit about 17% of the time. The |error| < 2% rule is unchanged.
- **A-3 (§§4.5, 5 settlement uniform for the L family).** U_L is the single draw from the legacy settlement stream (child 4), not stream 10. That draw feeds no generator in any arm, so it is a valid Θ-free settlement uniform. With it, L0 → L0b changes only the flow signs, and the whole L chain shares one settlement uniform. Stream 10 is still spawned, so the spawn order is unchanged. The L0 rule (compare with displayed p) and the L arms' rule (compare with unclipped p_close) differ only when p_close lies outside the display clip.
