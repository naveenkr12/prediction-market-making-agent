# Reproduction of Feil & Nendel (2026), "Optimal Market Making in Prediction Markets"

This is the baseline-reproduction deliverable for the capstone "A market-making
agent for prediction markets, benchmarked against the published optimum."
Before building or evaluating any RL quoting agent, the capstone needs a
trustworthy, independently-implemented copy of the theoretical optimum it is
benchmarked against. That is what this folder is: a from-scratch
implementation of Feil & Nendel's model and HJB solution, verified against
their published Figures 1-4 and Table 2.

## What's implemented

| File | Paper section | Contents |
|---|---|---|
| `params.py` | Table 1 | Model parameters (volatility, activity, intensity shape, risk aversion, inventory limit, trade size) |
| `price_process.py` | Sec. 2.1 | Logistic-transform latent belief diffusion; the market's fair-value price process on (0,1) |
| `intensities.py` | Sec. 2.1.3, 4.1 | Order arrival intensities Λ^b, Λ^a and their concrete Section 4.1 specification |
| `hjb_solver.py` | Sec. 4.2 | Finite-difference solver for the reduced (t,p,q) HJB PDE: implicit Euler in time, central differences in p with Neumann BCs, fixed-point iteration over the inventory-coupled bid/ask terms |
| `quotes.py` | Thm. 3.3, Sec. 4.2 | Optimal quote extraction from V via trilinear interpolation + maximization |
| `monte_carlo.py` | Sec. 4.4 | Monte Carlo simulation of the optimal strategy vs. a myopic (inventory-blind) benchmark, reproducing Table 2 |
| `figures.py` | Sec. 4.3-4.4 | Figures 1-5 |
| `convergence.py` | — | Grid-refinement convergence check |

## Key implementation decisions worth knowing about

**Damped fixed-point iteration.** The paper's Theorem D.2 contraction result
is a continuous-time statement (a Banach fixed point in a weighted sup-norm
over the whole horizon). It does *not* guarantee that the naive discrete
per-time-step Picard iteration (replace V by the freshly re-solved V each
pass) is itself non-expansive. In this implementation it isn't: Λ^b/Λ^a grow
without an explicit cap near certain quote boundaries under the concrete
Section 4.1 specification, so the naive iteration diverges to NaN within a
handful of iterations. Fixed with standard under-relaxation
(`V_new = 0.25 * V_solved + 0.75 * V_iter`), which changes the path to the
fixed point but not the fixed point itself. With it, all resolutions tested
converge stably in well under 60 iterations per time step.

**NaN-safe Monte Carlo.** A side that's switched off at an inventory boundary
returns `NaN` quotes with 0 fill intensity; naive `NaN * 0` still propagates
`NaN` in numpy, so quotes are `nan_to_num`'d before the cash update.

**Myopic benchmark = z=0 in the same H^o maximization.** The paper describes
the benchmark only in prose ("maximizes the instantaneous expected
mark-to-market profit"). We derived via Ito's lemma that the drift of
X_t + q_t p_t decomposes exactly into the same Λ(π)(G(π) − z) terms with z
(the inventory/continuation-value term) set to zero — i.e. dropping inventory
risk entirely. This reproduces the paper's Table 2 baseline row closely (see
below) and is the natural reading of the stated definition. Only the optimal
strategy respects the inventory bound Q, matching the paper's footnote 4
("the myopic benchmark is simulated without an inventory bound").

## Results

### Convergence (`convergence.py`, `output/data/convergence.json`)

V(0, 0.5, 0) — the time-0 value at mid-price, flat inventory — approaches the
paper's published mean optimal-strategy PnL (12.39) monotonically as the grid
is refined:

| Grid (n_t, n_p, n_π) | V(0,0.5,0) | |error| vs. paper | solve time |
|---|---|---|---|
| 40, 41, 61 | 11.905 | 0.485 | 2.9 s |
| 100, 81, 101 | 12.133 | 0.257 | 18.4 s |
| 150, 121, 201 | 12.217 | 0.173 | 76.8 s |

Not a formal order-of-convergence study (three dimensions refined at once,
not a controlled single-variable h→0 sweep), but the monotone approach is the
evidence that the discretization is behaving correctly rather than producing
arbitrary numbers.

### Table 2 reproduction (`output/data/table2_comparison.{json,csv}`)

Monte Carlo, 10,000 paths × 1,000 steps, using the production-resolution
V (150, 121, 201):

| Metric | Optimal (reproduced) | Optimal (paper) | Baseline (reproduced) | Baseline (paper) |
|---|---|---|---|---|
| PnL mean | 12.35 | 12.39 | 13.00 | 12.47 |
| PnL std | 10.36 | 10.34 | 35.78 | 28.11 |
| mean \|q_T\| | 15.13 | 15.23 | 59.29 | 49.37 |
| VaR 5% | 4.40 | 4.20 | 44.35 | 32.41 |
| ES 5% | 9.32 | 9.68 | 62.66 | 40.50 |

**The optimal strategy — the actual output of the HJB solve, and the harder
number to get right — matches the paper closely on every statistic.** This is
the part that matters most for the capstone: it's the benchmark the RL agent
will be compared against.

**The myopic baseline runs systematically higher-risk than published**
(std/VaR/ES roughly 25-55% above the paper, consistent across random seeds,
MC path counts, time-step counts, and grid resolutions — this was
investigated, not just observed once). The direction and magnitude of the
gap between optimal and baseline is qualitatively right (the optimal
strategy is dramatically safer, as the paper claims), and the *optimal* row's
excellent match indicates the core HJB solve and quote extraction are
correct. The most likely explanation is a difference in the myopic
benchmark's exact operational definition — the paper doesn't spell it out
symbol-by-symbol, only in prose, and there may be a detail (e.g. an implicit
quote-magnitude cap, or a slightly different z=0 convention) not
recoverable from the text alone. Documented here transparently rather than
silently tuned to match, since the point of a baseline reproduction is to be
checkable.

### Figures (`output/figures/`)

- **Figure 1** (spread vs. time, flat inventory): spread narrows in discrete
  steps from ~0.06 to ~0.04 as t → T, for prices ranging 0.10-0.50. Matches
  the paper's qualitative claim that uncertainty resolution near expiry
  tightens quotes. (Visible step structure is the finite π-grid resolution,
  not noise.)
- **Figure 2** (skew vs. price, t=0): positive skew (quotes shifted up) for
  short inventory, negative skew for long inventory, magnitude largest near
  the boundaries p→0,1. Matches the paper's stated sign convention.
- **Figure 3** (skew heatmap, p=0.5, (t,q)): skew grows with both |q| and
  proximity to T — makes sense as the risk of ending with unhedged inventory
  becomes more urgent near settlement.
- **Figure 4** (risk-aversion sensitivity, reduced-resolution grid since it's
  a 4-point parameter sweep, each point its own HJB solve): as γ increases
  1e-3 → 6.4e-2, the optimal spread widens (0.056 → 0.061), PnL volatility
  falls (10.7 → 8.0), and mean terminal inventory shrinks (15.3 → 12.2) — the
  expected monotone risk-aversion trade-off. Not literally the paper's own
  Figure 4 parameterization (not fully recoverable from the extracted text),
  but reproduces the qualitative claim it supports.
- **Figure 5** (PnL / terminal-inventory histograms): the optimal strategy's
  PnL distribution is visibly tighter and better centered than the myopic
  baseline's, and its terminal-inventory distribution is concentrated near
  zero while the baseline's has a long tail — the same story Table 2 tells,
  as a picture.

## How to reproduce

```
pip install numpy scipy matplotlib

python3 convergence.py                    # ~100s, writes output/data/convergence.json

python3 -c "
from hjb_solver import build_grid, solve_hjb
from params import DEFAULT
import numpy as np
grid = build_grid(DEFAULT, n_t=150, n_p=121, n_pi=201)
V = solve_hjb(DEFAULT, grid, damping=0.25, verbose=True)   # ~75s
np.save('output/data/V_production.npy', V)
np.savez('output/data/grid_production.npz', t=grid.t, p=grid.p, q=grid.q, pi=grid.pi)
"

python3 -c "
from figures import *
from hjb_solver import Grid
import numpy as np
d = np.load('output/data/grid_production.npz')
grid = Grid(t=d['t'], p=d['p'], q=d['q'], pi=d['pi'])
V = np.load('output/data/V_production.npy')
figure1_spread_vs_time(DEFAULT, grid, V, 'output/figures/figure1_spread_vs_time.png')
figure2_skew_vs_price(DEFAULT, grid, V, 'output/figures/figure2_skew_vs_price.png')
figure3_skew_heatmap(DEFAULT, grid, V, 'output/figures/figure3_skew_heatmap.png')
"                                           # ~8s total

python3 -c "
from monte_carlo import run_monte_carlo
from hjb_solver import Grid
from params import DEFAULT
import numpy as np, json
d = np.load('output/data/grid_production.npz')
grid = Grid(t=d['t'], p=d['p'], q=d['q'], pi=d['pi'])
V = np.load('output/data/V_production.npy')
res = run_monte_carlo(DEFAULT, grid, V, n_paths=10000, n_steps=1000)
print(res['baseline']); print(res['optimal'])
"                                           # ~250s (10k paths x 1k steps)
```

## Where this fits in the capstone

This is the **published-optimum benchmark**, not the RL agent. Next steps per
the project roadmap:

1. Use this same environment (price process + intensities + inventory
   mechanics) as the simulator the RL agent trains and is evaluated in, so
   the comparison is apples-to-apples.
2. Calibrate `params.py`'s intensity/activity parameters to the Kalshi data
   collected in `../kalshi_pipeline/` (fit k(t), A(t,p) etc. to observed
   trade arrival patterns) as a *robustness* variant — a second benchmark run
   under data-calibrated rather than paper-default parameters — while
   keeping this default-parameter reproduction as the primary, checkable
   baseline.
3. Train the RL agent against this benchmark's PnL/VaR/ES numbers directly.

## Known limitations

- Baseline (myopic) Monte Carlo risk statistics run 25-55% higher than the
  paper's published numbers; not yet root-caused (see Table 2 section above).
- Figure 4 uses a reduced grid resolution (50,51,71) per parameter point,
  not the production resolution (150,121,201) used for Figures 1-3, since
  it requires re-solving the full PDE at each of several γ values.
- Convergence check is a 3-point, jointly-refined-in-all-dimensions sweep,
  not a formal single-variable order-of-convergence estimate.
