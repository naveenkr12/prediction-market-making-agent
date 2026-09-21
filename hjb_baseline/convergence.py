"""
Grid-refinement convergence check for the HJB solver.

Solves at three successive grid resolutions and tracks V(0, 0.5, 0) -- the
model's time-0 value at the mid price and flat inventory, which is also
mean optimal-strategy PnL under the risk-neutral solution concept used
here -- and the fixed-point iteration counts, as a basic sanity check that
the finite-difference / fixed-point scheme is converging as the grid is
refined, not just producing numbers.

Not a formal order-of-convergence estimate (the grid is refined in three
dimensions at once -- t, p, pi -- and the two coarser runs were originally
run as ad hoc "smoke tests" rather than a controlled dt/dp/dpi -> 0 study),
but the monotone approach toward the paper's published value (12.39) across
three roughly 2x-in-each-dimension refinements is the intended evidence of
convergence.
"""

from __future__ import annotations

import json
import time

import numpy as np

from hjb_solver import build_grid, solve_hjb
from params import DEFAULT

RESOLUTIONS = [
    (40, 41, 61),
    (100, 81, 101),
    (150, 121, 201),
]

PAPER_OPTIMAL_MEAN_PNL = 12.39


def run(out_path: str = "output/data/convergence.json") -> list[dict]:
    rows = []
    for n_t, n_p, n_pi in RESOLUTIONS:
        grid = build_grid(DEFAULT, n_t=n_t, n_p=n_p, n_pi=n_pi)
        t0 = time.time()
        V = solve_hjb(DEFAULT, grid, damping=0.25, verbose=False)
        elapsed = time.time() - t0
        v0 = float(V[0, len(grid.p) // 2, len(grid.q) // 2])
        rows.append(
            {
                "n_t": n_t,
                "n_p": n_p,
                "n_pi": n_pi,
                "V(0,0.5,0)": v0,
                "abs_error_vs_paper": abs(v0 - PAPER_OPTIMAL_MEAN_PNL),
                "solve_time_s": elapsed,
            }
        )
        print(rows[-1])
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    run()
