"""
Finite-difference solver for the reduced Hamilton-Jacobi-Bellman equation,
Section 4.2 of Feil & Nendel.

    0 = -d_t V - (1/2) varsigma(t,p)^2 d_pp V + gamma q^2 varsigma(t,p)^2
          - 1{q<Q} H^b(t,p; z_b) - 1{q>-Q} H^a(t,p; z_a)

    V(T,p,q) = Phi(p,q) = -gamma_T q^2 p(1-p)          (terminal condition)

    z_b(t,p,q) = (V(t,p,q) - V(t,p,q+Delta)) / Delta
    z_a(t,p,q) = (V(t,p,q) - V(t,p,q-Delta)) / Delta

    H^o(t,p;z) = Delta * sup_{pi in [0,1]} Lambda^o(t,p,pi) (G^o(p,pi) - z)

Discretization (Section 4.2): implicit Euler backward in time, second-order
central differences in p with homogeneous Neumann boundary conditions,
and -- because H^b/H^a couple neighbouring inventory levels through z_b/z_a
-- a fixed-point iteration at each time step. The system matrix

    A_n := I - (h/2) diag(varsigma_n^2) D2

does not depend on q or on the fixed-point iteration index, only on n, so
its LU factorization is computed once per time step and reused across
every inventory level and every fixed-point iterate (exactly as the paper
does it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from params import ModelParams
import intensities as I
import price_process as PP


@dataclass
class Grid:
    t: np.ndarray   # (Nt+1,)
    p: np.ndarray   # (Np,)
    q: np.ndarray   # (Nq,)
    pi: np.ndarray  # (Npi,) candidate quotes for the inner maximization


def build_grid(par: ModelParams, n_t: int, n_p: int, n_pi: int,
                p_min: float = 1e-3, p_max: float | None = None) -> Grid:
    if p_max is None:
        p_max = 1 - 1e-3
    t = np.linspace(0.0, par.T, n_t + 1)
    p = np.linspace(p_min, p_max, n_p)
    n_q = int(round(2 * par.Q / par.delta)) + 1
    q = -par.Q + par.delta * np.arange(n_q)
    pi = np.linspace(1e-6, 1 - 1e-6, n_pi)
    return Grid(t=t, p=p, q=q, pi=pi)


def _build_D2(n_p: int, dp: float) -> sp.csc_matrix:
    """Second-derivative operator on the p-grid, homogeneous Neumann BC
    (ghost-node reflection) at both ends."""
    main = -2.0 * np.ones(n_p)
    off = np.ones(n_p - 1)
    D2 = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    # Neumann: mirror the first interior neighbour so the boundary row reads
    # 2*(V_1 - V_0)/dp^2  and  2*(V_{N-2} - V_{N-1})/dp^2
    D2[0, 0] = -2.0
    D2[0, 1] = 2.0
    D2[-1, -1] = -2.0
    D2[-1, -2] = 2.0
    return (D2 / dp ** 2).tocsc()


def terminal_condition(par: ModelParams, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Phi(p,q) = -gamma_T * q^2 * p(1-p), shape (Np, Nq)."""
    return -par.gamma_T * (q[None, :] ** 2) * (p[:, None] * (1 - p[:, None]))


def _max_H(t_n: float, p: np.ndarray, pi_grid: np.ndarray, z: np.ndarray,
           par: ModelParams, side: str) -> np.ndarray:
    """
    Vectorized H^o(t_n, p_j; z[j,q]) = Delta * max_l Lambda(p_j,pi_l) * (G(p_j,pi_l) - z[j,q])
    over the candidate quote grid pi_l, for every (j,q) at once.
    z has shape (Np, Nq_active); returns shape (Np, Nq_active).
    """
    if side == "b":
        lam = I.lambda_bid(t_n, p[:, None], pi_grid[None, :], par)   # (Np, Npi)
        g = I.G_bid(p[:, None], pi_grid[None, :])                    # (Np, Npi)
    else:
        lam = I.lambda_ask(t_n, p[:, None], pi_grid[None, :], par)
        g = I.G_ask(p[:, None], pi_grid[None, :])

    # Psi[j,l,q] = lam[j,l] * (g[j,l] - z[j,q])
    psi = lam[:, :, None] * g[:, :, None] - lam[:, :, None] * z[:, None, :]
    return par.delta * psi.max(axis=1)


def solve_hjb(
    par: ModelParams,
    grid: Grid,
    max_fp_iter: int = 60,
    fp_tol: float = 1e-6,
    damping: float = 0.25,
    verbose: bool = False,
) -> np.ndarray:
    """
    Returns V of shape (Nt+1, Np, Nq): V[n, j, iq] approximates V(t_n, p_j, q_iq).

    Implementation note (not in the paper's text, standard numerical-PDE
    practice): the raw Picard update ("replace V_n^(k) by the freshly
    solved V_n^(k+1)") is only a *local* contraction for this problem --
    empirically it diverges here because Lambda^b(t,p,pi) grows without
    an explicit cap as pi -> 1 for small p (and symmetrically for
    Lambda^a as pi -> 0), so a very negative/positive z_b/z_a can push
    the argmax toward that regime and blow up H^b/H^a within a couple of
    iterations. Under-relaxing the update (V_iter <- damping*V_new +
    (1-damping)*V_iter) does not change the fixed point being solved for
    -- only the path taken to reach it -- and restores stable
    convergence. This mirrors what any implementation of this scheme has
    to do in practice; the paper's Theorem D.2 contraction argument is a
    continuous-time statement (a Banach fixed point in a weighted sup
    norm over the whole [0,T] interval), not a claim that the naive
    per-time-step Picard iteration below is itself non-expansive.
    """
    n_t = len(grid.t) - 1
    n_p = len(grid.p)
    n_q = len(grid.q)
    dp = grid.p[1] - grid.p[0]
    D2 = _build_D2(n_p, dp)
    I_sp = sp.identity(n_p, format="csc")

    V = np.empty((n_t + 1, n_p, n_q))
    V[n_t] = terminal_condition(par, grid.p, grid.q)

    t0 = time.time()
    for n in range(n_t - 1, -1, -1):
        t_n = grid.t[n]
        h = grid.t[n + 1] - grid.t[n]

        varsigma_n = grid.p * (1 - grid.p) * PP.sigma_bar(t_n, grid.p, par)  # (Np,)
        varsigma2 = varsigma_n ** 2

        A_n = (I_sp - (h / 2) * sp.diags(varsigma2) @ D2).tocsc()
        lu = spla.splu(A_n)

        V_next = V[n + 1]
        V_iter = V_next.copy()  # fixed-point initial guess V_n^(0) = V_{n+1}

        for k in range(max_fp_iter):
            H_b = np.zeros((n_p, n_q))
            H_a = np.zeros((n_p, n_q))

            # bid term active for q < Q, i.e. inventory indices 0..n_q-2
            z_b = (V_iter[:, :-1] - V_iter[:, 1:]) / par.delta
            H_b[:, :-1] = _max_H(t_n, grid.p, grid.pi, z_b, par, side="b")

            # ask term active for q > -Q, i.e. inventory indices 1..n_q-1
            z_a = (V_iter[:, 1:] - V_iter[:, :-1]) / par.delta
            H_a[:, 1:] = _max_H(t_n, grid.p, grid.pi, z_a, par, side="a")

            rhs = V_next - h * (par.gamma * (grid.q[None, :] ** 2) * varsigma2[:, None] - H_b - H_a)
            V_solved = lu.solve(rhs)  # solves A_n V_new[:,q] = rhs[:,q] for every q at once

            V_new = damping * V_solved + (1 - damping) * V_iter  # under-relaxation, see docstring
            diff = np.max(np.abs(V_new - V_iter))
            V_iter = V_new
            if not np.isfinite(diff):
                raise FloatingPointError(
                    f"fixed-point iteration diverged at step n={n}, iter={k} -- try a smaller damping."
                )
            if diff < fp_tol:
                break

        V[n] = V_iter
        if verbose and (n % max(1, n_t // 10) == 0):
            print(f"  step {n}/{n_t}  fp_iters={k+1}  diff={diff:.2e}  elapsed={time.time()-t0:.1f}s")

    return V
