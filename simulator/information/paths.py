"""Per-(episode, arm) precomputed latent / price / book paths (spec Sections 6.1, 6.2, 7).

Fills and snipes never move prices and every kernel is policy-independent, so
the complete latent path L at every skeleton node, the realized execution and
jump signs, the external books (SpreadSampler called online, in legacy order,
on a fresh copy of the post-init spread stream) and the settlement are fixed
before the policy loop. `environment.py` walks the result.

Arms (Section 7):
- golden: L0-equivalent branch; legacy nodes only (executions, window ends),
  empirical window innovation bridged on the legacy bridge stream, bank signs,
  `ImpactPriceProcess.advance` calls in legacy order, legacy settlement stream.
- L0b: golden with bank signs replaced by S_u (and recomputed signed flow).
- L0c: sigma_c Gaussian legacy-law background + legacy-law jumps + FL flow.
- L1J / L1: Theta-free martingale background (+ tilted jumps), FL flow.
- FGH-L / FGH-LJ: Theta-free background, Theta-free FG-half flow (+ tilted jumps).
- M0 / M1 / M2 / M3 / M1-native: Theta-known background with scale s; F0 or FG
  flow; with or without sign-tilted jumps; settle Theta = 1{U_M < p0}.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np
from scipy.special import expit

from ..execution import BookSnapshot
from ..fees import decimal
from ..price_impact import ImpactPriceProcess, displayed_probability
from . import kernels as K
from .skeleton import (DECISION, EXECUTION, JUMP, PROPOSAL_ASK, PROPOSAL_BID, WINDOW_END, EpisodeSkeleton)

BACKGROUNDS = ("legacy_bridge", "m", "theta_free", "legacy_gaussian")
FLOWS = ("bank_fl", "su_fl", "f0", "fg", "fg_constant", "fg_half_theta_free")
JUMPS = ("none", "m", "theta_free", "legacy")
SETTLEMENTS = ("legacy", "theta", "u_l")
# Market-row flow law labels (also used by the independent ledger auditor).
FLOW_LAW = {"bank_fl": "fl", "su_fl": "fl", "f0": "uninformed", "fg": "fg", "fg_constant": "constant",
            "fg_half_theta_free": "fg_half"}


@dataclass(frozen=True)
class Arm:
    name: str
    family: str          # "legacy" (golden), "M", "L"
    background: str
    flow: str
    jumps: str
    settlement: str
    s: float = 1.0
    p0: float = 0.5

    def __post_init__(self):
        if (self.background not in BACKGROUNDS or self.flow not in FLOWS or self.jumps not in JUMPS or
                self.settlement not in SETTLEMENTS or self.family not in {"legacy", "M", "L"}):
            raise ValueError("Unknown arm component")
        if not np.isfinite(self.s) or self.s < 0 or not 0 < self.p0 < 1:
            raise ValueError("Background scale must be >= 0 and p0 in (0,1)")
        if self.family == "M" and (self.background != "m" or self.flow not in {"f0", "fg", "fg_constant"} or
                                   self.jumps not in {"none", "m"} or self.settlement != "theta"):
            raise ValueError("M arms use Theta-known background, F0/FG flow, M jumps and settle Theta")
        if self.family == "L" and (self.background not in {"legacy_bridge", "legacy_gaussian", "theta_free"} or
                                   self.flow not in {"su_fl", "fg_half_theta_free"} or
                                   self.jumps not in {"none", "legacy", "theta_free"} or self.settlement != "u_l"):
            raise ValueError("Invalid L-family arm")
        if self.family == "legacy" and (self.background, self.flow, self.jumps, self.settlement) != (
                "legacy_bridge", "bank_fl", "none", "legacy"):
            raise ValueError("The golden branch is exactly the legacy construction")
        if self.background == "legacy_bridge" and (self.flow not in {"bank_fl", "su_fl"} or self.jumps != "none"):
            raise ValueError("Legacy-node arms (golden, L0b) have FL flow and no jump nodes")
        if self.flow == "bank_fl" and self.background != "legacy_bridge":
            raise ValueError("Bank signs are used only on the golden branch")

    @property
    def legacy_nodes(self) -> bool:
        """Golden branch and L0b use only legacy nodes (executions, window ends)."""
        return self.background == "legacy_bridge"


def make_arm(name: str, *, s: float | None = None, p0: float = 0.5, constant_alpha: bool = False) -> Arm:
    """Arm constructor for every Section 7 arm. M0-M2 need the (P4-matched) scale s."""
    fg = "fg_constant" if constant_alpha else "fg"
    if constant_alpha and name not in {"M1", "M3", "M1-native"}:
        raise ValueError("Constant-alpha variant applies to FG arms only")
    label = name+("-const" if constant_alpha else "")
    if name in {"M0", "M1", "M2"}:
        if s is None:
            raise ValueError(f"{name} requires its matched background scale s")
        flow, jumps = {"M0": ("f0", "none"), "M1": (fg, "none"), "M2": ("f0", "m")}[name]
        return Arm(label, "M", "m", flow, jumps, "theta", float(s), p0)
    if name == "M3":
        return Arm(label, "M", "m", fg, "m", "theta", 1.0 if s is None else float(s), p0)
    if name == "M1-native":
        return Arm(label, "M", "m", fg, "none", "theta", 1.0, p0)
    fixed = {"golden": ("legacy", "legacy_bridge", "bank_fl", "none", "legacy"),
             "L0b": ("L", "legacy_bridge", "su_fl", "none", "u_l"),
             "L0c": ("L", "legacy_gaussian", "su_fl", "legacy", "u_l"),
             "L1J": ("L", "theta_free", "su_fl", "theta_free", "u_l"),
             "L1": ("L", "theta_free", "su_fl", "none", "u_l"),
             "FGH-L": ("L", "theta_free", "fg_half_theta_free", "none", "u_l"),
             "FGH-LJ": ("L", "theta_free", "fg_half_theta_free", "theta_free", "u_l")}
    if name not in fixed:
        raise KeyError(f"Unknown arm {name}")
    family, background, flow, jumps, settlement = fixed[name]
    return Arm(name, family, background, flow, jumps, settlement, 1.0 if s is None else float(s), p0)


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            a = np.ascontiguousarray(part)
            h.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode())
    return h.hexdigest()


@dataclass
class EpisodePath:
    """Precomputed public path of one (episode, arm); identical for every policy and S level."""
    skeleton: EpisodeSkeleton
    arm: Arm
    sigma2: float                       # background variance rate actually used (log-odds^2 / s)
    sigma_c: float | None
    beta: float
    initial_log_odds: float
    initial_book: BookSnapshot
    theta: int | None                   # hidden (M family only); never observed
    # Node arrays (Python lists for the environment walk).
    kinds: list
    refs: list
    elapsed: list
    offsets: list
    x: list                             # background log-odds increment arriving at the node
    l_bg: list                          # L after the background piece, before the node's own update
    l_after: list                       # L after the node's own update (flow / jump)
    p_bg: list                          # unclipped expit(l_bg) (sniping crossing uses this)
    books: dict                         # node index -> BookSnapshot (executions, window ends)
    window_node_slices: list
    # Executions (episode order).
    exec_sign: np.ndarray
    exec_increment: np.ndarray
    exec_u: np.ndarray
    exec_alpha: np.ndarray
    exec_one_minus_alpha: np.ndarray
    exec_informed: np.ndarray           # hidden diagnostic (M FG only); never in observations or ledgers
    window_signed_flow: list
    window_background: list
    # Jumps (applied only if the arm has jumps).
    jump_applied: bool
    jump_sign: np.ndarray
    jump_step: np.ndarray
    settlement: dict
    saturated_nodes: int
    flow_parameters: dict = field(default_factory=dict)
    spread_draws: int = 0               # SpreadSampler.sample calls (1 + executions + windows in every arm)
    spread_state_final: dict | None = None
    # Per-jump application mask (len = skeleton jumps). None -> every jump applied iff jump_applied.
    # A thinned (0.5x) path keeps all jump nodes in the node set; dropped jumps are inert nodes.
    jump_active: np.ndarray | None = None
    jump_keep_probability: float = 1.0  # thinning keep probability (intensity factor relative to the skeleton)

    def __post_init__(self):
        n = len(self.skeleton.jump_elapsed)
        if self.jump_active is None:
            self.jump_active = np.full(n, bool(self.jump_applied))
        self.jump_active = np.asarray(self.jump_active, dtype=bool)
        if len(self.jump_active) != n or (not self.jump_applied and self.jump_active.any()):
            raise ValueError("Jump mask must cover every skeleton jump and be empty when jumps are off")

    @property
    def active_jump_count(self) -> int:
        return int(np.count_nonzero(self.jump_active))

    @property
    def jumps_thinned(self) -> bool:
        return bool(self.jump_applied and not self.jump_active.all())

    @property
    def node_count(self) -> int:
        return len(self.kinds)

    # ------------------------------------------------------------------ V2
    def signals(self) -> tuple[list[dict], list[float]]:
        """Logged public signals (M family) and the path L after each one, in node order."""
        if self.arm.family != "M":
            raise ValueError("Bayesian signal logs are defined for M arms")
        sk, law = self.skeleton, FLOW_LAW[self.arm.flow]
        signals, values = [], []
        for i, kind in enumerate(self.kinds):
            signals.append({"kind": "background", "x": self.x[i], "variance": self.sigma2*float(sk.node_h[i])})
            values.append(self.l_bg[i])
            if kind == EXECUTION:
                j = self.refs[i]
                signal = {"kind": "flow", "law": law, "sign": int(self.exec_sign[j]),
                          "size": float(sk.exec_quantity[j]), "u": float(self.exec_u[j])}
                if law == "fg":
                    signal["eta"] = self.flow_parameters["eta"]
                elif law == "constant":
                    signal["alpha"] = self.flow_parameters["alpha0"]
                signals.append(signal)
                values.append(self.l_after[i])
            elif kind == JUMP and self.jump_active[self.refs[i]]:
                j = self.refs[i]
                signals.append({"kind": "jump", "sign": int(self.jump_sign[j]), "magnitude": float(sk.jump_magnitude[j])})
                values.append(self.l_after[i])
        return signals, values

    # ------------------------------------------------------------------ V8
    def public_nodes(self) -> list[int]:
        """Nodes every walk records regardless of policy/S/decision interval: executions, applied jumps, ends."""
        active = self.jump_active
        return [i for i, k in enumerate(self.kinds)
                if k in (EXECUTION, WINDOW_END) or (k == JUMP and active[self.refs[i]])]

    def signal_path_hash(self) -> str:
        """(b) realized signs, jump signs and times, L at all nodes."""
        sk = self.skeleton
        jumps = (self.jump_sign, sk.jump_elapsed) if self.jump_applied else (np.zeros(0, np.int8), np.zeros(0))
        if self.jumps_thinned:
            jumps = (self.jump_sign[self.jump_active], sk.jump_elapsed[self.jump_active])
        return _sha(self.arm.name, self.exec_sign, *jumps, np.asarray(self.l_bg), np.asarray(self.l_after))

    def book_hash(self) -> str:
        """(c) every sampled external book (reset, executions, window ends), in order."""
        books = [(self.initial_book.bid, self.initial_book.ask)]
        books += [(self.books[i].bid, self.books[i].ask) for i in sorted(self.books)]
        return _sha(books)

    def settlement_hash(self) -> str:
        """(d) settlement rule, uniform, closing p and outcome."""
        return _sha(self.settlement)

    def walk_trace_hashes(self) -> dict:
        """Hashes of the public-node trace an environment walk must reproduce (see environment.py)."""
        trace = [(i, self.l_bg[i], self.l_after[i]) for i in self.public_nodes()]
        books = [(self.initial_book.bid, self.initial_book.ask)]+[
            (self.books[i].bid, self.books[i].ask) for i in self.public_nodes() if i in self.books]
        return {"signal": _sha(trace), "book": _sha(books), "settlement": self.settlement_hash()}


def _settle(arm: Arm, sk: EpisodeSkeleton, l_close: float, theta: int | None) -> dict:
    p_unclipped = float(expit(l_close))
    p_display = float(displayed_probability(l_close))
    if arm.settlement == "legacy":
        # TradingEnvironment: int(settlement_rng.random() < self.price.probability) (displayed p).
        uniform, outcome = sk.u_legacy_settlement, int(sk.u_legacy_settlement < p_display)
    elif arm.settlement == "theta":
        uniform, outcome = sk.u_m, int(theta)
    else:
        uniform, outcome = sk.u_l, K.settle_l(sk.u_l, p_unclipped)
    return {"rule": arm.settlement, "uniform": uniform, "close_log_odds": l_close, "p_close": p_unclipped,
            "p_close_display": p_display, "outcome": outcome}


def _uninformed_rates(sk: EpisodeSkeleton, calibration) -> np.ndarray:
    if sk.n_executions == 0:
        return np.zeros(0)
    return np.asarray(calibration.uninformed_rate(sk.ticker, sk.exec_quantity, sk.exec_phase), dtype=float)


def _book(sk, value, rng, tick):
    return sk.spreads.sample(value, rng, tick)


def build_path(skeleton: EpisodeSkeleton, arm: Arm, *, calibration=None, jump_keep=None,
               jump_keep_probability: float | None = None) -> EpisodePath:
    """Precompute the full public path of `arm` on `skeleton` (calibration may be None for golden only).

    `jump_keep` (optional boolean mask over the skeleton's jumps, with its keep probability) thins the
    jumps of a jump arm: dropped jumps stay in the node set as inert nodes (no update, no ledger row), so
    W, proposals, executions and every other input are identical to the unthinned path.
    """
    if arm.legacy_nodes:
        if jump_keep is not None:
            raise ValueError("Legacy-node arms have no jump nodes")
        return _build_legacy_node_path(skeleton, arm, calibration)
    if calibration is None:
        raise ValueError("Non-golden arms require the information calibration")
    if skeleton.jump_extra is not None and arm.family != "M":
        raise ValueError("Superposed (Brownian-bridge refined) skeletons support M-family arms only")
    if jump_keep is not None:
        jump_keep = np.asarray(jump_keep, dtype=bool)
        if arm.jumps == "none" or len(jump_keep) != len(skeleton.jump_elapsed) or jump_keep_probability is None:
            raise ValueError("Thinning needs a jump arm, one flag per skeleton jump and the keep probability")
    return _build_skeleton_node_path(skeleton, arm, calibration, jump_keep, jump_keep_probability)


def _build_legacy_node_path(sk: EpisodeSkeleton, arm: Arm, calibration) -> EpisodePath:
    """Golden branch (and L0b): legacy nodes, pieces, arithmetic and spread order."""
    cfg = sk.config
    if arm.family == "legacy" and arm.p0 != cfg.initial_probability:
        raise ValueError("Golden branch p0 must equal the legacy initial probability")
    tick = decimal(cfg.price_tick)
    spread_rng = sk.spread_rng()
    price = ImpactPriceProcess(arm.p0, sk.dynamics.impact)
    initial_log_odds = price.log_odds
    initial_book = _book(sk, price.log_odds, spread_rng, tick)
    if arm.flow == "bank_fl":
        signs = sk.exec_bank_sign
        u = np.full(sk.n_executions, np.nan)
    else:   # L0b: S_u signs
        if calibration is None:
            raise ValueError("L0b requires uninformed sign rates")
        u = _uninformed_rates(sk, calibration)
        signs = K.uninformed_sign(sk.v_u, u)
    kinds, refs, elapsed, offsets, x, l_bg, l_after, books, slices = [], [], [], [], [], [], [], {}, []
    increments = np.zeros(sk.n_executions)
    window_flow, window_bg = [], []
    saturated = 0
    for w in sk.windows:
        first = len(kinds)
        offs = w.window["event_times_seconds"]
        for local, j in enumerate(range(w.exec_slice.start, w.exec_slice.stop)):
            price.advance(0, w.pieces[local])
            kinds.append(EXECUTION)
            refs.append(j)
            elapsed.append(float(sk.exec_elapsed[j]))
            offsets.append(offs[local])
            x.append(float(w.pieces[local]))
            l_bg.append(price.log_odds)
            books[len(kinds)-1] = _book(sk, price.log_odds, spread_rng, tick)
            sign, quantity = signs[j], sk.exec_quantity[j]
            update = price.advance(float(sign*quantity))
            increments[j] = update["impact_logodds"]
            saturated += int(update["float_probability_saturated"])
            l_after.append(price.log_odds)
        price.advance(0, w.pieces[-1])
        kinds.append(WINDOW_END)
        refs.append(w.index)
        elapsed.append(w.end)
        offsets.append(w.duration)
        x.append(float(w.pieces[-1]))
        l_bg.append(price.log_odds)
        l_after.append(price.log_odds)
        books[len(kinds)-1] = _book(sk, price.log_odds, spread_rng, tick)
        slices.append(slice(first, len(kinds)))
        q = w.window["quantities"]
        window_flow.append(float(w.window["signed_flow_contracts"]) if arm.flow == "bank_fl"
                           else float(np.dot(q, signs[w.exec_slice])))
        window_bg.append(float(np.sum(np.asarray(x[first:len(kinds)]))))
    settlement = _settle(arm, sk, price.log_odds, None)
    nan = np.full(sk.n_executions, np.nan)
    return EpisodePath(
        skeleton=sk, arm=arm, sigma2=sk.dynamics.sigma**2, sigma_c=None, beta=sk.dynamics.impact.beta,
        initial_log_odds=initial_log_odds, initial_book=initial_book, theta=None, kinds=kinds, refs=refs,
        elapsed=elapsed, offsets=offsets, x=x, l_bg=l_bg, l_after=l_after, p_bg=[float(expit(v)) for v in l_bg],
        books=books, window_node_slices=slices, exec_sign=np.asarray(signs), exec_increment=increments,
        exec_u=u, exec_alpha=nan, exec_one_minus_alpha=nan, exec_informed=np.zeros(sk.n_executions, bool),
        window_signed_flow=window_flow, window_background=window_bg, jump_applied=False,
        jump_sign=np.zeros(0, np.int8), jump_step=np.zeros(0), settlement=settlement, saturated_nodes=saturated,
        flow_parameters={} if arm.flow == "bank_fl" else {"beta": sk.dynamics.impact.beta},
        spread_draws=1+len(books), spread_state_final=spread_rng.bit_generator.state)


def _build_skeleton_node_path(sk: EpisodeSkeleton, arm: Arm, calibration, jump_keep=None,
                              jump_keep_probability=None) -> EpisodePath:
    cfg = sk.config
    tick = decimal(cfg.price_tick)
    cls = sk.information_class
    beta = float(sk.dynamics.impact.beta)
    if cfg.impact_scenario == "base" and beta != calibration.beta(cls):
        raise ValueError("Legacy base impact differs from the information calibration's base beta")
    sigma_c = calibration.sigma_c(sk.noise_ticker)
    sigma2 = (arm.s if arm.background == "m" else 1.0)*sigma_c**2
    h, dw = sk.node_h, sk.node_dw
    n_nodes, n_exec = len(sk.node_kind), sk.n_executions
    theta = K.settle_m(sk.u_m, arm.p0) if arm.family == "M" else None
    initial_log_odds = ImpactPriceProcess(arm.p0, sk.dynamics.impact).log_odds   # float(logit(p0))

    # ---- p-independent pieces, vectorized through the kernels.
    if arm.background == "m":
        x_fixed = K.background_m(np.zeros(n_nodes), theta, sigma2, h, dw)[1]
    elif arm.background == "legacy_gaussian":
        x_fixed = K.background_legacy(np.zeros(n_nodes), sigma2, dw)[1]
    else:   # theta_free: sequential; precompute the p-independent factors with the kernels' operation order
        x_fixed = None
        half = sigma2*h/2
        noise = np.sqrt(sigma2)*dw
    u = _uninformed_rates(sk, calibration)
    s_u = K.uninformed_sign(sk.v_u, u) if n_exec else np.zeros(0, np.int8)
    alpha = one_minus = np.full(n_exec, np.nan)
    informed = np.zeros(n_exec, bool)
    params = {}
    exec_sign = np.zeros(n_exec, np.int8)
    exec_inc = np.zeros(n_exec)
    if arm.flow == "f0":
        exec_sign = s_u
    elif arm.flow in {"fg", "fg_constant"}:
        if arm.flow == "fg":
            params["eta"] = calibration.eta(cls)
            alpha, one_minus = K.fg_alpha_terms(sk.exec_quantity, params["eta"])
        else:
            params["alpha0"] = calibration.constant_alpha(cls)
            alpha, one_minus = K.constant_alpha_terms(params["alpha0"], (n_exec,))
        if n_exec:
            exec_sign, informed = K.flow_sign_m(theta, sk.v_i, sk.v_u, alpha, u)
            exec_inc = K.gm_llr(exec_sign, alpha, one_minus, u)
    elif arm.flow == "su_fl":
        exec_sign = s_u
        params["beta"] = beta
    elif arm.flow == "fg_half_theta_free":
        params["beta"] = beta
        alpha, one_minus = K.fg_half_alpha_terms(sk.exec_quantity, beta)
        u = np.full(n_exec, 0.5)
    n_jump = len(sk.jump_elapsed)
    jump_sign, jump_step = np.zeros(n_jump, np.int8), np.zeros(n_jump)
    active = np.full(n_jump, arm.jumps != "none") if jump_keep is None else jump_keep.copy()
    if arm.jumps == "m" and n_jump:
        jump_sign = K.jump_sign_m(theta, sk.v_j, sk.jump_magnitude)
        jump_step = K.jump_update(np.zeros(n_jump), jump_sign, sk.jump_magnitude)[1]
    elif arm.jumps == "legacy" and n_jump:
        jump_sign = K.jump_sign_legacy(sk.v_j)
        jump_step = K.jump_update(np.zeros(n_jump), jump_sign, sk.jump_magnitude)[1]

    # ---- sequential walk over all nodes (online spread draws in legacy order).
    spread_rng = sk.spread_rng()
    initial_book = _book(sk, initial_log_odds, spread_rng, tick)
    kinds = sk.node_kind.tolist()
    refs = sk.node_ref.tolist()
    x, l_bg, l_after, books = [0.0]*n_nodes, [0.0]*n_nodes, [0.0]*n_nodes, {}
    L = initial_log_odds
    theta_free_bg = arm.background == "theta_free"
    v_aux = sk.node_v_aux
    if not theta_free_bg:
        x = x_fixed.tolist()
    for i in range(n_nodes):
        if theta_free_bg:
            # == kernels.background_theta_free(L, v_aux, sigma2, h, dW) (same operations; tested bitwise)
            theta_tilde = 1 if v_aux[i] < expit(L) else 0
            xi = float(half[i]*(2*theta_tilde-1)+noise[i])
            x[i] = xi
        else:
            xi = x[i]
        L = L+xi
        l_bg[i] = L
        kind = kinds[i]
        if kind == EXECUTION:
            j = refs[i]
            books[i] = _book(sk, L, spread_rng, tick)
            if arm.flow == "fg_half_theta_free":
                sign = int(K.flow_sign_theta_free(L, sk.v_f[j], alpha[j], one_minus[j], u[j]))
                exec_sign[j] = sign
                inc = float(K.fg_half_llr(sign, sk.exec_quantity[j], beta))
                exec_inc[j] = inc
                L = L+inc
            elif arm.flow == "su_fl":
                L, inc = K.fl_update(L, exec_sign[j], sk.exec_quantity[j], beta)
                exec_inc[j] = inc
            elif arm.flow in {"fg", "fg_constant"}:
                L = L+float(exec_inc[j])
            # f0: sign only, no update
        elif kind == JUMP:
            j = refs[i]
            if arm.jumps == "theta_free":
                sign = int(K.jump_sign_theta_free(L, sk.v_j[j], sk.jump_magnitude[j]))
                jump_sign[j] = sign
                jump_step[j] = float(sign*sk.jump_magnitude[j])
            if active[j]:
                L = L+float(jump_step[j])
        elif kind == WINDOW_END:
            books[i] = _book(sk, L, spread_rng, tick)
        l_after[i] = L
    K.check_signs(exec_sign)
    if arm.jumps != "none":
        K.check_signs(jump_sign)
    p_bg = expit(np.asarray(l_bg)).tolist()
    saturated = int(np.count_nonzero(K.saturated(np.asarray(l_after))))
    settlement = _settle(arm, sk, L, theta)
    slices = [w.node_slice for w in sk.windows]
    window_flow, window_bg = [], []
    xa = np.asarray(x)
    for w in sk.windows:
        window_flow.append(float(np.dot(sk.exec_quantity[w.exec_slice], exec_sign[w.exec_slice])))
        window_bg.append(float(xa[w.node_slice].sum()))
    return EpisodePath(
        skeleton=sk, arm=arm, sigma2=float(sigma2), sigma_c=float(sigma_c), beta=beta,
        initial_log_odds=initial_log_odds, initial_book=initial_book, theta=theta, kinds=kinds, refs=refs,
        elapsed=sk.node_elapsed.tolist(), offsets=sk.node_offset.tolist(), x=x, l_bg=l_bg, l_after=l_after,
        p_bg=p_bg, books=books, window_node_slices=slices, exec_sign=exec_sign, exec_increment=np.asarray(exec_inc),
        exec_u=np.asarray(u, dtype=float), exec_alpha=np.asarray(alpha, dtype=float),
        exec_one_minus_alpha=np.asarray(one_minus, dtype=float), exec_informed=np.asarray(informed, bool),
        window_signed_flow=window_flow, window_background=window_bg, jump_applied=arm.jumps != "none",
        jump_sign=jump_sign, jump_step=jump_step, settlement=settlement, saturated_nodes=saturated,
        flow_parameters=params, spread_draws=1+len(books), spread_state_final=spread_rng.bit_generator.state,
        jump_active=active, jump_keep_probability=1.0 if jump_keep is None else float(jump_keep_probability))


def with_p0(arm: Arm, p0: float) -> Arm:
    """Same arm at another starting probability (e.g. the p0_emp sensitivity)."""
    return replace(arm, p0=float(p0))


__all__ = ["Arm", "EpisodePath", "make_arm", "build_path", "with_p0", "FLOW_LAW",
           "EXECUTION", "JUMP", "PROPOSAL_BID", "PROPOSAL_ASK", "DECISION", "WINDOW_END"]
