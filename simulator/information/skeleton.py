"""Policy-, arm- and S-level-independent base random inputs per episode (spec Section 5).

Seeding mirrors `simulator.evaluate_hjb.run`: path_seed =
int(SeedSequence([seed, event_index, replicate]).generate_state(1)[0]) with the
global event_index over broad_based then sports, tickers sorted within group
(preserved when a subset is run). The episode SeedSequence(path_seed) spawns 12
children: 0-4 are used exactly as `TradingEnvironment.reset` uses them
(selection, market, spread, bridge, settlement), then 5 jump, 6 flow sign
(V_u, V_I, V_F), 7 background W, 8 L-aux, 9 U_M, 10 U_L, 11 snipe proposals.

Everything here is generated before any policy acts. Spread draws are NOT
precomputed: the spread stream is left exactly at its post-`SpreadSampler`-init
state and replayed online, in legacy order, by `paths.py`.

The jump-intensity sensitivity (coupled doses) uses two further children, 12
(thinning) and 13 (superposition), spawned positionally after these twelve;
see `dose_rng`, `thinning_draws` and `superpose_jumps` below.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from ..market_dynamics import DEFAULT_CONFIG as DYNAMICS_CONFIG, MarketDynamics
from ..spreads import DEFAULT_CONFIG as EXECUTION_CONFIG, SpreadSampler
from ..trading_env import EnvironmentConfig, bridged_innovations
from . import kernels as K

SEED = 20261009
GROUPS = ("broad_based", "sports")
STREAMS = ("selection", "market", "spread", "bridge", "settlement",
           "jump", "flow_sign", "background", "l_aux", "u_m", "u_l", "snipe")
DECISION_SPACING_SECONDS = 60.0
MEAN_LATENCY_SECONDS = 2.0
BOUNDARY_TOLERANCE_SECONDS = 1e-6

# Node kinds. Ties in time are broken by this order (window end always last).
EXECUTION, JUMP, PROPOSAL_BID, PROPOSAL_ASK, DECISION, WINDOW_END = range(6)
KIND_NAMES = ("execution", "jump", "proposal_bid", "proposal_ask", "decision", "window_end")


def event_roster(dynamics_path: Path = DYNAMICS_CONFIG) -> list[tuple[str, str, int]]:
    """(group, ticker, global event_index): broad_based then sports, tickers sorted."""
    config = json.loads(Path(dynamics_path).read_text())
    rows, index = [], 0
    for group in GROUPS:
        for ticker in sorted(config["groups"][group]["event_tickers"]):
            rows.append((group, ticker, index))
            index += 1
    return rows


def event_index(ticker: str, dynamics_path: Path = DYNAMICS_CONFIG) -> tuple[str, int]:
    for group, name, index in event_roster(dynamics_path):
        if name == ticker:
            return group, index
    raise KeyError(f"{ticker} is not a training event template")


def path_seed(event_index: int, replicate: int, seed: int = SEED) -> int:
    """Exactly `evaluate_hjb.run`'s per-(event, replicate) path seed."""
    return int(np.random.SeedSequence([seed, event_index, replicate]).generate_state(1)[0])


@dataclass
class LegacyWindow:
    """One legacy 300-s market window: legacy draws in legacy order."""
    index: int
    start: float                 # elapsed seconds at window start (legacy float arithmetic)
    duration: float
    end: float                   # min(lifetime, start + duration)
    window: dict                 # MarketDynamics.sample_window output, unmodified
    innovation: float            # legacy sample_innovation draw
    pieces: np.ndarray           # bridged_innovations pieces (len n + 1), legacy bridge stream
    exec_slice: slice            # this window's executions in the episode execution arrays
    node_slice: slice = None     # this window's nodes in the episode node arrays


@dataclass
class EpisodeSkeleton:
    ticker: str
    group: str
    information_class: str
    event_index: int
    replicate: int
    seed: int
    path_seed: int
    config: EnvironmentConfig
    dynamics_path: Path
    execution_path: Path
    dynamics: MarketDynamics
    spreads: SpreadSampler
    spread_rng_after_init: np.random.Generator   # pristine copy; never consumed here
    lifetime: float
    seconds: float
    start: float
    windows: list[LegacyWindow]
    # Executions (episode order).
    exec_window: np.ndarray
    exec_offset: np.ndarray      # window-relative legacy offsets
    exec_elapsed: np.ndarray
    exec_quantity: np.ndarray
    exec_bank_sign: np.ndarray
    exec_phase: np.ndarray
    exec_piece: np.ndarray
    v_u: np.ndarray
    v_i: np.ndarray
    v_f: np.ndarray
    # Jumps (shared; applied only when an arm has jumps).
    jump_elapsed: np.ndarray
    jump_magnitude: np.ndarray
    v_j: np.ndarray
    jump_rates: dict
    ttr_at_close: float
    # Snipe proposals and decision nodes.
    mean_latency: float
    proposal_bid_elapsed: np.ndarray
    proposal_ask_elapsed: np.ndarray
    decision_elapsed: np.ndarray
    # Node set (all windows, time order).
    node_kind: np.ndarray
    node_ref: np.ndarray         # index into exec/jump/proposal/decision arrays, window index for ends
    node_window: np.ndarray
    node_elapsed: np.ndarray
    node_offset: np.ndarray      # window-relative offset (legacy offsets for executions, duration for ends)
    node_h: np.ndarray           # interval since the previous node (first: since episode start)
    node_dw: np.ndarray          # W increments, N(0, h), stream 7
    node_v_aux: np.ndarray       # L-aux uniforms per node interval, stream 8
    u_m: float
    u_l: float
    u_legacy_settlement: float   # legacy settlement stream's single draw
    decision_spacing: float = DECISION_SPACING_SECONDS
    jump_rate_scale: float = 1.0
    _hash: str | None = field(default=None, repr=False)
    # Coupled jump-intensity dose (sensitivity B; None on every skeleton built by `build_skeleton`).
    # On a superposed (2x) skeleton: jump_extra marks the child-13 jumps, node_original the 1x nodes.
    jump_extra: np.ndarray | None = None
    node_original: np.ndarray | None = None
    dose_input_hash: str | None = None

    @property
    def noise_ticker(self) -> str:
        """Legacy donor drawn on the selection stream (own ticker when quote-supported)."""
        return self.dynamics.noise_template_ticker

    @property
    def n_executions(self) -> int:
        return len(self.exec_quantity)

    def spread_rng(self) -> np.random.Generator:
        """A fresh copy of the spread stream exactly after `SpreadSampler.__init__`."""
        return copy.deepcopy(self.spread_rng_after_init)

    def base_input_hash(self) -> str:
        """V8(a): sha256 of every base input (recomputed; detects mutation)."""
        h = hashlib.sha256()
        scalars = {"ticker": self.ticker, "group": self.group, "path_seed": self.path_seed,
                   "event_index": self.event_index, "replicate": self.replicate,
                   "donor": self.noise_ticker, "spread_ticker": self.spreads.ticker,
                   "lifetime": self.lifetime, "seconds": self.seconds, "start": self.start,
                   "mean_latency": self.mean_latency, "u_m": self.u_m, "u_l": self.u_l,
                   "u_legacy_settlement": self.u_legacy_settlement, "sigma_legacy": self.dynamics.sigma,
                   "jump_rates": self.jump_rates, "ttr_at_close": self.ttr_at_close,
                   "spread_state": self.spread_rng_after_init.bit_generator.state,
                   "windows": [[w.start, w.duration, w.end, w.innovation, int(w.window["phase"]),
                                bool(w.window["phase_mark_fallback"])] for w in self.windows]}
        h.update(json.dumps(scalars, sort_keys=True, default=str).encode())
        for array in (self.exec_window, self.exec_offset, self.exec_elapsed, self.exec_quantity, self.exec_bank_sign,
                      self.exec_phase, self.exec_piece, self.v_u, self.v_i, self.v_f, self.jump_elapsed,
                      self.jump_magnitude, self.v_j, self.proposal_bid_elapsed, self.proposal_ask_elapsed,
                      self.decision_elapsed, self.node_kind, self.node_ref, self.node_elapsed, self.node_offset,
                      self.node_h, self.node_dw, self.node_v_aux, *[w.pieces for w in self.windows]):
            a = np.ascontiguousarray(array)
            h.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        return h.hexdigest()


def _poisson_times(rng, rate: float, seconds: float) -> np.ndarray:
    n = int(rng.poisson(rate*seconds))
    return np.sort(rng.uniform(0.0, seconds, size=n))


def build_skeleton(ticker: str, replicate: int, *, calibration, seed: int = SEED, config: EnvironmentConfig | None = None,
                   hours_to_close: float = 1.0, mean_latency: float = MEAN_LATENCY_SECONDS,
                   decision_spacing: float = DECISION_SPACING_SECONDS, jump_rate_scale: float = 1.0,
                   dynamics_path: Path = DYNAMICS_CONFIG, execution_path: Path = EXECUTION_CONFIG) -> EpisodeSkeleton:
    """Generate every policy-independent base input of one (template, replicate) episode."""
    group, index = event_index(ticker, dynamics_path)
    config = config or EnvironmentConfig(group=group, ticker=ticker)
    if config.group != group or config.ticker != ticker:
        raise ValueError("Environment config group/ticker differs from the training roster")
    if not np.isfinite(mean_latency) or mean_latency <= 0:
        raise ValueError("Zero-latency (grid) sniping is a Tier 2 extension and is not implemented here")
    if not np.isfinite(decision_spacing) or decision_spacing <= 0 or not np.isfinite(jump_rate_scale) or jump_rate_scale < 0:
        raise ValueError("Invalid decision spacing or jump-rate scale")
    information_class = calibration.information_class(ticker)
    if information_class != group:
        raise ValueError("Information class differs from the legacy group")
    pseed = path_seed(index, replicate, seed)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(pseed).spawn(len(STREAMS))]
    selection, market, spread, bridge, settlement, jump_rng, flow_rng, w_rng, aux_rng, um_rng, ul_rng, snipe_rng = rngs

    # Children 0-4 exactly as TradingEnvironment.reset (selection -> donor; spread init may draw).
    dynamics = MarketDynamics(config.group, selection, config.ticker, config.impact_scenario, config.arrival_scenario,
                              config.noise_scenario, config_path=dynamics_path)
    spreads = SpreadSampler(config.group, dynamics.ticker, spread, dynamics_path, execution_path)
    spread_after_init = copy.deepcopy(spread)
    lifetime = float(dynamics.template["lifetime_seconds"])
    raw_dynamics = json.loads(Path(dynamics_path).read_text())
    seconds = float(min(hours_to_close*3600, raw_dynamics["event_templates"][ticker]["lifetime_seconds"]))
    if not np.isfinite(seconds) or not 0 < seconds <= lifetime:
        raise ValueError("seconds_to_close must be within the market lifetime")
    start = lifetime-seconds

    # Legacy driver: per window sample_window then sample_innovation (market), bridge pieces (bridge).
    windows, elapsed = [], start
    q_parts, s_parts, off_parts, piece_parts, phase_parts, win_parts = [], [], [], [], [], []
    n_total = 0
    while True:
        w_start = elapsed
        window = dynamics.sample_window(w_start/lifetime, market)
        duration = float(window["duration_seconds"])
        if not 0 < duration <= 300 or duration > lifetime-w_start+1e-7:
            raise ValueError("Market driver returned invalid window duration")
        innovation = float(dynamics.sample_innovation(market, duration_seconds=duration)[0])
        pieces = bridged_innovations(window["event_times_seconds"], duration, innovation, dynamics.sigma, bridge)
        n = len(window["quantities"])
        if n != len(window["signs"]) or n != len(window["event_times_seconds"]):
            raise ValueError("Market driver returned mismatched execution arrays")
        end = min(lifetime, w_start+duration)
        windows.append(LegacyWindow(len(windows), w_start, duration, end, window, innovation, pieces,
                                    slice(n_total, n_total+n)))
        q_parts.append(window["quantities"])
        s_parts.append(window["signs"])
        off_parts.append(window["event_times_seconds"])
        piece_parts.append(pieces[:-1])
        phase_parts.append(np.full(n, int(window["phase"]), dtype=np.int64))
        win_parts.append(np.full(n, len(windows)-1, dtype=np.int64))
        n_total += n
        elapsed = end
        if lifetime-elapsed < 1e-7:
            break
    u_legacy = float(settlement.random())   # TradingEnvironment draws exactly once, at termination

    exec_quantity = np.concatenate(q_parts) if q_parts else np.zeros(0)
    exec_bank_sign = np.concatenate(s_parts) if s_parts else np.zeros(0, dtype=np.int8)
    exec_offset = np.concatenate(off_parts) if off_parts else np.zeros(0)
    exec_piece = np.concatenate(piece_parts) if piece_parts else np.zeros(0)
    exec_phase = np.concatenate(phase_parts) if phase_parts else np.zeros(0, dtype=np.int64)
    exec_window = np.concatenate(win_parts) if win_parts else np.zeros(0, dtype=np.int64)
    exec_elapsed = np.empty(n_total)
    for w in windows:
        # Legacy: self.elapsed = start + float(offset)
        exec_elapsed[w.exec_slice] = [w.start+float(o) for o in w.window["event_times_seconds"]]
    K.check_signs(exec_bank_sign)

    # Stream 6: shared flow-sign uniforms per execution.
    v_u, v_i, v_f = flow_rng.random(n_total), flow_rng.random(n_total), flow_rng.random(n_total)

    # Stream 5: jump times (piecewise Poisson in TTR), magnitudes from the class bank, V_J.
    rates = calibration.jump_rates(information_class, jump_rate_scale)
    ttr_close = calibration.ttr_at_close(ticker)
    jump_offsets = K.poisson_jump_offsets(ttr_close+seconds, seconds, rates, jump_rng)
    bank = calibration.magnitude_bank(information_class)
    if len(jump_offsets) and not len(bank):
        raise ValueError("Empty jump magnitude bank")
    jump_magnitude = jump_rng.choice(bank, size=len(jump_offsets)) if len(jump_offsets) else np.zeros(0)
    v_j = jump_rng.random(len(jump_offsets))

    # Stream 11: two independent homogeneous Poisson proposal processes (bid then ask).
    bid_offsets = _poisson_times(snipe_rng, 1.0/mean_latency, seconds)
    ask_offsets = _poisson_times(snipe_rng, 1.0/mean_latency, seconds)

    ends = np.array([w.end for w in windows])
    starts = np.array([w.start for w in windows])
    boundaries = np.r_[starts, ends]
    decision = []
    j = 1
    while decision_spacing*j < seconds:
        t = start+decision_spacing*j
        if np.min(np.abs(boundaries-t)) > BOUNDARY_TOLERANCE_SECONDS:
            decision.append(t)
        j += 1
    decision_elapsed = np.asarray(decision, dtype=float)

    def locate(elapsed_values):
        e = np.minimum(start+np.asarray(elapsed_values, dtype=float), ends[-1])
        return e, np.minimum(np.searchsorted(ends, e, side="right"), len(windows)-1)

    jump_elapsed, jump_win = locate(jump_offsets)
    bid_elapsed, bid_win = locate(bid_offsets)
    ask_elapsed, ask_win = locate(ask_offsets)
    dec_win = np.minimum(np.searchsorted(ends, decision_elapsed, side="right"), len(windows)-1)

    kinds = np.r_[np.full(n_total, EXECUTION), np.full(len(jump_elapsed), JUMP), np.full(len(bid_elapsed), PROPOSAL_BID),
                  np.full(len(ask_elapsed), PROPOSAL_ASK), np.full(len(decision_elapsed), DECISION),
                  np.full(len(windows), WINDOW_END)].astype(np.int8)
    refs = np.r_[np.arange(n_total), np.arange(len(jump_elapsed)), np.arange(len(bid_elapsed)),
                 np.arange(len(ask_elapsed)), np.arange(len(decision_elapsed)), np.arange(len(windows))].astype(np.int64)
    wins = np.r_[exec_window, jump_win, bid_win, ask_win, dec_win, np.arange(len(windows))].astype(np.int64)
    times = np.r_[exec_elapsed, jump_elapsed, bid_elapsed, ask_elapsed, decision_elapsed, ends].astype(float)
    order = np.lexsort((kinds, times, wins))      # window, then time, then kind priority
    node_kind, node_ref, node_window, node_elapsed = kinds[order], refs[order], wins[order], times[order]
    node_offset = node_elapsed-starts[node_window]
    is_exec, is_end = node_kind == EXECUTION, node_kind == WINDOW_END
    node_offset[is_exec] = exec_offset[node_ref[is_exec]]
    node_offset[is_end] = [windows[k].duration for k in node_ref[is_end]]
    h = np.diff(np.r_[start, node_elapsed])
    if np.any(h < -1e-9):
        raise AssertionError("Skeleton nodes are not in time order")
    h = np.maximum(h, 0.0)
    node_dw = np.sqrt(h)*w_rng.standard_normal(len(h))     # stream 7: exact variances between nodes
    node_v_aux = aux_rng.random(len(h))                   # stream 8
    # Amendment A-3: U_L is the legacy settlement draw, so the whole L chain (L0 included) shares one
    # settlement uniform. That draw feeds no generator; stream 10 is still spawned, so the spawn order is unchanged.
    u_m, u_l = float(um_rng.random()), u_legacy

    bounds = np.searchsorted(node_window, np.arange(len(windows)+1), side="left")
    for k, w in enumerate(windows):
        w.node_slice = slice(int(bounds[k]), int(bounds[k+1]))
        if node_kind[w.node_slice.stop-1] != WINDOW_END:
            raise AssertionError("Each window must end with its window-end node")

    return EpisodeSkeleton(
        ticker=ticker, group=group, information_class=information_class, event_index=index, replicate=replicate,
        seed=seed, path_seed=pseed, config=config, dynamics_path=Path(dynamics_path), execution_path=Path(execution_path),
        dynamics=dynamics, spreads=spreads, spread_rng_after_init=spread_after_init, lifetime=lifetime, seconds=seconds,
        start=start, windows=windows, exec_window=exec_window, exec_offset=exec_offset, exec_elapsed=exec_elapsed,
        exec_quantity=exec_quantity, exec_bank_sign=exec_bank_sign, exec_phase=exec_phase, exec_piece=exec_piece,
        v_u=v_u, v_i=v_i, v_f=v_f, jump_elapsed=jump_elapsed, jump_magnitude=np.asarray(jump_magnitude, dtype=float),
        v_j=v_j, jump_rates=rates, ttr_at_close=ttr_close, mean_latency=float(mean_latency),
        proposal_bid_elapsed=bid_elapsed, proposal_ask_elapsed=ask_elapsed, decision_elapsed=decision_elapsed,
        node_kind=node_kind, node_ref=node_ref, node_window=node_window, node_elapsed=node_elapsed,
        node_offset=node_offset, node_h=h, node_dw=node_dw, node_v_aux=node_v_aux, u_m=u_m, u_l=u_l,
        u_legacy_settlement=u_legacy, decision_spacing=float(decision_spacing), jump_rate_scale=float(jump_rate_scale))


# ---------------------------------------------------------------------------------------------------
# Coupled jump-intensity doses (docs/paper3/SENSITIVITY_SPEC.md, experiment B).
#
# Two extra episode streams are spawned after the twelve above. SeedSequence spawn keys are positional,
# so SeedSequence(path_seed).spawn(14)[:12] are exactly the twelve Tier 1 children; children 0-11 and
# every Tier 1 draw are unchanged. Child 12 thins the native (1x) jumps; child 13 draws an independent
# Poisson(lambda_J) superposition and the Brownian-bridge normals that insert its nodes into W.
DOSE_STREAMS = ("jump_thinning", "jump_superposition")
THINNING_KEEP_PROBABILITY = 0.5


def dose_rng(pseed: int, stream: str) -> np.random.Generator:
    """Generator of episode child 12 (thinning) or 13 (superposition); children 0-11 are untouched."""
    children = np.random.SeedSequence(pseed).spawn(len(STREAMS)+len(DOSE_STREAMS))
    return np.random.default_rng(children[len(STREAMS)+DOSE_STREAMS.index(stream)])


def thinning_draws(sk: EpisodeSkeleton, keep: float = THINNING_KEEP_PROBABILITY) -> tuple[np.ndarray, np.ndarray]:
    """(uniforms, keep mask) for the native jumps of a 1x skeleton: keep jump j iff U_j < keep (child 12).

    One uniform per native jump, in the skeleton's (time) order. Independent p-thinning of a Poisson
    process of intensity lambda_J is Poisson(keep * lambda_J) with the same i.i.d. marks.
    """
    if sk.jump_extra is not None or sk.jump_rate_scale != 1.0:
        raise ValueError("Thinning applies to the native (1x) skeleton")
    if not 0 <= keep <= 1:
        raise ValueError("Keep probability must be in [0, 1]")
    uniforms = dose_rng(sk.path_seed, "jump_thinning").random(len(sk.jump_elapsed))
    return uniforms, uniforms < keep


def superposition_draws(ttr_start: float, seconds: float, rates: dict, bank: np.ndarray, rng) -> tuple:
    """Independent Poisson(lambda_J) jumps on [0, seconds): offsets, bank magnitudes, V_J (child 13 order).

    Same generator calls and order as the native jumps in `build_skeleton` (piecewise Poisson times,
    then magnitudes, then V_J), on the superposition stream.
    """
    offsets = K.poisson_jump_offsets(ttr_start, seconds, rates, rng)
    if len(offsets) and not len(bank):
        raise ValueError("Empty jump magnitude bank")
    magnitude = rng.choice(bank, size=len(offsets)) if len(offsets) else np.zeros(0)
    v_j = rng.random(len(offsets))
    return offsets, np.asarray(magnitude, dtype=float), v_j


def brownian_bridge_insert(total_h: float, total_dw: float, positions, normals) -> tuple[np.ndarray, np.ndarray]:
    """Insert points into one Brownian increment W(H) - W(0) = total_dw by sequential bridge sampling.

    positions: sorted points u in [0, H]; normals: one N(0,1) per point. Given W(t_prev) = w and
    W(H) = total_dw, W(u) ~ N(w + (u - t_prev)/(H - t_prev) (total_dw - w), (u - t_prev)(H - u)/(H - t_prev)).
    Returns (pieces, h): len(positions) + 1 increments and interval lengths; the pieces telescope to
    total_dw, so W at the end point (the original node) is unchanged up to float rounding.
    """
    H, d = float(total_h), float(total_dw)
    u = np.asarray(positions, dtype=float)
    z = np.asarray(normals, dtype=float)
    if H < 0 or len(u) != len(z) or np.any(np.diff(u) < 0) or np.any(u < 0) or np.any(u > H):
        raise ValueError("Bridge points must be sorted inside [0, H]")
    pieces, hs = np.empty(len(u)+1), np.empty(len(u)+1)
    w_prev, t_prev = 0.0, 0.0
    for k, (uk, zk) in enumerate(zip(u, z)):
        remaining = H-t_prev
        if uk >= H:
            w = d                                   # the end point itself: W(H) is known exactly
        elif remaining > 0:
            mean = w_prev+(uk-t_prev)/remaining*(d-w_prev)
            var = max((uk-t_prev)*(H-uk)/remaining, 0.0)
            w = mean+np.sqrt(var)*zk
        else:
            w = w_prev
        pieces[k], hs[k] = w-w_prev, uk-t_prev
        w_prev, t_prev = w, uk
    pieces[-1], hs[-1] = d-w_prev, H-t_prev
    return pieces, hs


def superpose_jumps(sk: EpisodeSkeleton, calibration) -> EpisodeSkeleton:
    """The 2x dose skeleton: native jumps plus an independent Poisson(lambda_J) process (child 13).

    Every native input (executions, native jumps, proposals, decision nodes, windows, uniforms, spread
    state, settlement uniforms) is shared with `sk`. The new jump nodes are inserted into the node set by
    Brownian-bridge refinement of W on the same stream: the W increment of every native node interval is
    split into pieces that sum to it, so W at every native node is unchanged (up to float rounding of the
    telescoping sum). L-aux uniforms are not defined at inserted nodes (NaN): only M-family (Theta-known)
    paths may be built on this skeleton.
    """
    if sk.jump_extra is not None or sk.jump_rate_scale != 1.0:
        raise ValueError("Superposition applies to the native (1x) skeleton")
    rng = dose_rng(sk.path_seed, "jump_superposition")
    rates = calibration.jump_rates(sk.information_class, 1.0)
    if rates != sk.jump_rates:
        raise ValueError("Skeleton jump rates differ from the calibration's native rates")
    bank = calibration.magnitude_bank(sk.information_class)
    offsets, magnitude, v_j = superposition_draws(sk.ttr_at_close+sk.seconds, sk.seconds, rates, bank, rng)
    n1, n2 = len(sk.jump_elapsed), len(offsets)
    ends = np.array([w.end for w in sk.windows])
    extra_elapsed = np.minimum(sk.start+offsets, ends[-1])
    extra_window = np.minimum(np.searchsorted(ends, extra_elapsed, side="right"), len(sk.windows)-1)

    # Combined jump arrays in time order (stable: a native jump precedes an inserted one at equal times).
    all_elapsed = np.r_[sk.jump_elapsed, extra_elapsed]
    jorder = np.argsort(all_elapsed, kind="stable")
    new_index = np.empty(n1+n2, dtype=np.int64)
    new_index[jorder] = np.arange(n1+n2)
    jump_elapsed = all_elapsed[jorder]
    jump_magnitude = np.r_[sk.jump_magnitude, magnitude][jorder]
    jump_v = np.r_[sk.v_j, v_j][jorder]
    jump_extra = np.r_[np.zeros(n1, bool), np.ones(n2, bool)][jorder]

    # Merged node set: native nodes first in their own order, then inserted jump nodes; lexsort is stable,
    # so the native nodes keep their relative order.
    n0 = len(sk.node_kind)
    refs = sk.node_ref.copy()
    native_jump = sk.node_kind == JUMP
    refs[native_jump] = new_index[refs[native_jump]]
    kinds = np.r_[sk.node_kind, np.full(n2, JUMP, dtype=np.int8)].astype(np.int8)
    refs = np.r_[refs, new_index[n1+np.arange(n2)]].astype(np.int64)
    wins = np.r_[sk.node_window, extra_window].astype(np.int64)
    times = np.r_[sk.node_elapsed, extra_elapsed].astype(float)
    order = np.lexsort((kinds, times, wins))
    is_extra = order >= n0
    original_pos = np.flatnonzero(~is_extra)
    if not np.array_equal(order[original_pos], np.arange(n0)):
        raise AssertionError("Native nodes changed order after insertion")
    node_kind, node_ref, node_window, node_elapsed = kinds[order], refs[order], wins[order], times[order]
    starts = np.array([w.start for w in sk.windows])
    node_offset = np.empty(len(order))
    node_offset[original_pos] = sk.node_offset
    extra_pos = np.flatnonzero(is_extra)
    node_offset[extra_pos] = node_elapsed[extra_pos]-starts[node_window[extra_pos]]

    # Brownian-bridge insertion: inserted nodes split the native interval of the next native node.
    node_h, node_dw = np.empty(len(order)), np.empty(len(order))
    node_h[original_pos], node_dw[original_pos] = sk.node_h, sk.node_dw
    normals = rng.standard_normal(n2)
    next_native = np.searchsorted(original_pos, extra_pos)
    if n2 and next_native.max() >= n0:
        raise AssertionError("An inserted node must precede a native node (every window ends with one)")
    native_elapsed = np.r_[sk.start, sk.node_elapsed]
    for k in np.unique(next_native):
        sel = np.flatnonzero(next_native == k)
        positions = np.clip(node_elapsed[extra_pos[sel]]-native_elapsed[k], 0.0, sk.node_h[k])
        pieces, hs = brownian_bridge_insert(sk.node_h[k], sk.node_dw[k], positions, normals[sel])
        node_dw[extra_pos[sel]], node_h[extra_pos[sel]] = pieces[:-1], hs[:-1]
        node_dw[original_pos[k]], node_h[original_pos[k]] = pieces[-1], hs[-1]
    node_v_aux = np.full(len(order), np.nan)
    node_v_aux[original_pos] = sk.node_v_aux

    bounds = np.searchsorted(node_window, np.arange(len(sk.windows)+1), side="left")
    windows = [replace(w, node_slice=slice(int(bounds[k]), int(bounds[k+1]))) for k, w in enumerate(sk.windows)]
    for w in windows:
        if node_kind[w.node_slice.stop-1] != WINDOW_END:
            raise AssertionError("Each window must end with its window-end node")
    dose_hash = hashlib.sha256()
    for a in (offsets, magnitude, v_j, normals):
        a = np.ascontiguousarray(a)
        dose_hash.update(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return replace(sk, windows=windows, jump_elapsed=jump_elapsed, jump_magnitude=jump_magnitude, v_j=jump_v,
                   jump_rates=calibration.jump_rates(sk.information_class, 2.0), node_kind=node_kind,
                   node_ref=node_ref, node_window=node_window, node_elapsed=node_elapsed, node_offset=node_offset,
                   node_h=node_h, node_dw=node_dw, node_v_aux=node_v_aux, jump_rate_scale=2.0,
                   jump_extra=jump_extra, node_original=~is_extra, dose_input_hash=dose_hash.hexdigest())
