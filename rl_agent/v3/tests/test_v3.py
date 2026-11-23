"""Checks for the v3 agent's env, features and live adapter."""
import numpy as np
import pytest

from rl_agent.v3.env_v3 import SimAgentDriver, V3QuotingEnv, sample_scenario
from rl_agent.v3.features import OBS_DIM, FeatureTracker, decode_action


def test_reward_decomposition_reconstructs_realized_pnl():
    """training reward (lambda=0) + diffusion noise + settlement noise == realized cash PnL."""
    rng = np.random.default_rng(3)
    for i in range(60):
        d = SimAgentDriver(sample_scenario(rng), i)
        d.obs()
        total, done = 0.0, False
        while not done:
            r, done, _ = d.step(d.quote(rng.uniform(-1, 1, 4)), 0.0)
            total += r
        noise, prev_p, prev_q, settle = 0.0, d.scenario["p0"], 0.0, 0.0
        for row in d.env.ledger:
            if row["kind"] == "noise":
                noise += float(row["inventory"]) * (row["probability"] - prev_p)
            if row["kind"] == "settlement":
                settle = prev_q * (row["settlement_outcome"] - prev_p)
            prev_p, prev_q = row["probability"], float(row["inventory"])
        pnl = float(d.env.account.cash - d.env.account.initial_cash)
        assert pnl == pytest.approx(total + noise + settle, abs=2e-3)


def test_flip_mirrors_taker_flow():
    scn = {"group": "sports", "impact": "base", "fill": "base", "flip": False, "p0": 0.5}
    a = SimAgentDriver(scn, 7)
    b = SimAgentDriver({**scn, "flip": True}, 7)
    assert np.array_equal(a.env.dynamics.signs, -b.env.dynamics.signs)


def test_env_spaces_and_episode_length():
    env = V3QuotingEnv(seed=1)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    steps, done = 0, False
    while not done:
        obs, r, done, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(r) and env.observation_space.contains(obs)
        steps += 1
    assert 1 <= steps <= 13


def test_decode_action_bounds_and_disable():
    bid, ask, bs, as_ = decode_action([0, 0, 0, 0], 0.5, 0)
    assert bid < 0.5 < ask and bs == as_ == 10
    _, _, bs, as_ = decode_action([0, 0, -1, -1], 0.5, 0)
    assert bs == 0 and as_ == 0
    _, _, bs, as_ = decode_action([0, 0, 1, 1], 0.5, 95)
    assert bs == 5 and as_ == 20          # bid capped by inventory headroom


def test_tracker_features_are_live_computable():
    t = FeatureTracker()
    v0 = t.observe(0.48, 0.52, 0, 0.0)
    t.record_window([1, 1, -1], [100, 50, 20])
    v1 = t.observe(0.50, 0.54, 10, 0.1)
    assert v0.shape == v1.shape == (OBS_DIM,)
    assert v1[4] > 0 and v1[7] > 0              # net YES-taker flow is positive


def test_dashboard_adapter_quotes_post_only(tmp_path):
    from pathlib import Path
    model = Path(__file__).resolve().parents[1] / "models" / "ppo_v3.zip"
    if not model.exists():
        pytest.skip("no trained v3 model")
    from dashboard.strategies import Context, V3Agent
    agent = V3Agent(model)
    assert agent.available, agent.reason
    trades = ({"ts": 1.0, "qty": 120.0, "side": "yes"}, {"ts": 2.0, "qty": 30.0, "side": "no"})
    for i, (bid, ask) in enumerate([(0.47, 0.49), (0.48, 0.50), (0.50, 0.53)]):
        ctx = Context(bid, ask, 0.0, now=1000.0 + 300 * i, open_ts=0.0, close_ts=3000.0,
                      tick="0.01", ticker="T", trades=trades)
        action, info = agent.quote(ctx)
        assert action.bid is None or action.bid < ask
        assert action.ask is None or action.ask > bid


def test_sixty_second_episodes_cover_the_hour():
    scn = {"group": "sports", "impact": "base", "fill": "base", "flip": False, "p0": 0.5}
    d = SimAgentDriver(scn, 5, decision_seconds=60)
    d.obs()
    steps, done = 0, False
    while not done:
        _, done, _ = d.step(d.quote(np.zeros(4)), 0.0)
        steps += 1
    assert steps == 60 and d.env.elapsed - d.env.start_elapsed == pytest.approx(3600)


def test_dashboard_picks_model_matching_cadence():
    from dashboard import config
    from dashboard.strategies import V3Agent
    assert V3Agent.model_for(60) == config.V3_60_MODEL
    assert V3Agent.model_for(300) == config.V3_MODEL
