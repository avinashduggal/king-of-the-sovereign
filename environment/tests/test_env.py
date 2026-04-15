"""Gymnasium API conformance + seeded determinism for SovereignEnv."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import sovereign  # noqa: F401  -- registers env ids


def test_make_v0():
    env = gym.make("Sovereign-v0")
    assert env is not None
    env.close()


def test_reset_returns_dict_obs_and_info():
    env = sovereign.SovereignEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    for key in (
        "territory_control",
        "invader_units",
        "defender_units",
        "legitimacy",
        "supply",
        "theta",
        "occupation_duration",
        "timestep",
        "sanctions_active",
        "neutral_joined_defender",
        "neutral_allied_invader",
    ):
        assert key in obs


def test_observation_space_contains_obs_after_reset():
    env = sovereign.SovereignEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_observation_space_contains_obs_after_step():
    env = sovereign.SovereignEnv()
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)


def test_step_returns_five_tuple():
    env = sovereign.SovereignEnv()
    env.reset(seed=0)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, r, term, trunc, info = out
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


def test_action_space_sample_is_valid():
    env = sovereign.SovereignEnv()
    a = env.action_space.sample()
    assert env.action_space.contains(a)


def test_seeded_determinism_two_envs():
    """Two envs with the same seed and the same action sequence must
    produce identical observations and rewards."""
    e1 = sovereign.SovereignEnv()
    e2 = sovereign.SovereignEnv()
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)

    rng = np.random.default_rng(7)
    for _ in range(40):
        a = e1.action_space.sample()  # action_space sampling is local
        # Use a deterministic action sequence so tests are reproducible
        a = np.array(
            [int(rng.integers(0, 5)), int(rng.integers(0, 4)), int(rng.integers(0, 9))],
            dtype=np.int64,
        )
        s1 = e1.step(a)
        s2 = e2.step(a)
        for k in s1[0]:
            assert np.array_equal(s1[0][k], s2[0][k]), f"key {k} differs"
        assert s1[1] == s2[1]
        if s1[2] or s1[3]:
            break


def test_episode_terminates_eventually():
    env = sovereign.SovereignEnv()
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 1000:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert terminated or truncated
    assert steps <= 1000


def test_render_ansi_returns_string():
    env = sovereign.SovereignEnv(render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "SOVEREIGN" in out


def test_invalid_action_raises():
    env = sovereign.SovereignEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step((9, 9, 99))


def test_check_env_passes():
    """Run gymnasium's built-in env checker."""
    from gymnasium.utils.env_checker import check_env

    env = sovereign.SovereignEnv()
    # check_env expects the unwrapped env; warnings are OK, errors are not.
    check_env(env, skip_render_check=True)
