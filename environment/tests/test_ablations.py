"""Ablation correctness: each preset disables exactly the right mechanics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import sovereign  # noqa: F401  -- registers env ids
from sovereign.config import (
    ABLATION_PRESETS,
    baseline,
    full_model,
    no_legitimacy,
    no_neutral_posture,
    no_occupation_cost,
)


@pytest.mark.parametrize("preset_name", list(ABLATION_PRESETS))
def test_ablation_preset_makes_env(preset_name):
    """Each registered ablation id constructs and runs one step."""
    env_id = f"Sovereign-{preset_name}-v0"
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert info["config_preset"] == preset_name
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert isinstance(r, float)
    env.close()


def test_no_legitimacy_holds_L_constant():
    cfg = no_legitimacy()
    env = sovereign.SovereignEnv(config=cfg)
    obs, _ = env.reset(seed=42)
    initial_L = float(obs["legitimacy"][0])
    for _ in range(50):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        assert float(obs["legitimacy"][0]) == initial_L
        if term or trunc:
            break


def test_no_occupation_cost_holds_t_occ_at_zero():
    cfg = no_occupation_cost()
    env = sovereign.SovereignEnv(config=cfg)
    obs, _ = env.reset(seed=42)
    for _ in range(40):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        assert int(obs["occupation_duration"][0]) == 0
        if term or trunc:
            break


def test_no_neutral_posture_holds_theta_at_zero():
    cfg = no_neutral_posture()
    env = sovereign.SovereignEnv(config=cfg)
    obs, _ = env.reset(seed=42)
    for _ in range(40):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        assert float(obs["theta"][0]) == 0.0
        assert int(obs["sanctions_active"][0]) == 0
        if term or trunc:
            break


def test_baseline_holds_all_off():
    cfg = baseline()
    env = sovereign.SovereignEnv(config=cfg)
    obs, _ = env.reset(seed=42)
    initial_L = float(obs["legitimacy"][0])
    for _ in range(40):
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        assert float(obs["legitimacy"][0]) == initial_L
        assert int(obs["occupation_duration"][0]) == 0
        assert float(obs["theta"][0]) == 0.0
        if term or trunc:
            break


def test_full_model_can_modify_all_three():
    """Under the full model, an aggressive policy should be able to move
    L, t_occ, and theta away from their initial values."""
    cfg = full_model()
    env = sovereign.SovereignEnv(config=cfg)
    obs, _ = env.reset(seed=7)
    # Force aggressive actions: ISSUE_THREAT + ADVANCE on adjacent territory.
    moved_L = moved_theta = moved_tocc = False
    for _ in range(60):
        action = np.array([2, 0, 1], dtype=np.int64)  # THREAT, ADVANCE, target=C1
        obs, _, term, trunc, _ = env.step(action)
        if float(obs["legitimacy"][0]) < 1.0:
            moved_L = True
        if abs(float(obs["theta"][0])) > 0.0:
            moved_theta = True
        if int(obs["occupation_duration"][0]) > 0:
            moved_tocc = True
        if term or trunc:
            break
    assert moved_L and moved_theta and moved_tocc
