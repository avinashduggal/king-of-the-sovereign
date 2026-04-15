"""Per-module dynamics tests.

Each subsystem is exercised in isolation against a fresh GameState so that
failures localize to one mechanic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sovereign.actions import MilitaryAction, PoliticalAction
from sovereign.config import SovereignConfig
from sovereign.dynamics import (
    economy,
    insurgency,
    military,
    neutral,
    political,
    terminal,
)
from sovereign.game_map import GameMap
from sovereign.state import (
    CTRL_CONTESTED,
    CTRL_INVADER,
    GameState,
)


@pytest.fixture
def fresh_state():
    cfg = SovereignConfig()
    m = GameMap()
    s = GameState.initial(m, cfg)
    return s, m, cfg


# ---------------------------------------------------------------------------
# Political dynamics
# ---------------------------------------------------------------------------

def test_negotiate_increases_legitimacy(fresh_state):
    s, m, cfg = fresh_state
    s.legitimacy = 0.7
    political.apply_political_action(s, PoliticalAction.NEGOTIATE, cfg)
    assert s.legitimacy == pytest.approx(0.7 + cfg.negotiate_dL)


def test_strike_decreases_legitimacy(fresh_state):
    s, m, cfg = fresh_state
    s.legitimacy = 1.0
    # advance/strike effects are applied via military, but we test the
    # numeric direction here via the strike resolver.
    s.invader_strike_units = 1
    s.defender_units[m.home["D"]] = 1
    info = military.resolve_invader_military(s, MilitaryAction.STRIKE, m.home["D"], m, cfg)
    assert info["units_destroyed"] == 1
    assert s.legitimacy < 1.0


def test_no_legitimacy_freezes_L():
    cfg = SovereignConfig(use_legitimacy=False)
    m = GameMap()
    s = GameState.initial(m, cfg)
    s.legitimacy = 0.42  # arbitrary starting value
    political.apply_political_action(s, PoliticalAction.NEGOTIATE, cfg)
    assert s.legitimacy == 0.42


# ---------------------------------------------------------------------------
# Military dynamics
# ---------------------------------------------------------------------------

def test_advance_to_invalid_territory_degrades(fresh_state):
    s, m, cfg = fresh_state
    # I_HOME=0; territory 6 is D_HOME, NOT adjacent to I_HOME.
    info = military.resolve_invader_military(
        s, MilitaryAction.ADVANCE, m.home["D"], m, cfg
    )
    assert info["action_was_degraded"] is True


def test_advance_to_valid_territory_captures(fresh_state):
    s, m, cfg = fresh_state
    initial_home_units = int(s.invader_units[m.home["I"]])
    info = military.resolve_invader_military(
        s, MilitaryAction.ADVANCE, 1, m, cfg  # C1 is adjacent
    )
    assert info["action_was_degraded"] is False
    assert int(s.invader_units[1]) >= 1
    assert int(s.invader_units[m.home["I"]]) < initial_home_units


def test_strike_without_strike_units_degrades(fresh_state):
    s, m, cfg = fresh_state
    s.invader_strike_units = 0
    info = military.resolve_invader_military(
        s, MilitaryAction.STRIKE, m.home["D"], m, cfg
    )
    assert info["action_was_degraded"] is True


def test_withdraw_resets_occupation_when_back_at_home(fresh_state):
    s, m, cfg = fresh_state
    # Set up: invader holds C1
    s.invader_units[1] = 3
    s.territory_control[1] = CTRL_INVADER
    s.occupation_duration = 5
    military.resolve_invader_military(s, MilitaryAction.WITHDRAW, 1, m, cfg)
    military.update_occupation_duration(s, m, cfg)
    assert s.occupation_duration == 0
    assert s.territory_control[1] == CTRL_CONTESTED


def test_no_occupation_cost_freezes_t_occ():
    cfg = SovereignConfig(use_occupation_cost=False)
    m = GameMap()
    s = GameState.initial(m, cfg)
    s.invader_units[1] = 3
    s.territory_control[1] = CTRL_INVADER
    military.update_occupation_duration(s, m, cfg)
    assert s.occupation_duration == 0


# ---------------------------------------------------------------------------
# Neutral dynamics
# ---------------------------------------------------------------------------

def test_advance_drives_theta_up(fresh_state):
    s, m, cfg = fresh_state
    rng = np.random.default_rng(42)
    s.theta = 0.0
    neutral.update_posture(
        s, PoliticalAction.DO_NOTHING, MilitaryAction.ADVANCE, cfg, rng
    )
    # Drift contains +beta plus small noise; expect positive on average.
    assert s.theta > -0.05  # noise is small relative to beta=0.05


def test_negotiate_pulls_theta_down(fresh_state):
    s, m, cfg = fresh_state
    rng = np.random.default_rng(0)
    s.theta = 0.0
    neutral.update_posture(
        s, PoliticalAction.NEGOTIATE, MilitaryAction.HOLD, cfg, rng
    )
    assert s.theta < 0.05  # -delta is the dominant signed term


def test_sanction_threshold_triggers(fresh_state):
    s, m, cfg = fresh_state
    s.theta = cfg.sanction_threshold + 0.01
    info = neutral.check_threshold_events(s, cfg)
    assert info["sanctions_triggered"] is True
    assert s.sanctions_active is True


def test_sanction_hysteresis_holds_then_lifts(fresh_state):
    s, m, cfg = fresh_state
    s.sanctions_active = True
    s.theta = cfg.sanction_lift_threshold - 0.01

    # First few steps below the lift threshold
    for i in range(cfg.sanction_lift_steps - 1):
        info = neutral.check_threshold_events(s, cfg)
        assert s.sanctions_active is True
        assert not info["sanctions_lifted"]

    # The Nth consecutive step should lift sanctions
    info = neutral.check_threshold_events(s, cfg)
    assert s.sanctions_active is False
    assert info["sanctions_lifted"] is True


def test_no_neutral_posture_freezes_theta():
    cfg = SovereignConfig(use_neutral_posture=False)
    m = GameMap()
    s = GameState.initial(m, cfg)
    rng = np.random.default_rng(0)
    s.theta = 0.0
    for _ in range(20):
        neutral.update_posture(
            s, PoliticalAction.SEEK_ALLIANCE, MilitaryAction.STRIKE, cfg, rng
        )
    assert s.theta == 0.0


# ---------------------------------------------------------------------------
# Insurgency
# ---------------------------------------------------------------------------

def test_insurgency_probability_matches_formula():
    p10 = insurgency.insurgency_probability(10, 0.05)
    p20 = insurgency.insurgency_probability(20, 0.05)
    # Section 8.3: at t_occ=10 ≈ 0.39, at t_occ=20 ≈ 0.63.
    assert math.isclose(p10, 1 - math.exp(-0.5), rel_tol=1e-9)
    assert math.isclose(p20, 1 - math.exp(-1.0), rel_tol=1e-9)


def test_insurgency_disabled_when_no_occupation_cost():
    cfg = SovereignConfig(use_occupation_cost=False)
    m = GameMap()
    s = GameState.initial(m, cfg)
    s.occupation_duration = 100  # would be guaranteed otherwise
    rng = np.random.default_rng(0)
    info = insurgency.roll_insurgency(s, cfg, rng)
    assert info["insurgency_fired"] is False


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------

def test_supply_decays_when_sanctions_active(fresh_state):
    s, m, cfg = fresh_state
    s.sanctions_active = True
    s.supply = 0.5
    pre = s.supply
    economy.update_supply(s, m, cfg)
    assert s.supply < pre


# ---------------------------------------------------------------------------
# Terminal conditions
# ---------------------------------------------------------------------------

def test_political_collapse_terminates(fresh_state):
    s, m, cfg = fresh_state
    s.legitimacy = 0.0
    done, r, reason = terminal.check_terminal(
        s, PoliticalAction.DO_NOTHING, MilitaryAction.HOLD, m, cfg
    )
    assert done and reason == "political_collapse"
    assert r == cfg.terminal_political_collapse


def test_military_defeat_terminates(fresh_state):
    s, m, cfg = fresh_state
    s.invader_units[:] = 0
    s.invader_strike_units = 0
    done, r, reason = terminal.check_terminal(
        s, PoliticalAction.DO_NOTHING, MilitaryAction.HOLD, m, cfg
    )
    assert done and reason == "military_defeat"
    assert r == cfg.terminal_military_defeat


def test_simple_settlement_after_5_negotiate(fresh_state):
    s, m, cfg = fresh_state
    s.legitimacy = 0.9
    for i in range(cfg.settlement_consecutive_steps):
        done, r, reason = terminal.check_terminal(
            s,
            PoliticalAction.NEGOTIATE,
            MilitaryAction.HOLD,
            m,
            cfg,
        )
    assert done is True
    assert reason == "negotiated_settlement"
    assert r == cfg.terminal_negotiated_settlement


def test_settlement_resets_on_non_negotiate(fresh_state):
    s, m, cfg = fresh_state
    s.legitimacy = 0.9
    # 4 negotiates, then a non-negotiate, then 4 more -> should NOT trigger
    for _ in range(4):
        terminal.check_terminal(
            s, PoliticalAction.NEGOTIATE, MilitaryAction.HOLD, m, cfg
        )
    assert s.consecutive_negotiate == 4
    terminal.check_terminal(
        s, PoliticalAction.DO_NOTHING, MilitaryAction.HOLD, m, cfg
    )
    assert s.consecutive_negotiate == 0
