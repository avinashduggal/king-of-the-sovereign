"""Neutral nation dynamics: drift-diffusion + threshold events + hysteresis.

Implements rulebook Section 7. The drift function ``μ(s, a)`` encodes the
deterministic response of the international community to observable
Invader behaviour; the noise term ``ε`` models the irreducible uncertainty
of international politics.

All randomness MUST flow through the ``rng`` argument (a
``numpy.random.Generator`` produced by ``gymnasium.utils.seeding``) for
seed-deterministic rollouts.
"""

from __future__ import annotations

import numpy as np

from ..actions import MilitaryAction, PoliticalAction
from ..config import SovereignConfig
from ..state import GameState


def update_posture(
    state: GameState,
    political_action: PoliticalAction,
    military_action: MilitaryAction,
    config: SovereignConfig,
    rng: np.random.Generator,
) -> None:
    """Drift-diffusion update: ``θ_{t+1} = clip(θ_t + μ + ε, -1, +1)``.

    No-op when ``use_neutral_posture`` is disabled.
    """
    if not config.use_neutral_posture:
        return

    L_term = config.alpha * (1.0 - state.legitimacy) if config.use_legitimacy else 0.0
    advance_term = config.beta if military_action == MilitaryAction.ADVANCE else 0.0
    strike_term = config.gamma if military_action == MilitaryAction.STRIKE else 0.0
    negotiate_term = -config.delta if political_action == PoliticalAction.NEGOTIATE else 0.0
    alliance_term = -config.epsilon if political_action == PoliticalAction.SEEK_ALLIANCE else 0.0
    occupation_term = (
        config.zeta * (state.occupation_duration / max(1, config.max_steps))
        if config.use_occupation_cost
        else 0.0
    )

    mu = (
        L_term
        + advance_term
        + strike_term
        + negotiate_term
        + alliance_term
        + occupation_term
    )
    noise = float(rng.normal(0.0, config.sigma))
    state.theta = float(np.clip(state.theta + mu + noise, -1.0, 1.0))


def check_threshold_events(
    state: GameState,
    config: SovereignConfig,
) -> dict:
    """Fire / lift threshold events from rulebook Section 7.3.

    Returns an info dict noting which events fired this step.
    """
    info: dict = {
        "sanctions_triggered": False,
        "sanctions_lifted": False,
        "coalition_triggered": False,
        "supply_routes_opened": False,
        "alliance_triggered": False,
    }

    if not config.use_neutral_posture:
        return info

    # ---- Sanctions (with hysteresis) -------------------------------------
    if not state.sanctions_active and state.theta > config.sanction_threshold:
        state.sanctions_active = True
        state.sanctions_below_threshold_count = 0
        info["sanctions_triggered"] = True
    elif state.sanctions_active:
        if state.theta < config.sanction_lift_threshold:
            state.sanctions_below_threshold_count += 1
            if state.sanctions_below_threshold_count >= config.sanction_lift_steps:
                state.sanctions_active = False
                state.sanctions_below_threshold_count = 0
                info["sanctions_lifted"] = True
        else:
            state.sanctions_below_threshold_count = 0

    # ---- Coalition (irreversible) ---------------------------------------
    if (
        not state.neutral_joined_defender
        and state.theta > config.coalition_threshold
    ):
        state.neutral_joined_defender = True
        # Defender's +2 unit bonus is applied by the env via
        # apply_coalition_unit_bonus() once we return.
        if config.use_legitimacy:
            state.legitimacy -= config.coalition_legitimacy_penalty
            state.clip_legitimacy()
        info["coalition_triggered"] = True

    # ---- Supply routes opened (favourable to Invader) -------------------
    if (
        not state.supply_routes_open
        and state.theta < config.supply_route_threshold
    ):
        state.supply_routes_open = True
        info["supply_routes_opened"] = True

    # ---- Formal alliance with Invader -----------------------------------
    if (
        not state.neutral_allied_invader
        and state.theta < config.alliance_threshold
    ):
        state.neutral_allied_invader = True
        if config.use_legitimacy:
            # Mild legitimacy hit: alliance is itself a controversial move.
            state.legitimacy -= config.alliance_legitimacy_penalty
            state.clip_legitimacy()
        info["alliance_triggered"] = True

    return info


def apply_coalition_unit_bonus(state: GameState, config: SovereignConfig, game_map) -> None:
    """Add the +2 unit bonus to the Defender after the coalition fires.

    Called once per step from the env *after* :pyfunc:`check_threshold_events`
    when ``info["coalition_triggered"]`` is True.
    """
    d_home = game_map.home["D"]
    state.defender_units[d_home] += config.coalition_unit_bonus
