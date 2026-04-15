"""Reward function (rulebook Section 8).

``r_t = r_pos(s_t, a_t) - r_neg(s_t, a_t)``

Positive terms
--------------
* w_T · Σ resource_value(v) for v controlled by Invader
* w_R · Δ(controlled resources) since previous step

Negative terms
--------------
* w_O · t_occ / T_max
* w_L · (1 - L)
* w_S · 1[θ > sanction_threshold] · (1 - E)
* w_I · Bernoulli(p(insurgency | t_occ))    -- handled in env via flag

Weights and toggles flow from :pyclass:`SovereignConfig`.
"""

from __future__ import annotations

from .config import SovereignConfig
from .game_map import GameMap
from .state import GameState, CTRL_INVADER


def compute_invader_resources(
    state: GameState, game_map: GameMap
) -> float:
    """Sum of ``resource_value`` for Invader-controlled territories.

    Disconnected occupied territories do not contribute (they yield no
    resources per Section 3.2). 'Disconnected' = not in the connected
    component of Invader's home territory.
    """
    i_home = game_map.home["I"]
    if state.territory_control[i_home] != CTRL_INVADER:
        return 0.0
    controlled = state.invader_controlled(game_map)
    connected = game_map.connected_component(i_home, controlled)
    return float(
        sum(game_map.territories[tid].resource_value for tid in connected)
    )


def compute_step_reward(
    state: GameState,
    game_map: GameMap,
    config: SovereignConfig,
    insurgency_fired: bool,
) -> tuple[float, dict]:
    """Compute the per-step reward and a per-term breakdown for logging.

    The territory-control component uses the *post-step* state. The Δ
    resources component uses ``state.prev_invader_resource`` set by the
    env at the start of the step.
    """
    breakdown: dict[str, float] = {}

    # ---- positive ---------------------------------------------------------
    current_resources = compute_invader_resources(state, game_map)
    territory_term = config.w_territory * current_resources
    breakdown["territory"] = territory_term

    delta_resources = max(0.0, current_resources - state.prev_invader_resource)
    capture_term = config.w_resource_capture * delta_resources
    breakdown["resource_capture"] = capture_term

    # ---- negative ---------------------------------------------------------
    if config.use_occupation_cost:
        occupation_term = (
            config.w_occupation
            * state.occupation_duration
            / max(1, config.max_steps)
        )
    else:
        occupation_term = 0.0
    breakdown["occupation"] = -occupation_term

    if config.use_legitimacy:
        legitimacy_term = config.w_legitimacy * (1.0 - state.legitimacy)
    else:
        legitimacy_term = 0.0
    breakdown["legitimacy"] = -legitimacy_term

    if state.sanctions_active and config.use_neutral_posture:
        sanction_term = config.w_sanction * (1.0 - state.supply)
    else:
        sanction_term = 0.0
    breakdown["sanction"] = -sanction_term

    insurgency_term = (
        config.w_insurgency if (insurgency_fired and config.use_occupation_cost) else 0.0
    )
    breakdown["insurgency"] = -insurgency_term

    reward = (
        territory_term
        + capture_term
        - occupation_term
        - legitimacy_term
        - sanction_term
        - insurgency_term
    )

    # Update bookkeeping for next step's Δ.
    state.prev_invader_resource = current_resources

    return reward, breakdown
