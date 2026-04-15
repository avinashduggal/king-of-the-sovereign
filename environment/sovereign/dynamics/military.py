"""Military action resolution and combat.

Implements ADVANCE, HOLD, WITHDRAW, STRIKE for the Invader plus a
minimal combat resolver used when the Invader and Defender contest the
same territory in a single step.

Design notes
------------
* All actions are *target-aware*: the agent picks a territory id. Invalid
  targets degrade to HOLD and the env records ``info["action_was_degraded"]``.
* ADVANCE moves a fixed fraction (half, ≥1 unit) of units from the
  nearest Invader-held neighbour into the target territory.
* WITHDRAW always cedes one Invader-held non-home territory (the one
  with the fewest units). The vacated units retreat to the Invader home.
  ``t_occ`` resets only when the Invader holds *only* its home territory
  after the withdrawal.
* STRIKE consumes one Invader strike unit and destroys one Defender
  unit from the targeted territory.
"""

from __future__ import annotations

from ..actions import MilitaryAction
from ..config import SovereignConfig
from ..game_map import GameMap
from ..state import (
    GameState,
    CTRL_INVADER,
    CTRL_DEFENDER,
    CTRL_NEUTRAL,
    CTRL_CONTESTED,
)


def resolve_invader_military(
    state: GameState,
    military_action: MilitaryAction,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
) -> dict:
    """Apply the Invader's military action to the state.

    Returns an info dict containing diagnostic flags such as
    ``action_was_degraded`` and ``units_destroyed``.
    """
    info: dict = {
        "action_was_degraded": False,
        "units_destroyed": 0,
        "territory_captured": None,
        "territory_ceded": None,
    }

    if military_action == MilitaryAction.ADVANCE:
        ok = _resolve_advance(state, target, game_map, config, info)
        if not ok:
            info["action_was_degraded"] = True

    elif military_action == MilitaryAction.STRIKE:
        ok = _resolve_strike(state, target, game_map, config, info)
        if not ok:
            info["action_was_degraded"] = True

    elif military_action == MilitaryAction.WITHDRAW:
        _resolve_withdraw(state, target, game_map, config, info)

    elif military_action == MilitaryAction.HOLD:
        pass  # no-op; t_occ tick handled by env after this returns

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_advance(
    state: GameState,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
    info: dict,
) -> bool:
    """Move units into ``target`` from the strongest adjacent Invader holding.

    Returns True if the advance succeeded, False if it had to be skipped.
    """
    if target < 0 or target >= game_map.n:
        return False
    # Already Invader-controlled -- no-op (treat as degraded).
    if state.territory_control[target] == CTRL_INVADER:
        return False

    # Find adjacent Invader-held source territories.
    sources = [
        nb for nb in game_map.neighbors(target)
        if state.territory_control[nb] == CTRL_INVADER
        and state.invader_units[nb] >= 1
    ]
    if not sources:
        return False

    # Pick the source with the most units (and break ties by id for determinism).
    sources.sort(key=lambda s: (-int(state.invader_units[s]), s))
    src = sources[0]

    # Move half (rounded up, ≥1) of the source's units.
    available = int(state.invader_units[src])
    moving = max(1, available // 2)

    state.invader_units[src] -= moving
    state.invader_units[target] += moving

    # Apply legitimacy hit (Section 6.2). Only if legitimacy is active.
    if config.use_legitimacy:
        state.legitimacy += config.advance_dL
        state.clip_legitimacy()

    # Resolve combat at the target if Defender or Neutral units are present.
    _resolve_contested_territory(state, target, game_map, config)

    info["territory_captured"] = (
        target if state.territory_control[target] == CTRL_INVADER else None
    )
    return True


def _resolve_strike(
    state: GameState,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
    info: dict,
) -> bool:
    if target < 0 or target >= game_map.n:
        return False
    if state.invader_strike_units < 1:
        return False
    if state.defender_units[target] < 1:
        return False

    state.invader_strike_units -= 1
    state.defender_units[target] -= 1
    info["units_destroyed"] = 1

    if config.use_legitimacy:
        state.legitimacy += config.strike_dL
        state.clip_legitimacy()

    # If a defender's territory was cleared by the strike and the Invader
    # already has units adjacent, mark it contested so a future ADVANCE
    # can take it (don't auto-flip control on a strike).
    if (
        state.territory_control[target] == CTRL_DEFENDER
        and state.defender_units[target] == 0
    ):
        state.territory_control[target] = CTRL_CONTESTED

    return True


def _resolve_withdraw(
    state: GameState,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
    info: dict,
) -> None:
    """Cede one Invader-held non-home territory.

    If ``target`` is a valid Invader-held non-home territory, withdraw from
    that one; otherwise withdraw from the Invader-held non-home territory
    with the fewest units. Vacated units retreat to the Invader home.
    """
    i_home = game_map.home["I"]
    held_non_home = [
        t.id for t in game_map.territories
        if state.territory_control[t.id] == CTRL_INVADER and t.id != i_home
    ]
    if not held_non_home:
        return  # nothing to withdraw from

    if 0 <= target < game_map.n and target in held_non_home:
        ceded = target
    else:
        ceded = min(
            held_non_home, key=lambda t: int(state.invader_units[t])
        )

    # Move all Invader units from the ceded territory back to home.
    state.invader_units[i_home] += state.invader_units[ceded]
    state.invader_units[ceded] = 0
    state.territory_control[ceded] = CTRL_CONTESTED

    if config.use_legitimacy:
        state.legitimacy += config.withdraw_dL
        state.clip_legitimacy()

    info["territory_ceded"] = ceded


def _resolve_contested_territory(
    state: GameState,
    tid: int,
    game_map: GameMap,
    config: SovereignConfig,
) -> None:
    """Resolve combat at ``tid`` after Invader units have moved in.

    Simple deterministic rule: whichever side has more *effective* units
    takes (or keeps) control. Effective unit count applies the
    home-territory bonus (Section 2.1) for the Defender on its home tile.
    Losing units are reduced by the difference, capped at zero.
    """
    inv = int(state.invader_units[tid])
    dfd = int(state.defender_units[tid])
    ntr = int(state.neutral_units[tid])

    if dfd == 0 and ntr == 0:
        # Empty territory -- Invader simply takes control.
        if inv > 0:
            state.territory_control[tid] = CTRL_INVADER
        return

    inv_eff = inv  # Invader gets no home bonus when attacking
    dfd_eff = dfd
    if dfd > 0 and game_map.territories[tid].home_of == "D":
        dfd_eff = int(round(dfd * (1.0 + config.defender_home_bonus)))

    # Simple deterministic combat: the difference removes that many units
    # from the loser; the winner retains the remainder.
    if inv_eff > dfd_eff + ntr:
        losses = max(1, dfd_eff + ntr)
        state.defender_units[tid] = max(0, dfd - losses)
        state.neutral_units[tid] = max(0, ntr - max(0, losses - dfd))
        # Reduce attacker by the defender's effective strength (cost of attack)
        attacker_loss = max(0, dfd_eff + ntr - 1)
        state.invader_units[tid] = max(0, inv - attacker_loss)
        if state.invader_units[tid] > 0:
            state.territory_control[tid] = CTRL_INVADER
        else:
            state.territory_control[tid] = CTRL_CONTESTED
    elif dfd_eff > inv_eff:
        # Defender holds; Invader loses all attacking units here.
        state.invader_units[tid] = 0
        # Defender takes a small loss too (cost of defending).
        attacker_strength = max(1, inv_eff)
        state.defender_units[tid] = max(0, dfd - max(0, attacker_strength - 1))
        if state.defender_units[tid] > 0:
            state.territory_control[tid] = CTRL_DEFENDER
        else:
            state.territory_control[tid] = CTRL_CONTESTED
    else:
        # Tie -- territory becomes contested, both sides take equal losses.
        loss = max(1, min(inv, dfd))
        state.invader_units[tid] = max(0, inv - loss)
        state.defender_units[tid] = max(0, dfd - loss)
        state.territory_control[tid] = CTRL_CONTESTED


def update_occupation_duration(
    state: GameState,
    game_map: GameMap,
    config: SovereignConfig,
) -> None:
    """Tick (or reset) the occupation counter at the end of the step.

    The counter increments by 1 whenever the Invader holds at least one
    non-home territory. It resets to 0 only when the Invader has fully
    retreated to its own home territory (i.e., its only owned territory
    is the home one).

    With ``use_occupation_cost=False`` the counter is held at zero.
    """
    if not config.use_occupation_cost:
        state.occupation_duration = 0
        return

    if state.invader_holds_only_home(game_map):
        state.occupation_duration = 0
    else:
        state.occupation_duration += 1


def apply_defender_action(
    state: GameState,
    political_action,
    military_action: MilitaryAction,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
) -> dict:
    """Apply the Defender's chosen action.

    Symmetric to :pyfunc:`resolve_invader_military` but for the Defender.
    Defender ADVANCE moves units into adjacent Defender-controlled or
    contested territories; STRIKE destroys one Invader unit at the target.
    Returns an info dict for diagnostics.
    """
    info: dict = {"defender_action_degraded": False, "defender_struck": 0}

    if military_action == MilitaryAction.ADVANCE:
        ok = _defender_advance(state, target, game_map, config)
        if not ok:
            info["defender_action_degraded"] = True

    elif military_action == MilitaryAction.STRIKE:
        if (
            state.defender_strike_units >= 1
            and 0 <= target < game_map.n
            and state.invader_units[target] >= 1
        ):
            state.defender_strike_units -= 1
            state.invader_units[target] -= 1
            info["defender_struck"] = 1
        else:
            info["defender_action_degraded"] = True

    # Defender HOLD / WITHDRAW are no-ops at this MVP level: the rule
    # policy mostly uses ADVANCE/STRIKE/HOLD to defend.
    return info


def _defender_advance(
    state: GameState,
    target: int,
    game_map: GameMap,
    config: SovereignConfig,
) -> bool:
    if target < 0 or target >= game_map.n:
        return False
    if state.territory_control[target] == CTRL_DEFENDER:
        return False

    sources = [
        nb for nb in game_map.neighbors(target)
        if state.territory_control[nb] == CTRL_DEFENDER
        and state.defender_units[nb] >= 1
    ]
    if not sources:
        return False
    sources.sort(key=lambda s: (-int(state.defender_units[s]), s))
    src = sources[0]
    moving = max(1, int(state.defender_units[src]) // 2)
    state.defender_units[src] -= moving
    state.defender_units[target] += moving

    # Resolve combat (mirror of invader resolution, swapping roles).
    inv = int(state.invader_units[target])
    dfd = int(state.defender_units[target])
    if inv == 0:
        state.territory_control[target] = CTRL_DEFENDER
    else:
        # Defender attacking Invader-held territory: no home bonus for
        # either side here.
        if dfd > inv:
            state.invader_units[target] = 0
            state.defender_units[target] = max(1, dfd - max(0, inv - 1))
            state.territory_control[target] = CTRL_DEFENDER
        elif inv > dfd:
            state.defender_units[target] = 0
            state.invader_units[target] = max(1, inv - max(0, dfd - 1))
        else:
            loss = max(1, min(inv, dfd))
            state.invader_units[target] = max(0, inv - loss)
            state.defender_units[target] = max(0, dfd - loss)
            state.territory_control[target] = CTRL_CONTESTED
    return True
