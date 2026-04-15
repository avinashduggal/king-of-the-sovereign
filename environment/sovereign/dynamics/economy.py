"""Supply / economy updates.

The Invader's supply index ``E ∈ [0, 1]`` reflects logistical health.
This module:

* Recomputes ``E`` based on which territories are connected back to the
  Invader's home (Section 3.2: occupied non-contiguous territories
  generate no resource value).
* Drains ``E`` while sanctions are active (Section 7.3 threshold event).
"""

from __future__ import annotations

from ..config import SovereignConfig
from ..game_map import GameMap
from ..state import GameState, CTRL_INVADER


def update_supply(
    state: GameState,
    game_map: GameMap,
    config: SovereignConfig,
) -> None:
    """Refresh ``E`` and apply sanction drain.

    Supply is computed as: base supply + bonus for each territory in the
    Invader's connected home component (weighted by resource_value).
    Disconnected occupied territories contribute *nothing* (they require
    extra logistics that cost more than they yield).
    """
    i_home = game_map.home["I"]
    controlled = set(state.invader_controlled(game_map))
    if i_home not in controlled:
        # Invader has lost its home territory -- supply collapses fast.
        state.supply = max(0.0, state.supply - 0.10)
        state.clip_supply()
        return

    connected = game_map.connected_component(i_home, controlled)
    contributing = sum(
        game_map.territories[tid].resource_value for tid in connected
    )

    # Base supply level slowly regenerates toward 1.0.
    state.supply += 0.01 * (1.0 - state.supply)
    # Connected resources add a small bonus (capped by clip).
    state.supply += 0.01 * contributing

    # Sanctions drain (Section 7.3).
    if state.sanctions_active:
        state.supply -= config.sanction_drain_rate

    state.clip_supply()
