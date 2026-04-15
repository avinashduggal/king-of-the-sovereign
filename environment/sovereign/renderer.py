"""Text/ASCII rendering of the game state.

Used by ``SovereignEnv.render()`` for both ``"human"`` and ``"ansi"``
modes. No external dependencies (no pygame, no matplotlib).
"""

from __future__ import annotations

from .actions import military_name, political_name
from .config import SovereignConfig
from .game_map import GameMap
from .state import (
    GameState,
    CTRL_INVADER,
    CTRL_DEFENDER,
    CTRL_NEUTRAL,
    CTRL_CONTESTED,
)


_CTRL_GLYPH = {
    CTRL_INVADER: "I",
    CTRL_DEFENDER: "D",
    CTRL_NEUTRAL: "N",
    CTRL_CONTESTED: "?",
}


def render_text(
    state: GameState,
    game_map: GameMap,
    config: SovereignConfig,
    last_action: tuple[int, int, int] | None = None,
    last_reward: float | None = None,
    cumulative_reward: float | None = None,
) -> str:
    """Format the current state as a multi-line string."""
    n = game_map.n
    lines: list[str] = []
    lines.append(f"=== SOVEREIGN  Step {state.timestep}/{config.max_steps} ===")
    lines.append(
        f"L: {state.legitimacy:.2f}  E: {state.supply:.2f}  "
        f"theta: {state.theta:+.2f}  t_occ: {state.occupation_duration}"
    )
    flags = []
    if state.sanctions_active:
        flags.append("SANCTIONS")
    if state.neutral_joined_defender:
        flags.append("COALITION")
    if state.supply_routes_open:
        flags.append("SUPPLY-OPEN")
    if state.neutral_allied_invader:
        flags.append("ALLIED-I")
    lines.append("Flags: " + (", ".join(flags) if flags else "none"))
    lines.append("")

    lines.append("Territories:")
    for tid in range(n):
        t = game_map.territories[tid]
        ctrl = int(state.territory_control[tid])
        glyph = _CTRL_GLYPH[ctrl]
        i_u = int(state.invader_units[tid])
        d_u = int(state.defender_units[tid])
        n_u = int(state.neutral_units[tid])
        home = f" (home of {t.home_of})" if t.home_of else ""
        lines.append(
            f"  [{tid}] {t.name:<7} ctrl={glyph}  I={i_u:2d} D={d_u:2d} N={n_u:2d}"
            f"  res={t.resource_value:.2f}{home}"
        )

    lines.append("")
    lines.append(
        f"Invader strike units: {state.invader_strike_units}  |  "
        f"Defender strike units: {state.defender_strike_units}"
    )

    if last_action is not None:
        pol, mil, tgt = last_action
        lines.append(
            f"Last invader action: {political_name(pol)} + "
            f"{military_name(mil)} (target territory {tgt})"
        )
    if last_reward is not None:
        cum_str = (
            f"  Cumulative: {cumulative_reward:.2f}"
            if cumulative_reward is not None
            else ""
        )
        lines.append(f"Reward: {last_reward:+.3f}{cum_str}")

    return "\n".join(lines)
