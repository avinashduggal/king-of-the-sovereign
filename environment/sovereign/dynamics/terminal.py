"""Terminal-condition checks (rulebook Section 9) plus settlement trigger.

Three settlement-trigger variants are kept side-by-side here. The first
("simple consecutive negotiation") is **active**; the other two are
preserved as commented blocks. Switching is a one-paragraph edit.
"""

from __future__ import annotations

from ..actions import MilitaryAction, PoliticalAction
from ..config import SovereignConfig
from ..game_map import GameMap
from ..state import GameState, CTRL_INVADER


def check_terminal(
    state: GameState,
    political_action: PoliticalAction,
    military_action: MilitaryAction,
    game_map: GameMap,
    config: SovereignConfig,
) -> tuple[bool, float, str]:
    """Return ``(terminated, terminal_reward, reason)``.

    Examined in priority order: political collapse, military defeat,
    negotiated settlement, total conquest.
    """
    # 1. Political collapse (only if legitimacy mechanic is active)
    if config.use_legitimacy and state.legitimacy <= 0.0:
        return True, config.terminal_political_collapse, "political_collapse"

    # 2. Military defeat
    if state.total_invader_units() <= 0:
        return True, config.terminal_military_defeat, "military_defeat"

    # 3. Negotiated settlement
    settled, reward = _check_settlement(state, political_action, military_action, game_map, config)
    if settled:
        return True, reward, "negotiated_settlement"

    # 4. Total conquest
    n = game_map.n
    invader_count = sum(
        1 for tid in range(n) if state.territory_control[tid] == CTRL_INVADER
    )
    if invader_count == n:
        return True, config.terminal_total_conquest, "total_conquest"

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Settlement triggers
# ---------------------------------------------------------------------------

def _check_settlement(
    state: GameState,
    political_action: PoliticalAction,
    military_action: MilitaryAction,
    game_map: GameMap,
    config: SovereignConfig,
) -> tuple[bool, float]:
    """Active settlement mechanic: simple consecutive negotiation.

    Triggers when the Invader plays NEGOTIATE for ``settlement_consecutive_steps``
    consecutive turns while keeping legitimacy above ``settlement_min_legitimacy``.
    """
    if (
        political_action == PoliticalAction.NEGOTIATE
        and state.legitimacy > config.settlement_min_legitimacy
    ):
        state.consecutive_negotiate += 1
    else:
        state.consecutive_negotiate = 0

    if state.consecutive_negotiate >= config.settlement_consecutive_steps:
        return True, config.terminal_negotiated_settlement
    return False, 0.0


# ---------------------------------------------------------------------------
# Alternative settlement triggers (kept as reference for future experiments)
# ---------------------------------------------------------------------------
#
# ---- ALTERNATIVE 1: Strict diplomatic posture --------------------------
# def _check_settlement_strict(state, political_action, military_action,
#                              game_map, config):
#     """Settlement only after sustained diplomatic credibility.
#
#     Trigger when ALL of the following hold simultaneously on the same step:
#       - political_action == NEGOTIATE
#       - military_action  in {HOLD, WITHDRAW}
#       - state.legitimacy >= 0.60
#       - state.theta < 0.40
#       - Invader does NOT control Defender's home territory
#       - state.timestep >= 10
#     """
#     d_home = game_map.home["D"]
#     if (
#         political_action == PoliticalAction.NEGOTIATE
#         and military_action in (MilitaryAction.HOLD, MilitaryAction.WITHDRAW)
#         and state.legitimacy >= 0.60
#         and state.theta < 0.40
#         and state.territory_control[d_home] != CTRL_INVADER
#         and state.timestep >= 10
#     ):
#         return True, config.terminal_negotiated_settlement
#     return False, 0.0
#
# ---- ALTERNATIVE 2: Defender-initiated --------------------------------
# Implementation sketch:
#   1. In defender.py, set ``state.settlement_offered = True`` and
#      ``state.offer_expires_at = state.timestep + 3`` whenever:
#         - state.legitimacy > 0.55
#         - Invader controls no Defender-home territory
#         - state.theta < 0.6
#   2. Add a ``settlement_offered`` MultiBinary(1) field to the
#      observation Dict in env.py + state.py so the agent can see the offer.
#   3. Replace the body of ``_check_settlement`` with:
#         if (
#             state.settlement_offered
#             and political_action == PoliticalAction.NEGOTIATE
#             and state.timestep <= state.offer_expires_at
#         ):
#             return True, config.terminal_negotiated_settlement
#         # offer expires
#         if state.timestep > state.offer_expires_at:
#             state.settlement_offered = False
#         return False, 0.0
