"""Political action effects.

Implements the legitimacy/posture/economy deltas from rulebook Section 6.1.
The neutral-posture component (Δθ) is *not* applied here -- it is folded
into the drift function in :pymod:`.neutral`. This module only handles the
direct, deterministic effects of the political action on the Invader's
internal state (legitimacy, supply when sanctioned).
"""

from __future__ import annotations

from ..actions import PoliticalAction
from ..config import SovereignConfig
from ..state import GameState


def apply_political_action(
    state: GameState,
    political_action: PoliticalAction,
    config: SovereignConfig,
) -> None:
    """Apply the immediate L / E effects of a political action.

    Mutates ``state`` in place. When ``use_legitimacy`` is False, all L
    updates are skipped (legitimacy is held at its initial value of 1.0).
    """
    if not config.use_legitimacy and political_action != PoliticalAction.IMPOSE_SANCTION:
        # Even with legitimacy disabled, IMPOSE_SANCTION still has an
        # economic effect on the target -- but L itself is frozen.
        return

    if political_action == PoliticalAction.SEEK_ALLIANCE:
        if config.use_legitimacy:
            state.legitimacy += config.seek_alliance_dL

    elif political_action == PoliticalAction.IMPOSE_SANCTION:
        if config.use_legitimacy:
            state.legitimacy += config.impose_sanction_dL
        # Sanctions imposed by the Invader are modelled as a small drag on
        # its own supply (the cost of running an enforcement regime).
        state.supply += config.impose_sanction_dE

    elif political_action == PoliticalAction.ISSUE_THREAT:
        if config.use_legitimacy:
            state.legitimacy += config.issue_threat_dL

    elif political_action == PoliticalAction.NEGOTIATE:
        if config.use_legitimacy:
            state.legitimacy += config.negotiate_dL

    elif political_action == PoliticalAction.DO_NOTHING:
        # Slow legitimacy decay if the Invader is already losing standing.
        if config.use_legitimacy and state.legitimacy < 0.5:
            state.legitimacy -= config.do_nothing_decay

    state.clip_legitimacy()
    state.clip_supply()
