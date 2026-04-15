"""Rule-based Defender policy.

The Defender follows a deterministic priority cascade:

1. **Defend home** -- if any Invader units occupy the Defender home
   territory, all available Defender units move to defend it. The
   Defender also issues SEEK_ALLIANCE diplomatically.
2. **Counterattack** -- if the Invader holds a contested territory
   adjacent to a Defender-controlled territory AND the Defender has
   ≥3 units in that adjacent territory, retake the weakest Invader
   position.
3. **Fortify** -- otherwise, hold position. Issue NEGOTIATE if
   ``θ < 0`` (Neutral drifting toward Invader, courting it back).

The Defender also retaliates with STRIKE if the Invader struck on the
previous step and the Defender has strike units remaining.

Determinism is intentional: the research question is about *the
Invader's* learned policy, and a stochastic Defender would obscure the
learning signal.
"""

from __future__ import annotations

from .actions import MilitaryAction, PoliticalAction
from .config import SovereignConfig
from .game_map import GameMap
from .state import GameState, CTRL_INVADER, CTRL_DEFENDER


class DefenderPolicy:
    """Deterministic priority-cascade defender."""

    def __init__(self) -> None:
        self.invader_struck_last_step: bool = False

    def reset(self) -> None:
        self.invader_struck_last_step = False

    def select_action(
        self,
        state: GameState,
        game_map: GameMap,
        config: SovereignConfig,
    ) -> tuple[PoliticalAction, MilitaryAction, int]:
        d_home = game_map.home["D"]

        # ---- Priority 1: defend home if invaded ---------------------------
        if state.invader_units[d_home] >= 1:
            return PoliticalAction.SEEK_ALLIANCE, MilitaryAction.HOLD, d_home

        # ---- Priority 1b: retaliate strikes -------------------------------
        if (
            self.invader_struck_last_step
            and state.defender_strike_units >= 1
        ):
            target = self._pick_strike_target(state, game_map)
            if target is not None:
                self.invader_struck_last_step = False  # consume
                return PoliticalAction.ISSUE_THREAT, MilitaryAction.STRIKE, target

        # ---- Priority 2: counterattack ------------------------------------
        target = self._pick_counterattack_target(state, game_map)
        if target is not None:
            return PoliticalAction.ISSUE_THREAT, MilitaryAction.ADVANCE, target

        # ---- Priority 3: fortify ------------------------------------------
        if state.theta < 0.0:
            pol = PoliticalAction.NEGOTIATE
        else:
            pol = PoliticalAction.DO_NOTHING
        return pol, MilitaryAction.HOLD, d_home

    def notify_invader_strike(self) -> None:
        """Tell the policy the Invader played STRIKE this step."""
        self.invader_struck_last_step = True

    # ---- helpers ---------------------------------------------------------

    def _pick_strike_target(
        self, state: GameState, game_map: GameMap
    ) -> int | None:
        """Highest-population Invader-held territory."""
        best: tuple[int, int] | None = None
        for tid in range(game_map.n):
            if state.territory_control[tid] == CTRL_INVADER:
                count = int(state.invader_units[tid])
                if count >= 1 and (best is None or count > best[1]):
                    best = (tid, count)
        return best[0] if best else None

    def _pick_counterattack_target(
        self, state: GameState, game_map: GameMap
    ) -> int | None:
        """Adjacent Invader-held territory with the fewest units (and we
        have ≥3 units stationed in some adjacent Defender territory)."""
        # Find Defender territories with the manpower for an attack.
        strong_def = [
            tid for tid in range(game_map.n)
            if state.territory_control[tid] == CTRL_DEFENDER
            and state.defender_units[tid] >= 3
        ]
        if not strong_def:
            return None

        candidates: list[tuple[int, int]] = []
        for src in strong_def:
            for nb in game_map.neighbors(src):
                if state.territory_control[nb] == CTRL_INVADER:
                    candidates.append((nb, int(state.invader_units[nb])))

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c[1], c[0]))
        return candidates[0][0]
