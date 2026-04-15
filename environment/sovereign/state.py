"""Mutable game state and observation builder.

Controller codes used inside :pyattr:`GameState.territory_control`:

==========  ==================================
0           Invader (I)
1           Defender (D)
2           Neutral (N)
3           Contested / unoccupied
==========  ==================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SovereignConfig
from .game_map import GameMap


# Controller code constants
CTRL_INVADER = 0
CTRL_DEFENDER = 1
CTRL_NEUTRAL = 2
CTRL_CONTESTED = 3


@dataclass
class GameState:
    """Mutable simulation state.

    All numeric vectors are sized by the number of territories ``|V|``.
    The observation Dict produced by :pymeth:`to_observation` matches the
    space defined in :pymeth:`SovereignEnv._build_observation_space`.
    """

    # ---- spatial state ----------------------------------------------------
    territory_control: np.ndarray         # shape (|V|,), int8 controller code
    invader_units: np.ndarray             # shape (|V|,), int32 ground units
    defender_units: np.ndarray            # shape (|V|,), int32 ground units
    neutral_units: np.ndarray             # shape (|V|,), int32 ground units

    # ---- strike unit pools (not bound to a territory) --------------------
    invader_strike_units: int = 0
    defender_strike_units: int = 0

    # ---- scalar state ----------------------------------------------------
    legitimacy: float = 1.0
    supply: float = 1.0
    theta: float = 0.0
    occupation_duration: int = 0
    timestep: int = 0

    # ---- threshold-event flags -------------------------------------------
    sanctions_active: bool = False
    neutral_joined_defender: bool = False
    neutral_allied_invader: bool = False
    supply_routes_open: bool = False

    # ---- internal counters (not part of obs) -----------------------------
    sanctions_below_threshold_count: int = 0  # hysteresis counter
    consecutive_negotiate: int = 0            # for simple settlement trigger

    # ---- bookkeeping for reward delta ------------------------------------
    prev_invader_resource: float = 0.0

    @classmethod
    def initial(cls, game_map: GameMap, config: SovereignConfig) -> "GameState":
        """Create the canonical starting state described in the rulebook.

        Each nation begins with all its ground units stationed on its home
        territory; contested territories have no units and no controller.
        """
        n = game_map.n
        control = np.full(n, CTRL_CONTESTED, dtype=np.int8)
        i_units = np.zeros(n, dtype=np.int32)
        d_units = np.zeros(n, dtype=np.int32)
        n_units = np.zeros(n, dtype=np.int32)

        i_home = game_map.home["I"]
        d_home = game_map.home["D"]
        n_home = game_map.home["N"]

        control[i_home] = CTRL_INVADER
        control[d_home] = CTRL_DEFENDER
        control[n_home] = CTRL_NEUTRAL

        i_units[i_home] = config.invader_ground
        d_units[d_home] = config.defender_ground
        n_units[n_home] = config.neutral_ground

        return cls(
            territory_control=control,
            invader_units=i_units,
            defender_units=d_units,
            neutral_units=n_units,
            invader_strike_units=config.invader_strike,
            defender_strike_units=config.defender_strike,
        )

    # ---- queries ---------------------------------------------------------

    def total_invader_units(self) -> int:
        return int(self.invader_units.sum()) + self.invader_strike_units

    def total_defender_units(self) -> int:
        return int(self.defender_units.sum()) + self.defender_strike_units

    def invader_controlled(self, game_map: GameMap) -> set[int]:
        return {
            t.id for t in game_map.territories
            if self.territory_control[t.id] == CTRL_INVADER
        }

    def invader_holds_only_home(self, game_map: GameMap) -> bool:
        i_home = game_map.home["I"]
        owned = self.invader_controlled(game_map)
        return owned == {i_home}

    # ---- observation -----------------------------------------------------

    def to_observation(self, game_map: GameMap) -> dict[str, np.ndarray]:
        """Build the observation dict matching ``observation_space``.

        The territory_control field is one-hot encoded over {I, D, N};
        contested territories produce an all-zero one-hot row, which is
        valid MultiBinary content.
        """
        n = game_map.n
        one_hot = np.zeros((n, 3), dtype=np.int8)
        for tid in range(n):
            c = int(self.territory_control[tid])
            if c < 3:  # skip contested
                one_hot[tid, c] = 1

        return {
            "territory_control": one_hot.flatten(),
            "invader_units": self.invader_units.astype(np.int32, copy=True),
            "defender_units": self.defender_units.astype(np.int32, copy=True),
            "legitimacy": np.array([self.legitimacy], dtype=np.float32),
            "supply": np.array([self.supply], dtype=np.float32),
            "theta": np.array([self.theta], dtype=np.float32),
            "occupation_duration": np.array(
                [self.occupation_duration], dtype=np.int32
            ),
            "timestep": np.array([self.timestep], dtype=np.int32),
            "sanctions_active": np.array(
                [int(self.sanctions_active)], dtype=np.int8
            ),
            "neutral_joined_defender": np.array(
                [int(self.neutral_joined_defender)], dtype=np.int8
            ),
            "neutral_allied_invader": np.array(
                [int(self.neutral_allied_invader)], dtype=np.int8
            ),
        }

    # ---- helpers ---------------------------------------------------------

    def clip_legitimacy(self) -> None:
        self.legitimacy = float(np.clip(self.legitimacy, 0.0, 1.0))

    def clip_supply(self) -> None:
        self.supply = float(np.clip(self.supply, 0.0, 1.0))

    def clip_theta(self) -> None:
        self.theta = float(np.clip(self.theta, -1.0, 1.0))
