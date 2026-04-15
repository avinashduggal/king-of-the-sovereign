"""``SovereignEnv``: the main Gymnasium environment.

Wires every subsystem into the canonical 12-step turn order
(rulebook Section 5):

    1.  Observe state s_t
    2.  Invader selects political action  a_pol
    3.  Invader selects military action   a_mil
    4.  Defender responds (rule-based)
    5.  Resolve military outcomes (deterministic)
    6.  Update territory control map M
    7.  Update L, E, t_occ
    8.  Sample neutral posture shift  Δθ
    9.  Check threshold events (sanctions, coalition)
    10. Check terminal conditions
    11. Compute reward r_t
    12. Emit  (s_{t+1}, r_t, done, info)

Key external behaviour
----------------------
* observation_space: ``spaces.Dict`` (see ``_build_observation_space``).
* action_space:      ``spaces.MultiDiscrete([5, 4, |V|])``.
* All randomness flows through ``self.np_random`` so seeded rollouts are
  bit-reproducible across machines.
* Construct directly, via ``gymnasium.make("Sovereign-v0")``, or via one
  of the registered ablation variants (``Sovereign-no_legitimacy-v0`` ...).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import (
    MilitaryAction,
    N_MILITARY,
    N_POLITICAL,
    PoliticalAction,
    military_name,
    political_name,
)
from .config import SovereignConfig, get_preset
from .defender import DefenderPolicy
from .dynamics import economy, insurgency, military, neutral, political, terminal
from .game_map import GameMap
from .renderer import render_text
from .reward import compute_invader_resources, compute_step_reward
from .state import GameState


class SovereignEnv(gym.Env):
    """SOVEREIGN: three-nation strategic-conflict environment.

    Parameters
    ----------
    config:
        Pre-built :class:`SovereignConfig`. Takes precedence over
        ``config_preset``.
    config_preset:
        Name of a preset (``"full"``, ``"no_legitimacy"``,
        ``"no_occupation_cost"``, ``"no_neutral_posture"``,
        ``"baseline"``). Used when ``config`` is None. Defaults to
        ``"full"`` if neither is supplied.
    game_map:
        Custom :class:`GameMap`. Defaults to the 9-node default topology.
    render_mode:
        ``"human"``, ``"ansi"``, or None.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: SovereignConfig | None = None,
        config_preset: str | None = None,
        game_map: GameMap | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if config is not None:
            self.config: SovereignConfig = config
        elif config_preset is not None:
            self.config = get_preset(config_preset)
        else:
            self.config = SovereignConfig()

        self.game_map: GameMap = game_map if game_map is not None else GameMap()
        self.render_mode = render_mode

        self.action_space: spaces.MultiDiscrete = spaces.MultiDiscrete(
            [N_POLITICAL, N_MILITARY, self.game_map.n]
        )
        self.observation_space: spaces.Dict = self._build_observation_space()

        self.defender = DefenderPolicy()
        self.state: GameState = GameState.initial(self.game_map, self.config)

        # Diagnostics for render()
        self._last_action: tuple[int, int, int] | None = None
        self._last_reward: float = 0.0
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------ API

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)  # seeds self.np_random
        self.state = GameState.initial(self.game_map, self.config)
        self.state.prev_invader_resource = compute_invader_resources(
            self.state, self.game_map
        )
        self.defender.reset()
        self._last_action = None
        self._last_reward = 0.0
        self._cumulative_reward = 0.0

        info: dict[str, Any] = {"config_preset": self._infer_preset_name()}
        return self.state.to_observation(self.game_map), info

    def step(
        self, action: np.ndarray | tuple[int, int, int]
    ) -> tuple[
        dict[str, np.ndarray], float, bool, bool, dict[str, Any]
    ]:
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.shape != (3,):
            raise ValueError(
                f"Action must be a length-3 sequence; got shape {action_arr.shape}"
            )
        political_idx = int(action_arr[0])
        military_idx = int(action_arr[1])
        target = int(action_arr[2])

        if not (0 <= political_idx < N_POLITICAL):
            raise ValueError(f"Invalid political action: {political_idx}")
        if not (0 <= military_idx < N_MILITARY):
            raise ValueError(f"Invalid military action: {military_idx}")
        if not (0 <= target < self.game_map.n):
            raise ValueError(f"Invalid target territory: {target}")

        political_action = PoliticalAction(political_idx)
        military_action = MilitaryAction(military_idx)

        # Steps 2-3 already happened (the agent picked the action).
        # Snapshot prev resources before any state mutation this step.
        self.state.prev_invader_resource = compute_invader_resources(
            self.state, self.game_map
        )

        info: dict[str, Any] = {}

        # Step 4: Defender chooses its action.
        d_pol, d_mil, d_target = self.defender.select_action(
            self.state, self.game_map, self.config
        )

        # Step 5+6: Resolve military outcomes (Invader first, then Defender).
        political.apply_political_action(self.state, political_action, self.config)
        inv_info = military.resolve_invader_military(
            self.state, military_action, target, self.game_map, self.config
        )
        info.update(inv_info)
        if military_action == MilitaryAction.STRIKE:
            self.defender.notify_invader_strike()

        # Defender side
        political.apply_political_action(self.state, d_pol, self.config)  # cheap symmetry
        def_info = military.apply_defender_action(
            self.state, d_pol, d_mil, d_target, self.game_map, self.config
        )
        info.update(def_info)

        # Step 7: Update L, E, t_occ.
        military.update_occupation_duration(self.state, self.game_map, self.config)
        economy.update_supply(self.state, self.game_map, self.config)

        # Step 8: Sample neutral posture shift.
        neutral.update_posture(
            self.state, political_action, military_action, self.config, self.np_random
        )

        # Step 9: Check threshold events.
        threshold_info = neutral.check_threshold_events(self.state, self.config)
        info.update(threshold_info)
        if threshold_info["coalition_triggered"]:
            neutral.apply_coalition_unit_bonus(self.state, self.config, self.game_map)

        # Insurgency roll (also part of step 7 / 8 conceptually).
        insurgency_info = insurgency.roll_insurgency(
            self.state, self.config, self.np_random
        )
        info.update(insurgency_info)

        # Step 10: Terminal conditions.
        self.state.timestep += 1
        terminated, terminal_reward, reason = terminal.check_terminal(
            self.state, political_action, military_action, self.game_map, self.config
        )
        truncated = (
            (not terminated) and self.state.timestep >= self.config.max_steps
        )
        if truncated:
            reason = "time_limit"

        # Step 11: Compute reward.
        step_reward, breakdown = compute_step_reward(
            self.state,
            self.game_map,
            self.config,
            insurgency_info["insurgency_fired"],
        )
        reward = step_reward + (terminal_reward if terminated else 0.0)

        info["reward_breakdown"] = breakdown
        info["terminal_reward"] = terminal_reward if terminated else 0.0
        info["termination_reason"] = reason
        info["defender_action"] = (int(d_pol), int(d_mil), int(d_target))

        self._last_action = (political_idx, military_idx, target)
        self._last_reward = float(reward)
        self._cumulative_reward += float(reward)

        # Step 12: Emit transition.
        return (
            self.state.to_observation(self.game_map),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self) -> str | None:
        if self.render_mode is None:
            return None
        text = render_text(
            self.state,
            self.game_map,
            self.config,
            last_action=self._last_action,
            last_reward=self._last_reward,
            cumulative_reward=self._cumulative_reward,
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text  # "ansi"

    def close(self) -> None:  # noqa: D401 - gymnasium API
        return None

    # ---- helpers --------------------------------------------------------

    def _build_observation_space(self) -> spaces.Dict:
        n = self.game_map.n
        return spaces.Dict(
            {
                "territory_control": spaces.MultiBinary(n * 3),
                "invader_units": spaces.Box(
                    low=0, high=200, shape=(n,), dtype=np.int32
                ),
                "defender_units": spaces.Box(
                    low=0, high=200, shape=(n,), dtype=np.int32
                ),
                "legitimacy": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "supply": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "theta": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "occupation_duration": spaces.Box(
                    low=0,
                    high=self.config.max_steps,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "timestep": spaces.Box(
                    low=0,
                    high=self.config.max_steps,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "sanctions_active": spaces.MultiBinary(1),
                "neutral_joined_defender": spaces.MultiBinary(1),
                "neutral_allied_invader": spaces.MultiBinary(1),
            }
        )

    def _infer_preset_name(self) -> str:
        from .config import (
            ABLATION_PRESETS,
        )  # local to avoid circular at import time

        for name, factory in ABLATION_PRESETS.items():
            if factory() == self.config:
                return name
        return "custom"

    # ---- diagnostics ----------------------------------------------------

    def action_meaning(
        self, action: np.ndarray | tuple[int, int, int]
    ) -> str:
        a = np.asarray(action).reshape(-1)
        return (
            f"{political_name(int(a[0]))} + {military_name(int(a[1]))} "
            f"-> territory {int(a[2])}"
        )
