"""Observation adapter for the Sovereign GNN pipeline.

Converts the SovereignEnv Dict observation (numpy arrays) into PyTorch
tensors for the GNN encoder. Used by smoke tests and standalone eval scripts;
GNNFeaturesExtractor works directly with the torch tensors SB3 provides.

Node features — per territory, dynamic only, shape [N, 5]:
  col 0  is_invader_ctrl    territory_control one-hot col 0
  col 1  is_defender_ctrl   territory_control one-hot col 1
  col 2  is_neutral_ctrl    territory_control one-hot col 2
  col 3  invader_units / 200.0   normalised to [0, 1]
  col 4  defender_units / 200.0  normalised to [0, 1]

Global features — scalar game state, shape [8]:
  0  legitimacy                      float [0, 1]
  1  supply                          float [0, 1]
  2  theta                           float [-1, 1]
  3  occupation_duration / max_steps float [0, 1]
  4  timestep / max_steps            float [0, 1]
  5  float(sanctions_active)         {0.0, 1.0}
  6  float(neutral_joined_defender)  {0.0, 1.0}
  7  float(neutral_allied_invader)   {0.0, 1.0}
"""

from __future__ import annotations

import numpy as np
import torch

# Exported constants used by gnn_policy.py and config_gnn.py
NODE_FEAT_DIM: int = 5
GLOBAL_FEAT_DIM: int = 8


class ObsAdapter:
    """Converts SovereignEnv Dict observations to (node_features, global_features)."""

    def __init__(
        self,
        game_map,
        max_steps: int = 200,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Args:
            game_map: sovereign.game_map.GameMap — used only for n_nodes.
            max_steps: SovereignConfig.max_steps for normalising occupation_duration
                       and timestep. Must match the config the env was created with.
            device: target device for returned tensors.
        """
        self.n = game_map.n
        self._max_steps = float(max_steps)
        self.device = torch.device(device) if isinstance(device, str) else device

    def to_tensors(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a single-env observation dict to tensors.

        Args:
            obs: dict from env.reset() / env.step(); values are numpy arrays
                 with shapes defined by SovereignEnv._build_observation_space.

        Returns:
            node_features:  float32 tensor [N, 5]
            global_features: float32 tensor [8]
        """
        n = self.n

        # --- node features ---
        # territory_control is (n*3,) flattened one-hot; reshape to (n, 3)
        ctrl = obs["territory_control"].astype(np.float32).reshape(n, 3)
        inv_u = (obs["invader_units"].astype(np.float32) / 200.0).reshape(n, 1)
        def_u = (obs["defender_units"].astype(np.float32) / 200.0).reshape(n, 1)
        node_np = np.concatenate([ctrl, inv_u, def_u], axis=1)  # [N, 5]

        # --- global features ---
        global_np = np.array([
            float(obs["legitimacy"][0]),
            float(obs["supply"][0]),
            float(obs["theta"][0]),
            float(obs["occupation_duration"][0]) / self._max_steps,
            float(obs["timestep"][0]) / self._max_steps,
            float(obs["sanctions_active"][0]),
            float(obs["neutral_joined_defender"][0]),
            float(obs["neutral_allied_invader"][0]),
        ], dtype=np.float32)  # [8]

        return (
            torch.from_numpy(node_np).to(self.device),
            torch.from_numpy(global_np).to(self.device),
        )

    def to_tensors_batched(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a vectorised-env observation dict (from DummyVecEnv) to tensors.

        DummyVecEnv adds a leading batch dimension B to each key.

        Args:
            obs: dict where each value has shape [B, *original_shape].

        Returns:
            node_features:  float32 tensor [B, N, 5]
            global_features: float32 tensor [B, 8]
        """
        n = self.n
        B = obs["territory_control"].shape[0]

        ctrl = obs["territory_control"].astype(np.float32).reshape(B, n, 3)
        inv_u = (obs["invader_units"].astype(np.float32) / 200.0).reshape(B, n, 1)
        def_u = (obs["defender_units"].astype(np.float32) / 200.0).reshape(B, n, 1)
        node_np = np.concatenate([ctrl, inv_u, def_u], axis=2)  # [B, N, 5]

        global_np = np.stack([
            obs["legitimacy"][:, 0].astype(np.float32),
            obs["supply"][:, 0].astype(np.float32),
            obs["theta"][:, 0].astype(np.float32),
            obs["occupation_duration"][:, 0].astype(np.float32) / self._max_steps,
            obs["timestep"][:, 0].astype(np.float32) / self._max_steps,
            obs["sanctions_active"][:, 0].astype(np.float32),
            obs["neutral_joined_defender"][:, 0].astype(np.float32),
            obs["neutral_allied_invader"][:, 0].astype(np.float32),
        ], axis=1)  # [B, 8]

        return (
            torch.from_numpy(node_np).to(self.device),
            torch.from_numpy(global_np).to(self.device),
        )
