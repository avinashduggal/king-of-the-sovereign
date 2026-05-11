"""Recurrent GAT policy for Sovereign-v0 — GATv2Conv encoder + LSTM.

Architecture overview
---------------------

GATRecurrentFeaturesExtractor (SB3 BaseFeaturesExtractor)
  ├── Dict obs → node_features [B, N, 5]  and  global_features [B, 8]
  ├── GATEncoder (GATv2Conv, same as gat_policy.py)
  │     → node_embeddings [B, N, D]   (stashed for target-head routing)
  │     → graph_embedding [B, D]      (mean pool over N)
  └── Output: cat(graph_emb, global_feats)  [B, D+G = 72]
      ← compact LSTM input; only 72 dims carry the temporal signal

GATRecurrentPolicy (RecurrentMultiInputActorCriticPolicy)
  ├── GATRecurrentFeaturesExtractor (features_dim = 72)
  ├── LSTM (72 → lstm_hidden) — managed by RecurrentActorCriticPolicy base class
  │     → episode resets and sequence padding handled automatically
  ├── IdentityMlpExtractor — passes LSTM output unchanged
  └── Asymmetric head routing:
        pol   = political_head(lstm_out)               [B, 5]
        mil   = military_head(lstm_out)                [B, 4]
        tgt   = target_score(cat(node_embs, lstm_exp)) [B, N]
          where node_embs comes from the stash set by the extractor
        value = value_net(lstm_out)                    [B, 1]

Default dims: D=64, G=8, N=9, L=lstm_hidden=128
  features_dim = D + G = 72  (LSTM input)
  target_score input dim = D + L = 192
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .gat_policy import GATEncoder, _build_static_edge_attr
from .gnn_policy import IdentityMlpExtractor
from .graph_utils import build_static_edge_index
from .obs_adapter import GLOBAL_FEAT_DIM, NODE_FEAT_DIM


# ---------------------------------------------------------------------------
# Features extractor
# ---------------------------------------------------------------------------

class GATRecurrentFeaturesExtractor(BaseFeaturesExtractor):
    """Dict-obs extractor for the recurrent GAT policy.

    Outputs cat(graph_emb, global_feats) [B, D+G=72] as the compact temporal
    signal that feeds the LSTM.  Node embeddings [B, N, D] are stashed as
    self._last_node_embs so the policy's target-score head can read them
    immediately after _process_sequence returns.
    """

    def __init__(
        self,
        observation_space,
        game_map,
        gat_hidden: int = 64,
        gat_output: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        max_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        self.n_nodes: int = game_map.n
        self.gat_output: int = gat_output
        self._max_steps: float = float(max_steps)
        self._last_node_embs: torch.Tensor | None = None

        # features_dim = D + G — only this compact vector goes to the LSTM.
        features_dim = gat_output + GLOBAL_FEAT_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.encoder = GATEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            hidden_dim=gat_hidden,
            out_dim=gat_output,
            heads=gat_heads,
            dropout=gat_dropout,
        )

        edge_index = build_static_edge_index(game_map)
        self.register_buffer("edge_index", edge_index)              # [2, E]

        edge_attr = _build_static_edge_attr(game_map, edge_index)
        self.register_buffer("edge_attr", edge_attr)                # [E, 2]

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        n = self.n_nodes
        B = obs["territory_control"].shape[0]

        ctrl  = obs["territory_control"].float().view(B, n, 3)
        inv_u = obs["invader_units"].float().view(B, n, 1) / 200.0
        def_u = obs["defender_units"].float().view(B, n, 1) / 200.0
        node_feats = torch.cat([ctrl, inv_u, def_u], dim=-1)        # [B, N, 5]

        global_feats = torch.stack([
            obs["legitimacy"][:, 0].float(),
            obs["supply"][:, 0].float(),
            obs["theta"][:, 0].float(),
            obs["occupation_duration"][:, 0].float() / self._max_steps,
            obs["timestep"][:, 0].float() / self._max_steps,
            obs["sanctions_active"][:, 0].float(),
            obs["neutral_joined_defender"][:, 0].float(),
            obs["neutral_allied_invader"][:, 0].float(),
        ], dim=1)  # [B, 8]

        node_embs, graph_emb = self.encoder(
            node_feats, self.edge_index, self.edge_attr
        )
        # Stash node_embs so the policy head can read them after LSTM.
        # Safe because extract_features() is always called before
        # _get_action_dist_from_latent() within a single forward pass.
        self._last_node_embs = node_embs                            # [B, N, D]

        return torch.cat([graph_emb, global_feats], dim=-1)         # [B, D+G]


# ---------------------------------------------------------------------------
# Recurrent GAT policy
# ---------------------------------------------------------------------------

class GATRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    """RecurrentPPO policy with GATv2Conv encoder + LSTM temporal memory.

    The GAT encoder produces a per-step graph embedding which feeds the LSTM.
    The LSTM output is then routed to asymmetric actor heads (political,
    military, per-node target) and a simple MLP value head.

    All LSTM state management (episode resets, sequence padding, rollout
    buffering) is handled automatically by the RecurrentPPO base class.

    Pass lstm_hidden_size and n_lstm_layers via policy_kwargs to RecurrentPPO,
    NOT as top-level constructor arguments:
        RecurrentPPO(GATRecurrentPolicy, env, policy_kwargs=dict(
            game_map=..., lstm_hidden_size=128, ...
        ))
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        game_map=None,
        gat_hidden: int = 64,
        gat_output: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        max_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        # Cache before super().__init__, which calls _build() immediately.
        self._gat_output    = gat_output
        self._actor_hidden  = actor_hidden
        self._critic_hidden = critic_hidden

        # Disable parent's ortho init — we do our own in _build().
        kwargs.setdefault("ortho_init", False)

        kwargs["features_extractor_class"]  = GATRecurrentFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            game_map=game_map,
            gat_hidden=gat_hidden,
            gat_output=gat_output,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            max_steps=max_steps,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ------------------------------------------------------------------
    # MLP extractor override (called from ActorCriticPolicy._build)
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self) -> None:
        # self.lstm_output_dim is set by RecurrentActorCriticPolicy.__init__
        # before super().__init__() calls _build() → _build_mlp_extractor().
        self.mlp_extractor = IdentityMlpExtractor(self.lstm_output_dim)

    # ------------------------------------------------------------------
    # Build custom actor / critic heads
    # ------------------------------------------------------------------

    def _build(self, lr_schedule) -> None:
        # Calls _build_mlp_extractor() (our override above), then creates
        # default action_net and value_net based on mlp_extractor dims.
        super()._build(lr_schedule)

        L  = self.lstm_output_dim   # lstm hidden size (default 128)
        D  = self._gat_output       # GATEncoder output dim (default 64)
        Ha = self._actor_hidden     # actor MLP hidden (default 64)
        Hc = self._critic_hidden    # critic MLP hidden (default 64)

        # --- actor heads (take lstm_out [B, L] as context) ---
        self.political_head = nn.Sequential(
            nn.Linear(L, Ha), nn.ReLU(),
            nn.Linear(Ha, 5),
        )
        self.military_head = nn.Sequential(
            nn.Linear(L, Ha), nn.ReLU(),
            nn.Linear(Ha, 4),
        )
        # Per-node: cat(node_emb [D], lstm_out [L]) → scalar score
        self.target_score = nn.Linear(D + L, 1)

        # --- critic: simple MLP on lstm_out ---
        self.value_net = nn.Sequential(
            nn.Linear(L, Hc), nn.ReLU(),
            nn.Linear(Hc, 1),
        )

        # action_net is bypassed by _get_action_dist_from_latent; keep valid.
        self.action_net = nn.Identity()

        def _ortho(module: nn.Module, gain: float = 0.01) -> None:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=gain)
                        nn.init.zeros_(m.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

        _ortho(self.political_head)
        _ortho(self.military_head)
        _ortho(self.target_score)
        _ortho(self.value_net)

        # NOTE: Do NOT rebuild optimizer here.
        # RecurrentActorCriticPolicy.__init__ creates the LSTMs and rebuilds
        # the optimizer AFTER _build() returns, correctly capturing all params.

    # ------------------------------------------------------------------
    # Asymmetric action routing
    # ------------------------------------------------------------------

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """Route LSTM output to asymmetric actor heads.

        latent_pi: [B, L] — LSTM output passed through IdentityMlpExtractor.
        node_embs: [B, N, D] — retrieved from features extractor stash.
        """
        D = self._gat_output
        N = self.features_extractor.n_nodes

        node_embs = self.features_extractor._last_node_embs  # [B, N, D]

        pol = self.political_head(latent_pi)   # [B, 5]
        mil = self.military_head(latent_pi)    # [B, 4]

        B = latent_pi.shape[0]
        lstm_exp  = latent_pi.unsqueeze(1).expand(B, N, -1)          # [B, N, L]
        node_ctx  = torch.cat([node_embs, lstm_exp], dim=-1)         # [B, N, D+L]
        tgt       = self.target_score(node_ctx).squeeze(-1)          # [B, N]

        logits = torch.cat([pol, mil, tgt], dim=-1)                  # [B, 5+4+N]
        return self.action_dist.proba_distribution(action_logits=logits)
