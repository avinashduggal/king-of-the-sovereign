"""GNN-based actor-critic policy for Sovereign-v0.

Architecture overview
---------------------

GNNFeaturesExtractor (SB3 BaseFeaturesExtractor)
  ├── Dict obs → node_features [B, N, 5] and global_features [B, 8]
  ├── GNNEncoder (2-layer GCNConv via torch_geometric)
  │     → node_embeddings [B, N, D]
  │     → graph_embedding  [B, D]   (mean pooling over N)
  └── Output: structured flat tensor [B, D + N*D + G]  (648 by default)

        components packed left-to-right:
            graph_embedding [D]  |  node_embs_flat [N*D]  |  global_feats [G]

GNNSovereignPolicy (SB3 MultiInputActorCriticPolicy)
  ├── GNNFeaturesExtractor (above) — shared actor/critic extractor
  ├── IdentityMlpExtractor — passes features unchanged; satisfies SB3 interface
  └── Asymmetric head routing (latent_pi = latent_vf = features):
        ctx = cat(graph_emb, global_feats)                   [B, D+G = 72]
        political_head(ctx)          → logits                [B, 5]
        military_head(ctx)           → logits                [B, 4]
        target_score per node        → logits                [B, N = 9]
          input: cat(node_emb, ctx) per node                 [B, N, D+D+G]
        _CriticRoutingHead(ctx)      → value                 [B, 1]

Default dimensions (D=64, N=9, G=8): features_dim = 64 + 576 + 8 = 648.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .graph_utils import build_static_edge_index, make_batch_edge_index
from .obs_adapter import GLOBAL_FEAT_DIM, NODE_FEAT_DIM


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """2-layer GCN encoder using torch_geometric GCNConv.

    Processes a batch of node-feature matrices sharing the same graph
    topology (static Sovereign map). Returns per-node embeddings and a
    mean-pooled graph embedding.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        # Deferred import so the module can be imported before torch_geometric
        # is installed (e.g. during pyproject.toml introspection).
        from torch_geometric.nn import GCNConv  # noqa: PLC0415

        self.output_dim = output_dim

        # GCNConv adds self-loops and applies Kipf-Welling D^{-1/2} A D^{-1/2}
        # normalisation internally when add_self_loops=True and normalize=True.
        self.conv1 = GCNConv(node_feat_dim, hidden_dim,
                             add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, output_dim,
                             add_self_loops=True, normalize=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # GCNConv (PyG ≥ 2.0) exposes its linear transform as conv.lin
        for conv in (self.conv1, self.conv2):
            lin = getattr(conv, "lin", None)
            if lin is not None:
                nn.init.orthogonal_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

    def forward(
        self,
        node_features: torch.Tensor,  # [B, N, F]
        edge_index: torch.Tensor,     # [2, E]  single-graph, no self-loops
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: batched node feature matrices [B, N, F].
            edge_index: single-graph edge index [2, E].  The same topology is
                        used for every graph in the batch.

        Returns:
            node_embeddings: [B, N, D]
            graph_embedding:  [B, D]  (mean over node dimension)
        """
        B, N, _ = node_features.shape

        # Flatten batch: treat B graphs as one disconnected super-graph [B*N, F]
        x = node_features.reshape(B * N, -1)
        batch_ei = make_batch_edge_index(edge_index, B, N)  # [2, E*B]

        x = F.relu(self.norm1(self.conv1(x, batch_ei)))     # [B*N, H]
        x = F.relu(self.norm2(self.conv2(x, batch_ei)))     # [B*N, D]

        node_embs = x.view(B, N, self.output_dim)           # [B, N, D]
        graph_emb = node_embs.mean(dim=1)                   # [B, D]
        return node_embs, graph_emb


# ---------------------------------------------------------------------------
# SB3 FeaturesExtractor
# ---------------------------------------------------------------------------

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """SB3-compatible extractor that runs the GNN on the SovereignEnv Dict obs.

    Output is a structured flat tensor packed as:
        [graph_embedding (D) | node_embs_flat (N*D) | global_features (G)]

    This single vector is carried through SB3's mlp_extractor (which is an
    identity pass-through) and then sliced back into its components by
    GNNSovereignPolicy for asymmetric head routing.
    """

    def __init__(
        self,
        observation_space,
        game_map,
        gnn_hidden: int = 64,
        gnn_output: int = 64,
        max_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        self.n_nodes: int = game_map.n
        self.gnn_output: int = gnn_output
        self._max_steps: float = float(max_steps)

        features_dim = gnn_output + self.n_nodes * gnn_output + GLOBAL_FEAT_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.encoder = GNNEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            hidden_dim=gnn_hidden,
            output_dim=gnn_output,
        )

        # Register as buffer so edge_index moves with .to(device) calls
        edge_index = build_static_edge_index(game_map)
        self.register_buffer("edge_index", edge_index)  # [2, E]

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert Dict obs (torch tensors from SB3) to structured features.

        SB3 pre-converts numpy obs to torch tensors before calling forward,
        so we work directly with tensors here (no numpy round-trip).

        Returns:
            features: [B, D + N*D + G]
        """
        n = self.n_nodes
        B = obs["territory_control"].shape[0]

        # Node features [B, N, 5]
        # territory_control: [B, n*3] flattened one-hot → [B, N, 3]
        ctrl  = obs["territory_control"].float().view(B, n, 3)
        inv_u = obs["invader_units"].float().view(B, n, 1) / 200.0
        def_u = obs["defender_units"].float().view(B, n, 1) / 200.0
        node_feats = torch.cat([ctrl, inv_u, def_u], dim=-1)  # [B, N, 5]

        # Global features [B, 8]
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

        node_embs, graph_emb = self.encoder(node_feats, self.edge_index)
        return torch.cat(
            [graph_emb, node_embs.flatten(1), global_feats], dim=-1
        )  # [B, features_dim]


# ---------------------------------------------------------------------------
# SB3 MlpExtractor shim
# ---------------------------------------------------------------------------

class IdentityMlpExtractor(nn.Module):
    """Pass-through replacement for SB3's MlpExtractor.

    SB3 requires mlp_extractor to expose latent_dim_pi, latent_dim_vf, and
    forward / forward_actor / forward_critic methods.  This shim returns
    features unchanged; all routing logic lives in the policy overrides.
    """

    def __init__(self, features_dim: int) -> None:
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return features, features

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


# ---------------------------------------------------------------------------
# Critic routing head (value_net replacement)
# ---------------------------------------------------------------------------

class _CriticRoutingHead(nn.Module):
    """Value head that routes through graph_emb + global_feats.

    Replaces SB3's default value_net so the standard code paths
    (forward, evaluate_actions, predict_values) all compute values correctly
    without any extra overrides.
    """

    def __init__(self, gnn_output: int, n_nodes: int, hidden_dim: int) -> None:
        super().__init__()
        self._D = gnn_output
        self._N = n_nodes
        ctx_dim = gnn_output + GLOBAL_FEAT_DIM
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """Slice graph_emb and global_feats from structured features, compute value.

        Args:
            latent_vf: [B, D + N*D + G] — the full structured features tensor.

        Returns:
            value: [B, 1]
        """
        D, N = self._D, self._N
        graph_emb    = latent_vf[:, :D]           # [B, D]
        global_feats = latent_vf[:, D + N * D:]   # [B, G]
        ctx = torch.cat([graph_emb, global_feats], dim=-1)
        return self.mlp(ctx)  # [B, 1]


# ---------------------------------------------------------------------------
# GNN PPO Policy
# ---------------------------------------------------------------------------

class GNNSovereignPolicy(MultiInputActorCriticPolicy):
    """SB3 actor-critic policy with an asymmetric GNN-based architecture.

    The GNNFeaturesExtractor encodes the Dict observation into a structured
    flat tensor [graph_emb | node_embs_flat | global_feats].  The policy
    slices this tensor to route different components to different heads:

      ctx = cat(graph_emb, global_feats)              [B, D+G = 72]
      political_head(ctx)    → political logits        [B, 5]
      military_head(ctx)     → military logits         [B, 4]
      target head per node   → target logits           [B, N = 9]
        input: cat(node_emb, ctx) per node             [B, N, D+D+G = 136]
      value_net(features)    → state value             [B, 1]

    Compatible with SB3 PPO's standard training loop; no custom trainer needed.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        game_map=None,
        gnn_hidden: int = 64,
        gnn_output: int = 64,
        actor_hidden: int = 64,
        critic_hidden: int = 64,
        max_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        # Cache before super().__init__, which calls _build() immediately
        self._gnn_output   = gnn_output
        self._actor_hidden = actor_hidden
        self._critic_hidden = critic_hidden

        kwargs["features_extractor_class"]  = GNNFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            game_map=game_map,
            gnn_hidden=gnn_hidden,
            gnn_output=gnn_output,
            max_steps=max_steps,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:  # type: ignore[override]
        """Build GNN actor/critic heads, replacing SB3's default MLP components."""
        # Let SB3 create features_extractor, default mlp_extractor, action_net,
        # value_net, and optimizer.  We replace the components below.
        super()._build(lr_schedule)

        D   = self._gnn_output
        N   = self.features_extractor.n_nodes
        G   = GLOBAL_FEAT_DIM
        ctx = D + G          # graph_emb || global_feats
        Ha  = self._actor_hidden
        Hc  = self._critic_hidden

        # --- replace mlp_extractor with identity pass-through ---
        # self.features_dim is set in ActorCriticPolicy.__init__ before _build()
        self.mlp_extractor = IdentityMlpExtractor(self.features_dim)

        # --- actor heads ---
        self.political_head = nn.Sequential(
            nn.Linear(ctx, Ha), nn.ReLU(),
            nn.Linear(Ha, 5),
        )
        self.military_head = nn.Sequential(
            nn.Linear(ctx, Ha), nn.ReLU(),
            nn.Linear(Ha, 4),
        )
        # Per-node scoring: each node receives (node_emb [D] concat ctx [D+G])
        self.target_score = nn.Linear(D + ctx, 1)

        # --- critic: replace value_net with routing head ---
        self.value_net = _CriticRoutingHead(D, N, Hc)

        # action_net is never reached (overridden in _get_action_dist_from_latent),
        # but SB3 may introspect it; keep a valid identity module.
        self.action_net = nn.Identity()

        # Orthogonal init (small gain on output layers, PPO best practice)
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

        # Rebuild the optimizer so it captures all newly-added parameters
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """Asymmetric routing to political, military, and target actor heads.

        latent_pi is the full structured features vector [B, D + N*D + G]
        (IdentityMlpExtractor passes it through unchanged).
        """
        D = self._gnn_output
        N = self.features_extractor.n_nodes
        G = GLOBAL_FEAT_DIM

        graph_emb    = latent_pi[:, :D]                      # [B, D]
        node_embs    = latent_pi[:, D : D + N * D].view(-1, N, D)  # [B, N, D]
        global_feats = latent_pi[:, D + N * D:]              # [B, G]

        ctx = torch.cat([graph_emb, global_feats], dim=-1)   # [B, D+G]

        pol = self.political_head(ctx)   # [B, 5]
        mil = self.military_head(ctx)    # [B, 4]

        B = latent_pi.shape[0]
        ctx_exp  = ctx.unsqueeze(1).expand(B, N, -1)          # [B, N, D+G]
        node_ctx = torch.cat([node_embs, ctx_exp], dim=-1)    # [B, N, D+(D+G)]
        tgt      = self.target_score(node_ctx).squeeze(-1)    # [B, N]

        # TODO: action masking extension point.
        # If valid target masks are available (e.g. from node_feats cols 0-2
        # + edge_index), apply them here before passing to proba_distribution:
        #   tgt = tgt + (1 - target_mask) * -1e9

        logits = torch.cat([pol, mil, tgt], dim=-1)           # [B, 5+4+N]
        return self.action_dist.proba_distribution(action_logits=logits)
