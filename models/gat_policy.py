"""GAT-based actor-critic policy for Sovereign-v0.

Replaces the GCNConv backbone with a 2-layer GATv2Conv encoder (Brody et al.
2022) that learns per-edge attention weights conditioned on both the source and
target node.  Unlike GCN's fixed degree-based normalisation, learned attention
lets the model up-weight strategically important neighbors (e.g. central hub C4).

Architecture overview
---------------------

GATFeaturesExtractor (SB3 BaseFeaturesExtractor)
  ├── Dict obs → node_features [B, N, 5] and global_features [B, 8]
  ├── GATEncoder (2-layer GATv2Conv via torch_geometric)
  │     → node_embeddings [B, N, D]
  │     → graph_embedding  [B, D]   (mean pooling over N)
  └── Output: structured flat tensor [B, D + N*D + G]  (648 by default)

        components packed left-to-right:
            graph_embedding [D]  |  node_embs_flat [N*D]  |  global_feats [G]

GATSovereignPolicy (SB3 MultiInputActorCriticPolicy)
  ├── GATFeaturesExtractor (above) — shared actor/critic extractor
  ├── IdentityMlpExtractor — passes features unchanged; satisfies SB3 interface
  └── Asymmetric head routing (latent_pi = latent_vf = features):
        ctx = cat(graph_emb, global_feats)                   [B, D+G = 72]
        political_head(ctx)          → logits                [B, 5]
        military_head(ctx)           → logits                [B, 4]
        target_score per node        → logits                [B, N = 9]
          input: cat(node_emb, ctx) per node                 [B, N, D+D+G]
        _CriticRoutingHead(ctx)      → value                 [B, 1]

Default dimensions (D=64, N=9, G=8): features_dim = 64 + 576 + 8 = 648.
Packed tensor format is identical to GNNFeaturesExtractor so downstream heads
and SB3 integration points are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .graph_utils import build_static_edge_index, make_batch_edge_index
from .gnn_policy import IdentityMlpExtractor, _CriticRoutingHead
from .obs_adapter import GLOBAL_FEAT_DIM, NODE_FEAT_DIM

# Each directed edge (src → dst) is annotated with:
#   [resource_value(dst), strategic_value(dst)]
_EDGE_ATTR_DIM: int = 2


# ---------------------------------------------------------------------------
# Edge attribute builder
# ---------------------------------------------------------------------------

def _build_static_edge_attr(
    game_map,
    edge_index: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build per-edge static attribute tensor from GameMap territory values.

    Destination-node convention: the edge (src → dst) is annotated with
    the *destination* territory's static properties so the attention
    mechanism can directly see how strategically important the node being
    updated is — information orthogonal to the dynamic node features.

    Args:
        game_map: sovereign.game_map.GameMap instance.
        edge_index: [2, E] edge index (from build_static_edge_index).
        device: target device.

    Returns:
        edge_attr: float32 tensor [E, 2] with values already in [0, 1].
    """
    territories = game_map.territories
    rows = [
        [territories[dst].resource_value, territories[dst].strategic_value]
        for dst in edge_index[1].tolist()
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Weight init helper
# ---------------------------------------------------------------------------

def _init_gatv2_weights(conv) -> None:
    """Orthogonal init for GATv2Conv projection matrices.

    GATv2Conv has lin_l (left/source) and lin_r (right/target) projection
    matrices plus an optional lin_edge for edge attributes.  The 3-D attention
    scoring vector `att` is left at PyG's default glorot initialisation because
    orthogonal_ is undefined on rank-3 tensors.
    """
    nn.init.orthogonal_(conv.lin_l.weight)
    if conv.lin_l.bias is not None:
        nn.init.zeros_(conv.lin_l.bias)
    nn.init.orthogonal_(conv.lin_r.weight)
    if conv.lin_r.bias is not None:
        nn.init.zeros_(conv.lin_r.bias)
    if getattr(conv, "lin_edge", None) is not None:
        nn.init.orthogonal_(conv.lin_edge.weight)
        if conv.lin_edge.bias is not None:
            nn.init.zeros_(conv.lin_edge.bias)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


# ---------------------------------------------------------------------------
# GAT Encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """2-layer GATv2Conv encoder with static edge attributes.

    Layer 1 uses multi-head attention (concat=True) so each of the `heads`
    attention patterns operates in its own head_dim = hidden_dim // heads
    subspace, giving the model multiple independent views of the neighbourhood.

    Layer 2 uses a single head (concat=False) to collapse back to out_dim,
    ensuring features_dim is independent of the layer-1 head count.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = 64,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        edge_attr_dim: int = _EDGE_ATTR_DIM,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv  # noqa: PLC0415

        self.out_dim = out_dim

        # Layer 1: heads × head_dim = hidden_dim channels total after concat.
        self.conv1 = GATv2Conv(
            node_feat_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_attr_dim,
            add_self_loops=True,
            fill_value="mean",
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: single head → out_dim (features_dim independent of heads).
        self.conv2 = GATv2Conv(
            hidden_dim,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_attr_dim,
            add_self_loops=True,
            fill_value="mean",
        )
        self.norm2 = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in (self.conv1, self.conv2):
            _init_gatv2_weights(conv)

    def forward(
        self,
        node_features: torch.Tensor,  # [B, N, F]
        edge_index: torch.Tensor,     # [2, E]  single graph
        edge_attr: torch.Tensor,      # [E, edge_attr_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_embs: [B, N, out_dim]
            graph_emb: [B, out_dim]  (mean pooled over node dimension)
        """
        B, N, _ = node_features.shape

        x = node_features.reshape(B * N, -1)                    # [B*N, F]
        batch_ei = make_batch_edge_index(edge_index, B, N)      # [2, E*B]
        # Replicate edge_attr for every graph in the batch; ordering matches
        # the node-offset layout that make_batch_edge_index produces.
        batch_ea = edge_attr.repeat(B, 1)                       # [E*B, 2]

        x = F.elu(self.norm1(self.conv1(x, batch_ei, batch_ea)))  # [B*N, H]
        x = F.elu(self.norm2(self.conv2(x, batch_ei, batch_ea)))  # [B*N, D]

        node_embs = x.view(B, N, self.out_dim)   # [B, N, D]
        graph_emb = node_embs.mean(dim=1)         # [B, D]
        return node_embs, graph_emb


# ---------------------------------------------------------------------------
# SB3 FeaturesExtractor
# ---------------------------------------------------------------------------

class GATFeaturesExtractor(BaseFeaturesExtractor):
    """SB3-compatible extractor that runs GATv2Conv on the SovereignEnv Dict obs.

    Output packed tensor format is identical to GNNFeaturesExtractor:
        [graph_embedding (D) | node_embs_flat (N*D) | global_features (G)]

    Both edge_index and edge_attr are registered as buffers so they follow
    .to(device) calls automatically.
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

        features_dim = gat_output + self.n_nodes * gat_output + GLOBAL_FEAT_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.encoder = GATEncoder(
            node_feat_dim=NODE_FEAT_DIM,
            hidden_dim=gat_hidden,
            out_dim=gat_output,
            heads=gat_heads,
            dropout=gat_dropout,
        )

        edge_index = build_static_edge_index(game_map)
        self.register_buffer("edge_index", edge_index)           # [2, E]

        edge_attr = _build_static_edge_attr(game_map, edge_index)
        self.register_buffer("edge_attr", edge_attr)             # [E, 2]

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        n = self.n_nodes
        B = obs["territory_control"].shape[0]

        ctrl  = obs["territory_control"].float().view(B, n, 3)
        inv_u = obs["invader_units"].float().view(B, n, 1) / 200.0
        def_u = obs["defender_units"].float().view(B, n, 1) / 200.0
        node_feats = torch.cat([ctrl, inv_u, def_u], dim=-1)     # [B, N, 5]

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
        return torch.cat(
            [graph_emb, node_embs.flatten(1), global_feats], dim=-1
        )  # [B, features_dim]


# ---------------------------------------------------------------------------
# GAT PPO Policy
# ---------------------------------------------------------------------------

class GATSovereignPolicy(MultiInputActorCriticPolicy):
    """SB3 actor-critic policy with a GATv2Conv-based feature extractor.

    Head routing and packed tensor slicing are identical to GNNSovereignPolicy;
    only the graph encoder changes (GATv2Conv instead of GCNConv).

      ctx = cat(graph_emb, global_feats)              [B, D+G = 72]
      political_head(ctx)    → political logits        [B, 5]
      military_head(ctx)     → military logits         [B, 4]
      target head per node   → target logits           [B, N = 9]
        input: cat(node_emb, ctx) per node             [B, N, D+D+G = 136]
      value_net(features)    → state value             [B, 1]
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

        kwargs["features_extractor_class"]  = GATFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            game_map=game_map,
            gat_hidden=gat_hidden,
            gat_output=gat_output,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            max_steps=max_steps,
        )
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:  # type: ignore[override]
        super()._build(lr_schedule)

        D   = self._gat_output
        N   = self.features_extractor.n_nodes
        G   = GLOBAL_FEAT_DIM
        ctx = D + G
        Ha  = self._actor_hidden
        Hc  = self._critic_hidden

        self.mlp_extractor = IdentityMlpExtractor(self.features_dim)

        self.political_head = nn.Sequential(
            nn.Linear(ctx, Ha), nn.ReLU(),
            nn.Linear(Ha, 5),
        )
        self.military_head = nn.Sequential(
            nn.Linear(ctx, Ha), nn.ReLU(),
            nn.Linear(Ha, 4),
        )
        self.target_score = nn.Linear(D + ctx, 1)

        self.value_net  = _CriticRoutingHead(D, N, Hc)
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

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        D = self._gat_output
        N = self.features_extractor.n_nodes
        G = GLOBAL_FEAT_DIM

        graph_emb    = latent_pi[:, :D]                          # [B, D]
        node_embs    = latent_pi[:, D : D + N * D].view(-1, N, D)  # [B, N, D]
        global_feats = latent_pi[:, D + N * D:]                  # [B, G]

        ctx = torch.cat([graph_emb, global_feats], dim=-1)       # [B, D+G]

        pol = self.political_head(ctx)   # [B, 5]
        mil = self.military_head(ctx)    # [B, 4]

        B = latent_pi.shape[0]
        ctx_exp  = ctx.unsqueeze(1).expand(B, N, -1)             # [B, N, D+G]
        node_ctx = torch.cat([node_embs, ctx_exp], dim=-1)       # [B, N, D+(D+G)]
        tgt      = self.target_score(node_ctx).squeeze(-1)       # [B, N]

        logits = torch.cat([pol, mil, tgt], dim=-1)              # [B, 5+4+N]
        return self.action_dist.proba_distribution(action_logits=logits)
