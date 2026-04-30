"""Graph utilities for the Sovereign GNN pipeline.

Converts the GameMap adjacency dict into PyTorch tensors compatible with
torch_geometric's GNN layers. All graph-building operations live here so
the rest of the pipeline only sees tensors.
"""

from __future__ import annotations

import torch


def adj_dict_to_edge_index(
    adjacency: dict[int, list[int]],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert an adjacency dict to a torch_geometric-style edge_index.

    Args:
        adjacency: node_id → list of neighbor_ids. The GameMap enforces
                   symmetry, so every directed edge already appears in both
                   directions.
        device: target device for the returned tensor.

    Returns:
        edge_index: LongTensor shape [2, num_directed_edges].
                    Self-loops are NOT included; GCNConv handles that internally.
                    Default 9-node map → shape [2, 22].
    """
    rows: list[int] = []
    cols: list[int] = []
    for src, neighbors in adjacency.items():
        for dst in neighbors:
            rows.append(src)
            cols.append(dst)
    return torch.tensor([rows, cols], dtype=torch.long, device=device)


def make_batch_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """Create a batched edge_index for B identical graphs.

    Standard torch_geometric batching: nodes in graph i are offset by i*N,
    so all B graphs are concatenated into one large disconnected graph.

    Args:
        edge_index: [2, E] single-graph edge index (no self-loops).
        batch_size: B — number of graph copies (one per env).
        num_nodes: N — nodes per graph.

    Returns:
        batched_edge_index: [2, E*B] with node-id offsets applied.
    """
    E = edge_index.shape[1]
    # offsets[i] = i * num_nodes, tiled across E edges for graph i
    offsets = (
        torch.arange(batch_size, device=edge_index.device)
        .repeat_interleave(E)   # [B*E]
        .mul(num_nodes)
    )
    return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)


def build_static_edge_index(
    game_map,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return edge_index built from a GameMap's adjacency dict.

    Args:
        game_map: sovereign.game_map.GameMap instance.
        device: target device.

    Returns:
        edge_index: [2, E]. Default 9-node map → [2, 22].
    """
    return adj_dict_to_edge_index(game_map.adjacency, device=device)
