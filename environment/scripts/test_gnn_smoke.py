"""Smoke test for the GNN-PPO pipeline.

Instantiates the env, runs one forward pass through every component,
samples an action, and steps the env.  Verifies tensor shapes at each
stage.  No training — completes in a few seconds.

Usage:
    python scripts/test_gnn_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the models/ package importable from king-of-the-sovereign/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    import torch
    from stable_baselines3 import PPO

    import sovereign  # noqa: F401

    from _train_common import make_env, make_vec_env
    from models.graph_utils import build_static_edge_index, make_batch_edge_index
    from models.gnn_policy import GNNEncoder, GNNFeaturesExtractor, GNNSovereignPolicy
    from models.obs_adapter import GLOBAL_FEAT_DIM, NODE_FEAT_DIM, ObsAdapter
    from sovereign.game_map import GameMap

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    def check(label: str, condition: bool, detail: str = "") -> None:
        status = PASS if condition else FAIL
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")
        if not condition:
            raise AssertionError(f"FAILED: {label}{suffix}")

    print("\n=== Sovereign GNN smoke test ===\n")

    # ---- 1. Graph utilities --------------------------------------------------------
    print("1. graph_utils")
    game_map = GameMap()
    edge_index = build_static_edge_index(game_map)
    check(
        "edge_index shape [2, 22]",
        edge_index.shape == (2, 22),
        str(tuple(edge_index.shape)),
    )
    check("edge_index dtype int64", edge_index.dtype == torch.int64)

    batch_ei = make_batch_edge_index(edge_index, batch_size=4, num_nodes=9)
    check(
        "batched edge_index shape [2, 88]",
        batch_ei.shape == (2, 88),
        str(tuple(batch_ei.shape)),
    )
    print()

    # ---- 2. Observation adapter ----------------------------------------------------
    print("2. obs_adapter (single env)")
    env = make_env(preset="full")
    obs, _ = env.reset(seed=42)

    adapter = ObsAdapter(game_map)
    node_feats, global_feats = adapter.to_tensors(obs)
    check(
        f"node_features shape [{game_map.n}, {NODE_FEAT_DIM}]",
        node_feats.shape == (game_map.n, NODE_FEAT_DIM),
        str(tuple(node_feats.shape)),
    )
    check(
        f"global_features shape [{GLOBAL_FEAT_DIM}]",
        global_feats.shape == (GLOBAL_FEAT_DIM,),
        str(tuple(global_feats.shape)),
    )
    check("node features in finite range", node_feats.isfinite().all().item())
    check("global features in finite range", global_feats.isfinite().all().item())
    print()

    # ---- 3. GNN encoder ------------------------------------------------------------
    print("3. GNNEncoder forward pass")
    encoder = GNNEncoder(node_feat_dim=NODE_FEAT_DIM, hidden_dim=64, output_dim=64)
    nf_b = node_feats.unsqueeze(0)  # [1, N, 5]
    node_embs, graph_emb = encoder(nf_b, edge_index)
    check(
        "node_embeddings shape [1, 9, 64]",
        node_embs.shape == (1, 9, 64),
        str(tuple(node_embs.shape)),
    )
    check(
        "graph_embedding shape [1, 64]",
        graph_emb.shape == (1, 64),
        str(tuple(graph_emb.shape)),
    )
    check("no NaN in node_embeddings", node_embs.isfinite().all().item())
    print()

    # ---- 4. Batched obs adapter ----------------------------------------------------
    print("4. obs_adapter (batched, DummyVecEnv)")
    vec_env = make_vec_env(preset="full", n_envs=4, seed=0)
    obs_batch = vec_env.reset()  # DummyVecEnv.reset() returns obs only (no info tuple)
    nf_b2, gf_b2 = adapter.to_tensors_batched(obs_batch)
    check(
        "batched node_features shape [4, 9, 5]",
        nf_b2.shape == (4, 9, 5),
        str(tuple(nf_b2.shape)),
    )
    check(
        "batched global_features shape [4, 8]",
        gf_b2.shape == (4, 8),
        str(tuple(gf_b2.shape)),
    )
    vec_env.close()
    print()

    # ---- 5. GNNFeaturesExtractor ---------------------------------------------------
    print("5. GNNFeaturesExtractor")
    extractor = GNNFeaturesExtractor(
        env.observation_space, game_map=game_map, gnn_hidden=64, gnn_output=64
    )
    expected_fdim = 64 + 9 * 64 + 8  # 648
    check(
        f"features_dim = {expected_fdim}",
        extractor.features_dim == expected_fdim,
        str(extractor.features_dim),
    )

    # Simulate SB3's obs→tensor conversion (float32, batch dim 1)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).float() for k, v in obs.items()}
    feats = extractor(obs_t)
    check(
        f"extractor output shape [1, {expected_fdim}]",
        feats.shape == (1, expected_fdim),
        str(tuple(feats.shape)),
    )
    check("extractor output finite", feats.isfinite().all().item())
    print()

    # ---- 6. Full SB3 PPO policy forward --------------------------------------------
    print("6. GNNSovereignPolicy (SB3 PPO)")
    vec_env2 = make_vec_env(preset="full", n_envs=2, seed=1)
    model = PPO(
        GNNSovereignPolicy,
        vec_env2,
        policy_kwargs=dict(game_map=game_map, gnn_hidden=64, gnn_output=64),
        n_steps=64,  # small for smoke test
        batch_size=32,
        verbose=0,
    )

    # predict() runs a full forward pass
    action, _state = model.predict(obs, deterministic=True)
    check("action shape (3,)", action.shape == (3,), str(action.shape))
    check("political action in [0,4]", 0 <= int(action[0]) <= 4, str(action[0]))
    check("military action in [0,3]", 0 <= int(action[1]) <= 3, str(action[1]))
    check(
        f"target action in [0,{game_map.n - 1}]",
        0 <= int(action[2]) <= game_map.n - 1,
        str(action[2]),
    )
    vec_env2.close()
    print()

    # ---- 7. Env step with sampled action -------------------------------------------
    print("7. Env step")
    obs2, reward, terminated, truncated, info = env.step(action)
    check("obs2 has same keys", set(obs2.keys()) == set(obs.keys()))
    check("reward is finite", abs(float(reward)) < 1e6)
    check("termination_reason present", "termination_reason" in info)
    print(
        f"     reward={reward:.4f}  done={terminated or truncated}"
        f"  reason={info['termination_reason']!r}"
    )
    env.close()
    print()

    print("=== All checks passed ===\n")


if __name__ == "__main__":
    main()
