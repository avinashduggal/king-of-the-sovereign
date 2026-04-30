"""Evaluate a saved GNN-PPO checkpoint on Sovereign-v0.

Loads a model.zip saved by train_ppo_gnn.py and runs deterministic rollouts,
printing mean return, episode length, and termination-reason breakdown.

Usage:
    python scripts/evaluate_ppo_gnn.py \\
        --checkpoint models/gnn_ppo_full_20250430-120000_500000ts/model.zip \\
        [--episodes 50] [--preset full] [--seed 2025] [--stochastic] [--render]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the models/ package importable from king-of-the-sovereign/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a GNN-PPO checkpoint on Sovereign-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default: deterministic)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the first episode to stdout"
    )
    args = parser.parse_args()

    # ---- imports -------------------------------------------------------------------
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 not installed.") from exc

    import sovereign  # noqa: F401

    from _train_common import evaluate_model, make_env
    from models.gnn_policy import GNNSovereignPolicy

    # ---- load checkpoint -----------------------------------------------------------
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    # custom_objects tells SB3 how to reconstruct the custom policy class
    model = PPO.load(
        str(checkpoint),
        custom_objects={"policy_class": GNNSovereignPolicy},
    )
    print(f"Loaded  {checkpoint.name}")
    print(f"Policy  features_dim = {model.policy.features_extractor.features_dim}")

    # ---- optional: render first episode --------------------------------------------
    if args.render:
        env = make_env(preset=args.preset)
        obs, _ = env.reset(seed=args.seed)
        terminated = truncated = False
        ep_return = 0.0
        print("\n--- First episode (rendered) ---")
        while not (terminated or truncated):
            print(env.render())
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, terminated, truncated, info = env.step(action)
            ep_return += float(r)
        print(
            f"Episode return: {ep_return:.3f}  "
            f"reason: {info.get('termination_reason', '?')}\n"
        )
        env.close()

    # ---- multi-episode evaluation --------------------------------------------------
    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(
        model,
        eval_env,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    print(f"GNN-PPO/{args.preset}  {stats.pretty()}")
    print("\nTermination breakdown:")
    for reason, count in sorted(stats.reasons.items()):
        pct = 100.0 * count / args.episodes
        print(f"  {reason:<30} {count:>4}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
