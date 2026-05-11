"""Evaluate a saved RecGAT-PPO checkpoint on Sovereign-v0.

Loads a model.zip saved by train_recurrent_gat.py and runs deterministic
rollouts, printing mean return, episode length, and termination-reason
breakdown.

Usage:
    python scripts/evaluate_recgat_ppo.py \\
        --checkpoint checkpoints/recgat_ppo_full_20260508-185400_500000ts/model.zip \\
        [--episodes 50] [--preset full] [--seed 2025]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a RecGAT-PPO checkpoint on Sovereign-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise SystemExit("sb3-contrib not installed.") from exc

    import sovereign  # noqa: F401

    from _train_common import evaluate_model, make_env
    from models.recurrent_gat_policy import GATRecurrentPolicy

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model = RecurrentPPO.load(
        str(checkpoint),
        custom_objects={"policy_class": GATRecurrentPolicy},
    )
    print(f"Loaded  {checkpoint.name}")

    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(
        model,
        eval_env,
        episodes=args.episodes,
        deterministic=True,
        seed=args.seed,
    )
    print(f"RecGAT-PPO/{args.preset}  {stats.pretty()}")
    print("\nTermination breakdown:")
    for reason, count in sorted(stats.reasons.items()):
        pct = 100.0 * count / args.episodes
        print(f"  {reason:<30} {count:>4}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
