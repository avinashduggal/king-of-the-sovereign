"""Train a PPO agent with a GAT policy on Sovereign-v0.

Replaces the flat MultiInputPolicy backbone with a graph-aware encoder
(GATSovereignPolicy) that uses GATv2Conv attention over the territory
adjacency structure, learning to up-weight strategically important neighbors.

Requires the ``[train]`` extra and torch_geometric:
    pip install -e ".[train]" torch_geometric

Usage:
    python scripts/train_ppo_gat.py [options]

    --total-timesteps 500000   total env steps (default 500 000)
    --preset          full     SovereignConfig preset (default "full")
    --n-envs          8        parallel envs (default 8)
    --n-steps         512      rollout steps per env per update (default 512)
    --batch-size      256      PPO mini-batch size (default 256)
    --n-epochs        10       PPO update epochs per rollout (default 10)
    --lr              3e-4     learning rate (default 3e-4)
    --gat-hidden      64       GATv2Conv hidden channels (default 64)
    --gat-output      64       GATv2Conv output channels (default 64)
    --gat-heads       4        attention heads in layer 1 (default 4)
    --gat-dropout     0.1      attention dropout probability (default 0.1)
    --actor-hidden    64       actor MLP hidden units (default 64)
    --critic-hidden   64       critic MLP hidden units (default 64)
    --ent-coef        0.01     entropy bonus coefficient (default 0.01)
    --seed            0        RNG seed (default 0)
    --device          auto     torch device: cpu / cuda / mps / auto
    --eval-episodes   30       evaluation episodes after training (default 30)
    --save            PATH     override save path (default: auto timestamped dir)
    --tb                       enable TensorBoard logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the models/ package importable from king-of-the-sovereign/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO + GAT on Sovereign-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gat-hidden", type=int, default=64)
    parser.add_argument("--gat-output", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-dropout", type=float, default=0.1)
    parser.add_argument("--actor-hidden", type=int, default=64)
    parser.add_argument("--critic-hidden", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save", default=None, help="Override model save path")
    parser.add_argument("--tb", action="store_true", help="Enable TensorBoard")
    parser.add_argument(
        "--load-checkpoint", type=Path, default=None, metavar="PATH",
        help="Path to a model.zip to resume training from.",
    )
    args = parser.parse_args()

    # ---- imports (after path is set) -----------------------------------------------
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            'stable-baselines3 not installed.  Run: pip install -e ".[train]"'
        ) from exc

    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "torch_geometric not installed.  Run: pip install torch_geometric"
        ) from exc

    import sovereign  # noqa: F401 — registers Sovereign-*-v0 gym IDs

    from _train_common import (
        evaluate_model,
        make_env,
        make_run_dir,
        make_vec_env,
        tb_dir,
    )
    from models.gat_policy import GATSovereignPolicy
    from sovereign.game_map import GameMap
    from utils.logging_utils import TrainingProgressCallback, setup_logging

    # ---- run directory + logging ---------------------------------------------------
    if args.save:
        run_dir = None
        log = setup_logging(log_path=None)
    else:
        run_dir = make_run_dir("gat_ppo", args.preset, args.total_timesteps)
        log = setup_logging(log_path=run_dir, name="gat_ppo")

    # ---- environment ---------------------------------------------------------------
    train_env = make_vec_env(preset=args.preset, n_envs=args.n_envs, seed=args.seed)

    # Build the static game map (same topology for all presets)
    game_map = GameMap()

    # ---- model ---------------------------------------------------------------------
    if args.load_checkpoint is not None:
        checkpoint = Path(args.load_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"[resume] Loading checkpoint: {checkpoint}")
        model = PPO.load(
            str(checkpoint), env=train_env,
            custom_objects={"policy_class": GATSovereignPolicy},
        )
        print(f"[resume] Resuming from {model.num_timesteps:,} timesteps")
    else:
        model = PPO(
            GATSovereignPolicy,
            train_env,
            policy_kwargs=dict(
                game_map=game_map,
                gat_hidden=args.gat_hidden,
                gat_output=args.gat_output,
                gat_heads=args.gat_heads,
                gat_dropout=args.gat_dropout,
                actor_hidden=args.actor_hidden,
                critic_hidden=args.critic_hidden,
            ),
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(tb_dir()) if args.tb else None,
        )

    log.info(
        "GAT-PPO | preset=%s | envs=%d | D=%d | heads=%d | dropout=%.2f | device=%s | features_dim=%d | from_ts=%d",
        args.preset,
        args.n_envs,
        args.gat_output,
        args.gat_heads,
        args.gat_dropout,
        args.device,
        model.policy.features_extractor.features_dim,
        model.num_timesteps,
    )

    # ---- training ------------------------------------------------------------------
    callback = TrainingProgressCallback(args.total_timesteps, log, algo_name="GAT-PPO")
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)

    # ---- save ----------------------------------------------------------------------
    save_path = args.save if args.save else str(run_dir / "model.zip")  # type: ignore[operator]
    model.save(save_path)
    log.info("Saved model → %s", save_path)

    # ---- evaluation ----------------------------------------------------------------
    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(
        model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000
    )
    log.info("GAT-PPO/%s  %s", args.preset, stats.pretty())


if __name__ == "__main__":
    main()
