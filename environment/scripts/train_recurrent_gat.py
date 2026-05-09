"""Train a Recurrent PPO agent with a GATv2Conv encoder on Sovereign-v0.

Combines the GATv2Conv graph encoder (spatial inductive bias over territory
adjacency) with an LSTM (temporal memory for legitimacy trends, occupation
dynamics, and supply trajectories).  Uses sb3_contrib RecurrentPPO.

Requires the ``[train]`` extra, sb3-contrib, and torch_geometric:
    pip install -e ".[train]" torch_geometric

Usage:
    python scripts/train_recurrent_gat.py [options]

    --timesteps       500000  total env steps (default 500 000)
    --preset          full    SovereignConfig preset (default "full")
    --n-envs          4       parallel envs (default 4)
    --n-steps         256     rollout steps per env per update (default 256)
    --batch-size      128     PPO mini-batch size (default 128)
    --lr              3e-4    learning rate (default 3e-4)
    --gat-hidden      64      GATv2Conv hidden channels (default 64)
    --gat-output      64      GATv2Conv output channels (default 64)
    --gat-heads       4       attention heads in GAT layer 1 (default 4)
    --gat-dropout     0.1     attention dropout probability (default 0.1)
    --lstm-hidden     128     LSTM hidden units (default 128)
    --lstm-layers     1       number of LSTM layers (default 1)
    --actor-hidden    64      actor MLP hidden units (default 64)
    --critic-hidden   64      critic MLP hidden units (default 64)
    --ent-coef        0.01    entropy bonus coefficient (default 0.01)
    --seed            0       RNG seed (default 0)
    --eval-episodes   30      evaluation episodes after training (default 30)
    --save            PATH    override model save path (default: auto timestamped dir)
    --tb                      enable TensorBoard logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the models/ package importable from king-of-the-sovereign/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO + GAT on Sovereign-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timesteps",    type=int,   default=500_000)
    parser.add_argument("--preset",       default="full")
    parser.add_argument("--n-envs",       type=int,   default=4)
    parser.add_argument("--n-steps",      type=int,   default=256)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--gat-hidden",   type=int,   default=64)
    parser.add_argument("--gat-output",   type=int,   default=64)
    parser.add_argument("--gat-heads",    type=int,   default=4)
    parser.add_argument("--gat-dropout",  type=float, default=0.1)
    parser.add_argument("--lstm-hidden",  type=int,   default=128)
    parser.add_argument("--lstm-layers",  type=int,   default=1)
    parser.add_argument("--actor-hidden", type=int,   default=64)
    parser.add_argument("--critic-hidden",type=int,   default=64)
    parser.add_argument("--ent-coef",     type=float, default=0.01)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--eval-episodes",type=int,   default=30)
    parser.add_argument("--save",         default=None, help="Override model save path")
    parser.add_argument("--tb",           action="store_true", help="Enable TensorBoard")
    parser.add_argument(
        "--load-checkpoint", type=Path, default=None, metavar="PATH",
        help="Path to a model.zip to resume training from.",
    )
    args = parser.parse_args()

    # ---- imports (after path is set) -----------------------------------------------
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise SystemExit(
            "sb3-contrib is not installed.\n"
            'Install with: pip install -e ".[train]"'
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
    from models.recurrent_gat_policy import GATRecurrentPolicy
    from sovereign.game_map import GameMap
    from utils.logging_utils import TrainingProgressCallback, setup_logging

    # ---- run directory + logging ---------------------------------------------------
    if args.save:
        run_dir = None
        log = setup_logging(log_path=None, name="recgat")
    else:
        run_dir = make_run_dir("recgat_ppo", args.preset, args.timesteps)
        log = setup_logging(log_path=run_dir, name="recgat")

    # ---- environment ---------------------------------------------------------------
    train_env = make_vec_env(preset=args.preset, n_envs=args.n_envs, seed=args.seed)

    # Build the static game map (same topology for all presets)
    game_map = GameMap()

    # ---- model ---------------------------------------------------------------------
    policy_kwargs = dict(
        game_map=game_map,
        gat_hidden=args.gat_hidden,
        gat_output=args.gat_output,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        # lstm_hidden_size and n_lstm_layers go in policy_kwargs,
        # NOT as top-level RecurrentPPO arguments.
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=args.lstm_layers,
    )

    if args.load_checkpoint is not None:
        checkpoint = Path(args.load_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"[resume] Loading checkpoint: {checkpoint}")
        model = RecurrentPPO.load(
            str(checkpoint), env=train_env,
            custom_objects={"policy_class": GATRecurrentPolicy},
        )
        print(f"[resume] Resuming from {model.num_timesteps:,} timesteps")
    else:
        model = RecurrentPPO(
            GATRecurrentPolicy,
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=args.seed,
            tensorboard_log=str(tb_dir()) if args.tb else None,
        )

    fe = model.policy.features_extractor
    log.info(
        "RecGAT-PPO | preset=%s | envs=%d | D=%d | heads=%d | lstm=%d×%d"
        " | features_dim=%d | from_ts=%d",
        args.preset,
        args.n_envs,
        args.gat_output,
        args.gat_heads,
        args.lstm_hidden,
        args.lstm_layers,
        fe.features_dim,
        model.num_timesteps,
    )

    # ---- training ------------------------------------------------------------------
    callback = TrainingProgressCallback(args.timesteps, log, algo_name="RecGAT-PPO")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    # ---- save ----------------------------------------------------------------------
    save_path = args.save if args.save else str(run_dir / "model.zip")  # type: ignore[operator]
    model.save(save_path)
    log.info("Saved model → %s", save_path)

    # ---- evaluation ----------------------------------------------------------------
    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(
        model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000
    )
    log.info("RecGAT-PPO/%s  %s", args.preset, stats.pretty())


if __name__ == "__main__":
    main()
