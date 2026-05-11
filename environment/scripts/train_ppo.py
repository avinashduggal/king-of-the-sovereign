"""Train a PPO agent on Sovereign-v0 using Stable-Baselines3.

PPO is the default recommendation for this env: on-policy, handles the
``Dict`` observation natively via ``MultiInputPolicy``, and is robust to
the mixed-scale rewards (per-step O(0.5), terminals O(50)).

Requires the ``[train]`` extra: ``pip install -e ".[train]"``.

Usage:
    python scripts/train_ppo.py [--timesteps 200000] [--preset full] [--n-envs 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _train_common import evaluate_model, load_or_init_model, make_env, make_run_dir, make_vec_env, tb_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save", default=None, help="Path to save the model.")
    parser.add_argument("--tb", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument(
        "--load-checkpoint", type=Path, default=None, metavar="PATH",
        help="Path to a model.zip to resume training from.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    from utils.logging_utils import TrainingProgressCallback, setup_logging

    if args.save:
        run_dir = None
        log = setup_logging(log_path=None, name="ppo")
    else:
        run_dir = make_run_dir("ppo", args.preset, args.timesteps)
        log = setup_logging(log_path=run_dir, name="ppo")

    train_env = make_vec_env(preset=args.preset, n_envs=args.n_envs, seed=args.seed)

    model = load_or_init_model(
        PPO, args.load_checkpoint, train_env,
        policy="MultiInputPolicy",
        verbose=0,
        seed=args.seed,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(tb_dir()) if args.tb else None,
    )

    log.info("PPO | preset=%s | envs=%d | device=cpu | from_ts=%d", args.preset, args.n_envs, model.num_timesteps)

    callback = TrainingProgressCallback(args.timesteps, log, algo_name="PPO")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    save_path = args.save if args.save else str(run_dir / "model.zip")  # type: ignore[operator]
    model.save(save_path)
    log.info("Saved model → %s", save_path)

    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000)
    log.info("PPO/%s  %s", args.preset, stats.pretty())


if __name__ == "__main__":
    main()
