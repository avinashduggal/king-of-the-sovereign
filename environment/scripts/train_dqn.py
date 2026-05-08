"""Train a DQN agent on Sovereign-v0 using Stable-Baselines3.

DQN is value-based and off-policy; it requires a ``Discrete`` action
space and a flat observation vector. We wrap the env with
``FlattenObservation`` and ``FlattenMultiDiscreteAction`` (5*4*9=180).

Requires the ``[train]`` extra: ``pip install -e ".[train]"``.

Usage:
    python scripts/train_dqn.py [--timesteps 200000] [--preset full]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _train_common import evaluate_model, make_env, make_run_dir, tb_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save", default=None)
    parser.add_argument("--tb", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    from utils.logging_utils import TrainingProgressCallback, setup_logging

    if args.save:
        run_dir = None
        log = setup_logging(log_path=None, name="dqn")
    else:
        run_dir = make_run_dir("dqn", args.preset, args.timesteps)
        log = setup_logging(log_path=run_dir, name="dqn")

    train_env = make_env(
        preset=args.preset, flatten_obs=True, flatten_action=True, seed=args.seed
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=args.seed,
        learning_starts=1_000,
        buffer_size=100_000,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        gamma=0.99,
        learning_rate=5e-4,
        tensorboard_log=str(tb_dir()) if args.tb else None,
    )

    log.info("DQN | preset=%s | device=cpu", args.preset)

    callback = TrainingProgressCallback(args.timesteps, log, algo_name="DQN")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    save_path = args.save if args.save else str(run_dir / "model.zip")  # type: ignore[operator]
    model.save(save_path)
    log.info("Saved model → %s", save_path)

    eval_env = make_env(
        preset=args.preset, flatten_obs=True, flatten_action=True
    )
    stats = evaluate_model(model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000)
    log.info("DQN/%s  %s", args.preset, stats.pretty())


if __name__ == "__main__":
    main()
