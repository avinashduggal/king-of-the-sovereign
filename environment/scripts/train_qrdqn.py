"""Train a QR-DQN agent on Sovereign-v0 using sb3-contrib.

Quantile-Regression DQN learns a distribution over returns rather than a
point estimate; in our setting (sparse, mixed-scale terminal rewards
ranging from -50 to +40) it usually outperforms vanilla DQN.

Same wrappers as DQN: FlattenObservation + flat-action.

Requires the ``[train]`` extra (which pulls in sb3-contrib).

Usage:
    python scripts/train_qrdqn.py [--timesteps 200000] [--preset full]
"""

from __future__ import annotations

import argparse

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
        from sb3_contrib import QRDQN
    except ImportError as exc:
        raise SystemExit(
            "sb3-contrib is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    train_env = make_env(
        preset=args.preset, flatten_obs=True, flatten_action=True, seed=args.seed
    )

    model = QRDQN(
        "MlpPolicy",
        train_env,
        verbose=1,
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
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    if args.save:
        save_path = args.save
    else:
        run_dir = make_run_dir("qrdqn", args.preset, args.timesteps)
        save_path = str(run_dir / "model.zip")
    model.save(save_path)
    print(f"saved model to {save_path}")

    eval_env = make_env(
        preset=args.preset, flatten_obs=True, flatten_action=True
    )
    stats = evaluate_model(model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000)
    print(f"QR-DQN/{args.preset}  {stats.pretty()}")


if __name__ == "__main__":
    main()
