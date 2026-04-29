"""Train a Recurrent PPO agent on Sovereign-v0 using sb3-contrib.

The Sovereign observation is technically Markov (full state is exposed),
but legitimacy / posture / occupation dynamics have long-horizon
consequences that an LSTM policy can summarise more compactly than a
feed-forward MLP. Useful as a robustness comparison.

Requires the ``[train]`` extra (which pulls in sb3-contrib).

Usage:
    python scripts/train_recurrent_ppo.py [--timesteps 200000] [--preset full]
"""

from __future__ import annotations

import argparse

from _train_common import evaluate_model, make_env, make_run_dir, make_vec_env, tb_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save", default=None)
    parser.add_argument("--tb", action="store_true")
    args = parser.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise SystemExit(
            "sb3-contrib is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    train_env = make_vec_env(preset=args.preset, n_envs=args.n_envs, seed=args.seed)

    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        n_steps=256,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(tb_dir()) if args.tb else None,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    if args.save:
        save_path = args.save
    else:
        run_dir = make_run_dir("recppo", args.preset, args.timesteps)
        save_path = str(run_dir / "model.zip")
    model.save(save_path)
    print(f"saved model to {save_path}")

    eval_env = make_env(preset=args.preset)
    stats = evaluate_model(model, eval_env, episodes=args.eval_episodes, seed=args.seed + 1000)
    print(f"RecurrentPPO/{args.preset}  {stats.pretty()}")


if __name__ == "__main__":
    main()
