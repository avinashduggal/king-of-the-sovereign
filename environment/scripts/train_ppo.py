"""Train a PPO agent on Sovereign-v0 using Stable-Baselines3.

Requires the ``[train]`` extra: ``pip install -e ".[train]"``.

Usage:
    python scripts/train_ppo.py [--timesteps 50000] [--preset full]
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save the trained model (optional).",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    import gymnasium as gym

    import sovereign  # noqa: F401  -- registers env ids

    env_id = f"Sovereign-{args.preset}-v0"
    env = gym.make(env_id)

    # MultiInputPolicy handles Dict observation spaces natively.
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    if args.save:
        model.save(args.save)
        print(f"saved model to {args.save}")

    # Quick eval rollout
    obs, _ = env.reset(seed=args.seed)
    cum = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        cum += r
    print(f"eval episode return: {cum:.3f}  reason: {info.get('termination_reason')}")


if __name__ == "__main__":
    main()
