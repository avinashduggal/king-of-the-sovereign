"""Random-policy rollout. Sanity check that the env runs end-to-end.

Usage:
    python scripts/random_rollout.py [--seed N] [--preset NAME] [--render]
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import sovereign  # noqa: F401  -- registers env ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-policy rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=list(sovereign.ABLATION_PRESETS),
        default="full",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print the state every step (slow but informative).",
    )
    args = parser.parse_args()

    env_id = f"Sovereign-{args.preset}-v0"
    env = gym.make(env_id, render_mode="human" if args.render else None)
    obs, info = env.reset(seed=args.seed)

    if args.render:
        env.render()

    cumulative = 0.0
    steps = 0
    terminated = truncated = False
    last_info: dict = {}
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        steps += 1
        last_info = info
        if args.render:
            env.render()

    print()
    print(f"=== Episode finished ({env_id}) ===")
    print(f"  steps:       {steps}")
    print(f"  cumulative:  {cumulative:.3f}")
    print(f"  reason:      {last_info.get('termination_reason', '?')}")
    print(f"  terminal r:  {last_info.get('terminal_reward', 0.0):+.3f}")
    env.close()


if __name__ == "__main__":
    main()
