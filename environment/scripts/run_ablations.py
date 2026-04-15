"""Run the five ablation presets with a random policy and report stats.

Useful as a smoke test: each preset should produce visibly different
distributions over termination reasons and episode lengths, confirming
that the toggles actually change behaviour.

Usage:
    python scripts/run_ablations.py [--episodes 30] [--seed 0]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import gymnasium as gym
import numpy as np

import sovereign  # noqa: F401


def run_preset(
    preset: str, episodes: int, seed_start: int
) -> dict:
    env = gym.make(f"Sovereign-{preset}-v0")
    lengths: list[int] = []
    returns: list[float] = []
    reasons: Counter[str] = Counter()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        cum = 0.0
        steps = 0
        terminated = truncated = False
        last_reason = "?"
        while not (terminated or truncated):
            action = env.action_space.sample()
            _, r, terminated, truncated, info = env.step(action)
            cum += r
            steps += 1
            last_reason = info.get("termination_reason", "?")
        lengths.append(steps)
        returns.append(cum)
        reasons[last_reason] += 1

    env.close()
    return {
        "lengths": lengths,
        "returns": returns,
        "reasons": dict(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"{'preset':<22}{'mean_len':>10}{'mean_ret':>12}{'reasons':>30}"
    )
    print("-" * 74)
    for preset in sovereign.ABLATION_PRESETS:
        stats = run_preset(preset, args.episodes, args.seed)
        mean_len = float(np.mean(stats["lengths"]))
        mean_ret = float(np.mean(stats["returns"]))
        # Compact "reason1=N reason2=M" for column display
        reasons_str = " ".join(
            f"{k}={v}" for k, v in sorted(stats["reasons"].items())
        )
        print(
            f"{preset:<22}{mean_len:>10.1f}{mean_ret:>12.2f}  {reasons_str}"
        )


if __name__ == "__main__":
    main()
