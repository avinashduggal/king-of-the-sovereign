"""Train a DQN agent on Sovereign-v0 using Stable-Baselines3.

DQN requires a ``Discrete`` action space and a flat observation; we wrap
the env with ``FlattenObservation`` and a custom flat-action wrapper to
make this work.

Requires the ``[train]`` extra: ``pip install -e ".[train]"``.

Usage:
    python scripts/train_dqn.py [--timesteps 30000] [--preset full]
"""

from __future__ import annotations

import argparse

import numpy as np


class FlattenMultiDiscreteAction:
    """Wrapper converting MultiDiscrete([5, 4, 9]) -> Discrete(5*4*9=180).

    Implemented as a custom wrapper because gymnasium's built-in does not
    cover the case where individual sub-action sizes differ.
    """

    def __init__(self, env):
        import gymnasium as gym

        self.env = env
        self.observation_space = env.observation_space
        sizes = env.action_space.nvec
        self._sizes = tuple(int(s) for s in sizes)
        self._n = int(np.prod(self._sizes))
        self.action_space = gym.spaces.Discrete(self._n)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        a = np.asarray(self._unflatten(int(action)), dtype=np.int64)
        return self.env.step(a)

    def _unflatten(self, flat_idx):
        # Row-major decoding consistent with np.unravel_index
        return np.unravel_index(flat_idx, self._sizes)

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Install with: pip install -e \".[train]\""
        ) from exc

    import gymnasium as gym
    from gymnasium.wrappers import FlattenObservation

    import sovereign  # noqa: F401

    env_id = f"Sovereign-{args.preset}-v0"
    env = gym.make(env_id)
    env = FlattenObservation(env)
    env = FlattenMultiDiscreteAction(env)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_starts=200,
        buffer_size=10_000,
        target_update_interval=500,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    obs, _ = env.reset(seed=args.seed)
    cum = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        cum += r
    print(f"eval episode return: {cum:.3f}  reason: {info.get('termination_reason')}")


if __name__ == "__main__":
    main()
