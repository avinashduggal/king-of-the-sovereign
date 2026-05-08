"""Shared utilities for the training scripts.

Centralises env construction, the discrete-action wrapper that DQN and
QR-DQN both need, multi-episode evaluation, and a small SB3 callback
that logs per-episode termination reasons. Keeping this in one place
avoids drift between the per-algorithm scripts.

This module is import-light at the top level so scripts that don't have
stable-baselines3 installed (e.g. random_rollout.py) still work; the
SB3 imports happen inside the helpers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium.wrappers import FlattenObservation


class FlattenMultiDiscreteAction(gymnasium.Wrapper):
    """Wrap MultiDiscrete([5, 4, 9]) -> Discrete(180) for value-based agents.

    DQN-family agents in SB3 only support Discrete action spaces. This
    wrapper bijectively maps a flat index to the joint (political,
    military, target) action via row-major unravelling.
    """

    def __init__(self, env):
        super().__init__(env)
        sizes = env.action_space.nvec  # type: ignore[attr-defined]
        self._sizes = tuple(int(s) for s in sizes)
        self._n = int(np.prod(self._sizes))
        self.action_space = gymnasium.spaces.Discrete(self._n)

    def step(self, action):
        a = np.asarray(np.unravel_index(int(action), self._sizes), dtype=np.int64)
        return self.env.step(a)


def make_env(
    preset: str = "full",
    flatten_obs: bool = False,
    flatten_action: bool = False,
    seed: int | None = None,
):
    """Build a single Sovereign env with optional wrappers."""
    import sovereign  # noqa: F401  -- registers env ids

    env = gymnasium.make(f"Sovereign-{preset}-v0")
    if flatten_obs:
        env = FlattenObservation(env)
    if flatten_action:
        env = FlattenMultiDiscreteAction(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(
    preset: str = "full",
    n_envs: int = 1,
    flatten_obs: bool = False,
    flatten_action: bool = False,
    seed: int = 0,
):
    """Vectorised env (DummyVecEnv) for on-policy algorithms."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _factory(rank: int) -> Callable[[], Any]:
        def _thunk():
            return make_env(
                preset=preset,
                flatten_obs=flatten_obs,
                flatten_action=flatten_action,
                seed=seed + rank,
            )
        return _thunk

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


@dataclass
class EvalStats:
    mean_return: float
    std_return: float
    mean_length: float
    reasons: dict[str, int]

    def pretty(self) -> str:
        reasons = " ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
        return (
            f"return={self.mean_return:+.2f}±{self.std_return:.2f} "
            f"len={self.mean_length:.1f}  reasons[{reasons}]"
        )


def evaluate_model(
    model,
    env,
    episodes: int = 20,
    deterministic: bool = True,
    seed: int = 1000,
) -> EvalStats:
    """Run ``episodes`` rollouts and report mean return + termination mix."""
    returns: list[float] = []
    lengths: list[int] = []
    reasons: Counter[str] = Counter()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        cum = 0.0
        steps = 0
        terminated = truncated = False
        last_reason = "?"
        # RecurrentPPO needs the lstm hidden state threaded through predict().
        state = None
        episode_starts = np.array([True])
        while not (terminated or truncated):
            try:
                action, state = model.predict(
                    obs,
                    state=state,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
            except TypeError:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            cum += float(r)
            steps += 1
            episode_starts = np.array([False])
            last_reason = info.get("termination_reason", "?")
        returns.append(cum)
        lengths.append(steps)
        reasons[last_reason] += 1

    return EvalStats(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_length=float(np.mean(lengths)),
        reasons=dict(reasons),
    )


def models_dir() -> Path:
    """Resolve scripts/../models, creating it if needed."""
    here = Path(__file__).resolve().parent.parent
    out = here.parent / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_run_dir(algo: str, preset: str, timesteps: int) -> Path:
    """Create and return models/<algo>_<preset>_<timestamp>_<timesteps>ts/.

    The timestamp encodes when the run started so concurrent or repeated
    runs of the same (algo, preset) configuration don't clobber each
    other. ``model.zip`` and any sidecar artifacts (eval log, etc.) live
    inside the returned directory.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{algo}_{preset}_{ts}_{timesteps}ts"
    out = models_dir() / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def tb_dir() -> Path:
    """TensorBoard log directory."""
    here = Path(__file__).resolve().parent
    out = here.parent / "tb_logs"
    out.mkdir(parents=True, exist_ok=True)
    return out
