"""Load a saved agent and run an evaluation rollout suite.

Accepts either:
  * a run directory like ``models/ppo_full_20260429-153012_200000ts/``
    (the script will load ``model.zip`` inside it), or
  * a direct path to a ``.zip`` model file.

The algorithm and preset are inferred from the run-directory name
(``<algo>_<preset>_<timestamp>_<timesteps>ts``); pass ``--preset`` to
override.

Usage:
    python scripts/evaluate.py models/ppo_full_20260429-153012_200000ts
    python scripts/evaluate.py models/dqn_baseline_20260429-160500_50000ts --episodes 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _train_common import evaluate_model, make_env


PREFIX_TO_LOADER = {
    "ppo": ("stable_baselines3", "PPO", False),
    "a2c": ("stable_baselines3", "A2C", False),
    "dqn": ("stable_baselines3", "DQN", True),
    "qrdqn": ("sb3_contrib", "QRDQN", True),
    "recppo": ("sb3_contrib", "RecurrentPPO", False),
}

VALID_PRESETS = {
    "full",
    "no_legitimacy",
    "no_occupation_cost",
    "no_neutral_posture",
    "baseline",
}


def parse_run_name(name: str) -> tuple[str, str | None]:
    """Pull (algo, preset) out of '<algo>_<preset>_<timestamp>_<N>ts'.

    Falls back to (prefix, None) when only the algorithm prefix matches —
    callers can then default the preset to 'full'.
    """
    lowered = name.lower()
    for prefix in PREFIX_TO_LOADER:
        head = prefix + "_"
        if not lowered.startswith(head):
            continue
        rest = lowered[len(head):]
        # Try to greedily match the longest preset that the rest starts with.
        for preset in sorted(VALID_PRESETS, key=len, reverse=True):
            if rest == preset or rest.startswith(preset + "_"):
                return prefix, preset
        return prefix, None
    raise SystemExit(
        f"Cannot infer algorithm from {name!r}. "
        f"Name must start with one of: {list(PREFIX_TO_LOADER)}"
    )


def resolve(model_arg: Path) -> tuple[Path, str, str]:
    """Return (zip_path, algo, preset) for either a run dir or a .zip path."""
    if model_arg.is_dir():
        zip_path = model_arg / "model.zip"
        if not zip_path.exists():
            raise SystemExit(f"No model.zip found inside {model_arg}")
        algo, preset = parse_run_name(model_arg.name)
    else:
        zip_path = model_arg
        # Use parent dir name if it follows the run convention; else the stem.
        parent_name = model_arg.parent.name
        try:
            algo, preset = parse_run_name(parent_name)
        except SystemExit:
            algo, preset = parse_run_name(model_arg.stem)
    return zip_path, algo, preset or "full"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path,
                        help="Run directory or .zip model file.")
    parser.add_argument("--preset", default=None,
                        help="Override preset. Default: infer from run name.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    zip_path, algo, preset = resolve(args.model_path)
    if args.preset:
        preset = args.preset

    module, classname, needs_flatten = PREFIX_TO_LOADER[algo]
    mod = __import__(module, fromlist=[classname])
    cls = getattr(mod, classname)
    model = cls.load(str(zip_path))

    env = make_env(
        preset=preset, flatten_obs=needs_flatten, flatten_action=needs_flatten
    )
    stats = evaluate_model(model, env, episodes=args.episodes, seed=args.seed)
    print(f"{args.model_path}  algo={algo}  preset={preset}  {stats.pretty()}")


if __name__ == "__main__":
    main()
