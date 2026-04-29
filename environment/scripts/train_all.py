"""Sweep every algorithm across every ablation preset.

Drives one Python process per (algorithm, preset) pair so a crash in
one combination doesn't bring down the whole sweep. Each child writes
its model under ``models/<algo>_<preset>.zip`` and prints an evaluation
line that this driver scrapes for the final summary table.

Usage:
    python scripts/train_all.py [--timesteps 200000]
                                [--algos ppo,a2c,dqn,qrdqn,recppo]
                                [--presets full,no_legitimacy,...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ALGO_TO_SCRIPT = {
    "ppo": "train_ppo.py",
    "a2c": "train_a2c.py",
    "dqn": "train_dqn.py",
    "qrdqn": "train_qrdqn.py",
    "recppo": "train_recurrent_ppo.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--algos",
        default="ppo,a2c,dqn,qrdqn,recppo",
        help="Comma-separated subset of algorithms to run.",
    )
    parser.add_argument(
        "--presets",
        default="full,no_legitimacy,no_occupation_cost,no_neutral_posture,baseline",
        help="Comma-separated subset of ablation presets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    scripts_dir = Path(__file__).resolve().parent

    summary: list[tuple[str, str, float, str]] = []
    for algo in algos:
        if algo not in ALGO_TO_SCRIPT:
            print(f"[skip] unknown algorithm: {algo}")
            continue
        script = scripts_dir / ALGO_TO_SCRIPT[algo]
        for preset in presets:
            label = f"{algo}/{preset}"
            print(f"\n{'=' * 70}\n>>> training {label}\n{'=' * 70}")
            t0 = time.time()
            cmd = [
                sys.executable,
                str(script),
                "--timesteps", str(args.timesteps),
                "--preset", preset,
                "--seed", str(args.seed),
                "--eval-episodes", str(args.eval_episodes),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - t0
            if result.returncode != 0:
                print(f"[FAIL] {label} after {duration:.1f}s")
                print(result.stderr[-2000:])
                summary.append((algo, preset, duration, "FAILED"))
                continue
            tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            print(tail)
            summary.append((algo, preset, duration, tail))

    print(f"\n{'=' * 70}\nSWEEP SUMMARY\n{'=' * 70}")
    for algo, preset, duration, tail in summary:
        print(f"{algo:<8}{preset:<22}{duration:>7.1f}s   {tail}")


if __name__ == "__main__":
    main()
