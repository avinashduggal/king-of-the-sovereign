"""Sweep every algorithm across every ablation preset.

Drives one Python process per (algorithm, preset) pair so a crash in
one combination doesn't bring down the whole sweep. Each child writes
its model under ``checkpoints/<algo>_<preset>_<timestamp>_<N>ts/`` and
the sweep writes a persistent log to ``checkpoints/sweep_<timestamp>.log``.

Usage:
    python scripts/train_all.py [--timesteps 200000]
                                [--algos ppo,a2c,dqn,qrdqn,recppo,gnn_ppo]
                                [--presets full,no_legitimacy,...]
                                [--resume]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ALGO_TO_SCRIPT = {
    "ppo": "train_ppo.py",
    "a2c": "train_a2c.py",
    "dqn": "train_dqn.py",
    "qrdqn": "train_qrdqn.py",
    "recppo": "train_recurrent_ppo.py",
    "gnn_ppo": "train_ppo_gnn.py",
}

# Off-policy algorithms that save/load a replay buffer.
_OFF_POLICY = {"dqn", "qrdqn"}

# GNN-PPO uses --total-timesteps instead of --timesteps.
_TOTAL_TS_FLAG = {"gnn_ppo"}


def _find_latest_checkpoint(algo: str, preset: str) -> Path | None:
    """Return the most recent model.zip for (algo, preset), or None."""
    base = Path(__file__).resolve().parents[2] / "checkpoints"
    candidates = sorted(base.glob(f"{algo}_{preset}_*/model.zip"))
    return candidates[-1] if candidates else None


def _setup_sweep_log(sweep_ts: str) -> tuple[logging.Logger, Path]:
    """Create a logger that writes to both console and a sweep log file."""
    checkpoints_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoints_dir / f"sweep_{sweep_ts}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(f"sweep_{sweep_ts}")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log, log_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--algos",
        default="ppo,a2c,dqn,qrdqn,recppo",
        help="Comma-separated algorithms to run. Add gnn_ppo to include the GNN policy.",
    )
    parser.add_argument(
        "--presets",
        default="full,no_legitimacy,no_occupation_cost,no_neutral_posture,baseline",
        help="Comma-separated subset of ablation presets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-load the latest checkpoint for each (algo, preset) pair and train for additional --timesteps.",
    )
    args = parser.parse_args()

    sweep_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log, log_file = _setup_sweep_log(sweep_ts)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    scripts_dir = Path(__file__).resolve().parent

    log.info(
        "Sweep start | algos=%s | presets=%s | timesteps=%d | resume=%s",
        ",".join(algos), ",".join(presets), args.timesteps, args.resume,
    )
    log.info("Sweep log → %s", log_file)

    summary: list[tuple[str, str, float, str]] = []
    for algo in algos:
        if algo not in ALGO_TO_SCRIPT:
            log.warning("[skip] unknown algorithm: %s", algo)
            continue
        script = scripts_dir / ALGO_TO_SCRIPT[algo]
        for preset in presets:
            label = f"{algo}/{preset}"
            log.info("%s", "=" * 70)
            log.info(">>> training %s", label)
            log.info("%s", "=" * 70)

            ts_flag = "--total-timesteps" if algo in _TOTAL_TS_FLAG else "--timesteps"
            cmd = [
                sys.executable,
                str(script),
                ts_flag, str(args.timesteps),
                "--preset", preset,
                "--seed", str(args.seed),
                "--eval-episodes", str(args.eval_episodes),
            ]

            if args.resume:
                ckpt = _find_latest_checkpoint(algo, preset)
                if ckpt:
                    cmd += ["--load-checkpoint", str(ckpt)]
                    log.info("[resume] %s ← %s", label, ckpt)
                    if algo in _OFF_POLICY:
                        rb = ckpt.parent / "replay_buffer.pkl"
                        if rb.exists():
                            cmd += ["--load-replay-buffer", str(rb)]
                            log.info("[resume] replay buffer ← %s", rb)
                else:
                    log.info("[resume] no checkpoint found for %s — starting fresh", label)

            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - t0

            if result.returncode != 0:
                log.error("[FAIL] %s after %.1fs", label, duration)
                if result.stderr.strip():
                    for line in result.stderr.strip().splitlines()[-50:]:
                        log.error("  %s", line)
                summary.append((algo, preset, duration, "FAILED"))
                continue

            for line in result.stdout.strip().splitlines():
                log.info("  %s", line)
            tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            summary.append((algo, preset, duration, tail))

    log.info("%s", "=" * 70)
    log.info("SWEEP SUMMARY")
    log.info("%s", "=" * 70)
    for algo, preset, dur, tail in summary:
        log.info("%-8s %-22s %7.1fs   %s", algo, preset, dur, tail)
    log.info("Sweep log → %s", log_file)


if __name__ == "__main__":
    main()
