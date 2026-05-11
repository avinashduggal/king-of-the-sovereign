"""Utilities for discovering and parsing train.log files from checkpoint directories."""
from pathlib import Path
import re
import pandas as pd

CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"

# Ordered longest-first so greedy prefix matching works correctly
PRESETS = [
    "no_neutral_posture",
    "no_occupation_cost",
    "no_legitimacy",
    "baseline",
    "full",
]

PPO_METRICS = [
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
    "time/fps",
]

DQN_METRICS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "rollout/exploration_rate",
    "train/loss",
    "time/fps",
]

DQN_ALGORITHMS = {"dqn", "qrdqn"}
PPO_ALGORITHMS = {"ppo", "a2c", "recppo", "gnn_ppo", "gat_ppo", "recgat_ppo"}


def get_family_metrics(algorithm: str) -> list[str]:
    if algorithm in DQN_ALGORITHMS:
        return DQN_METRICS
    return PPO_METRICS


def parse_dir_name(name: str) -> dict | None:
    m = re.match(r"^(.+)_(\d{8}-\d{6})_(\d+)ts$", name)
    if not m:
        return None
    algo_preset, timestamp, timesteps = m.group(1), m.group(2), int(m.group(3))
    for preset in PRESETS:
        if algo_preset.endswith("_" + preset):
            algorithm = algo_preset[: -(len(preset) + 1)]
            return {
                "algorithm": algorithm,
                "preset": preset,
                "timestamp": timestamp,
                "target_timesteps": timesteps,
            }
    return None


def parse_log_file(log_path: Path) -> list[dict]:
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\d{2}:\d{2}:\d{2}\s+DEBUG\s+(.+)$", line.strip())
            if not m:
                continue
            kv: dict = {}
            for part in m.group(1).split(" | "):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                try:
                    kv[k.strip()] = float(v.strip())
                except ValueError:
                    pass
            if kv:
                rows.append(kv)
    return rows


def parse_final_return(log_path: Path) -> dict | None:
    pattern = re.compile(r"return=([+-]?[\d.]+)±([\d.]+)\s+len=([\d.]+)")
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return {
                    "final_return": float(m.group(1)),
                    "return_std": float(m.group(2)),
                    "final_ep_len": float(m.group(3)),
                }
    return None


def _discover_runs(
    checkpoints_dir: Path, target_timesteps: int
) -> list[dict]:
    """Return one metadata dict per run, deduplicated to the most recent timestamp."""
    candidates: dict[tuple, dict] = {}
    for d in checkpoints_dir.iterdir():
        if not d.is_dir():
            continue
        meta = parse_dir_name(d.name)
        if not meta or meta["target_timesteps"] != target_timesteps:
            continue
        log_path = d / "train.log"
        if not log_path.exists():
            continue
        key = (meta["algorithm"], meta["preset"])
        if key not in candidates or meta["timestamp"] > candidates[key]["timestamp"]:
            candidates[key] = {**meta, "log_path": log_path, "dir": d.name}
    return list(candidates.values())


def load_all_runs(
    checkpoints_dir: Path = CHECKPOINTS_DIR, target_timesteps: int = 500_000
) -> pd.DataFrame:
    runs = _discover_runs(checkpoints_dir, target_timesteps)
    all_rows: list[dict] = []
    for run in runs:
        rows = parse_log_file(run["log_path"])
        final = parse_final_return(run["log_path"])
        meta = {k: run[k] for k in ("algorithm", "preset", "timestamp", "target_timesteps", "dir")}
        run_label = f"{run['algorithm']}/{run['preset']}"
        for row in rows:
            row.update(meta)
            row["run_label"] = run_label
            if final:
                row.update(final)
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def get_training_times(
    checkpoints_dir: Path = CHECKPOINTS_DIR, target_timesteps: int = 500_000
) -> pd.DataFrame:
    runs = _discover_runs(checkpoints_dir, target_timesteps)
    records = []
    for run in runs:
        rows = parse_log_file(run["log_path"])
        elapsed_values = [r["time/time_elapsed"] for r in rows if "time/time_elapsed" in r]
        training_time_s = max(elapsed_values) if elapsed_values else 0.0
        final = parse_final_return(run["log_path"])
        records.append({
            "algorithm": run["algorithm"],
            "preset": run["preset"],
            "timestamp": run["timestamp"],
            "target_timesteps": run["target_timesteps"],
            "run_label": f"{run['algorithm']}/{run['preset']}",
            "training_time_s": training_time_s,
            "training_time_min": training_time_s / 60,
            "final_return": final["final_return"] if final else None,
            "return_std": final["return_std"] if final else None,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Checkpoints dir: {CHECKPOINTS_DIR}\n")
    times = get_training_times()
    print(f"Discovered {len(times)} runs (500k timesteps):\n")
    print(times[["algorithm", "preset", "training_time_min", "final_return"]].to_string(index=False))
    print()
    df = load_all_runs()
    print(f"Total rows loaded: {len(df)}")
    print(f"Columns: {sorted(df.columns.tolist())}")
