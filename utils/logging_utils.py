"""Logging setup and tqdm progress callback for GNN-PPO training."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from tqdm import tqdm


def setup_logging(
    log_path: Path | None = None, name: str = "train"
) -> logging.Logger:
    """Return a configured logger.

    Attaches a console handler (INFO) and, when *log_path* is given,
    a file handler writing to ``log_path/train.log``.
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "train.log", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class _PyLogFormat(KVWriter):
    """SB3 KVWriter that routes each metrics dump through Python logging.

    Produces one pipe-delimited INFO line per SB3 dump:
        12:34:01  INFO  time/fps=3609 | train/approx_kl=0.0101 | ...
    """

    def __init__(self, py_logger: logging.Logger) -> None:
        self._log = py_logger

    def write(self, key_values: dict, key_excluded: dict, step: int = 0) -> None:
        if not key_values:
            return
        line = " | ".join(f"{k}={v}" for k, v in key_values.items())
        self._log.debug(line)

    def close(self) -> None:
        pass


class TrainingProgressCallback(BaseCallback):
    """SB3 callback: tqdm progress bar + structured Python logging.

    Injects _PyLogFormat into the model's SB3 logger on training start so
    every SB3 metric dump is written as one timestamped pipe-delimited line.
    The tqdm bar postfix is updated per rollout with the latest available
    metrics from name_to_value.
    """

    def __init__(
        self,
        total_timesteps: int,
        logger: logging.Logger,
        algo_name: str = "Training",
    ) -> None:
        super().__init__(verbose=0)
        self._total = total_timesteps
        self._log = logger
        self._algo_name = algo_name
        self._pbar: tqdm | None = None
        self._last_ts: int = 0

    def _on_training_start(self) -> None:
        # model.logger is live here (set up by _setup_learn before this call).
        self.model.logger.output_formats.append(_PyLogFormat(self._log))
        self._pbar = tqdm(
            total=self._total,
            desc=f"Training {self._algo_name}",
            unit="step",
            dynamic_ncols=True,
        )
        self._last_ts = 0

    def _on_step(self) -> bool:
        delta = self.num_timesteps - self._last_ts
        if self._pbar is not None and delta > 0:
            self._pbar.update(delta)
        self._last_ts = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        # name_to_value holds train/* from the previous update and any
        # rollout/* metrics already recorded this iteration.
        vals = self.model.logger.name_to_value
        ep_rew = vals.get("rollout/ep_rew_mean", float("nan"))
        ep_len = vals.get("rollout/ep_len_mean", float("nan"))
        pol = vals.get("train/policy_gradient_loss", float("nan"))
        val = vals.get("train/value_loss", float("nan"))
        ent = vals.get("train/entropy_loss", float("nan"))

        if self._pbar is not None:
            self._pbar.set_postfix(
                {
                    "ep_rew": f"{ep_rew:.2f}" if not math.isnan(ep_rew) else "n/a",
                    "ep_len": f"{ep_len:.1f}" if not math.isnan(ep_len) else "n/a",
                    "pol": f"{pol:.4f}" if not math.isnan(pol) else "n/a",
                    "val": f"{val:.4f}" if not math.isnan(val) else "n/a",
                    "ent": f"{ent:.4f}" if not math.isnan(ent) else "n/a",
                },
                refresh=False,
            )

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        self._log.info("Training complete — %d timesteps.", self.num_timesteps)
