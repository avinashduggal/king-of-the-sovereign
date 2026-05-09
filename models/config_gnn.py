"""Hyperparameter config for the GNN-PPO training pipeline.

All settings are collected in one dataclass so they can be serialised,
logged, and reproduced easily.  CLI scripts map argparse args onto this
dataclass; tests can construct it with overrides without touching argparse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GNNPPOConfig:
    # ---- GNN architecture -----------------------------------------------
    gnn_hidden_dim: int = 64  # hidden channels in GCNConv layer 1
    gnn_output_dim: int = 64  # output channels in GCNConv layer 2 (= D)
    actor_hidden_dim: int = 64  # hidden units in political / military heads
    critic_hidden_dim: int = 64  # hidden units in the value head

    # ---- PPO hyperparameters (matched to existing train_ppo.py defaults) --
    learning_rate: float = 3e-4
    n_steps: int = 512  # rollout steps per env per update
    batch_size: int = 256  # PPO mini-batch size
    n_epochs: int = 10  # PPO update epochs per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ---- training setup -------------------------------------------------
    total_timesteps: int = 500_000
    n_envs: int = 8
    seed: int = 0
    preset: str = "full"  # SovereignConfig preset name
    max_steps: int = 200  # SovereignConfig.max_steps (for obs normalisation)
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"

    # ---- evaluation / checkpointing -------------------------------------
    eval_episodes: int = 30
