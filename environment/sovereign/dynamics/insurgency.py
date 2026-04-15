"""Insurgency rolls.

Hazard model from rulebook Section 8.3:

    p(insurgency | t_occ) = 1 - exp(-λ · t_occ)

A successful roll destroys one Invader unit (drawn uniformly from
territories with at least one Invader unit). Disabled when the
``use_occupation_cost`` toggle is off (since t_occ is held at 0 in that
ablation).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import SovereignConfig
from ..state import GameState


def insurgency_probability(t_occ: int, lam: float) -> float:
    """Per-step insurgency probability."""
    if t_occ <= 0:
        return 0.0
    return 1.0 - math.exp(-lam * t_occ)


def roll_insurgency(
    state: GameState,
    config: SovereignConfig,
    rng: np.random.Generator,
) -> dict:
    """Roll for an insurgency event and apply it if it fires.

    Returns an info dict with ``insurgency_fired`` and the territory id
    where the unit was destroyed (if any).
    """
    info: dict = {"insurgency_fired": False, "insurgency_territory": None}

    if not config.use_occupation_cost:
        return info

    p = insurgency_probability(state.occupation_duration, config.insurgency_lambda)
    if p <= 0.0:
        return info
    if rng.random() >= p:
        return info

    # Insurgency fires. Pick a random Invader-occupied territory.
    candidates = [
        tid for tid in range(len(state.invader_units))
        if state.invader_units[tid] >= 1
    ]
    if not candidates:
        return info  # nothing to destroy
    target = int(rng.choice(candidates))
    state.invader_units[target] -= 1

    info["insurgency_fired"] = True
    info["insurgency_territory"] = target
    return info
