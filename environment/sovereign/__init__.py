"""SOVEREIGN: a Gymnasium-compatible strategic-simulation environment.

Importing this package registers ``Sovereign-v0`` plus one variant per
ablation preset:

    Sovereign-full-v0
    Sovereign-no_legitimacy-v0
    Sovereign-no_occupation_cost-v0
    Sovereign-no_neutral_posture-v0
    Sovereign-baseline-v0

Construct via ``gymnasium.make("Sovereign-v0")`` or any of the variants.
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from .config import (
    SovereignConfig,
    ABLATION_PRESETS,
    full_model,
    no_legitimacy,
    no_occupation_cost,
    no_neutral_posture,
    baseline,
    get_preset,
)
from .env import SovereignEnv

__all__ = [
    "SovereignConfig",
    "SovereignEnv",
    "ABLATION_PRESETS",
    "full_model",
    "no_legitimacy",
    "no_occupation_cost",
    "no_neutral_posture",
    "baseline",
    "get_preset",
]


# Default registration uses the full model.
register(
    id="Sovereign-v0",
    entry_point="sovereign.env:SovereignEnv",
    max_episode_steps=200,
)

# One registered id per ablation preset, so users can switch experiments via
# the env id alone (e.g., ``gym.make("Sovereign-baseline-v0")``).
for _preset_name in ABLATION_PRESETS:
    register(
        id=f"Sovereign-{_preset_name}-v0",
        entry_point="sovereign.env:SovereignEnv",
        kwargs={"config_preset": _preset_name},
        max_episode_steps=200,
    )
