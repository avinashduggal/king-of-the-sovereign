"""Configuration for the SOVEREIGN environment.

Single source of truth for every tunable parameter and the three ablation
toggles (legitimacy, occupation cost, neutral posture). Five preset
factories cover the experimental protocol from `instructions.md` Section 10.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SovereignConfig:
    """All parameters governing a SOVEREIGN episode.

    The dataclass is frozen so configurations cannot be mutated mid-episode.
    Construct a new config via one of the preset factories
    (``full_model``, ``no_legitimacy``, ...) or by passing keyword arguments.
    """

    # ---- Ablation toggles -------------------------------------------------
    use_legitimacy: bool = True
    use_occupation_cost: bool = True
    use_neutral_posture: bool = True

    # ---- Episode limits ---------------------------------------------------
    max_steps: int = 1000

    # ---- Initial unit counts ---------------------------------------------
    invader_ground: int = 12
    invader_strike: int = 3
    defender_ground: int = 6
    defender_strike: int = 1
    neutral_ground: int = 4

    # ---- Drift-diffusion coefficients (Section 7.2) ----------------------
    alpha: float = 0.04   # legitimacy coupling
    beta: float = 0.05    # advance shock
    gamma: float = 0.10   # strike shock
    delta: float = 0.04   # negotiate pull
    epsilon: float = 0.03 # alliance pull
    zeta: float = 0.03    # occupation drift
    sigma: float = 0.02   # noise std dev

    # ---- Threshold events (Section 7.3) ----------------------------------
    sanction_threshold: float = 0.60
    coalition_threshold: float = 0.85
    supply_route_threshold: float = -0.60
    alliance_threshold: float = -0.85
    sanction_lift_threshold: float = 0.50
    sanction_lift_steps: int = 5
    sanction_drain_rate: float = 0.01
    occupation_cost_reduction: float = 0.30  # when neutral opens supply routes
    coalition_unit_bonus: int = 2            # extra defender units on coalition
    coalition_legitimacy_penalty: float = 0.10
    alliance_legitimacy_penalty: float = 0.05

    # ---- Political action effects (Section 6.1) --------------------------
    seek_alliance_dL: float = 0.01
    seek_alliance_dtheta: float = -0.05
    impose_sanction_dL: float = -0.02
    impose_sanction_dtheta: float = 0.04
    impose_sanction_dE: float = -0.03
    issue_threat_dL: float = -0.03
    issue_threat_dtheta: float = 0.03
    negotiate_dL: float = 0.03
    negotiate_dtheta: float = -0.04
    do_nothing_decay: float = 0.005          # slow L decay if L < 0.5

    # ---- Military action effects (Section 6.2) ---------------------------
    advance_dL: float = -0.05
    withdraw_dL: float = 0.02
    strike_dL: float = -0.08
    defender_home_bonus: float = 0.20         # +20% effectiveness on home

    # ---- Reward weights (Section 8) --------------------------------------
    w_territory: float = 0.30
    w_resource_capture: float = 0.20
    w_occupation: float = 0.25
    w_legitimacy: float = 0.15
    w_sanction: float = 0.20
    w_insurgency: float = 0.10

    # ---- Terminal rewards (Section 9) ------------------------------------
    terminal_political_collapse: float = -50.0
    terminal_military_defeat: float = -30.0
    terminal_negotiated_settlement: float = 40.0
    terminal_total_conquest: float = 10.0

    # ---- Insurgency (Section 8.3) ----------------------------------------
    insurgency_lambda: float = 0.05

    # ---- Settlement trigger ----------------------------------------------
    # Active mechanic: simple consecutive negotiation (see dynamics/terminal.py
    # for the two alternative settlement triggers preserved as commented blocks)
    settlement_consecutive_steps: int = 5
    settlement_min_legitimacy: float = 0.5


# ---------------------------------------------------------------------------
# Preset factories for the five ablation experiments (Section 10)
# ---------------------------------------------------------------------------

def full_model() -> SovereignConfig:
    """All mechanics active. Expected optimal: negotiate or deter."""
    return SovereignConfig()


def no_legitimacy() -> SovereignConfig:
    """Legitimacy frozen at 1.0. Expected: slower invasion."""
    return SovereignConfig(use_legitimacy=False)


def no_occupation_cost() -> SovereignConfig:
    """Occupation duration not tracked. Expected: partial invasion."""
    return SovereignConfig(use_occupation_cost=False)


def no_neutral_posture() -> SovereignConfig:
    """Neutral posture frozen at 0. Expected: invasion."""
    return SovereignConfig(use_neutral_posture=False)


def baseline() -> SovereignConfig:
    """All political/economic costs disabled. Expected: always invade."""
    return SovereignConfig(
        use_legitimacy=False,
        use_occupation_cost=False,
        use_neutral_posture=False,
    )


ABLATION_PRESETS = {
    "full": full_model,
    "no_legitimacy": no_legitimacy,
    "no_occupation_cost": no_occupation_cost,
    "no_neutral_posture": no_neutral_posture,
    "baseline": baseline,
}


def get_preset(name: str) -> SovereignConfig:
    """Look up a preset by name (raises KeyError on invalid name)."""
    if name not in ABLATION_PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {list(ABLATION_PRESETS)}"
        )
    return ABLATION_PRESETS[name]()
