"""Per-mechanic dynamics modules.

Each module owns one slice of the simulation (political, military, neutral
posture, economy, insurgency, terminal). The env imports these and
composes them in the canonical 12-step turn order.
"""
