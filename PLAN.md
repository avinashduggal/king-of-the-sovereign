# SOVEREIGN Gymnasium Environment — Implementation Plan

## Context

The `environment/` directory of the SOVEREIGN project is empty. SOVEREIGN is a CS 272 deep reinforcement learning research project that models a three-nation geopolitical conflict (Invader, Defender, Neutral) to investigate whether a militarily superior agent learns that invasion is a strategically dominated strategy. The full game rulebook lives in `instructions.md`.

We need to build a Gymnasium-compatible environment that:
- Implements the full game spec (state, joint action space, deterministic combat, stochastic neutral posture, threshold events, reward function, terminal conditions).
- Supports the **5 ablation experiments** that are the central research deliverable (toggle legitimacy, occupation cost, neutral posture).
- Works with multiple RL algorithms (PPO, A2C, DQN) via Stable-Baselines3 and similar libraries.
- Is modular enough that individual mechanics can be swapped or tested in isolation.

The intended outcome is a self-contained, installable Python package at `environment/sovereign/` with a registered Gym env (`Sovereign-v0`), 5 registered ablation variants, a rule-based Defender opponent, comprehensive tests, and runnable example scripts.

---

## Confirmed Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Package structure | **Modular package** (~19 files) | Clean ablation swappability; each dynamics module testable in isolation |
| Observation space | `spaces.Dict` with named components | Self-documenting, SB3 `MultiInputPolicy` handles natively, wrappable to flat for DQN |
| Action space | `MultiDiscrete([5, 4, 9])` (political, military, target) | Wide RL-library compatibility; flatten wrapper for DQN |
| Map | 9-territory adjacency dict (no networkx) | Tiny graph, zero deps |
| Defender policy | Priority cascade (defend home → counterattack → fortify) | Competent but predictable; not a self-play partner |
| Settlement trigger | **Simple consecutive negotiation** (5 consecutive NEGOTIATE while L > 0.5) | User-selected; easier discovery for early training |
| Settlement alternatives | **Strict** and **Defender-initiated** preserved as commented blocks | Easy to swap later for ablations / further experiments |
| Config | Frozen `SovereignConfig` dataclass + 5 preset factories | Single source of truth for all parameters and toggles |
| Stochasticity | Use `self.np_random` (gymnasium-managed RNG) everywhere | Seeded reproducibility |

---

## File / Package Layout

```
environment/
  pyproject.toml                          # package metadata; deps: gymnasium, numpy
  README.md                               # usage, install, ablation reference
  sovereign/
    __init__.py                           # gym.register Sovereign-v0 + 5 ablation variants
    config.py                             # SovereignConfig dataclass + 5 preset factories
    game_map.py                           # Territory dataclass, GameMap, default 9-node topology
    actions.py                            # PoliticalAction / MilitaryAction IntEnums
    state.py                              # GameState dataclass + to_observation() builder
    defender.py                           # DefenderPolicy (rule-based priority cascade)
    reward.py                             # RewardCalculator with per-term computation
    renderer.py                           # render_text() ASCII view for human/ansi modes
    env.py                                # SovereignEnv(gymnasium.Env) — main step loop
    dynamics/
      __init__.py
      political.py                        # apply_political_action(): L updates from political action
      military.py                         # resolve_advance/strike/withdraw/hold(); combat
      neutral.py                          # drift-diffusion update + threshold events + hysteresis
      economy.py                          # supply index update; sanctions drain
      insurgency.py                       # Bernoulli unit destruction based on t_occ
      terminal.py                         # check_terminal(); houses 3 settlement variants (1 active, 2 commented)
  tests/
    __init__.py
    test_env.py                           # gymnasium API compliance, seeded determinism
    test_dynamics.py                      # per-module unit tests
    test_ablations.py                     # verify 5 presets disable correct mechanics
    test_map.py                           # adjacency / connectivity checks
  scripts/
    random_rollout.py                     # sanity check: random action episode
    run_ablations.py                      # loop all 5 presets with random rollouts
    train_ppo.py                          # SB3 PPO with MultiInputPolicy
    train_dqn.py                          # SB3 DQN with FlattenObservation + flat-action wrappers
```

**Total: 23 files** (19 source + 4 tests/scripts grouping).

---

## Key Type Specifications

### Observation space

```python
spaces.Dict({
    "territory_control":       spaces.MultiBinary(9 * 3),                  # one-hot (I/D/N) per territory
    "invader_units":           spaces.Box(0, 20, (9,), np.int32),
    "defender_units":          spaces.Box(0, 20, (9,), np.int32),
    "legitimacy":              spaces.Box(0.0, 1.0, (1,), np.float32),
    "supply":                  spaces.Box(0.0, 1.0, (1,), np.float32),
    "theta":                   spaces.Box(-1.0, 1.0, (1,), np.float32),
    "occupation_duration":     spaces.Box(0, 200, (1,), np.int32),
    "timestep":                spaces.Box(0, 200, (1,), np.int32),
    "sanctions_active":        spaces.MultiBinary(1),
    "neutral_joined_defender": spaces.MultiBinary(1),
    "neutral_allied_invader":  spaces.MultiBinary(1),
})
```

### Action space

```python
spaces.MultiDiscrete([5, 4, 9])
# index 0: PoliticalAction  (SEEK_ALLIANCE=0, IMPOSE_SANCTION=1, ISSUE_THREAT=2, NEGOTIATE=3, DO_NOTHING=4)
# index 1: MilitaryAction   (ADVANCE=0, HOLD=1, WITHDRAW=2, STRIKE=3)
# index 2: target territory id 0..8 (ignored for HOLD/WITHDRAW/DO_NOTHING)
```

Invalid targets degrade to HOLD; `info["action_was_degraded"] = True` for diagnostics.

### Settlement trigger (active code)

```python
# Active: simple consecutive negotiation (per user choice)
def check_settlement(state, political_action, config):
    if political_action == PoliticalAction.NEGOTIATE and state.legitimacy > 0.5:
        state.consecutive_negotiate += 1
    else:
        state.consecutive_negotiate = 0
    if state.consecutive_negotiate >= config.settlement_consecutive_steps:  # default 5
        return True, config.terminal_negotiated_settlement   # +40
    return False, 0.0

# ---- ALTERNATIVE 1: Strict diplomatic posture (kept as commented block) ----
# def check_settlement(state, political_action, military_action, config, game_map):
#     if (
#         political_action == PoliticalAction.NEGOTIATE
#         and military_action in (MilitaryAction.HOLD, MilitaryAction.WITHDRAW)
#         and state.legitimacy >= 0.60
#         and state.theta < 0.40
#         and not invader_holds_defender_home(state, game_map)
#         and state.timestep >= 10
#     ):
#         return True, config.terminal_negotiated_settlement
#     return False, 0.0

# ---- ALTERNATIVE 2: Defender-initiated (kept as commented block) ----
# Defender policy sets state.settlement_offered = True when L > 0.55,
# Invader has no Defender-home control, theta < 0.6.
# Offer expires after 3 turns. Invader accepts via NEGOTIATE while flag active.
# Requires adding `settlement_offered` and `offer_expires_at` to GameState
# and exposing `settlement_offered` in observation Dict.
```

---

## Ablation Toggle → Code Path Mapping

| Toggle | When OFF | Affected modules |
|---|---|---|
| `use_legitimacy=False` | `political.apply_political_action()` skips L updates; L stays 1.0. Terminal `L<=0` check disabled. `w_legitimacy * (1-L)` = 0. Drift term `alpha * (1-L)` = 0. | `dynamics/political.py`, `reward.py`, `dynamics/terminal.py` |
| `use_occupation_cost=False` | `t_occ` not incremented (stays 0). `w_occupation` term = 0. Insurgency roll skipped (p=0). | `dynamics/military.py`, `dynamics/insurgency.py`, `reward.py` |
| `use_neutral_posture=False` | `neutral.update_posture()` returns early; θ=0. No threshold events fire. Defender gets no coalition bonus. | `dynamics/neutral.py`, `env.py` |

Implementation pattern — guard at top of each dynamics function:

```python
def update_posture(state, actions, config, rng):
    if not config.use_neutral_posture:
        return state
    # ... drift-diffusion logic
```

Five preset factories in `config.py`: `full_model()`, `no_legitimacy()`, `no_occupation_cost()`, `no_neutral_posture()`, `baseline()`.

---

## Critical Files (Must Get Right)

| File | Why critical |
|---|---|
| `sovereign/env.py` | Wires together every subsystem; orchestrates the 12-step turn structure from rulebook §5 |
| `sovereign/config.py` | Single source of truth — typos in defaults silently bias all experiments |
| `sovereign/dynamics/neutral.py` | Drift-diffusion + threshold events + hysteresis is the most subtle mechanic; must use `np_random` for reproducibility |
| `sovereign/dynamics/military.py` | Combat resolution + ADVANCE adjacency validation + WITHDRAW reset semantics |
| `sovereign/state.py` | Observation builder must produce values that always match `observation_space` |
| `sovereign/__init__.py` | Gym registration enables `gym.make("Sovereign-v0")` and ablation variants |

---

## Default Map Topology (9 territories)

```
    [I_HOME]──[C1]──[C3]──[D_HOME]
        │      │       │
       [C2]──[C4]────────┘
        │      │
       [C5]──[C6]──[N_HOME]
```

Territory IDs: `0=I_HOME, 1=C1, 2=C2, 3=C3, 4=C4, 5=C5, 6=D_HOME, 7=C6, 8=N_HOME`

Resource / strategic values balanced so C4 is the central flashpoint, Invader needs ≥2 edges to reach D_HOME, Neutral is reachable through C6.

---

## Implementation Order

1. **Foundation** — `config.py`, `game_map.py`, `actions.py`, `state.py` (no deps; testable immediately)
2. **Dynamics** — `dynamics/political.py`, `military.py`, `neutral.py`, `economy.py`, `insurgency.py`, `terminal.py`
3. **Agents & reward** — `defender.py`, `reward.py`
4. **Environment** — `env.py`, `renderer.py`, `__init__.py` (registration)
5. **Packaging** — `pyproject.toml`, `README.md`
6. **Tests** — `test_map.py`, `test_dynamics.py`, `test_ablations.py`, `test_env.py`
7. **Scripts** — `random_rollout.py`, `run_ablations.py`, `train_ppo.py`, `train_dqn.py`

---

## Verification

After implementation, validate end-to-end with:

```bash
cd "/Users/avinashduggal/CS 272/king-of-the-sovereign/environment"
pip install -e ".[test]"

# 1. Unit tests (must all pass)
pytest tests/ -v

# 2. Gymnasium API conformance (built-in checker)
python -c "import gymnasium as gym; import sovereign; \
  from gymnasium.utils.env_checker import check_env; \
  check_env(gym.make('Sovereign-v0').unwrapped)"

# 3. Random-policy sanity rollout
python scripts/random_rollout.py

# 4. Ablation differentiation (each preset should produce different stats)
python scripts/run_ablations.py

# 5. (Optional, requires sb3) PPO smoke train for ~10k steps
pip install -e ".[train]"
python scripts/train_ppo.py --timesteps 10000

# 6. Seed determinism manual check
python -c "
import gymnasium as gym, sovereign, numpy as np
e1 = gym.make('Sovereign-v0'); e2 = gym.make('Sovereign-v0')
o1, _ = e1.reset(seed=42); o2, _ = e2.reset(seed=42)
for _ in range(50):
    a = e1.action_space.sample()
    s1 = e1.step(a); s2 = e2.step(a)
    assert np.allclose(s1[0]['theta'], s2[0]['theta']), 'Non-deterministic!'
    if s1[2] or s1[3]: break
print('OK: seeded trajectories match.')
"
```

Expected results:
- All pytests pass
- `check_env` reports no warnings
- Random rollout produces an episode with bounded reward and explicit termination reason in `info`
- Ablation script shows visibly different mean episode lengths / termination distributions across the 5 presets
- Seed determinism check prints `OK`

---

## Risks and Sharp Edges

1. **Python 3.14 wheel availability** — Check `pip install gymnasium` succeeds before committing; fall back to Python 3.12 in the venv if needed.
2. **All randomness must use `self.np_random`** (set by `gym.Env.reset(seed=...)`), never global `np.random` or `random`. Otherwise seeded rollouts will not be reproducible across processes.
3. **Reward scale mismatch** — per-step rewards are O(0.5), terminals are O(50). PPO with GAE handles this; DQN may need reward clipping.
4. **Action degradation is silent to the agent** — log via `info["action_was_degraded"]` so training diagnostics can detect when this dominates.
5. **Settlement reachability under "simple consecutive"** — agent can in principle invade then negotiate. Document this as a known property of the chosen mechanic; it is the trade-off we accepted by picking simplicity over strict diplomatic posture.
6. **Defender policy stationarity** — deterministic given state, so exploitable. Acceptable for the research question (makes "invasion dominated" finding stronger), but document.
