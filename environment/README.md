# SOVEREIGN — Gymnasium Environment

A Gymnasium-compatible strategic-simulation environment for deep reinforcement
learning research (CS 272). Models a three-nation geopolitical conflict to
investigate whether a militarily superior **Invader** agent learns that
invasion is a strategically dominated strategy.

The full game rulebook is in [`../instructions.md`](../instructions.md).

## Install

From the `environment/` directory:

```bash
pip install -e .                  # core (gymnasium + numpy)
pip install -e ".[test]"          # also installs pytest
pip install -e ".[train]"         # also installs stable-baselines3 + torch
```

## Quickstart

```python
import gymnasium as gym
import sovereign  # registers the env ids

env = gym.make("Sovereign-v0", render_mode="human")
obs, info = env.reset(seed=42)

terminated = truncated = False
total = 0.0
while not (terminated or truncated):
    action = env.action_space.sample()              # MultiDiscrete([5, 4, 9])
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
print("episode return:", total, "reason:", info["termination_reason"])
```

## Action and observation spaces

* **Action**: `spaces.MultiDiscrete([5, 4, |V|])` -- `(political, military, target)`.
  * Political: `SEEK_ALLIANCE`, `IMPOSE_SANCTION`, `ISSUE_THREAT`, `NEGOTIATE`, `DO_NOTHING`.
  * Military: `ADVANCE`, `HOLD`, `WITHDRAW`, `STRIKE`.
  * Target: territory id `0..|V|-1`. Ignored for `HOLD`. Invalid targets
    (e.g., `STRIKE` on an empty territory) silently degrade to `HOLD`;
    the env sets `info["action_was_degraded"] = True` so you can monitor
    this during training.
* **Observation**: `spaces.Dict` with named components. Use SB3's
  `MultiInputPolicy` for native support, or wrap with
  `gymnasium.wrappers.FlattenObservation` for algorithms that need a flat
  vector (e.g., DQN).

## Ablation experiments

Each of the five experiments from rulebook Section 10 is registered as its
own gym id:

| Preset | Gym id | Expected optimal policy |
|---|---|---|
| Full model | `Sovereign-v0` or `Sovereign-full-v0` | Negotiate or deter |
| No legitimacy | `Sovereign-no_legitimacy-v0` | Slower invasion |
| No occupation cost | `Sovereign-no_occupation_cost-v0` | Partial invasion |
| No neutral posture | `Sovereign-no_neutral_posture-v0` | Invasion |
| Baseline (all off) | `Sovereign-baseline-v0` | Always invade |

Or build a custom config:

```python
from sovereign import SovereignConfig, SovereignEnv
cfg = SovereignConfig(use_legitimacy=False, w_occupation=0.5)
env = SovereignEnv(config=cfg)
```

## Verifying the install

```bash
# Unit tests
pytest tests/ -v

# Gymnasium API conformance
python -c "import gymnasium as gym, sovereign; \
  from gymnasium.utils.env_checker import check_env; \
  check_env(gym.make('Sovereign-v0').unwrapped)"

# Random-policy sanity rollout
python scripts/random_rollout.py

# Run all five ablation presets
python scripts/run_ablations.py

# (Optional) PPO smoke train, requires the [train] extra
python scripts/train_ppo.py --timesteps 5000
```

## Repo layout

```
environment/
  pyproject.toml
  README.md
  sovereign/                 -- the gym env package
    __init__.py              -- gym.register for Sovereign-v0 + 5 variants
    config.py                -- SovereignConfig + 5 preset factories
    game_map.py              -- Territory + GameMap (default 9-node topology)
    actions.py               -- PoliticalAction + MilitaryAction enums
    state.py                 -- GameState dataclass + observation builder
    defender.py              -- Rule-based DefenderPolicy
    reward.py                -- Per-term reward calculator
    renderer.py              -- ASCII state view
    env.py                   -- SovereignEnv (the main class)
    dynamics/
      political.py           -- L updates from political actions
      military.py            -- ADVANCE/HOLD/WITHDRAW/STRIKE + combat
      neutral.py             -- drift-diffusion + threshold events + hysteresis
      economy.py             -- supply index + sanction drain
      insurgency.py          -- Bernoulli unit destruction
      terminal.py            -- terminal checks; 3 settlement variants
                                 (1 active, 2 commented)
  tests/                     -- pytest test suite
  scripts/                   -- random rollout, ablation runner, training demos
```
