# King of the Sovereign — PPT Slide Draft

---

## Slide 1 — Title

**King of the Sovereign**
*Learning Geopolitical Strategy via Deep Reinforcement Learning*

- CS 272 Final Project
- Three-nation simulation: Invader vs. Defender vs. Neutral
- Core question: Can an RL agent learn that invasion is a dominated strategy?

---

## Slide 2 — Game Environment Overview

**The Sovereign Environment**

- Built as a custom Gymnasium-compatible environment
- 3 actors: Invader (RL agent), Defender (rule-based), Neutral (stochastic)
- Episode length: up to 200 steps
- 5 termination conditions: negotiated settlement, political collapse, military defeat, total conquest, max steps
- 5 ablation presets: `full`, `no_legitimacy`, `no_occupation_cost`, `no_neutral_posture`, `baseline`

---

## Slide 3 — Game Map Topology

**9-Territory Adjacency Graph**

```
[I_HOME]──[C1]──[C3]──[D_HOME]
    │       │      │
   [C2]──[C4]─────┘
    │       │
   [C5]──[C6]──[N_HOME]
```

- 9 territories, each with resource value (0.2–0.6) and strategic value (0.2–0.8)
- C4 is the central flashpoint — connects 4 territories, highest contention
- Invader needs ≥2 hops to reach Defender home; single path through C6 to Neutral
- Graph encoded as plain adjacency dict — no external library dependency

---

## Slide 4 — Observation & Action Space

**What the Agent Sees and Does**

**Observation (Dict, 11 components):**
- `territory_control` — one-hot [I/D/N] per territory (27-dim)
- `invader_units`, `defender_units` — units per territory (9-dim each)
- `legitimacy` (L), `supply` (E), `theta` (θ, Neutral posture)
- `occupation_duration`, `timestep`, `sanctions_active`, `neutral_joined_defender`, `neutral_allied_invader`

**Action Space:** `MultiDiscrete([5, 4, 9])`
- Political: SEEK_ALLIANCE / IMPOSE_SANCTION / ISSUE_THREAT / NEGOTIATE / DO_NOTHING
- Military: ADVANCE / HOLD / WITHDRAW / STRIKE
- Target territory: 0–8

---

## Slide 5 — Reward Function

**Shaping Incentives Across 4 Dimensions**

**Per-step rewards:**
- (+) Territory resource income for connected, Invader-held territories
- (+) Resource capture bonus on new territory gains
- (−) Occupation cost (grows with duration)
- (−) Legitimacy deficit penalty
- (−) Sanction drain when Neutral is hostile (θ > 0.6)
- (−) Insurgency damage cost

**Terminal rewards:**
- +40 Negotiated settlement | +10 Total conquest
- −30 Military defeat | −50 Political collapse

---

## Slide 6 — Models Trained

**Algorithm Zoo**

| Model | Algorithm | Timesteps |
|---|---|---|
| PPO | Proximal Policy Optimization (MultiInputPolicy) | 200K |
| A2C | Advantage Actor-Critic | 200K |
| DQN | Deep Q-Network (flattened actions) | 200K |
| Recurrent PPO | PPO + LSTM hidden state | 200K |
| GNN-PPO | PPO + Graph Neural Network encoder | 500K |

- All trained on the `full` preset (all mechanics active)
- Checkpoints saved as `.zip` with accompanying `train.log` per model

---

## Slide 7 — Model Intuition: PPO & A2C

**Actor-Critic Family**

**A2C (Advantage Actor-Critic):**
- Separate actor (policy) and critic (value function) networks
- Updates synchronously after each rollout batch
- Advantage = actual return − estimated value → reduces gradient variance
- Simple, fast, good baseline

**PPO (Proximal Policy Optimization):**
- Adds a clipped surrogate objective on top of A2C
- Prevents large policy updates (clip ε = 0.2) — more stable training
- Multiple gradient steps per rollout batch
- Handles Dict observation space natively via MultiInputPolicy

---

## Slide 8 — Model Intuition: DQN

**Value-Based Approach**

- Learns Q(s, a): expected return from state s taking action a
- Action selected greedily: argmax Q(s, a)
- Replay buffer breaks temporal correlation between samples
- Target network updated slowly to stabilize learning
- Challenge: MultiDiscrete action space → flattened to single discrete dimension (5×4×9 = 180 actions)
- ε-greedy exploration decays over training

---

## Slide 9 — Model Intuition: Recurrent PPO & GNN-PPO

**Memory and Structure**

**Recurrent PPO (LSTM):**
- Replaces MLP feature extractor with LSTM layer
- Maintains hidden state across timesteps within an episode
- Captures temporal dependencies (e.g., Neutral's posture shift over time)
- Useful when the current observation alone is insufficient for optimal decisions

**GNN-PPO:**
- Custom Graph Convolutional Network (GCN) encoder built on top of PPO
- Node features: units, control, resource value per territory
- 2-layer GCN → mean-pooled graph embedding + per-node embeddings
- Separate actor heads for political, military, and target dimensions
- Encodes map topology directly — spatially aware policy

---

## Slide 10 — Training Metrics

**What We Monitor During Training**

| Metric | Meaning |
|---|---|
| `rollout/ep_rew_mean` | Average episode return — primary performance signal |
| `rollout/ep_len_mean` | Average episode length — longer = more complex behavior |
| `train/value_loss` | Critic's MSE on predicted vs. actual returns |
| `train/explained_variance` | R² of critic predictions (0 = random, 1 = perfect) |
| `train/policy_gradient_loss` | PPO policy objective (should decrease then stabilize) |
| `train/approx_kl` | KL divergence from old to new policy (watch for spikes) |
| `train/clip_fraction` | % of updates hitting the PPO clip boundary |
| `train/entropy_loss` | Negative entropy — low = more deterministic policy |
| `time/fps` | Environment steps per second — training throughput |

---

## Slide 11 — Key Metric Deep Dives

**Reading the Training Signal**

**`value_loss`**
- Measures how well the critic estimates future rewards
- High early in training (random policy → noisy returns); should decrease as critic learns

**`explained_variance`**
- R² score: how much variance in actual returns is explained by the critic's prediction
- Starts near 0; target is close to 1.0 — sanity check on critic quality
- Negative values mean the critic is worse than predicting the mean return

**`approx_kl` & `clip_fraction`**
- KL spike → policy changed too aggressively; clip_fraction > 0.3 signals instability
- PPO's clipping mechanism keeps both in check

**`entropy_loss`**
- High entropy = exploratory policy; declining entropy = policy converging to deterministic strategy
- Too low too early → premature convergence

---

## Slide 12 — Inference & Results

**What Did the Models Learn?**

- Evaluation: deterministic rollouts from saved checkpoints, 50+ episodes each
- Metrics reported: mean return, mean episode length, termination reason breakdown

**Observed behaviors by model:**
- PPO / A2C: Learned mixed strategies — NEGOTIATE often triggered; political collapse rare
- DQN: Struggled with large flattened action space; more erratic behavior
- Recurrent PPO: Better handling of Neutral's posture evolution; fewer legitimacy collapses
- GNN-PPO: Most spatially coherent — targeted C4 strategically; best average return at 500K steps

**Termination breakdown (full preset, GNN-PPO):**
- Negotiated settlement: ~40–50% of episodes
- Political collapse: ~15%
- Max steps reached: ~20–25%
- Military defeat: ~10%

---

## Slide 13 — Ablation Study

**Which Mechanics Drive Strategy?**

| Preset | Key Behavior |
|---|---|
| `full` | Agent learns deterrence / negotiation |
| `no_legitimacy` | More aggressive invasion attempts |
| `no_occupation_cost` | Partial occupation of contested territories |
| `no_neutral_posture` | Invasion (no coalition threat to deter) |
| `baseline` | Always invades (no costs at all) |

- Legitimacy (L) and Neutral posture (θ) are the critical deterrence mechanisms
- Removing either causes the agent to shift toward invasion strategies

---

## Slide 14 — Conclusion

**Key Takeaways**

- A militarily superior RL agent can learn that invasion is strategically dominated — given the right reward shaping
- Legitimacy (L) and Neutral posture (θ) are the dominant deterrence levers
- GNN-PPO best captures the spatial structure of the game map; outperforms flat-observation baselines
- Recurrent memory helps with long-horizon Neutral dynamics
- Future work: self-play, multi-agent training, larger maps, real-world geopolitical calibration

*"Rational deterrence emerges from experience, not from programming."*
