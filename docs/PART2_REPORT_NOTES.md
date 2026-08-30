# Part II report notes

Use this as a compact source for the final ten-page report. Numeric claims below
come from the generated JSON artifacts and should be refreshed after retraining.

## Environment

Neon Rift Arena is a continuous 800×600 Pygame combat environment. A damageable
ship destroys steering enemies and their periodically spawning rifts. Destroying
all active rifts advances the phase, increasing spawner count, health, enemy
health/speed, and spawn rate. An episode ends on ship destruction or the fixed
simulation-step limit. Training is headless; evaluation renders the identical
state and mechanics.

## Observation

The 34-value normalized feature vector avoids expensive pixel learning. It
contains player kinematics/orientation/health, weapon readiness, nearest enemy
and rift relative features, entity counts, phase/time, aim alignment, and a
signed turn signal for the active target. Six progression features expose ship
level, XP progress, volley size, fire rate, damage, and laser state. Direction
uses unit vectors, angle is encoded with sine/cosine to avoid wrap discontinuity,
and missing targets use an unambiguous sentinel.

## Reward justification

Large sparse terms directly represent the objective: enemy kill, rift kill,
phase progress, damage penalty, and death penalty. Damage-dealt rewards assign
credit to projectiles before a delayed kill. Same-target distance and aim
potential differences help movement/rotation without paying for oscillation or
target replacement. Shot-quality shaping rewards intentional aligned fire.
Every component is returned in `info`, and shaping never changes the mechanics.
Combat XP is explicitly separate from RL reward: it unlocks weapons for gameplay
depth but is not added to the scalar reward or its breakdown.

## Training and tuning

Separate SB3 DQN agents use configurable MLPs, replay memory, target-network
updates, epsilon exploration, action repeat four, checkpoints, TensorBoard, and
held-out seeded model selection. Three configurations varied learning rate,
exploration fraction, and network width. The 20,000-step sweep selected
`balanced` for direct control (mean reward 429.58, mean phase 5.13) and
`fast_exploration` for rotation (mean reward 9.48, 37.5% phase progression).
The final longer budgets were 150,000 and 250,000 decisions respectively. Exact
settings remain in model metadata and
`logs/arena/tuning/hyperparameter_results.json`.

## Control comparison and originality

Direct movement is easier because one action chooses an absolute direction and
shooting auto-aims. Rotation control is harder because the policy must align,
thrust with momentum, and shoot forward. Compare final reward, phase progression,
survival, kills, and accuracy from `logs/arena/control_style_comparison.json`.
Across 20 held-out episodes, direct control achieved mean reward 622.98, 100%
phase progression, 90% time-limit survival, mean phase 6.9, and mean ship level
4.95. Rotation/thrust achieved 186.84 reward, 90% progression, 25% survival,
mean phase 3.1, and level 4.05. This supports the expected conclusion that the
direct action set is easier and more sample-efficient, while the rotation policy
still learned intentional progression and consistently unlocked upgrades.
Original elements include the continuous custom combat simulation, phase
director, normalized targeting/turn representation, auditable shaped reward,
held-out checkpoint selection, automatic five-tier combat progression, distinct
multi-beam/laser weapons, visual policy launcher, and procedural neon VFX.
