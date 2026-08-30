# Part II report notes

Use this as a compact source for the final ten-page report. Numeric claims below
come from the generated JSON artifacts and should be refreshed after retraining.

## Environment

Neon Rift Arena is a continuous 800×600 Pygame combat environment. A damageable
ship destroys steering enemies and their periodically spawning rifts. Destroying
all active rifts advances the phase, increasing spawner count, health, enemy
health/speed, contact damage, capacity, and spawn rate. Every third phase is a
boss-rift encounter with elite minions. An episode ends on ship destruction or
a missed 60-second phase deadline, with a separate long episode safety cap.
Clearing a phase resets the clock and removes surviving hostiles and projectiles
without treating them as kills or granting XP/reward. Training is headless;
evaluation renders the identical state and mechanics.

## Observation

The 43-value normalized feature vector avoids expensive pixel learning. It
contains player kinematics/orientation/health, weapon readiness, nearest enemy
and rift relative features, entity counts, phase/time, aim alignment, and a
signed turn signal for the active target. Six progression features expose ship
level, XP progress, volley size, fire rate, damage, and laser state. Direction
uses unit vectors, angle is encoded with sine/cosine to avoid wrap discontinuity,
and missing targets use an unambiguous sentinel. Nine further features expose
hull, shielding, range, piercing, splash, engines, wingman, bomb, and boss state,
so procedural builds remain Markov rather than hidden from the agent.

## Reward justification

Large sparse terms directly represent the objective: enemy kill, rift kill,
phase progress, damage penalty, and death penalty. Damage-dealt rewards assign
credit to projectiles before a delayed kill. Same-target distance and aim
potential differences help movement/rotation without paying for oscillation or
target replacement. Shot-quality shaping rewards intentional aligned fire.
Every component is returned in `info`, and shaping never changes the mechanics.
Combat XP and drafted upgrades are explicitly separate from RL reward: they add
gameplay depth but are not added to the scalar reward or its breakdown.

## Training and tuning

Separate SB3 DQN agents use configurable MLPs, replay memory, target-network
updates, epsilon exploration, action repeat four, checkpoints, TensorBoard, and
held-out seeded model selection. Three configurations varied learning rate,
exploration fraction, and network width. The 25,000-step sweep selected
`long_exploration` for direct control (mean reward 1518.12, 100% phase
progression) and `fast_exploration` for rotation (mean reward 14.35, 33.3% phase
progression). The final longer budgets were 200,000 and 300,000 decisions
respectively. Exact
settings remain in model metadata and
`logs/arena/tuning/hyperparameter_results.json`.

## Control comparison and originality

Direct movement is easier because one action chooses an absolute direction and
shooting receives close-range target assist; outside lock range, shots follow
the current heading. Rotation control must align,
thrust with momentum, and shoot forward. Compare final reward, phase progression,
survival, kills, and accuracy from `logs/arena/control_style_comparison.json`.
Across 20 held-out episodes, direct control achieved mean reward 872.52, 100%
phase progression, mean phase 9.9, mean ship level 8.3, and 2.75 boss-rift kills
per episode. Rotation/thrust achieved 241.72 reward, 100% progression, mean
phase 4.25, level 5.95, and 0.8 boss-rift kills. Neither policy merely waited for
a deadline: they pursued increasingly dangerous objectives until destroyed.
The direct policy progressed further because absolute movement and close target
assist make control easier; rotation remained effective while learning the
harder coupled aiming and momentum problem. The seeded random direct baseline
averaged phase 1.25 with 25% progression, while random rotation never left phase
1, supporting that the submitted policies learned purposeful progression.
Original elements include the continuous custom combat simulation, phase
director, normalized targeting/turn representation, auditable shaped reward,
held-out checkpoint selection, three-card build drafts, eleven stackable upgrade
types, support drops, boss rifts, distinct multi-beam/laser/piercing/splash
weapons, a visual policy launcher, and procedural neon VFX.
