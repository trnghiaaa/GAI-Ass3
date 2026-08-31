# Part II report notes

Use this as a compact source for the final ten-page report. Numeric claims below
come from the generated JSON artifacts and should be refreshed after retraining.

Use the final schema-7 figures in `PART2_SAFETY_EXPERIMENT.md`. Older schema-5
and schema-6 figures are historical tuning evidence, not final gameplay results.

## Environment

Neon Rift Arena is a continuous 800×600 Pygame combat environment. A damageable
ship destroys steering enemies and their periodically spawning rifts. Destroying
all active rifts advances the phase, increasing spawner count, health, enemy
health/speed, contact damage, capacity, and spawn rate. Every third phase is a
boss-rift encounter with a 30–45% shield, three health-gated miniboss summons,
and named multi-lane barrages. Random
Rift Hunter minibosses provide optional risk/reward encounters. An episode ends on ship destruction or
a missed 60-second phase deadline, with a separate long episode safety cap.
Clearing a phase resets the clock and removes surviving hostiles and projectiles
without treating them as kills or granting XP/reward. Training is headless;
evaluation renders the identical state and mechanics.

## Observation

The 89-value normalized feature vector avoids expensive pixel learning. It
contains player kinematics/orientation/health, weapon readiness, nearest enemy
and rift relative features, entity counts, phase/time, aim alignment, and a
signed turn signal for the active target. Six progression features expose ship
level, XP progress, volley size, fire rate, damage, and laser state. Direction
uses unit vectors, angle is encoded with sine/cosine to avoid wrap discontinuity,
and missing targets use an unambiguous sentinel. Further features expose
hull, shielding, range, piercing, splash, engines, wingmen, bomb, boss/miniboss
state, primary/combined/secondary boss-hazard escape and impact-time signals,
Aegis, overdrive, regeneration, homing, late-game mastery, critical chance,
hull siphon, and Riftbreaker power. Simultaneous lanes and procedural builds
are represented explicitly. Schema 6 also exposes second-enemy and crowd
features, closing speeds, wall/spawner clearance, boss shield/summon state,
and body-relative hazard escape. The feature summary is still partially
observable; it does not encode every projectile or enemy trajectory.

## Reward justification

Large sparse terms directly represent the objective: enemy kill, rift kill,
phase progress, damage penalty, and death penalty. Damage-dealt rewards assign
credit to projectiles before a delayed kill. Safe-range target distance, crowd separation, and aim
potential differences help movement/rotation without paying for oscillation or
target replacement. A same-barrage potential difference rewards movement toward
safety, and shot-quality shaping rewards intentional aligned fire.
Every component is returned in `info`, and shaping never changes the mechanics.
Combat XP and drafted upgrades are explicitly separate from RL reward. Miniboss
and boss-clear/full-dodge rewards encourage intentional optional combat and hazard evasion;
they remain named, auditable reward terms. Progression adds gameplay depth but
its XP is not added to the scalar reward or its breakdown.

## Training and tuning

Separate SB3 DQN agents use configurable MLPs, replay memory, target-network
updates, epsilon exploration, action repeat four, checkpoints, TensorBoard, and
held-out seeded, boss-aware model selection. Three configurations varied learning
rate, exploration fraction, and network width. In the final 35,000-step sweep,
direct long exploration led with 766.30 mean reward and 100% progression.
Rotation fast exploration learned fastest at the short budget (47.12 reward,
50% progression), while the wider long-exploration network was retained for the
much larger final budget so exploration continued through boss encounters. The
direct safety transfer added 450,000 decisions. Fresh 800,000-step and
300,000-step consolidation rotation attempts were rejected because their
safety gains did not preserve progression/dodge reliability; the proven
300,000-decision rotation checkpoint was retained. Exact settings remain in model metadata and
`logs/arena/tuning/hyperparameter_results.json`.

## Control comparison and originality

Direct movement is easier because one action chooses an absolute direction and
shooting receives close-range target assist; outside lock range, shots follow
the current heading. Rotation control must align,
thrust with momentum, and shoot forward. Compare final reward, phase progression,
survival, kills, and accuracy from `logs/arena/control_style_comparison.json`.
Across 30 final held-out episodes, direct achieved 510.30 reward, 100%
progression, mean/max Phase 5.70/12, and 1.10 boss kills. Rotation achieved
89.48 reward, 100% progression, and Phase 2.70/3. Direct averaged 4.77 dodges
versus 1.37 hits; rotation averaged 1.27 versus 0.60. Neither policy merely waited for
a deadline: they pursued increasingly dangerous objectives until destroyed.
The direct policy progressed further because absolute movement and close target
assist make control easier; rotation remained effective while learning the
harder coupled aiming and momentum problem. The matching seeded random baselines
scored -14.80 reward/3.33% progression for direct and -37.83/0% for rotation,
supporting that the submitted policies learned purposeful progression.
Original elements include the continuous custom combat simulation, phase
director, normalized targeting/turn representation, auditable shaped reward,
held-out boss-aware checkpoint selection, uncapped three-card build drafts, 21 upgrade and
repeatable mastery paths, permanent multi-drone squadrons, support drops, random
minibosses, post-boss relics, grouped cross/diagonal/trident/nova boss barrages,
distinct multi-beam/laser/piercing/splash/critical weapons, hull siphon,
specialist Riftbreaker damage, a visual policy launcher, and procedural neon VFX.
