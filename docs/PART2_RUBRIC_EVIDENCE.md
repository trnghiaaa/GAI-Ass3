# Part II rubric evidence map

This page maps each marking item to code, automated checks, and generated
artifacts. Numeric claims should be copied from the final JSON files rather than
typed manually into the report.

Schema-7 final status and common-seed results are in
`PART2_SAFETY_EXPERIMENT.md`. Older schema-5/6 artifacts remain as honest
hyperparameter and rejected-candidate evidence.

## G — Real-time Pygame arena

| Criterion | Implementation | Verification/evidence |
|---|---|---|
| Continuous animated arena | `ArenaEnv` uses floating-point position/velocity; `ArenaRenderer` is non-grid Pygame rendering | RGB render contract test and `logs/arena/environment_showcase.png` |
| Movement and shooting | Required direct and rotation action dictionaries plus `_apply_player_action()` and `_fire_projectile()` | Direct/rotation movement tests |
| Spawners and steering enemies | `_update_spawners()`, `_spawn_enemy()`, `_update_enemies()` | Spawn and navigation tests |
| Health and collisions | Circular projectile/entity collisions; player/enemy/spawner health | Damage, destruction, and death tests |
| Phase progression | Destroying all active spawners increments phase, cleanly withdraws old hostiles/projectiles, resets the phase clock, and starts a visible transition | Phase-cleanup/timer tests and final policy benchmarks |
| Terminal rules | Player health reaches zero, a 60-second phase deadline expires, or the long safety budget is exhausted | Termination/truncation tests |

Creative renderer-only presentation features include parallax background,
targeting reticle, projectile trails, hit particles, low-health vignette,
animated spawners, phase/boss/level-up transition VFX, fitted text, telemetry,
a visible keyboard/mouse pause overlay, and a visual launcher.
The additional combat-progression system grants non-RL XP for objectives and
offers seeded three-card build drafts. Uncapped ship levels, 21 upgrade/mastery
paths, permanent drone squadrons, random minibosses, phase support drops,
post-boss relics, and grouped multi-lane boss barrages deepen play without altering either
required action dictionary or the phase progression rule.

## H — API and observation

- Native `ArenaEnv` follows Gymnasium/SB3: `reset() -> (obs, info)` and
  `step() -> (obs, reward, terminated, truncated, info)`.
- `LegacyArenaEnv` exposes the assignment's four-value contract exactly:
  `reset() -> obs` and `step() -> (obs, reward, done, info)`.
- The fixed `float32` observation contains 89 normalized features. It includes
  player position, velocity, orientation, health, nearest enemy/spawner relative
  direction and distance, phase, targeting diagnostics, XP progress, ship level,
  the active weapon, full composed build, support systems, miniboss state, and
  primary, combined, and secondary boss-hazard escape/time signals plus the
  critical/leech/Riftbreaker build state.
- Nineteen appended features expose second-enemy/closing-speed/crowd information,
  wall and spawner clearance, finite boss shields/summons, and body-relative
  hazard escape. This is a compact scene summary, not a fully observed state.
- `ObservationIndex`, `OBSERVATION_NAMES`, and `observation_as_dict()` make every
  position explicit for tests and report tables.

## I — Two control schemes and models

| Control style | Actions | Final model | Visual evaluator |
|---|---|---|---|
| Rotation + thrust | no-op, thrust, rotate left/right, shoot | `models/arena/dqn_rotation.zip` | `python -m arena.evaluate_rotation` |
| Direct movement | no-op, up/down/left/right, shoot | `models/arena/dqn_direct.zip` | `python -m arena.evaluate_direct` |

Metadata beside each model records its control style, observation schema,
network, seed, action repeat, hyperparameters, versions, checkpoint selection,
and held-out benchmark. The submitted 30-episode schema-7 benchmarks report
100% progression for both controls, with mean rewards of 510.30 and 89.48.
Mean phases were 5.70 and 2.70. Direct averaged 1.10 boss kills and 4.77 dodges
versus 1.37 hits; rotation averaged 1.27 dodges versus 0.60 hits. Matching random
baselines scored -14.80 reward/3.33% progression for direct and -37.83/0% for rotation, supporting that progression and build
activation are learned behavior rather than random control.

## J — Reward and deep-RL quality

`ArenaEnv._calculate_reward()` returns a total and exact named breakdown.
Required event rewards are configured in `arena/config.json`: enemy/spawner
destruction, phase advancement, damage taken, and death. Miniboss destruction,
boss clearance, complete barrage dodges, and same-barrage escape improvement
are additional named terms. Small damage, safe-range/crowd-separation,
aim-improvement, and shot-quality terms improve temporal credit assignment and
are fully logged; they do not change environment mechanics. Combat XP is a
separate gameplay currency and is intentionally absent from the RL reward total.

`arena.train` uses Stable-Baselines3 DQN with a two-hidden-layer MLP, replay
buffer, target network, linear epsilon exploration, checkpoints, held-out model
selection, Monitor CSV, TensorBoard event logging, and final seeded benchmark.
`arena.tune` compares fast, balanced, and long-exploration configurations and
writes CSV/JSON/PNG evidence under `logs/arena/tuning`.

## Report-ready files

- `logs/arena/runs/dqn_*/training_curve.png`
- `logs/arena/runs/dqn_*/benchmark.json`
- `logs/arena/tuning/hyperparameter_comparison.png`
- `logs/arena/control_style_comparison.png`
- `logs/arena/boss_avoidance_comparison.png`
- `logs/arena/learning_vs_random.png`
- `logs/arena/environment_showcase.png`
- `logs/arena/pause_showcase.png`
- `logs/arena/upgrade_showcase.png`
- `logs/arena/boss_transition_showcase.png`
- `logs/arena/choice_showcase.png`
- `logs/arena/boss_showcase.png`
- `logs/arena/miniboss_showcase.png`
- `logs/arena/boss_reward_showcase.png`
- `logs/arena/evidence_manifest.json`
