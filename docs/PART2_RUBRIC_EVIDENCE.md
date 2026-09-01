# Part II rubric evidence map

This is the current schema-8 submission map. Generated JSON/CSV files are the
source for numeric report claims.

## G — Real-time Pygame arena

| Criterion | Implementation and evidence |
|---|---|
| Continuous animated arena | Float-coordinate `ArenaEnv` and `ArenaRenderer`; `logs/arena/evidence/environment_showcase.png` |
| Ship, shooting, spawners and steering enemies | `_apply_player_action()`, `_fire_projectile()`, `_update_spawners()`, `_update_enemies()`; mechanics tests |
| Health, collisions and phase system | Circular collisions, player/enemy/rift health, clean phase hand-off and per-phase clocks; phase/boss transition screenshots |
| Terminal conditions | Player death, phase deadline, or long safety cap; termination tests |

The creative additions (bosses, minibosses, XP/drafts, wingmen, support drops,
VFX and pause UI) do not replace any required action, health, collision, or
phase mechanic.

## H — Gym-style API and observation

- `ArenaEnv.reset()` and `ArenaEnv.step()` use Gymnasium/SB3 return values;
  `LegacyArenaEnv` exposes the four-value assignment adapter.
- `render()` supports visible Pygame evaluation and RGB arrays.
- The observation is a fixed 89-feature numeric `float32` vector: player
  position/velocity/orientation/health, nearest enemy/rift geometry, phase,
  targeting, build state, and boss-hazard escape information.
- `ObservationIndex`, `OBSERVATION_NAMES`, and `observation_as_dict()` make
  each field inspectable; corresponding tests validate shape/ranges.

## I — Two control schemes and saved models

| Style | Actions | Submitted model | Visual evaluation |
|---|---|---|---|
| Rotation + thrust | no-op, thrust, rotate L/R, shoot | `models/arena/dqn_rotation.zip` | `python -m arena.evaluate_rotation` |
| Direct | no-op, up/down/left/right, shoot | `models/arena/dqn_direct.zip` | `python -m arena.evaluate_direct` |

Each model has metadata and a deterministic 30-episode normal-start evaluation
under `logs/arena/evidence`. The direct policy also has a 30-episode fixed
Phase-3 boss evaluation to make dodge behaviour auditable.

## J — reward design and DQN training quality

- `arena/config.json` defines named required event rewards: enemy/rift kill,
  phase progress, damage, and death.
- `ArenaEnv._calculate_reward()` also logs shaping terms for damage, spacing,
  crowd escape, aim/shot quality, barrage escape, full dodges, boss hits, and
  exposure to an active telegraphed danger zone. XP remains separate.
- `arena.train` uses Stable-Baselines3 DQN with a two-hidden-layer MLP, replay
  memory, target network, epsilon schedule, checkpoints, Monitor CSV and
  TensorBoard. `arena.tune` compares fast/balanced/long exploration profiles.
- `BossCurriculumWrapper` only selects a reset phase during training; it does
  not control actions or rewrite state. Direct checkpoint selection evaluates
  both normal starts and fixed Phase-3 boss starts. This is documented in
  `logs/arena/evidence/selection/direct_checkpoint_sweep.json`.

## Report-ready artifacts

- `logs/arena/evidence/direct_schema8_final.{csv,json}`
- `logs/arena/evidence/direct_schema8_boss_focus.{csv,json}`
- `logs/arena/evidence/rotation_schema8_final.{csv,json}`
- `logs/arena/evidence/rotation_schema8_boss_focus.{csv,json}`
- `logs/arena/evidence/selection/*_checkpoint_sweep.json`
- `logs/arena/evidence/*_showcase.png`
- `logs/arena/evidence/evidence_manifest.json`
- `logs/arena/training/final_direct/training_curve.png`
- `logs/arena/training/final_rotation/training_curve.png`
- `logs/arena/tuning/hyperparameter_comparison.png`
- `logs/arena/tensorboard/` (SB3 event files)
