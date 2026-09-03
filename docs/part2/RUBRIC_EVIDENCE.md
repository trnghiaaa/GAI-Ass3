# Part II rubric evidence map

This is the current schema-12 submission map. Generated JSON/CSV files are the
source for numeric report claims.

## G — Real-time Pygame arena

| Criterion | Implementation and evidence |
|---|---|
| Continuous animated arena | Float-coordinate `ArenaEnv` and `ArenaRenderer`; `logs/arena/evidence/environment_showcase.png` |
| Ship, shooting, spawners and steering enemies | `_apply_player_action()`, `_fire_projectile()`, `_update_spawners()`, `_update_enemies()`; mechanics tests |
| Health, collisions and phase system | Circular collisions, player/enemy/rift health, clean phase hand-off and per-phase clocks; phase/boss transition screenshots |
| Terminal conditions | Player death, phase deadline, or long safety cap; termination tests |

The creative additions (bosses, finite Aegis sentry intermissions, telegraphed
finite-guidance missiles, minibosses, XP/drafts, wingmen, support drops, VFX and pause
UI) do not replace any required action, health, collision, or phase mechanic.
The balanced curve adds rifts every two phases, caps active enemies at 28, and
softens compound boss/minion health and cadence without removing progression.
Its visible `SURGE` director adds at most two enemies and 10% cadence only while
hull/time are comfortable; it is identical for humans and agents and disabled
during bosses.
Bosses use their own numbered tier curve: health/shields, minions, summons,
sentries, skill damage and cadence grow with encounter depth under fairness
caps. Early boss/minion health steps are regression-tested between 1.10x and
1.50x so the rise is noticeable but adaptable. `boss_threat_tier` is exposed
in `info` and the HUD.

## H — Gym-style API and observation

- `ArenaEnv.reset()` and `ArenaEnv.step()` use Gymnasium/SB3 return values;
  `LegacyArenaEnv` exposes the four-value assignment adapter.
- `render()` supports visible Pygame evaluation and RGB arrays.
- The observation is a fixed 126-feature numeric `float32` vector: player
  position/velocity/orientation/health, nearest enemy/rift geometry, phase,
  targeting, build state, boss-hazard escape information, boss vulnerability,
  sentry priority, nearest and second missile timing/velocity, combined volley
  pressure/escape/turn guidance, lock-on, and boss velocity.
- `ObservationIndex`, `OBSERVATION_NAMES`, and `observation_as_dict()` make
  each field inspectable; corresponding tests validate shape/ranges.
- Direct control uses a shared 420-pixel tactical-target assist because its
  required action set has no aim action: immediate threats defend first, then
  the damageable rift/boss objective. Rotation retains a measured 18-degree correction;
  neither mode receives a hidden human/AI targeting difference.

## I — Two control schemes and saved models

| Style | Actions | Submitted model | Visual evaluation |
|---|---|---|---|
| Rotation + thrust | no-op, thrust, rotate L/R, shoot | `models/arena/dqn_rotation.zip` | `python -m arena.evaluate_rotation` |
| Direct | no-op, up/down/left/right, shoot | `models/arena/dqn_direct.zip` | `python -m arena.evaluate_direct` |

Each model has metadata and a deterministic 24-episode normal-start evaluation
under `logs/arena/evidence`. Both policies also have a 24-episode fixed
Phase-3 boss evaluation to make dodge behaviour auditable.

## J — reward design and DQN training quality

- `arena/config.json` defines named required event rewards: enemy/rift kill,
  phase progress, damage, and death.
- `ArenaEnv._calculate_reward()` also logs shaping terms for damage, spacing,
  crowd escape, close-range pressure, explicit enemy and wall contact,
  feasible edge escape, aim/shot quality,
  barrage/missile escape, full dodges, boss or
  missile hits, sentry kills, immune-shot waste, and exposure to an active
  telegraphed danger zone. XP remains separate.
- `arena.train` uses Stable-Baselines3 DQN with a two-hidden-layer MLP, replay
  memory, target network, epsilon schedule, checkpoints, Monitor CSV and
  TensorBoard. `arena.tune` compares fast/balanced/long exploration profiles.
- `BossCurriculumWrapper` selects boss resets, reconstructs the skipped
  level/draft progression of a real run, and can expose the genuine finite
  Aegis intermission more often during training; it never controls actions and
  explicit evaluation resets remain unchanged. Checkpoint selection evaluates
  both normal starts and fixed Phase-3 boss starts. This is documented in
  `logs/arena/evidence/selection/dodgeable_missile_direct_schema10_sweep.json`.
- Rotation additionally supports reproducible observation-only demonstration
  initialisation. The teacher is absent from runtime: the launcher and
  evaluator load a cooldown-aware SB3 DQN. Optional DQfD-style auxiliary loss
  is exposed for experiments, while failed candidates remain unpromoted.
- On 12 identical fresh normal-start seeds, tactical Direct targeting retains
  100% progression while raising mean phase from 11.25 to 16.92, maximum phase
  from 38 to 54, boss clears from 3.17 to 5.08 per run, and safety-cap survival
  from 33% to 67%. Boss-skill hits fall from 1.25 to 0.58 per run. The model's
  action output is unchanged; its shared aim assist now protects against an
  immediate threat and otherwise attacks the damageable progression objective.
- The final Rotation checkpoint uses action repeat 2 and cooldown-aware targets
  for finer steering. Its combat-balanced demonstration pass keeps immediate
  boss/missile safety overrides but attacks through moderate ambient pressure.
  On 16 identical unseen seeds, mean phase rises 3.50 to 4.00, reward 86.4 to
  128.6, enemy kills 25.1 to 44.9, spawner kills 5.19 to 6.19, and boss clears
  0.31 to 0.50. Accuracy rises 83% to 87%; damage per 1,000 frames falls 38.86
  to 32.80, contacts 2.29 to 1.86, and wall contacts 2.69 to 1.09. Runtime is
  still the saved SB3 DQN—no teacher or scripted steering is loaded.
- The final pre-boss build planner recognizes both an active boss and the phase
  immediately before one. It guarantees a visible Shield option, prioritizes
  that option for non-interactive playback, and keeps manual selection free.
  Launcher seed 590009 provides a representative combat-balanced run that
  reaches Phase 5, clears Boss 1, and records only one resolved boss-skill hit.
- A later fixed-seed assist ablation keeps the learned Rotation model and action
  set unchanged: 18 degrees raises maximum phase from 3 to 5, lowers damage
  from 68.3 to 54.3 per 1,000 frames, and raises mean boss dodges from 1.6 to
  4.1 versus 12 degrees. Wider 21/24-degree variants and repeat 1 were rejected.

## Report-ready artifacts

- `logs/arena/evidence/direct_schema10_assist220_selected_final.{csv,json}`
- `logs/arena/evidence/direct_schema10_assist220_selected_boss.{csv,json}`
- `logs/arena/evidence/rotation_schema10_assist220_tiered_final.{csv,json}`
- `logs/arena/evidence/rotation_schema10_assist220_tiered_boss.{csv,json}`
- `logs/arena/evidence/selection/assist220_*_schema10_sweep.json`
- `logs/arena/evidence/edge_escape_final_*_holdout.{csv,json}`
- `logs/arena/evidence/safety_v12_*_final_holdout.{csv,json}`
- `logs/arena/evidence/safety_v12_rotation_preboss_shield_launcher_seed53006.{csv,json}`
- `logs/arena/evidence/rotation_refined_final_holdout.{csv,json}`
- `logs/arena/evidence/rotation_refined_final_boss_holdout.{csv,json}`
- `logs/arena/evidence/rotation_refined_launcher_seed530005.{csv,json}`
- `logs/arena/evidence/rotation_attack_balance_{normal,boss}_holdout.{csv,json}`
- `logs/arena/evidence/rotation_attack_balance_launcher_seed590009.{csv,json}`
- `logs/arena/evidence/*_showcase.png`
- `logs/arena/evidence/ai_mission_summary_showcase.png`
- `logs/arena/evidence/evidence_manifest.json`
- `logs/arena/training/final_direct/training_curve.png`
- `logs/arena/training/final_rotation/training_curve.png`
- `logs/arena/tuning/hyperparameter_comparison.png`
- `logs/arena/tensorboard/` (SB3 event files)
