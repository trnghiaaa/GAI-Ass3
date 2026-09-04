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

The launcher includes an eleven-chapter illustrated Pilot Guide for the video
demonstration. It presents the two action sets, phase loop, boss/sentry target
priority, reward cadence, all 21 ship upgrades, five phase caches, and four
boss relics on-screen. Catalogue text is loaded from `arena/config.json`, so
balance edits cannot silently make the manual inaccurate. A report-ready frame
is saved as `logs/arena/evidence/pilot_guide_showcase.png`.
Chapter 10 illustrates the live `Tab` build inspector and all five cosmetic
`C`-key ship palettes with matching thrust and laser audio. Chapter 11 is a
dedicated Other Features reference for pause/resume, seeded replay, persistent
mission summaries, procedural music and event SFX, mixer/help controls, and
trained-AI speed/single-step playback. Its report-ready frame is saved as
`logs/arena/evidence/pilot_other_features_showcase.png`.

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
  for finer steering. Its selective-pressure pass keeps unconditional boss,
  missile, and closing-enemy responses but does not flee from moderate ambient
  pressure. On 24 fresh identical seeds versus the prior checkpoint, mean
  phase rises 3.29 to 3.75, maximum phase 6 to 7, player shots 106.0 to 146.0,
  player hits 81.8 to 115.8, enemy kills 29.6 to 42.5, and boss clears 0.29 to
  0.54. Close approaches fall 27.8% to 25.5%, damage per 1,000 frames 37.7 to
  32.9, wall contacts 2.08 to 1.60, and boss-skill hits 2.42 to 1.79. Runtime
  is still the saved SB3 DQN—no teacher, hidden brake, or scripted steering is
  loaded.
- A forced-sentry holdout prevents a false pass caused by ordinary runs not
  reaching the optional Aegis intermission. On 12 unseen seeds the submitted
  Rotation DQN averages 32.75 sentry missiles evaded versus 0.75 hits (97.8%
  of resolved missiles evaded). The same benchmark is reproducible with
  `arena.tools.compare_candidate --sentry-start-phase 3`; no scripted action
  override or easier runtime is used.
- The final pre-boss build planner recognizes both an active boss and the phase
  immediately before one. It guarantees a visible Shield option, prioritizes
  that option for non-interactive playback, and keeps manual selection free.
  Launcher seed 590003 provides a reproducible selective-pressure run that
  reaches Phase 6, clears Boss 1, fires 255 player shots with 207 hits, and
  takes no missile hits.
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
- `logs/arena/evidence/rotation_anticipatory18_{normal,boss}_holdout.{csv,json}`
- `logs/arena/evidence/rotation_anticipatory18_launcher_seed590002.{csv,json}`
- `logs/arena/evidence/rotation_selective_baseline610_holdout.{csv,json}`
- `logs/arena/evidence/rotation_selective056_validation610_holdout.{csv,json}`
- `logs/arena/evidence/rotation_selective056_boss_holdout.{csv,json}`
- `logs/arena/evidence/rotation_selective056_launcher_seed590003.{csv,json}`
- `logs/arena/evidence/rotation_selective056_sentry_holdout650.{csv,json}`
- `logs/arena/evidence/*_showcase.png`
- `logs/arena/evidence/ai_mission_summary_showcase.png`
- `logs/arena/evidence/pilot_guide_showcase.png`
- `logs/arena/evidence/pilot_build_style_showcase.png`
- `logs/arena/evidence/pilot_other_features_showcase.png`
- `logs/arena/evidence/evidence_manifest.json`
- `logs/arena/training/final_direct/training_curve.png`
- `logs/arena/training/final_rotation/training_curve.png`
- `logs/arena/tuning/hyperparameter_comparison.png`
- `logs/arena/tensorboard/` (SB3 event files)
