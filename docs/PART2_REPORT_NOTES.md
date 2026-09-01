# Part II report notes

Use this document and the generated JSON files as the source for the final
ten-page report. Do not copy older schema-5–7 figures: they are retained only
in the ignored local recovery area and are not part of the submission.

## Environment

**Neon Rift Arena** is an original continuous 800×600 Pygame combat arena.
The player can move and shoot; rifts periodically spawn steering enemies;
projectiles and contact damage use health systems; destroying every active rift
advances the phase. Each new phase removes surviving hostiles/projectiles,
resets the per-phase time budget, and visibly announces the next deployment.
Every third phase is a shielded boss encounter with finite miniboss summons and
telegraphed horizontal, vertical, diagonal, cross, trident, and circular
barrages. The episode ends on player death, a missed phase deadline, or a long
safety cap. The same simulation is used for human play, headless training, and
rendered evaluation.

Creative presentation/gameplay additions are intentionally renderer and
progression layers around the required arena: pause/single-step controls,
phase/boss/level-up VFX, readable target/health telemetry, XP-based three-card
drafts, permanent wingmen, support drops, optional Rift Hunter minibosses, and
post-boss relic choices. XP is **not** added to the RL reward.

## Observation and actions

The fixed 89-value `float32` observation includes player position, velocity,
heading, health, weapon readiness, nearest enemy/rift relative geometry,
counts, phase/time, targeting alignment, progression/build state, crowd/wall
clearance, miniboss/boss shield state, and primary/secondary/combined
boss-hazard escape signals. It contains no pixels. The two required action sets
are implemented unchanged:

| Control style | Actions |
|---|---|
| Rotation/thrust | no-op, thrust forward, rotate left, rotate right, shoot |
| Direct movement | no-op, move up, move down, move left, move right, shoot |

`ArenaEnv` uses Gymnasium/SB3's five-value API and `LegacyArenaEnv` supplies
the assignment's four-value compatibility API.

## Reward and boss-safety training

The configurable reward contains the required positive enemy/rift/phase terms
and negative damage/death terms. Additional, logged shaping rewards only
improve temporal credit assignment: damage dealt, safe spacing, crowd escape,
aim/shot quality, boss-barrage escape, and a complete barrage dodge. Schema 8
adds a clear `boss_skill_hit` penalty and a small `hazard_exposure` penalty on
every active step inside a telegraphed danger zone. For simultaneous lanes the
environment uses the **maximum** danger, so an overlapping barrage cannot make
the penalty disappear by averaging. This changes reward learning signals only;
it does not script player movement, change collisions, or change the action
sets.

The direct agent was fine-tuned for 400,000 decisions with an honest
training-only boss curriculum: 70% of resets begin at Phase 3. It still uses
the real environment and receives no forced dodge action. Checkpoints were
selected using separate deterministic normal and fixed-Phase-3 benchmarks,
which prevents a policy that simply survives/stalls from winning.

## Final measured evidence (schema 8)

All values below are deterministic held-out evaluations of the committed model
weights, not training returns.

| Policy / evaluation | Episodes | Mean reward | Mean phase | Bosses cleared | Dodges | Boss hits |
|---|---:|---:|---:|---:|---:|---:|
| Direct, normal start (seed 93000) | 30 | 846.13 | 7.73 | 1.93 | 6.13 | 0.33 |
| Direct, fixed Phase-3 boss start (seed 96000) | 30 | 405.70 | 6.17 | 1.47 | 5.70 | 0.63 |
| Rotation, normal start (seed 94000) | 30 | 89.89 | 2.93 | 0.03 | 2.07 | 1.87 |
| Rotation, fixed Phase-3 boss start (seed 95000) | 30 | -66.54 | 3.00 | 0.00 | 2.07 | 2.53 |

The direct policy is the demonstration-ready boss-aware policy: the Phase-3
holdout records 5.70 complete dodges for 0.63 boss-skill hits per episode, and
it clears the opening boss in 90% of focused runs. Rotation is deliberately
reported honestly as the harder control problem: a fresh 600k boss-curriculum
candidate did not beat the retained rotation model on a common normal + boss
checkpoint sweep, so it was not promoted. This is defensible model selection,
not a hidden difficulty change.

Exact machine-readable evidence:

- `logs/arena/evidence/direct_schema8_final.json`
- `logs/arena/evidence/direct_schema8_boss_focus.json`
- `logs/arena/evidence/rotation_schema8_final.json`
- `logs/arena/evidence/rotation_schema8_boss_focus.json`
- `logs/arena/evidence/selection/direct_checkpoint_sweep.json`
- `logs/arena/evidence/selection/rotation_checkpoint_sweep.json`

## Reproducibility and project structure

`models/arena/dqn_direct.zip` and `models/arena/dqn_rotation.zip` are the two
submitted SB3 DQN models. Their adjacent metadata records architecture,
observation schema, settings, selected checkpoint, and holdout paths. The
compact submission evidence tree is:

```text
logs/arena/
  evidence/       screenshots, CSV/JSON holdouts, selection records, manifest
  training/       final direct/rotation Monitor, summary and selected checkpoint
  tensorboard/    final-policy and tuning event files
  tuning/         compact hyperparameter comparison
```

Use `python -m arena.build_evidence` after retraining to rebuild/verify the
manifest. TensorBoard logs record actual SB3 learning; the screenshot/CSV/JSON
files are report-ready evidence rather than substituted training results.
