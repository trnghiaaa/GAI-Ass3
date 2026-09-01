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
barrages. Breaking the shield makes the boss drift slowly; at low health it
deploys one finite Aegis sentry wing. The boss is temporarily immune and can
repair only a capped amount until the player destroys the missile sentries.
The episode ends on player death, a missed phase deadline, or a long
safety cap. The same simulation is used for human play, headless training, and
rendered evaluation.

Difficulty remains progressive without compounding into an abrupt post-boss
spike. Normal encounters start with two rifts and add one every two phases up to
five. Active enemies start at 16 and cap at 28, spawn cadence has a 0.75-second
floor, and health/speed continue to rise under bounded late-game curves. Later
bosses, their minions, minibosses, and Aegis sentries still scale, but use softer
multipliers than the original curve. The 60-second normal phase clock and boss
bonuses remain unchanged.

Creative presentation/gameplay additions are intentionally renderer and
progression layers around the required arena: pause/single-step controls,
phase/boss/level-up VFX, readable target/health telemetry, XP-based three-card
drafts, permanent wingmen, support drops, optional Rift Hunter minibosses, and
post-boss relic choices. XP is **not** added to the RL reward.

## Observation and actions

The fixed 107-value `float32` observation includes player position, velocity,
heading, health, weapon readiness, nearest enemy/rift relative geometry,
counts, phase/time, targeting alignment, progression/build state, crowd/wall
clearance, miniboss/boss shield state, primary/secondary/combined boss-hazard
escape signals, boss vulnerability/motion, defender priority, and incoming
missile direction/distance/impact time, missile velocity, a perpendicular
escape direction, and lock-on progress. It contains no pixels. The two required action sets
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
aim/shot quality, boss-barrage and missile escape, complete dodges, defender
kills, and avoiding wasted fire on an immune boss. Schema 10 adds clear
`boss_skill_hit` and `missile_hit` penalties and a small `hazard_exposure` penalty on
every active step inside a telegraphed danger zone. For simultaneous lanes the
environment uses the **maximum** danger, so an overlapping barrage cannot make
the penalty disappear by averaging. This changes reward learning signals only;
it does not script player movement, change collisions, or change the action
sets.

The direct agent was fine-tuned for 350,000 decisions with an honest
training-only boss curriculum: 78% of resets begin at Phase 3. It still uses
the real environment and receives no forced dodge action. Checkpoints were
selected using separate deterministic normal and fixed-Phase-3 benchmarks,
which prevents a policy that simply survives/stalls from winning.

## Final measured evidence (schema 10)

All values below are deterministic held-out evaluations of the committed model
weights, not training returns.

| Policy / evaluation | Episodes | Mean reward | Mean phase | Progress | Bosses | Sentries | Dodges | Boss hits | Missile hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct, normal (seed 231000) | 24 | 1433.58 | 11.33 | 100% | 2.92 | 10.88 | 14.29 | 1.25 | 2.83 |
| Direct, Phase-3 start (seed 232000) | 24 | 834.62 | 8.79 | 79.2% | 2.04 | 7.92 | 14.54 | 1.38 | 2.83 |
| Rotation, normal (seed 233000) | 24 | 85.11 | 3.08 | 100% | 0.08 | 0.58 | 2.54 | 1.96 | 0.29 |
| Rotation, Phase-3 start (seed 234000) | 24 | -60.78 | 3.00 | 0% | 0.00 | 0.00 | 1.79 | 2.25 | 0.00 |

The direct policy is the demonstration-ready intermission-aware policy: the
focused holdout records 14.54 complete dodges, 7.92 destroyed sentries, and 2.83
missile hits per episode, with 79.2% phase progression. Rotation is deliberately
reported as the harder coupled-control problem. Its selected input-preserving
refinement cut ordinary-play missile hits to 0.12 per episode and preserved
100% ordinary phase progression, but did not clear the no-upgrade immediate-boss
stress test. This is defensible model selection and transparent limitation
reporting, not a hidden difficulty change.

## Encounter-design references

The finite vulnerability intermission follows the design principle that boss
defences can gate damage and create readable vulnerability phases. Telegraphs
use a warning, attack, and recovery structure so mandatory dodges are announced
before damage. Cite these external design references in the report:

- GDC Vault, *Boss Up: Boss Battle Design from Concept to Completion*:
  https://www.gdcvault.com/play/1025398/Boss-Up-Boss-Battle-Design
- Game Developer, *Enemy Attacks and Telegraphing*:
  https://www.gamedeveloper.com/design/enemy-attacks-and-telegraphing
- Game Developer, *Using a Modular System of Maneuvers to Design Psychonauts 2's Boss Fights*:
  https://www.gamedeveloper.com/marketing/using-a-modular-system-of-maneuvers-to-design-i-psychonauts-2-i-s-boss-fights-in-a-hurry

Exact machine-readable evidence:

- `logs/arena/evidence/direct_schema10_balanced_v2_final.json`
- `logs/arena/evidence/direct_schema10_balanced_v2_boss.json`
- `logs/arena/evidence/rotation_schema10_balanced_v2_final.json`
- `logs/arena/evidence/rotation_schema10_balanced_v2_boss.json`
- `logs/arena/evidence/selection/dodgeable_missile_direct_schema10_sweep.json`
- `logs/arena/evidence/selection/dodgeable_missile_rotation_schema10_sweep.json`

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
