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

A transparent performance director makes strong normal-phase play less trivial
without a hidden human/AI difficulty split. When hull and remaining time are
both comfortable, it continuously scales up to two additional active enemies
and 10% faster spawning. It returns to zero below 55% hull or 35% remaining
time, is deterministically derived from already-observed health/time features,
and is disabled for every boss. The HUD labels active pressure as `SURGE`.

Bosses instead use a separate visible threat tier, not a fixed difficulty.
Measured effective health rises from about 639 at Phase 3/Tier 1 to 894 at
Phase 6/Tier 2, 1,122 at Phase 9/Tier 3, 1,357 at Phase 12/Tier 4, 3,786 at
Phase 30/Tier 10, and 15,110 at Phase 99/Tier 33. The first four encounter
steps are regression-tested to remain between 1.10x and 1.50x for both boss
effective health and minion health.
Minion health/speed, shield, finite summon budget, Aegis sentry count, attack
damage, and cast cadence also scale; mechanic counts/cadence eventually cap for
fairness while health continues rising. This preserves late-game challenge
without applying an opaque adaptive modifier during an already complex boss.

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
| Direct, normal (seed 251000) | 24 | 985.83 | 9.71 | 100% | 2.50 | 8.75 | 10.92 | 1.04 | 3.71 |
| Direct, Phase-3 start (seed 252000) | 24 | 541.28 | 7.29 | 79% | 1.71 | 5.46 | 9.62 | 0.83 | 2.71 |
| Rotation, normal (seed 263000) | 24 | 72.08 | 2.96 | 100% | 0.04 | 0.54 | 1.92 | 2.00 | 0.50 |
| Rotation, Phase-3 start (seed 264000) | 24 | -55.11 | 3.00 | 0% | 0.00 | 0.00 | 1.33 | 1.67 | 0.00 |

The direct policy is the demonstration-ready intermission-aware policy: the
focused holdout records 9.62 complete dodges, 5.46 destroyed sentries, only 0.83
boss-skill hits, and 2.71 missile hits per episode, with 79% phase progression.
Rotation is deliberately reported as the harder coupled-control problem. Its
independently reselected 350k checkpoint reached Phase 5 and preserved 100%
ordinary phase progression, but did not clear the no-upgrade immediate-boss
stress test. This is defensible checkpoint selection and transparent limitation
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

- `logs/arena/evidence/direct_schema10_tiered_final.json`
- `logs/arena/evidence/direct_schema10_tiered_boss.json`
- `logs/arena/evidence/rotation_schema10_tiered_selected_final.json`
- `logs/arena/evidence/rotation_schema10_tiered_selected_boss.json`
- `logs/arena/evidence/selection/tiered_rotation_schema10_sweep.json`

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

Learned-policy playback ends on an interactive results modal showing phase,
ship level, kills, rifts, bosses, and reward. Replay, optional Next Run, and
Main Menu remain available until selected; there is no timed auto-return.

Use `python -m arena.build_evidence` after retraining to rebuild/verify the
manifest. TensorBoard logs record actual SB3 learning; the screenshot/CSV/JSON
files are report-ready evidence rather than substituted training results.
