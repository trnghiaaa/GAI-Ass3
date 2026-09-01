# Reinforcement Learning & Deep RL Agents — Assignment 3

This branch combines both assignment projects: the polished **classical
reinforcement-learning Gridworld** from Part I and the **Neon Rift Arena**
deep-RL environment from Part II. It includes reproducible experiments, saved
policies, report-ready evidence, and interactive Pygame experiences for both
parts.

The assignment environments' rewards are never changed by the UI or training
tools.

## Setup and launch

From this repository directory, create the project environment and install all
Part I and Part II dependencies. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --part 1
.\.venv\Scripts\python.exe main.py --part 2
```

Using the `.venv` interpreter directly is recommended, especially for Part II:
the saved Stable-Baselines3 DQN policies were serialized with the NumPy version
installed in this project environment. A different global Python environment
can fail while loading a model with an error such as
`No module named 'numpy._core.numeric'`. Part II is the default when `--part`
is omitted. The equivalent package entry points are `python -m gridworld` and
`python -m arena` when run through the same `.venv` interpreter.

On Windows, double-click **`scripts/part1/run_part1.bat`** for Part I or
**`scripts/part2/run_part2.bat`** for Part II. Both automatically use the project `.venv`
when present, so nobody needs to type the interpreter path.

### Repository layout

```text
arena/
  core/                Part II simulation, entities, and API adapter
  presentation/        Pygame launcher, rendering, and manual play
  learning/            DQN training, wrappers, cooldown handling, and tuning
  evaluation/          Policy evaluation and deterministic benchmarks
  tools/               Evidence, comparison, selection, and promotion utilities
  config.json          Arena mechanics and training configuration
  settings.py          Shared model, log, and configuration paths
gridworld/             Part I environment and tabular agents
tests/
  part1_gridworld/     Part I tests and fixtures
  part2_arena/         Part II tests
  integration/         Combined-launcher tests
docs/
  part1/               Part I rubric evidence and screenshots
  part2/               Part II report notes, rubric map, and video plan
scripts/
  part1/               Part I Windows launch/evidence helpers
  part2/               Part II Windows launcher
models/                Saved policies grouped by assignment part
logs/                  Training and evaluation evidence grouped by part
main.py                Shared visual launcher
```

Thin compatibility modules preserve established `arena.*` imports and commands
after this physical package reorganization. VS Code hides those wrappers so the
Explorer shows the implementation folders above.

### VS Code run button

The generic **Run Python File** triangle is owned by the Python extension and may
still paste an unquoted cached interpreter path into PowerShell. This repository
therefore includes two shell-safe launch routes that never parse the `.venv`
path as PowerShell source:

- press `F5` and select **Part I: Run Gridworld (safe)** or
  **Part II: Run Neon Rift Arena**; or
- press `Ctrl+Shift+B` to run the default **Part I: Run Gridworld (shell-safe)** task.

These launch `main.py` without constructing the unsafe `& D:\A3_Game&AI...`
command. The task route uses a VS Code `process` task rather than a shell task.

After cloning or pulling these settings, run **Developer: Reload Window** once (or close all existing VS Code terminals) before using the configured run option. Existing terminals keep their previous environment. The folder name `A3_Game&AI` contains PowerShell's `&` operator, so the old unquoted absolute command cannot execute safely; `scripts/part1/run_part1.bat` remains an unaffected alternative.

If the top-right triangle still prints a command beginning with an absolute
`.venv\\Scripts\\python.exe` path, VS Code is retaining its old workspace choice.
Open the Command Palette, run **Python: Clear Workspace Interpreter Setting**,
then run **Developer: Reload Window**. The repository intentionally launches the
plain `python` command and lets `main.py` perform the shell-safe `.venv` hand-off.

## Part I: Gridworld AI Lab

The Part I launcher provides:

- **Campaign** — play Levels 0 through 6 continuously, or watch the trained AI tour.
- **Free Play** — choose any level and verify every mechanic manually.
- **AI Showcase** — select a saved Q-learning, SARSA, or intrinsic policy.
- **RL Inspector** — inspect current-state Q-values and exact greedy ties.
- **Policy Lens** — overlay learned actions on the grid.
- original looping procedural music and event-specific sound effects, with `V` mute.
- pause, single-step, playback speed, animated movement, particles, trails, responsive resizing, objective cards, and safe timeout/result screens.

Manual controls:

| Input | Action |
|---|---|
| Arrow keys or WASD | Move up, down, left, right |
| R | Retry level |
| P | Toggle policy lens when viewing an AI |
| Space | Pause/resume AI |
| `.` or Right Arrow while paused | Single AI step |
| `+` / `-` | Change AI playback speed |
| `1` or `0` on an AI result screen | Reset replay speed to 1x |
| V | Mute or unmute the procedural music and sound effects |
| M or Esc | Menu |

AI campaign transitions always begin the next level at 1x. The result screen also
provides `-`, `+`, and **Reset to 1x** controls before replaying, so a fast playback
setting never traps the player behind the completion popup.

Direct links remain available for assessors and quick recording:

```bash
python main.py --part 1 --mode manual --level 4
python main.py --part 1 --mode ai --level 1 --agent sarsa
python main.py --part 1 --mode ai --level 6 --agent qlearning --intrinsic
```

## Exact rules

| Mechanic | Implemented behavior | Environment reward |
|---|---|---:|
| Movement | Up, down, left, right | 0 |
| Rock / boundary | Agent remains in its current cell; the attempted action still advances the turn | 0 |
| Fire | Immediate death | 0 |
| Monster collision | Immediate death, whether the agent enters its tile or it enters the agent's tile | 0 |
| Apple | Consumed | +1 |
| Key | Consumed; enables chest opening | 0 |
| Locked chest | Remains until the key is held | 0 |
| Opened chest | Consumed | +2 |
| Monster phase | Each monster independently has the configured 40% chance to make one valid random move after an agent action | 0 |
| Episode terminal | All collectible rewards obtained, or agent death | — |

There is no hidden step penalty, death penalty, bonus environment reward, or altered terminal rule. A blocked input is still an agent action, so it counts toward the action limit and still triggers the post-action monster phase. Level 6's intrinsic bonus exists only in the agent's learning target and is logged separately.

## Levels and rubric coverage

| Level | Assignment task | Main evidence |
|---:|---|---|
| 0 | Task 1: basic Q-learning | Redesigned orchard; greedy Q-learning completes the verified optimum in **22 steps** |
| 1 | Task 2: SARSA | Q-learning uses a 7-step fire-edge route; SARSA uses a 9-step safe route |
| 2 | Task 3 | Branching garden with three apples, key, chest; Q-learning and SARSA |
| 3 | Task 3 | Dense maze with three apples, key, chest; both algorithms |
| 4 | Task 4 | One stochastic monster; both algorithms and learning curves |
| 5 | Task 4 | Two stochastic monsters; both algorithms and learning curves |
| 6 | Task 5 | Baseline versus exact per-episode count-bonus Q-learning |

Level definitions and display metadata live in `gridworld/levels/levels.py`.
Level 0's three apples are compliant with the Task 1 wording: they are all on
the right side and are the only collectible type present. Levels 2–3 introduce
the required combined planning problem of multiple apples, a key, and a chest.

The final independent seeded benchmark records **96.7% / 97.6%** success for
Level 4 Q-learning / SARSA and **97.5% / 97.5%** for Level 5 over 1,000 episodes
per monster policy. All deterministic policies and both Level 6 variants achieve
100% over 300 episodes. Full results are in `logs/gridworld/policy_benchmark.json`.

## RL implementation

The state is a stable hashable tuple:

```text
(agent_row, agent_column, has_key, collectible_flags, monster_observation)
```

Every collectible has a presence flag. On monster levels, threats within Manhattan distance two have exact relative offsets; a far-monster count represents every remaining monster without exploding the table. This local-threat abstraction generalises avoidance behavior while Pygame still renders all absolute monster positions.

Q-learning uses the off-policy target:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

SARSA uses the action actually selected by the same epsilon-greedy policy:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
```

Both implementations use:

- config-driven `alpha`, `gamma`, episode count, epsilon start/end, and step cap;
- linear epsilon decay, with the final training episode using `epsilon_end`;
- random tie-breaking among **exactly equal** maximal Q-values;
- independent seeded random generators;
- terminal targets without bootstrap and explicit time-limit truncation handling;
- versioned, metadata-rich model files that remain backward-compatible with the original pickle format.
- a level-layout fingerprint that rejects stale Q-tables after any future map edit.

## Task 5: intrinsic reward

At the start of every episode, visit counts are reset. For a transition into `next_state`, the agent reads its number of prior visits `n(next_state)`, applies the required formula, then records the visit:

```text
intrinsic_reward = intrinsicRewardStrength / sqrt(n(s) + 1)
learning_reward  = environment_reward + intrinsic_reward
```

Here `s` is the reached state—the `next_state` of the transition and the current state after the action.

The default strength is deliberately small (`0.0001`). A larger value can make repeated exploration more valuable than the fixed +3 available in Level 6, so the value was tuned rather than selected arbitrarily.

The five-seed comparison currently records:

- baseline mean first success: **episode 10.2**;
- intrinsic mean first success: **episode 8.0** — **21.6% earlier**;
- both final greedy policies: **100% success, 22 steps**;
- both last-100 training victory rates: **100%**.

The evidence also shows the honest trade-off: the small bonus improves first discovery but does not dominate baseline coverage or early repeat success. Environment, intrinsic, and combined learning returns remain separate in every CSV.

## Rebuild all submission evidence

One command trains every rubric-required policy and produces both comparisons:

```bash
python -m gridworld.build_evidence
```

This builds 13 rubric policy bundles, uses held-out multi-seed champion selection
for the four stochastic monster policies, produces the Level 1 and Level 6
comparisons, benchmarks every saved policy, then verifies all required artifacts.
A quick plumbing-only run is available, but should not be submitted:

```bash
python -m gridworld.build_evidence --quick
```

### Individual training

```bash
python -m gridworld.train --level 0 --agent qlearning
python -m gridworld.train --level 3 --agent sarsa
python -m gridworld.train --level 6 --agent qlearning --intrinsic
python -m gridworld.optimize --level 4 --agent qlearning
```

Useful overrides include `--episodes`, `--alpha`, `--gamma`, `--epsilon-start`, `--epsilon-end`, `--max-steps`, `--seed`, and `--intrinsic-strength`. Defaults and per-level profiles are in `gridworld/config.json`.

`gridworld.optimize` trains several configured candidate seeds, evaluates every
candidate on the same held-out stochastic episodes, and saves only the most
reliable model. Completion rate is ranked before timeouts, deaths, and path
length; environment rewards and the Q-learning/SARSA rules are unchanged.

### Visual or headless evaluation

```bash
python -m gridworld.evaluate --level 0 --agent qlearning --episodes 3
python -m gridworld.evaluate --level 5 --agent sarsa --episodes 10
python -m gridworld.evaluate --level 6 --agent qlearning --intrinsic
python -m gridworld.evaluate --level 4 --agent qlearning --episodes 100 --headless
```

Evaluation is seeded and bounded. It reports victories, deaths, and timeouts separately instead of mislabelling every non-victory as a death.

### Rubric comparisons

```bash
python -m gridworld.compare --comparison algorithms
python -m gridworld.compare --comparison intrinsic
```

The Level 1 experiment uses 10 paired seeds. Its saved evidence shows:

- greedy Q-learning: 5 steps, 4 hazard-adjacent steps;
- greedy SARSA: 7 steps, 0 hazard-adjacent steps;
- at epsilon 0.05 over 2,000 evaluations per algorithm: Q-learning averages 9.4 fire deaths per 200 episodes/seed, SARSA 0.4.

## Output layout

```text
models/gridworld/
  level0_qlearning.pkl
  ...
  level6_qlearning.pkl
  level6_qlearning_intrinsic.pkl
  evidence/                         # paired-seed Task 5 models

logs/gridworld/
  levelN_agent.png                  # four-panel training dashboards
  levelN_agent_rewards.csv          # full per-episode metrics
  levelN_agent_summary.json         # parameters + aggregate results
  level1_qlearning_vs_sarsa_evidence.png
  level1_algorithm_comparison_*.csv/json
  level6_intrinsic_comparison_evidence.png
  level6_intrinsic_comparison_*.csv/json

docs/part1/screenshots/
  *_preview.png                       # report/video UI captures
```

Saved model metadata includes algorithm, level, seed, episode count, alpha, gamma, epsilon range, intrinsic settings, profile, summary, and state schema.

## Tests

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The suite covers:

- every specified environment reward and terminal rule;
- blocked moves, locked/unlocked chests, and layered collision rendering;
- deterministic and statistical monster movement;
- Q-learning and SARSA equations;
- exact random ties and linear epsilon schedule;
- destination-state intrinsic counts, terminal bonuses, and episode reset;
- model save/load compatibility;
- Level 0 optimal behavior, the Level 1 policy difference, and the Level 6 A/B result;
- unified-app and CLI-renderer headless Pygame smoke tests.

## Originality and presentation features

All visual art is drawn procedurally with Pygame primitives; there are no copied sprites or third-party game assets. Creative work beyond the base specification includes the connected campaign, AI tour, responsive interface, animated interpolation, particles, trail rendering, policy lens, live Q-value inspector, deterministic playback controls, in-window benchmark evidence, held-out champion selection, evidence dashboards, exploration heatmap, multi-seed confidence bands, config provenance, and automated rubric acceptance tests.

See `docs/part1/RUBRIC_EVIDENCE.md` for the Part I code/artifact mapping.
## Part II: Neon Rift Action Arena

> Schema-10 boss-intermission learning, pause/UI, upgrade clarity, and tidy final
> evidence are complete. See `docs/part2/REPORT_NOTES.md` for current metrics
> and `docs/part2/RUBRIC_EVIDENCE.md` for the marking-evidence map.

The Part II environment is a continuous-coordinate Pygame combat arena named
**Neon Rift Arena**. On Windows, launch it through the project environment so
the saved DQN models use the compatible NumPy and Stable-Baselines3 versions:

```powershell
.\.venv\Scripts\python.exe main.py --part 2
# equivalent module entry point:
.\.venv\Scripts\python.exe -m arena
```

The simplest Windows alternative is `scripts/part2/run_part2.bat`, which selects `.venv`
automatically. The repository's shared VS Code launch configuration also uses
the project environment and avoids PowerShell parsing errors when the project
path contains an ampersand. The menu offers
manual play and deterministic trained-agent playback for both required control
schemes. Direct manual play is also available from the command line:

```bash
python -m arena.play --control-style direct
```

Use `WASD` or the arrow keys to move and hold `Space` to fire at the same time. Direct mode snaps
shots to the nearest hostile when its reticle turns green inside the balanced
220-pixel assist range;
outside that range it fires along the current heading. To try rotation and
thrust controls instead:

```bash
python -m arena.play --control-style rotation
```

For rotation controls, use `W` to thrust, `A`/`D` to rotate, and hold `Space` to
fire while thrusting or turning. Rotation shots receive a narrow seven-degree
aim correction only when the pilot is already aligned with a target; aiming is
still controlled by rotation. In both modes, `R` restarts an episode and
`Esc` quits. `P` or the visible PAUSE/RESUME button freezes the simulation and
phase timer. `Tab` opens a readable ship-build panel and pauses the manual battle.

The arena provides:

- A damageable player ship with continuous movement and projectile firing
- Damageable enemy spawners that periodically create hostiles
- Enemies that continuously steer toward and damage the player on contact
- Projectile collisions with separate enemy and spawner health bars
- Increasing phases after all active spawners are destroyed
- Clean phase hand-offs that withdraw surviving hostiles and clear old shots
  without awarding fake kills, XP, or RL reward
- A gradual phase director, random Rift Hunter minibosses, and a boss rift every third phase
- An onboarding Boss 1 with an 18% shield, one Hunter reinforcement, fewer
  simultaneous minions, and a 90-second deadline; later bosses use 32.5–45%
  shields, a 70-second deadline, and finite 2/3/4-Hunter budgets
- Bosses begin moving slowly after their shield breaks. At low health they
  deploy one finite, telegraphed Aegis sentry wing: the boss becomes immune and
  repairs only a capped amount until those missile turrets/interceptors fall
- Grouped horizontal, vertical, diagonal, cross, trident, and circular boss
  barrages with readable names/countdowns and one-hit-per-cast fairness
- A fresh 60-second deadline for ordinary phases, boss-specific time budgets,
  and a separate long episode safety cap
- Headless, human-window, and RGB-array rendering modes
- Targeting reticles, impact particles, projectile trails, damage feedback,
  readable HUD telemetry, animated level-up/phase/boss transitions, and
  report-ready RGB screenshots
- Uncapped combat levels and three-card drafts spanning 21 upgrade/mastery
  paths, with NEW/current/next-rank previews on every card
- Permanent, upgradable wingman squadrons plus five between-phase support choices
- Strong post-boss relic drafts and automatic repair/Aegis miniboss caches

### Arena API

Stable-Baselines3 2.x uses the modern Gymnasium contract:

```python
from arena import ArenaEnv

env = ArenaEnv(control_style="direct", render_mode="human")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(0)
done = terminated or truncated
env.render()
env.close()
```

The assignment handout describes the older four-value Gym contract. The
compatibility adapter exposes that exact interface when needed for a marker or
demonstration, while the native environment remains compatible with SB3:

```python
from arena import LegacyArenaEnv

env = LegacyArenaEnv(control_style="direct", render_mode="human")
observation = env.reset(seed=42)
observation, reward, done, info = env.step(0)
env.render()
env.close()
```

### Arena Observation Vector

The agent receives a one-dimensional `float32` vector with exactly 107
normalized features. It never receives the rendered pixels.

| Indices | Features | Range | Meaning |
|---|---|---|---|
| 0–1 | `player_x`, `player_y` | `[-1, 1]` | Player position within the playable arena |
| 2–3 | `player_velocity_x`, `player_velocity_y` | `[-1, 1]` | Velocity divided by the configured maximum speed |
| 4–5 | `player_heading_cos`, `player_heading_sin` | `[-1, 1]` | Continuous orientation without angle wrap-around |
| 6 | `player_health` | `[0, 1]` | Remaining player-health proportion |
| 7 | `weapon_ready` | `[0, 1]` | Fire-cooldown readiness |
| 8–11 | Nearest-enemy direction X/Y, distance, health | mixed normalized | Unit relative direction, arena-diagonal distance, and health |
| 12–15 | Nearest-spawner direction X/Y, distance, health | mixed normalized | Unit relative direction, arena-diagonal distance, and health |
| 16–17 | `enemy_count`, `spawner_count` | `[0, 1]` | Active counts divided by configured maxima |
| 18 | `phase` | `[0, 1]` | Current phase divided by the configured observation cap |
| 19 | `time_remaining` | `[0, 1]` | Fraction of the current phase deadline remaining |
| 20–21 | Enemy/spawner aim alignment | `[-1, 1]` | Cosine alignment between ship heading and each nearest target |
| 22–24 | Active-target direction X/Y and distance | mixed normalized | The exact closest target used by the targeting reticle |
| 25 | `active_target_is_spawner` | `[0, 1]` | Distinguishes a progression target from an enemy |
| 26–27 | Active-target alignment and signed turn direction | `[-1, 1]` | Tells rotation control how close its aim is and which way to turn |
| 28–29 | Ship level and XP progress | `[0, 1]` | Current build level and progress toward the next draft |
| 30–33 | Volley size, fire rate, damage and laser flag | `[0, 1]` | Active weapon characteristics needed to keep progression Markov |
| 34–39 | Hull, shield, range, piercing, splash and engine | `[0, 1]` | Normalized permanent build-upgrade state |
| 40–42 | Wingman, nova bomb and boss phase | `[0, 1]` | Temporary support and encounter state |
| 43–44 | Miniboss presence and health | `[0, 1]` | Makes optional Rift Hunter encounters observable |
| 45–50 | Hazard escape X/Y, safety distance, impact time, active/circle flags | mixed normalized | Provides the information needed to dodge telegraphed boss attacks |
| 51–57 | Drone level/count, Aegis, overdrive, regeneration, homing and mastery | `[0, 1]` | Exposes the expanded procedural build without hidden state |
| 58–60 | Hazard count and combined escape X/Y | mixed normalized | Summarizes simultaneous boss lanes instead of treating a barrage as one line |
| 61–66 | Secondary hazard escape X/Y, safety distance, impact time, active/circle flags | mixed normalized | Exposes a second lane so the policy can choose a genuinely safe position |
| 67–69 | Critical chance, leech strength and Riftbreaker power | `[0, 1]` | Keeps the three additional weapon/sustain upgrades observable |
| 70–74 | Second-nearest enemy direction/distance/closing speed; nearest closing speed | mixed normalized | Reveals approaching threats beyond a single target |
| 75–78 | Local enemy count, crowd pressure, escape X/Y | mixed normalized | Summarizes enemies within 200px of surface clearance |
| 79–83 | Four wall clearances and nearest-spawner surface clearance | `[0, 1]` | Provides explicit room to maneuver and target spacing |
| 84–85 | Boss shield fraction and remaining summon budget | `[0, 1]` | Exposes finite boss defenses and reinforcements |
| 86–88 | Body-relative hazard escape alignment/turn and boss summon cooldown | mixed normalized | Relates hazard direction to ship steering and summon readiness |
| 89–91 | Boss health, vulnerability and defender count | `[0, 1]` | Makes the Aegis intermission and its completion explicit |
| 92–95 | Nearest defender direction X/Y, distance and health | mixed normalized | Exposes the only damageable progression target while the boss is immune |
| 96–99 | Nearest missile direction X/Y, distance and impact time | mixed normalized | Supports anticipatory missile avoidance rather than reacting after damage |
| 100–101 | Boss velocity X/Y | `[-1, 1]` | Lets the policy track the shield-broken mobile boss |
| 102–106 | Missile velocity X/Y, recommended escape X/Y, lock-on fraction | mixed normalized | Makes the telegraph, finite guidance window, and safest perpendicular dodge directly observable |

If a target type is absent, its four target features are `(0, 0, 1, 0)`:
no direction, maximum normalized distance, and zero health. Stable feature
indices are exported as `ObservationIndex`; `env.observation_as_dict()` gives a
named view for debugging and report evidence.

### Required Action Sets

| Style | Discrete actions |
|---|---|
| Rotation/thrust | No-op, thrust forward, rotate left, rotate right, shoot |
| Direct | No-op, move up, move down, move left, move right, shoot |

The training wrapper holds a decision for four 60 Hz simulation frames. This
does not add or remove actions; it makes rotation and thrust decisions visible
enough for DQN to learn while the renderer remains smooth.

### Combat XP, Build Drafts, and Boss Phases

Combat XP is a creative gameplay system separate from the RL reward. Destroyed
enemies, minibosses, rifts, and completed phases grant XP within the current episode. A
manual player pauses at each level and chooses one of three seeded cards:
maximum hull, repair, damage, fire rate, multi-beam, range, prism laser,
piercing, splash, shield, engines, homing, regeneration, capacitors, permanent
drones, critical beams, kill-based hull siphon, or specialist Riftbreaker
damage. Once the regular paths mature, repeatable weapon, hull, and drone
masteries keep the ship growing with no combat-level cap. Clearing a phase offers
repair, an armed arena bomb, a permanent wingman tier, overdrive, or Aegis.
Every card shows the installed tier and concrete result before selection (for
example, beam count, cooldown, hull, resistance, or Wingman Core/mastery/drone
count). The build banner then confirms the applied result rather than repeating
only the generic upgrade name.
Random Rift Hunters drop XP plus a repair/Aegis cache. Every third phase replaces
ordinary rifts with a shielded boss that telegraphs named multi-lane sweeps, crosses,
diagonal lattices, trident walls, and a circular nova cage;
after the shield breaks the boss drifts around the arena, and a low-health
Aegis intermission temporarily redirects combat to a finite sentry wing. Sentry
missiles show a stationary warning trace, guide gently for only 0.3 seconds,
then commit to a ballistic path. This makes lateral evasion readable and
testably dodgeable rather than unavoidable permanent homing;
victory opens a stronger permanent-relic draft. Speed, damage, spawn rate, and
active-enemy counts use fairness caps while health and player mastery keep scaling.
Normal encounters begin with two rifts and add one only every two phases, up to
five, avoiding the old jump to five immediately after the first boss. Active
hostiles begin at 16 and cap at 28; spawn intervals never fall below 0.75 seconds.
Later boss health, minion multipliers, and sentry health are deliberately softer
than their original compound curve while still increasing each encounter. The
first four encounters are regression-tested to grow in adaptable 10–50% steps,
rather than receiving one fixed jump after Boss 1.
Normal phases also use a visible performance director derived entirely from the
already-observed hull and timer state. While both are comfortable it can add at
most two hostiles and 10% spawn cadence (`SURGE` in the HUD); pressure smoothly
returns to baseline as hull or time becomes scarce. Boss phases never receive
this normal-wave adjustment because they have an independent tier curve, and
manual and learned play use the identical rules. The HUD exposes both the boss
tier and phase (`BOSS T2 / P6`): effective boss health, shield fraction, minion health,
minion speed, summon budget, sentry count, skill damage, and cast cadence grow
across tiers. Summons/sentries and attack speed use fairness caps, while boss
and minion health continue scaling into late phases.
Optional Rift Hunter probability begins at 24% and rises by six percentage
points per phase after Phase 5, capped at 66%.
Ordinary kills grant 7 XP and cumulative thresholds follow
`52 * (level - 1)^1.85`. Phase 1 guarantees enough objective XP for the first
draft, while the superlinear curve still slows late upgrades without a level cap.

During headless training and learned-policy playback, a deterministic heuristic
chooses from the same seeded three-card offers using health, upcoming boss risk,
control style, owned-weapon synergies, and diminishing-stack value. This keeps both rubric-required
action dictionaries exactly unchanged: the existing `Shoot` action uses the
composed build. XP and drafts never add an unreported RL reward term or replace
the required phase rule.

### Reward Design

The configurable reward function contains the required progression terms:
enemy destruction, larger spawner destruction, phase advancement, damage
penalty, and a strong death penalty. Small shaping terms give credit for actual
damage, progress toward a safe range around a stable spawner target, crowd
separation, aim improvement, and
well-aligned shots. Shaping never changes health, collisions, entity movement,
or terminal rules. A potential-difference term rewards movement out of an
unchanged boss barrage, while a full dodge remains a separate event reward.
Schema 10 also logs boss-skill and missile-hit penalties, explicit sentry-kill
credit, a small penalty for wasting shots on an immune boss, potential-based
missile escape credit, and a small per-step hazard-exposure
penalty. Overlapping boss lanes use the highest danger value rather than an
average, so standing in one active lane cannot be masked by safer lanes.
Combat XP is deliberately excluded from this total. Every
`step()` exposes `info["reward_breakdown"]`, making the exact contribution of
every RL reward term auditable.

### Stable-Baselines3 Training

Use a unique `--run-name` for every new run. The submitted
`models/arena/dqn_direct.zip` and `dqn_rotation.zip` already exist, so the bare
`python -m arena.train` command intentionally stops instead of overwriting
them. Train direct and rotation separately with new names, incrementing the
suffix for later attempts. The `boss_intermission` profile can use a training-only
boss curriculum: it samples a genuine Phase-3 reset for a configured fraction
of training episodes. It never selects actions for the policy or makes the
boss easier. A changed game still requires training and fresh evaluation.

Both policies use SB3 DQN with a configurable MLP, replay buffer, target
network, epsilon schedule, checkpoints, held-out evaluation, TensorBoard, and
model metadata:

```bash
# Recommended safe retraining commands (300,000 decisions by default)
python -m arena.train --control-style direct --run-name retrain_direct_v1
python -m arena.train --control-style rotation --run-name retrain_rotation_v1

# Boss/missile-aware refinements used by the submitted policies
python -m arena.train --control-style direct --timesteps 350000 --profile boss_intermission --benchmark-episodes 20 --seed 46100 --run-name dodgeable_missile_direct_350k_s46100 --init-model models/arena/dqn_direct.zip --boss-curriculum 0.78 --curriculum-phases 3
python -m arena.train --control-style rotation --timesteps 400000 --profile boss_intermission --benchmark-episodes 20 --seed 47100 --run-name dodgeable_missile_rotation_400k_s47100 --init-model models/arena/dqn_rotation.zip --new-inputs-only --boss-curriculum 0.76 --curriculum-phases 3

# Generic from-scratch runs
python -m arena.train --control-style direct --timesteps 300000 --profile balanced --run-name new_direct
python -m arena.train --control-style rotation --timesteps 300000 --profile balanced --run-name new_rotation

# Reproduce the three-profile hyperparameter comparison
python -m arena.tune --control-style both --timesteps 35000 --benchmark-episodes 6 --seed 5100
```

Default models are saved separately as `models/arena/dqn_direct.zip` and
`models/arena/dqn_rotation.zip`. The compact evidence layout is
`logs/arena/evidence/` (screenshots and held-outs), `training/` (final monitor,
curve and selected checkpoint), `tensorboard/`, and `tuning/`.

On the current assist-220/tier-balanced schema-10 24-episode holdouts, direct
achieved 2549.38 mean reward, mean Phase 16.83, 4.79 boss clears, 13.50
boss-skill dodges, and 1.42 boss-skill hits. Its fixed-Phase-3 test achieved 92%
phase progression, 2.21 boss clears, 8.00 sentry kills, 9.25 dodges, 1.08
boss-skill hits, and 2.75 missile hits per episode. Rotation achieved 72.08 mean reward, mean Phase 2.96,
100% normal phase progression, reached Phase 5, and averaged 0.50 missile hits,
but did not clear the deliberately harsh no-upgrade Phase-3 stress start. The
stronger direct boss result is expected because direct movement is the easier
action set; both models are reported honestly in `docs/part2/REPORT_NOTES.md`.

### Visual Evaluation

```bash
python -m arena.evaluate --control-style direct
python -m arena.evaluate --control-style rotation

# Equivalent dedicated entry points required for an easy video demo
python -m arena.evaluate_direct
python -m arena.evaluate_rotation
```

Playback is deterministic (`model.predict(..., deterministic=True)`) and shows
an interactive mission summary on death or timeout. It remains open until the
viewer chooses Replay, Next Run (multi-episode evaluation), or Main Menu; it no
longer auto-closes after a short delay. Playback also shows
the saved model controlling the actual submitted environment. Use `P` to pause,
`.` to single-step, `+/-` to change speed, `Tab` to inspect the current build,
`R` to replay, and `Esc` to exit. Launcher playback runs one episode by default.
Each cleared phase refreshes its deadline: 60 seconds ordinarily, 90 seconds
for the onboarding boss, and 70 seconds for later bosses. Successful play can
continue while an agent that stalls still reaches a clear terminal condition.

Build and verify the final report evidence after training:

```bash
python -m arena.build_evidence
```

Run the focused mechanics tests with:

```bash
python -m unittest discover -s tests -v
```

See `docs/part2/REPORT_NOTES.md` for final figures,
`docs/part2/RUBRIC_EVIDENCE.md` for implementation/artifact mapping, and
`docs/part2/VIDEO_DEMO.md` for the recommended recording sequence.
