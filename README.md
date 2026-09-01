# Reinforcement Learning & Deep RL Agents (Assignment 3)

## 1. Setup & Installation
Make sure you have Python 3.10+ installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Interactive Manual Play (Gridworld)
You can play and test the mechanics of each level using keyboard controls:

```bash
python -m gridworld.play --level 0
```

### Try different levels:
- `--level 0` : Basic navigation & collecting apples (+1 reward)
- `--level 1` : Fire hazards (stepping on fire = instant death)
- `--level 2` : Key & chest mechanics (collect key first to open chest for +2 reward)
- `--level 4` : Stochastic monsters (monsters move with 40% probability)

### Controls:
- **Arrow Keys** : Move (Up / Down / Left / Right)
- **R** : Reset the current level
- **ESC / Q** : Quit game window

---

## 3. Part II Action Arena

> Schema-9 boss-intermission learning, pause/UI, upgrade clarity, and tidy final
> evidence are complete. See `docs/PART2_REPORT_NOTES.md` for current metrics
> and `docs/PART2_RUBRIC_EVIDENCE.md` for the marking-evidence map.

The Part II environment is a continuous-coordinate Pygame combat arena named
**Neon Rift Arena**. The easiest entry point is the unified visual launcher:

```bash
python main.py
# equivalent aliases: python part2.py or python -m arena
```

You can also open `main.py` and press the VS Code **Run Python File** button.
The repository's shared VS Code settings select `.venv` and use Command Prompt
on Windows, avoiding PowerShell parsing errors when the project path contains
an ampersand. `run_part2.bat` is the double-click fallback. The menu offers
manual play and deterministic trained-agent playback for both required control
schemes. Direct manual play is also available from the command line:

```bash
python -m arena.play --control-style direct
```

Use `WASD` or the arrow keys to move and hold `Space` to fire at the same time. Direct mode snaps
shots to the nearest hostile when its reticle turns green inside assist range;
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

Use a unique `--run-name` for new runs; training refuses to overwrite existing
models or run directories. The `boss_intermission` profile can use a training-only
boss curriculum: it samples a genuine Phase-3 reset for a configured fraction
of training episodes. It never selects actions for the policy or makes the
boss easier. A changed game still requires training and fresh evaluation.

Both policies use SB3 DQN with a configurable MLP, replay buffer, target
network, epsilon schedule, checkpoints, held-out evaluation, TensorBoard, and
model metadata:

```bash
# Boss/missile-aware refinements used by the submitted policies
python -m arena.train --control-style direct --timesteps 350000 --profile boss_intermission --benchmark-episodes 20 --seed 46100 --run-name dodgeable_missile_direct_350k_s46100 --init-model models/arena/dqn_direct.zip --boss-curriculum 0.78 --curriculum-phases 3
python -m arena.train --control-style rotation --timesteps 400000 --profile boss_intermission --benchmark-episodes 20 --seed 47100 --run-name dodgeable_missile_rotation_400k_s47100 --init-model models/arena/dqn_rotation.zip --new-inputs-only --boss-curriculum 0.76 --curriculum-phases 3

# Generic from-scratch runs
python -m arena.train --control-style direct --timesteps 300000 --profile balanced --run-name new_direct
python -m arena.train --control-style rotation --timesteps 300000 --profile balanced --run-name new_rotation

# Generic training is also supported (300,000 decisions by default)
python -m arena.train --control-style both

# Reproduce the three-profile hyperparameter comparison
python -m arena.tune --control-style both --timesteps 35000 --benchmark-episodes 6 --seed 5100
```

Default models are saved separately as `models/arena/dqn_direct.zip` and
`models/arena/dqn_rotation.zip`. The compact evidence layout is
`logs/arena/evidence/` (screenshots and held-outs), `training/` (final monitor,
curve and selected checkpoint), `tensorboard/`, and `tuning/`.

On the current schema-10 24-episode holdouts, direct achieved 998.48 mean
reward, mean Phase 9.04, 2.08 boss clears, 10.54 boss-skill dodges, and 1.25
boss-skill hits. Its fixed-Phase-3 test achieved 62.5% phase progression, 0.75
boss clears, 2.83 sentry kills, 6.17 dodges, 0.92 boss-skill hits, and 1.58
missile hits per episode. Rotation achieved 87.33 mean reward, mean Phase 2.83,
100% normal phase progression, and only 0.12 missile hits per episode, but did
not clear the deliberately harsh no-upgrade Phase-3 stress start. The stronger
direct boss result is expected because direct movement is the easier action
set; both models are reported honestly in `docs/PART2_REPORT_NOTES.md`.

### Visual Evaluation

```bash
python -m arena.evaluate --control-style direct
python -m arena.evaluate --control-style rotation

# Equivalent dedicated entry points required for an easy video demo
python -m arena.evaluate_direct
python -m arena.evaluate_rotation
```

Playback is deterministic (`model.predict(..., deterministic=True)`) and shows
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

See `docs/PART2_REPORT_NOTES.md` for final figures,
`docs/PART2_RUBRIC_EVIDENCE.md` for implementation/artifact mapping, and
`docs/PART2_VIDEO_DEMO.md` for the recommended recording sequence.
