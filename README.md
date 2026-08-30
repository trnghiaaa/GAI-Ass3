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

Use `WASD` or the arrow keys to move, then `Space` to fire. Direct mode snaps
shots to the nearest hostile when its reticle turns green inside assist range;
outside that range it fires along the current heading. To try rotation and
thrust controls instead:

```bash
python -m arena.play --control-style rotation
```

For rotation controls, use `W` to thrust, `A`/`D` to rotate, and `Space` to
fire in the ship's current direction. In both modes, `R` restarts an episode
and `Esc` quits. `Tab` opens a readable ship-build panel and pauses the manual
battle while you inspect every selected upgrade and live weapon statistic.

The arena provides:

- A damageable player ship with continuous movement and projectile firing
- Damageable enemy spawners that periodically create hostiles
- Enemies that continuously steer toward and damage the player on contact
- Projectile collisions with separate enemy and spawner health bars
- Increasing phases after all active spawners are destroyed
- Clean phase hand-offs that withdraw surviving hostiles and clear old shots
  without awarding fake kills, XP, or RL reward
- A gradual phase director plus a special boss rift every third phase
- A fresh 60-second combat deadline for every phase, plus a long episode safety cap
- Headless, human-window, and RGB-array rendering modes
- Targeting reticles, impact particles, projectile trails, damage feedback,
  readable HUD telemetry, phase banners, and report-ready RGB screenshots
- Per-episode combat XP and three-card level-up drafts with eleven upgrade types
- Between-phase support drafts: repair cache, nova bomb, or temporary wingman

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

The agent receives a one-dimensional `float32` vector with exactly 43
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
enemies, rifts, and completed phases grant XP within the current episode. A
manual player pauses at each level and chooses one of three seeded cards:
maximum hull, repair, damage, fire rate, multi-beam, range, prism laser,
piercing, splash, shield, or engines. Clearing a phase offers repair, an armed
arena bomb, or a one-phase combat drone. Every third phase replaces ordinary
rifts with one larger boss rift and elite minions. Difficulty grows gradually
through health, speed, contact damage, capacity, and spawn rate.

During headless training and learned-policy playback, a deterministic heuristic
chooses from the same seeded three-card offers. This keeps both rubric-required
action dictionaries exactly unchanged: the existing `Shoot` action uses the
composed build. XP and drafts never add an unreported RL reward term or replace
the required phase rule.

### Reward Design

The configurable reward function contains the required progression terms:
enemy destruction, larger spawner destruction, phase advancement, damage
penalty, and a strong death penalty. Small shaping terms give credit for actual
damage, progress toward a stable spawner target, aim improvement, and
well-aligned shots. Shaping never changes health, collisions, entity movement,
or terminal rules. Combat XP is deliberately excluded from this total. Every
`step()` exposes `info["reward_breakdown"]`, making the exact contribution of
every RL reward term auditable.

### Stable-Baselines3 Training

Both policies use SB3 DQN with a configurable MLP, replay buffer, target
network, epsilon schedule, checkpoints, held-out evaluation, TensorBoard, and
model metadata:

```bash
# Reproduce the two tuned final models
python -m arena.train --control-style direct --timesteps 200000 --profile long_exploration --benchmark-episodes 20 --seed 5200
python -m arena.train --control-style rotation --timesteps 300000 --profile fast_exploration --benchmark-episodes 20 --seed 6200

# Generic training is also supported (300,000 decisions by default)
python -m arena.train --control-style both

# Reproduce the three-profile hyperparameter comparison
python -m arena.tune --control-style both --timesteps 25000 --benchmark-episodes 6 --seed 7300
```

Final models are saved separately as `models/arena/dqn_direct.zip` and
`models/arena/dqn_rotation.zip`. TensorBoard event files, monitor CSVs,
checkpoints, deterministic seeded benchmarks, plots, and summaries are written
under `logs/arena`. In the submitted held-out 20-episode benchmarks, direct
control achieved mean reward 872.52, 100% phase progression, mean phase 9.9,
and mean ship level 8.3; rotation/thrust achieved 241.72, 100%, phase 4.25, and
level 5.95 respectively. The policies averaged 2.75 and 0.8 destroyed boss
rifts per episode. A seeded random direct baseline averaged phase 1.25 and only
25% progression, while random rotation never cleared phase 1.

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
Each cleared phase refreshes its 60-second deadline, so successful play can
continue while an agent that stalls still reaches a clear terminal condition.

Build and verify the final report evidence after training:

```bash
python -m arena.build_evidence
```

Run the focused mechanics tests with:

```bash
python -m unittest discover -s tests -v
```

See `docs/PART2_RUBRIC_EVIDENCE.md` for the implementation/artifact mapping and
`docs/PART2_VIDEO_DEMO.md` for the recommended recording sequence.
