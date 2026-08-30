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

Use `WASD` or the arrow keys to move and `Space` to auto-aim and fire. To try
rotation and thrust controls instead:

```bash
python -m arena.play --control-style rotation
```

For rotation controls, use `W` to thrust, `A`/`D` to rotate, and `Space` to
fire in the ship's current direction. In both modes, `R` restarts an episode
and `Esc` quits.

The arena provides:

- A damageable player ship with continuous movement and projectile firing
- Damageable enemy spawners that periodically create hostiles
- Enemies that continuously steer toward and damage the player on contact
- Projectile collisions with separate enemy and spawner health bars
- Increasing phases after all active spawners are destroyed
- Episode endings for player destruction and the configured maximum step count
- Headless, human-window, and RGB-array rendering modes
- Targeting reticles, impact particles, projectile trails, damage feedback,
  readable HUD telemetry, phase banners, and report-ready RGB screenshots
- Per-episode combat XP with five automatic ship levels: Pulse Cannon, Rapid
  Loader, Twin Pulse, Laser Array, and Nova Tri-Beam

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

The agent receives a one-dimensional `float32` vector with exactly 34
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
| 19 | `time_remaining` | `[0, 1]` | Fraction of the episode step budget remaining |
| 20–21 | Enemy/spawner aim alignment | `[-1, 1]` | Cosine alignment between ship heading and each nearest target |
| 22–24 | Active-target direction X/Y and distance | mixed normalized | The exact closest target used by the targeting reticle |
| 25 | `active_target_is_spawner` | `[0, 1]` | Distinguishes a progression target from an enemy |
| 26–27 | Active-target alignment and signed turn direction | `[-1, 1]` | Tells rotation control how close its aim is and which way to turn |
| 28–29 | Ship level and XP progress | `[0, 1]` | Current automatic upgrade tier and progress toward the next tier |
| 30–33 | Volley size, fire rate, damage and laser flag | `[0, 1]` | Active weapon characteristics needed to keep progression Markov |

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

### Combat XP and Automatic Upgrades

Combat XP is a creative gameplay system separate from the RL reward. Destroyed
enemies, destroyed rifts, and completed phases grant XP within the current
episode. Levels automatically unlock faster firing, multi-shot volleys, and
laser weapons. Automatic unlocks preserve the assignment's exact action sets:
the existing `Shoot` action simply fires the active tier. XP never replaces the
required phase system or adds an unreported term to the RL reward breakdown.

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
python -m arena.train --control-style direct --timesteps 150000 --profile balanced --benchmark-episodes 20 --seed 5200
python -m arena.train --control-style rotation --timesteps 250000 --profile fast_exploration --benchmark-episodes 20 --seed 6200

# Generic training is also supported (300,000 decisions by default)
python -m arena.train --control-style both

# Reproduce the three-profile hyperparameter comparison
python -m arena.tune --control-style both --timesteps 20000 --benchmark-episodes 8 --seed 4200
```

Final models are saved separately as `models/arena/dqn_direct.zip` and
`models/arena/dqn_rotation.zip`. TensorBoard event files, monitor CSVs,
checkpoints, deterministic seeded benchmarks, plots, and summaries are written
under `logs/arena`. In the submitted held-out 20-episode benchmarks, direct
control achieved mean reward 622.98, 100% phase progression, and mean ship level
4.95; rotation/thrust achieved 186.84, 90%, and level 4.05 respectively.

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
`.` to single-step, `+/-` to change speed, `R` to replay, and `Esc` to exit.

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
