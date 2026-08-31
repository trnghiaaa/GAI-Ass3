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

### Quick start: watch the trained Control Style 1 agent

Run these commands in **Windows PowerShell from the repository root** (the
folder containing `requirements.txt`). If you do not already have a working
`.venv`, create it and install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No virtual-environment activation is required when using the commands below.
The selected model is **v20**, a Stable-Baselines3 DQN agent using rotation and
thrust. You do **not** need to train it again to watch it play.

First, check that its saved checkpoint is available:

```powershell
Test-Path .\models\dqn_rotation\control_style_1_dqn_polished\final_model.zip
```

This should print `True`. If it prints `False`, obtain the saved model folder
with the project; installing dependencies alone does not provide the trained
checkpoint.

Watch three episodes in an animated Pygame window:

```powershell
.\.venv\Scripts\python.exe -m arena.evaluate --control-style rotation --episodes 3 --render
```

The **agent controls the ship automatically**; this is not keyboard-controlled
play. The default evaluator selects the promoted v20 checkpoint. To select it
explicitly and use a repeatable starting seed:

```powershell
.\.venv\Scripts\python.exe -m arena.evaluate --model models/dqn_rotation/control_style_1_dqn_polished/final_model.zip --control-style rotation --seed 62000 --episodes 3 --render
```

Playback defaults to `0.75` speed. Add `--playback-speed 0.5` for a slower
demonstration. This changes visual playback speed, not the learned policy.
Each episode ends when the ship dies or reaches the configured step limit;
the next episode starts automatically until the requested count is complete.
Use `Ctrl+C` in the terminal to stop evaluation early.

For a headless evaluation of the existing 60-seed comparison set
(`62000`–`62059`), saving results separately from the original evidence:

```powershell
.\.venv\Scripts\python.exe -m arena.evaluate --model models/dqn_rotation/control_style_1_dqn_polished/final_model.zip --control-style rotation --seed 62000 --episodes 60 --output logs/dqn_rotation/manual_v20_evaluation.json
```

Results are printed in the terminal and saved to the specified JSON file.
Reusing that output filename replaces only that evaluation report, not the
model. Keep the default cooldown-aware action masking enabled for normal
evaluation. The v19 comparison model remains available at
`models/dqn_rotation/control_style_1_dqn_progression/final_model.zip`.

### Play the arena yourself

The Part II environment is a continuous-coordinate Pygame combat arena. To
play it manually with direct directional movement:

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
and `Esc` quits. Human playback defaults to `0.75` speed (45 simulation steps
per wall-clock second) so the action is easier to follow. Override it with,
for example, `--playback-speed 0.5` or `--playback-speed 1.0`.

The arena currently provides:

- A damageable player ship with continuous movement and projectile firing
- Damageable enemy spawners that periodically create hostiles
- Enemies that continuously steer toward and damage the player on contact
- Projectile collisions with separate enemy and spawner health bars
- Increasing phases after all active spawners are destroyed
- Episode endings for player destruction and the configured maximum step count
- Headless, human-window, and RGB-array rendering modes

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

The agent receives a one-dimensional `float32` vector with exactly 47
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
| 20 | `player_angular_velocity` | `[-1, 1]` | Smoothed turn rate divided by maximum turn rate |
| 21–22 | Enemy forward alignment and turn direction | `[-1, 1]` | Body-relative aiming error for the nearest enemy |
| 23–24 | Spawner forward alignment and turn direction | `[-1, 1]` | Body-relative aiming error for the nearest spawner |
| 25 | Nearest-enemy closing speed | `[-1, 1]` | Positive when the closest enemy is approaching the ship |
| 26–27 | Forward and lateral velocity | `[-1, 1]` | Ship velocity in its own body frame for smoother thrust/evasion control |
| 28–31 | Second-nearest enemy direction, distance, and closing speed | mixed normalized | Prevents the policy from evading one enemy by steering into another |
| 32–33 | Crowd escape direction | `[-1, 1]` | Weighted direction away from all nearby approaching enemies |
| 34–35 | Crowd pressure and nearby-enemy count | `[0, 1]` | Overall local danger rather than only the closest threat |
| 36–37 | Escape alignment and turn direction | `[-1, 1]` | Shows which rotation aligns the ship with the safest escape corridor |
| 38–39 | Wall-safe corridor alignment and turn direction | `[-1, 1]` | Bends crowd evasion around arena boundaries instead of into corners |
| 40–43 | Left, right, top, and bottom wall clearance | `[0, 1]` | Explicit distance to each arena boundary within the safety range |
| 44–45 | Boundary-inward alignment and turn direction | `[-1, 1]` | Shows how to rotate back into the arena near an edge |
| 46 | Emergency threat | `[0, 1]` | Immediate collision urgency, separate from purple-spawner objective value |

If a target type is absent, its four target features are `(0, 0, 1, 0)`:
no direction, maximum normalized distance, and zero health. Stable feature
indices are exported as `ObservationIndex`; `env.observation_as_dict()` gives a
named view for debugging and report evidence.

Run the focused mechanics tests with:

```bash
python -m unittest discover -s tests -v
```

### Optional: train a new Control Style 1 DQN agent

This is separate from running the selected v20 model above. A new training run
does not reproduce v20's fine-tuning history merely by using the default
command, and is not needed for the demonstration.

Control style 1 uses the five discrete rotation-and-thrust actions required by
the assignment: no-op, thrust, rotate left, rotate right, and shoot. Train the
config-driven Stable-Baselines3 DQN agent headlessly with:

```bash
.venv\Scripts\python.exe -m arena.train --control-style rotation
```

The default run uses 500,000 timesteps and a two-hidden-layer `[256, 256]`
`MlpPolicy`.
Generated artifacts are kept in `models/dqn_rotation/` and
`logs/dqn_rotation/`; they include the best and final models, checkpoints,
TensorBoard event data, episode Monitor data, a PNG training curve, and a JSON
evaluation summary. To inspect TensorBoard while training:

```bash
.venv\Scripts\python.exe -m tensorboard.main --logdir logs/dqn_rotation
```

Evaluate the selected default model headlessly, or show animated evaluation.
To evaluate a new training run instead, pass its checkpoint with `--model`:

```bash
.venv\Scripts\python.exe -m arena.evaluate --control-style rotation --episodes 10
.venv\Scripts\python.exe -m arena.evaluate --control-style rotation --episodes 3 --render
```

The selected final Control Style 1 checkpoint is the early best model from
`polish_v20_150k_seed541`. Its promoted copy is stored at
`control_style_1_dqn_polished/final_model.zip`. Play it with:

```bash
.venv\Scripts\python.exe -m arena.evaluate --control-style rotation --episodes 3 --render
```

Rotation evaluation masks `SHOOT` while the observed weapon cooldown is active.
This is also used during training, including in the DQN Bellman target, so an
unavailable action cannot dominate `max Q(s', a)`. Use
`--allow-invalid-shots` only for ablation against an unmasked policy.

The reward hierarchy now prioritizes phase progression, spawner destruction,
and survival over repeatable enemy damage. Spawner-distance shaping is active
only below the configured emergency threshold, so it cannot reward a suicide
rush through a crowd.

On 60 identical held-out episodes (seed 62000), v19 reached Phase 2 or higher
in `48.3%` of episodes, compared with `18.3%` for both v14 and v18. It averaged
`1.62` destroyed spawners, `4.60` enemies, `50%` survival, and a `53.9%` shot
hit rate. Raw cooldown SHOOT attempts fell from v18's `34%` to `21%`, while
executed invalid shots remained zero. The main remaining weakness is near-wall
time (`36%`), which is reported rather than hidden. v14 and v18 remain
available. Full results and the requested comparison are in
`logs/dqn_rotation/comparison_seed62000_60.md`.

The final conservative polish run, `polish_v20_150k_seed541`, refined those
same signals: wall trapping activates only after sustained non-escape, and the
danger stagnation penalty applies only when a close enemy is still closing
while ship speed is low. On the same 60 seeds, its early best checkpoint raised
Phase 2+ progression from `48.3%` to `85.0%`, raised survival from `50.0%` to
`53.3%`, and reached Phase 3. It also reduced near-wall time from `35.8%` to
`33.2%`, danger time from `78.1%` to `74.6%`, and danger NOOP from `29.6%` to
`16.9%`. Shot accuracy declined from `53.9%` to `45.6%`, but progression is the
primary assignment objective and invalid shots remained zero. v20 is therefore
the final automatic default; v19 remains available at
`control_style_1_dqn_progression/final_model.zip` as the baseline comparison.
See `logs/dqn_rotation/polish_comparison_seed62000_60.md` for the complete
decision table. Control Style 1 training is now frozen.

Rendered evaluation is frame-limited and uses the configured `0.75` playback
speed by default. Headless training and evaluation remain uncapped for speed.

The legacy 20-feature `control_style_1_dqn_300k_seed42` run was trained for 300,000
timesteps. On 20 held-out episodes (seed 31415), its final model improved mean
reward from `-24.21` for a random policy to `-9.48`, destroyed `3.35` enemies
per episode instead of `0`, and reached phase 2 in 3 of 20 episodes while the
random policy never left phase 1. See the training curve and JSON evaluation
files in `logs/dqn_rotation/control_style_1_dqn_300k_seed42/`.

After adding frame-limited `0.75`-speed playback, the legacy 20-feature agent was retrained for
another 300,000 timesteps as `control_style_1_dqn_slow_playback_300k_seed73`.
On 20 held-out episodes (seed 27182), its final model improved mean reward from
`-22.81` for random control to `-12.80`, averaged `1.70` destroyed enemies,
and reached phase 2 in 3 of 20 episodes.

The current 28-feature combat run was trained for 500,000 timesteps as
`control_style_1_dqn_combat_v3_500k_seed91`. It adds smoothed turning,
body-relative motion and target-alignment features, approaching-enemy speed,
combat/evasion reward signals, and more visible beam projectiles. On 30 held-out
episodes (seed 42424), the selected best checkpoint reached phase 2 or higher in
29 episodes and phase 3 in 15 episodes. It averaged `49.02` reward, `3.40`
enemies destroyed, `4.23` spawners destroyed, and a `72.8%` projectile hit rate.
Random control never left phase 1 and achieved a `3.2%` hit rate. Episodes lasted
`1,046.6` steps on average versus `659.2` for random control. The arena becomes
progressively harder without a final winning phase, so the trained policy still
eventually died in all 30 trials; its improvement is measured by combat accuracy,
phase progression, and survival time. Detailed evidence is in
`logs/dqn_rotation/control_style_1_dqn_combat_v3_500k_seed91/`.

The crowd-evasion extension expands the observation from 28 to 40 features,
including the second-nearest enemy, closing threats, crowd pressure, an escape
direction, and a wall-safe corridor. Turning momentum is retained briefly while
thrusting or shooting, producing curved evasive movement instead of rigid
rotate-then-move paths. The original combat network is protected during the
fine-tune: only weights connected to the 12 new inputs are trained. Training
episodes randomly start in phases 1–3 to expose the policy to dense crowds;
evaluation and playback always start normally in phase 1.

On 40 identical held-out episodes (seed 52525), the selected curriculum model
kept the original phase-3 reach rate at `57.5%` while improving mean reward from
`31.55` to `33.63`, hit rate from `72.17%` to `76.33%`, and mean destroyed
spawners from `4.525` to `4.575`. It reduced danger-step exposure from `71.15%`
to `68.59%`, crowd pressure from `0.2064` to `0.2036`, and damage per 1,000
steps from `112.75` to `111.69`. The reproducible JSON results and training
curve are in
`logs/dqn_rotation/control_style_1_dqn_phase_curriculum_v8_150k_seed197/`.

The normal rotation controller now blends two low-level steering behaviours
without replacing the DQN's selected action. Firing retains a small forward
combat drift and, under crowd pressure, blends a turn toward the safe corridor;
this lets the ship move and evade while continuing to shoot. A heading-ray wall
check begins steering before a forward collision, but leaves wall-parallel
movement unchanged. Outward velocity is cleared immediately at a boundary.
On 40 evaluation episodes (seed 85858), the unshielded agent retained a `2.50`
mean maximum phase and reached phase 3, while crowd-danger exposure averaged
`64.20%` and dangerous spawner damage averaged `133.73` per episode.

For additional conservative evaluation, `--safety-shield` adds a narrow safety
layer around the DQN. The neural policy still selects every ordinary action.
The layer only vetoes (1) thrust aimed out of the arena at the actual boundary,
and (2) a shot aligned with a purple spawner during a severe red-enemy threat
when the ship is not aimed at that enemy. Defensive shots aimed at red enemies
remain allowed. On 40 matched episodes (seed 63636), this reduced outward-thrust
steps from `4.58%` to `2.97%`, near-wall time from `38.35%` to `34.79%`,
emergency shots from `21.35` to `19.30`, and dangerous spawner damage from
`153.92` to `142.42` per episode. Mean reward increased from `12.60` to `13.91`,
though the more defensive behavior reduced the phase-3 rate from `55.0%` to
`42.5%`; omit `--safety-shield` when measuring the unmodified DQN alone.
