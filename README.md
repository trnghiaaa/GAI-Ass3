# Assignment 3: Reinforcement Learning and Deep RL Agents

This repository contains the complete group project for **Games and Artificial
Intelligence Techniques**:

- **Part I - Gridworld AI Lab:** visual Q-learning and SARSA across seven
  Pygame levels.
- **Part II - Neon Rift Arena:** a continuous action arena with two separately
  trained Stable-Baselines3 DQN agents.

Both parts, the saved models, training logs, TensorBoard events, evaluation
evidence, and presentation features are available from one launcher.

## Quick start

Python 3.11 is recommended. From the repository root in Windows PowerShell:

### First-Time Setup
```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the visual project hub
python main.py
```

### Subsequent Runs
In any new terminal session, activate the environment and run:
```powershell
.\.venv\Scripts\activate
python main.py
```
*(VS Code automatically activates `.venv` whenever you open a new terminal).*

### Direct Launch Options
```powershell
python main.py --part 1
python main.py --part 2
```

> [!NOTE]
> If you choose to install dependencies into your global Python instead of a virtual environment, you can skip the `.venv` steps and simply run `pip install -r requirements.txt` followed by `python main.py`. If `.venv` is created but not activated, prefix commands with `.\.venv\Scripts\python.exe`.

Windows users can also double-click:

- `scripts/part1/run_part1.bat`
- `scripts/part2/run_part2.bat`

In VS Code, press `F5` and choose either supplied launch configuration. These
routes also avoid PowerShell parsing problems when a parent directory contains
an ampersand (`&`).

## Project summary

| Area | Implementation | Main evidence |
|---|---|---|
| Part I environment | Animated, interactive Pygame Gridworld | `gridworld/`, `logs/gridworld/` |
| Classical RL | Q-learning and SARSA with config-driven training | `gridworld/agents/`, `gridworld/config.json` |
| Intrinsic reward | Per-episode count-based reward for Level 6 | `gridworld/agents/`, comparison logs |
| Part II environment | Continuous Pygame combat arena with Gymnasium API | `arena/core/` |
| Deep RL | Separate Direct and Rotation DQN policies | `models/arena/dqn_direct.zip`, `dqn_rotation.zip` |
| Training evidence | Monitor CSV, curves, tuning and TensorBoard logs | `logs/arena/` |

## Part I - Gridworld AI Lab

### Required mechanics

- Four actions: up, down, left and right.
- Rocks and borders block movement.
- A blocked attempt is still an agent action: the player remains in place, the
  action counter advances, and the monster phase still occurs.
- Fire or monster contact causes immediate death.
- Apples give `+1`, keys give `0`, and an opened chest gives `+2`.
- A key is required before a chest can open.
- After every action, each monster independently has a 40% chance to move.
- An episode ends when all collectible rewards are obtained or the agent dies.

### Levels

| Level | Focus |
|---:|---|
| 0 | Three apples on the right; shortest-path Q-learning |
| 1 | Fire hazard; Q-learning versus conservative SARSA |
| 2 | Multiple apples, key, chest and branching routes |
| 3 | Multiple apples, key, chest and a denser maze |
| 4 | One probabilistically moving monster |
| 5 | Two independently moving monsters |
| 6 | Sparse maze; baseline versus intrinsic exploration reward |

The implementation includes epsilon-greedy selection, random tie-breaking,
config-driven linear epsilon decay, the correct off-policy Q-learning update,
and the correct on-policy SARSA update. Level 6 uses:

```text
intrinsic_reward = intrinsic_strength / sqrt(n(s) + 1)
learning_reward  = unchanged_environment_reward + intrinsic_reward
```

The visit counter is reset at the start of every episode. Intrinsic reward is
used only for learning and never changes the environment reward.

### Part I commands

```bash
# Train one policy
python -m gridworld.train --level 0 --agent qlearning

# Train the required Level 6 intrinsic policy
python -m gridworld.train --level 6 --agent qlearning --intrinsic

# Watch a saved greedy policy
python -m gridworld.evaluate --level 4 --agent sarsa

# Rebuild all Part I models, comparisons and benchmark evidence
python -m gridworld.build_evidence
```

The visual application adds a seven-level campaign, algorithm selection,
policy arrows, live Q-values, pause, single-step playback, adjustable speed,
training summaries, procedural effects, and audio.

## Part II - Neon Rift Arena

### Environment and progression

The Arena is a real-time, continuous-coordinate Pygame environment. It includes:

- A controllable ship with movement, shooting and health.
- Enemy rifts that periodically spawn enemies.
- Enemies that steer toward the player.
- Enemy health, projectile collisions and damage.
- Phase progression after all active rifts are destroyed.
- Per-phase deadlines, player death and a long safety cap as terminal cases.

`ArenaEnv` uses the Gymnasium reset/step API required by Stable-Baselines3.
`LegacyArenaEnv` provides the assignment-style
`step(action) -> observation, reward, done, info` adapter. Rendering supports
both a visible Pygame window and RGB arrays.

### Observation, actions and reward

The agent receives a fixed **126-value `float32` observation vector**. It
includes player position, velocity, orientation, health, phase, nearest enemy
and rift geometry, build state, walls, target information, boss vulnerability,
telegraphed hazards, sentry priority and missile escape information.

| Control style | Discrete actions | Submitted model |
|---|---|---|
| Direct | no-op, up, down, left, right, shoot | `models/arena/dqn_direct.zip` |
| Rotation + thrust | no-op, thrust, rotate left, rotate right, shoot | `models/arena/dqn_rotation.zip` |

Required event rewards are configured in `arena/config.json`: destroying
enemies and rifts, progressing a phase, taking damage, and dying. Documented
shaping encourages safe spacing, accurate shooting, target pressure, crowd and
wall escape, boss-hazard dodging, missile evasion, and correct sentry priority.
XP and ship upgrades remain separate from the RL reward.

### Manual controls

| Key | Direct mode | Rotation mode |
|---|---|---|
| `WASD` / arrows | Move | `W` thrust; `A/D` rotate |
| `Space` | Shoot | Shoot forward |
| `1`, `2`, `3` | Choose an upgrade/reward card | Choose an upgrade/reward card |
| `Tab` | Inspect the current ship build | Inspect the current ship build |
| `C` | Cycle ship skins | Cycle ship skins |
| `P` | Pause/resume | Pause/resume |
| `G` | Pilot Guide | Pilot Guide |
| `V` / `M` | Volume panel / mute | Volume panel / mute |
| `R` | Replay the current seed | Replay the current seed |

### Creative gameplay features

The required arena loop is extended with ship XP, uncapped levels, three-card
upgrade drafts, phase caches, boss relics, support wingmen, minibosses,
telegraphed boss attacks, and adaptive difficulty. Low-health bosses can enter
an Aegis intermission: sentries temporarily make the boss immune and repair it,
so players and agents must dodge their missiles and destroy the sentries first.

The launcher also includes an illustrated Pilot Guide, build inspector,
replays, mission summaries, five ship palettes, procedural music, event sound
effects, AI pause/single-step controls and adjustable playback speed.

### Evaluate the submitted DQN models

```bash
# Visual deterministic evaluation
python -m arena.evaluate_direct --episodes 3
python -m arena.evaluate_rotation --episodes 3

# Headless benchmark
python -m arena.evaluate --control-style direct --headless --episodes 12
```

### Train and compare DQN agents

```bash
# Separate training runs
python -m arena.train --control-style direct --timesteps 300000 --profile balanced --run-name my_direct
python -m arena.train --control-style rotation --timesteps 300000 --profile balanced --run-name my_rotation

# Reproduce the three-profile hyperparameter comparison
python -m arena.tune --control-style both --timesteps 40000

# Verify and rebuild report-ready Part II evidence
python -m arena.build_evidence
```

Training uses Stable-Baselines3 DQN with an MLP, replay buffer, target network,
epsilon schedule, checkpoints, held-out evaluation, Monitor CSV files and
TensorBoard logging.

### TensorBoard

```bash
tensorboard --logdir ./logs/arena/tensorboard --port 6006
```

Open `http://localhost:6006`. For a clear presentation comparison, select only:

- `volley_v11_direct_180k_s93100_1`
- `edge_escape_rotation_600k_r2_s92100_1`

Useful Scalars include `rollout/ep_rew_mean`, `eval/mean_reward`,
`rollout/exploration_rate`, `arena/phase`,
`arena/event_boss_skills_dodged`, and `arena/event_missiles_evaded`.

## Repository structure

```text
main.py                    Shared visual launcher
gridworld/                 Part I environment, agents, UI and evidence tools
arena/core/                Part II simulation and entities
arena/presentation/        Arena launcher, renderer, manual play and guidebook
arena/learning/            DQN training, wrappers and tuning
arena/evaluation/          Visual and headless policy evaluation
arena/tools/               Evidence and model-selection utilities
models/gridworld/          Saved tabular policies
models/arena/              Saved DQN policies and metadata
logs/                      Training, evaluation and TensorBoard evidence
docs/part1/                Detailed Part I rubric evidence map
docs/part2/                Detailed Part II rubric evidence map
tests/                     Part I, Part II and integration tests
scripts/                   Windows launch and evidence helpers
```

## Verification

Install the development dependency and run the complete suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Detailed rubric-to-code mappings and current numeric results are deliberately
kept outside this concise README:

- `docs/part1/RUBRIC_EVIDENCE.md`
- `docs/part2/RUBRIC_EVIDENCE.md`
- `logs/gridworld/`
- `logs/arena/evidence/evidence_manifest.json`

Use the saved models and evidence included in the repository for the report and
video demonstration. Retraining is optional and may take substantial time.
