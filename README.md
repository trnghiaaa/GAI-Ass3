# Gridworld AI Lab — Assignment 3, Part I

This branch contains the polished **classical reinforcement-learning Gridworld** for Part I of *Games and Artificial Intelligence Techniques*. It implements all seven levels, Q-learning, SARSA, stochastic monsters, the exact Level 6 count-based intrinsic reward, reproducible experiments, saved policies, report-ready evidence, and a unified Pygame experience.

The assignment's environment rewards are never changed by the UI or training tools.

## Start the complete game

From this repository directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

`main.py` automatically hands execution to the repository `.venv` when the
currently selected Python does not have Pygame. This hand-off does not pass the
virtual-environment path through PowerShell, so it remains safe when a parent
folder contains `&`, as in `A3_Game&AI`.

On Windows, the easiest option is to double-click **`scripts/part1/run_part1.bat`**. It automatically uses the project `.venv` when present, so nobody needs to type the interpreter path. The package-style command `python -m gridworld` remains an equivalent alternative.

### VS Code run button

The generic **Run Python File** triangle is owned by the Python extension and may
still paste an unquoted cached interpreter path into PowerShell. This repository
therefore includes two shell-safe launch routes that never parse the `.venv`
path as PowerShell source:

- press `F5` and select **Part I: Run Gridworld (safe)**; or
- press `Ctrl+Shift+B` to run the default **Part I: Run Gridworld (shell-safe)** task.

Both launch `main.py` without constructing the unsafe `& D:\A3_Game&AI...`
command. The second route uses a VS Code `process` task rather than a shell task.

After cloning or pulling these settings, run **Developer: Reload Window** once (or close all existing VS Code terminals) before using the configured run option. Existing terminals keep their previous environment. The folder name `A3_Game&AI` contains PowerShell's `&` operator, so the old unquoted absolute command cannot execute safely; `scripts/part1/run_part1.bat` remains an unaffected alternative.

If the top-right triangle still prints a command beginning with an absolute
`.venv\\Scripts\\python.exe` path, VS Code is retaining its old workspace choice.
Open the Command Palette, run **Python: Clear Workspace Interpreter Setting**,
then run **Developer: Reload Window**. The repository intentionally launches the
plain `python` command and lets `main.py` perform the shell-safe `.venv` hand-off.

The single launcher provides:

- **Campaign** — play Levels 0 through 6 continuously, or watch the trained AI tour.
- **Free Play** — choose any level and verify every mechanic manually.
- **AI Showcase** — select a saved Q-learning, SARSA, or intrinsic policy.
- **RL Inspector** — inspect current-state Q-values and exact greedy ties.
- **Policy Lens** — overlay learned actions on the grid.
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
| M or Esc | Menu |

AI campaign transitions always begin the next level at 1x. The result screen also
provides `-`, `+`, and **Reset to 1x** controls before replaying, so a fast playback
setting never traps the player behind the completion popup.

Direct links remain available for assessors and quick recording:

```bash
python main.py --mode manual --level 4
python main.py --mode ai --level 1 --agent sarsa
python main.py --mode ai --level 6 --agent qlearning --intrinsic
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

docs/screenshots/
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

See `docs/PART1_RUBRIC_EVIDENCE.md` for the code/artifact mapping and `docs/VIDEO_DEMO_PLAN.md` for a concise recording plan.
