# Gridworld AI Lab — Assignment 3, Part I

This branch contains the polished **classical reinforcement-learning Gridworld** for Part I of *Games and Artificial Intelligence Techniques*. It implements all seven levels, Q-learning, SARSA, stochastic monsters, the exact Level 6 count-based intrinsic reward, reproducible experiments, saved policies, report-ready evidence, and a unified Pygame experience.

The assignment's environment rewards are never changed by the UI or training tools.

## Start the complete game

From this repository directory:

```bash
python -m pip install -r requirements.txt
python main.py
```

On Windows, the easiest option is to double-click **`run_part1.bat`**. It automatically uses the project `.venv` when present, so nobody needs to type the interpreter path. The package-style command `python -m gridworld` remains an equivalent alternative.

### VS Code run button

The repository includes `.vscode/launch.json`. Open `main.py` and use the play-button dropdown to select **Part I: Run Gridworld**, or press `F5` and select **Part I: Debug Gridworld**. Both launch the fixed root entry point without constructing a PowerShell command from the workspace path.

If VS Code still shows an old malformed command, close its existing Python terminal once and use the configured run option. The folder name `A3_Game&AI` contains PowerShell's `&` operator, so an unquoted interpreter path cannot be executed safely; `run_part1.bat` and `python main.py` are also unaffected alternatives.

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
| M or Esc | Menu |

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
| Rock / boundary | Agent remains in its current cell | 0 |
| Fire | Immediate death | 0 |
| Monster collision | Immediate death, whether the agent enters its tile or it enters the agent's tile | 0 |
| Apple | Consumed | +1 |
| Key | Consumed; enables chest opening | 0 |
| Locked chest | Remains until the key is held | 0 |
| Opened chest | Consumed | +2 |
| Monster phase | Each monster independently has the configured 40% chance to make one valid random move after an agent action | 0 |
| Episode terminal | All collectible rewards obtained, or agent death | — |

There is no hidden step penalty, death penalty, bonus environment reward, or altered terminal rule. Level 6's intrinsic bonus exists only in the agent's learning target and is logged separately.

## Levels and rubric coverage

| Level | Assignment task | Main evidence |
|---:|---|---|
| 0 | Task 1: basic Q-learning | Greedy Q-learning completes the verified optimum in **16 steps** |
| 1 | Task 2: SARSA | Q-learning uses a 5-step fire-edge route; SARSA uses a 7-step safe route |
| 2 | Task 3 | Two apples, key, chest; Q-learning and SARSA |
| 3 | Task 3 | Harder maze with two apples, key, chest; both algorithms |
| 4 | Task 4 | One stochastic monster; both algorithms and learning curves |
| 5 | Task 4 | Two stochastic monsters; both algorithms and learning curves |
| 6 | Task 5 | Baseline versus exact per-episode count-bonus Q-learning |

Level definitions and display metadata live in `gridworld/levels/levels.py`.

The final seeded benchmark records 95.0% / 96.3% success for Level 4 Q-learning / SARSA and 97.7% / 99.0% for Level 5. All deterministic policies and both Level 6 variants achieve 100% benchmark success. Full results are in `logs/gridworld/policy_benchmark.json`.

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

This builds 13 main model runs plus the multi-seed Level 1 and Level 6 experiments, benchmarks every saved policy, then verifies 62 required artifacts. A quick plumbing-only run is available, but should not be submitted:

```bash
python -m gridworld.build_evidence --quick
```

### Individual training

```bash
python -m gridworld.train --level 0 --agent qlearning
python -m gridworld.train --level 3 --agent sarsa
python -m gridworld.train --level 6 --agent qlearning --intrinsic
```

Useful overrides include `--episodes`, `--alpha`, `--gamma`, `--epsilon-start`, `--epsilon-end`, `--max-steps`, `--seed`, and `--intrinsic-strength`. Defaults and per-level profiles are in `gridworld/config.json`.

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

All visual art is drawn procedurally with Pygame primitives; there are no copied sprites or third-party game assets. Creative work beyond the base specification includes the connected campaign, AI tour, responsive interface, animated interpolation, particles, trail rendering, policy lens, live Q-value inspector, deterministic playback controls, evidence dashboards, exploration heatmap, multi-seed confidence bands, config provenance, and automated rubric acceptance tests.

See `docs/PART1_RUBRIC_EVIDENCE.md` for the code/artifact mapping and `docs/VIDEO_DEMO_PLAN.md` for a concise recording plan.
