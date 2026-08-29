# Part I rubric evidence map

This file is a navigation aid for markers and report authors. Generated JSON and CSV files are the source of truth for numeric claims.

## A — Gridworld implementation and rules

| Criterion | Code evidence | Automated evidence |
|---|---|---|
| Visual, animated, interactive Pygame | `gridworld/app.py` (`GridworldApp`) and `gridworld/renderer.py` | `tests/test_app.py` |
| Four moves; rocks block | `GridWorldEnv.step()` | `test_rocks_and_boundaries_block_without_reward` |
| Fire/monster immediate death | `GridWorldEnv.step()` and `_move_monsters()` | fire, step-on-monster, and monster-onto-agent tests |
| Apple +1 | collectible phase in `GridWorldEnv.step()` | `test_apple_reward_and_reward_completion_terminal` |
| Key +0; chest +2 | key/chest phase in `GridWorldEnv.step()` | `test_key_locked_chest_and_exact_reward_accounting` |
| Correct terminal rules | `GridWorldEnv.step()` | apple, chest, fire, and monster tests |
| 40% monster movement | config plus `_move_monsters()` | exact 0/1 and 5,000-step statistical test |

The environment returns `info["env_reward"]` equal to the returned reward. Intrinsic reward is never inserted into the environment.

## B — Task 1: Q-learning Level 0

| Criterion | Evidence |
|---|---|
| Epsilon-greedy | `TabularAgentBase.choose_action()` |
| Off-policy maximum | `QLearningAgent.update()` and numeric unit test |
| Linear config-driven decay | `TabularAgentBase.decay_epsilon()`, `config.json`, exact schedule test |
| Random exact ties | `best_actions()` uses exact equality; tie test samples both best actions |
| Shortest path | `test_level0_q_learning_reaches_verified_16_step_optimum`; saved greedy model completes all three apples in 16 steps |

Artifacts: `level0_qlearning.pkl`, `level0_qlearning.png`, CSV, and summary JSON.

## C — Task 2: SARSA Level 1

`SARSAAgent.update()` uses the actual `next_action` selected once by the training loop. Q-learning and SARSA use the identical Level 1 profile and seed list.

Ten paired seeds currently produce the policy difference in all 10 runs:

| Metric | Q-learning | SARSA |
|---|---:|---:|
| Greedy route | Risk lane | Safe lane |
| Greedy steps | 5 | 7 |
| Hazard-adjacent steps | 4 | 0 |
| Minimum fire distance | 1 | 2 |
| Mean victory rate at epsilon 0.05 | 95.3% | 99.8% |
| Mean fire deaths per 200 eval episodes/seed | 9.4 | 0.4 |

Artifacts: `level1_qlearning_vs_sarsa_evidence.png`, comparison metrics CSV, rollouts CSV, and summary JSON.

## D — Task 3: Levels 2–3

Both maps contain two apples, one key, and one chest. Both algorithms have config-driven model/log bundles. Mechanics are covered by unit tests and trained policies complete the deterministic objectives.

Artifacts: Level 2 and 3 Q-learning/SARSA models, dashboards, metrics CSVs, and summary JSONs.

## Task 4 — Monster Levels 4–5

- Each monster independently samples the configured 0.4 movement probability after agent actions.
- Valid movement excludes bounds, rocks, fire, and occupied monster tiles.
- Collision is checked when the agent enters a monster tile and after monsters move.
- The state records exact relative offsets for threats within distance two plus a count of every farther monster. This produces useful local-threat generalisation without visually hiding any monster.
- Environment and agent randomness use independent seeded streams.

Artifacts: Level 4 and 5 Q-learning/SARSA models, dashboards, full metrics CSVs, summaries, and `policy_benchmark.json`.

Final seeded benchmark (300 episodes per policy): Level 4 Q-learning **95.0%**, Level 4 SARSA **96.3%**, Level 5 Q-learning **97.7%**, and Level 5 SARSA **99.0%** success.

## F — Task 5: intrinsic reward Level 6

`TabularAgentBase.reward_components()` performs the required sequence:

1. Read prior per-episode visits to `next_state`.
2. Calculate `intrinsic_strength / sqrt(n(s) + 1)`, where `s` is that reached/current state.
3. Set `learning_reward = unchanged_environment_reward + intrinsic_reward`.
4. Record the destination visit, including terminal destinations.
5. Reset all visit counts at the next episode.

Both Q-learning and SARSA inherit this exact implementation. Numeric tests check the first bonus, repeat bonus, terminal bonus, and reset behavior.

Five paired Level 6 seeds compare baseline and intrinsic Q-learning:

| Metric | Baseline | Intrinsic |
|---|---:|---:|
| Mean first-success episode | 10.2 | 8.0 |
| Last-100 victory rate | 100% | 100% |
| Greedy success | 100% | 100% |
| Greedy completion | 22 steps | 22 steps |

The observed improvement is a 21.6% earlier first discovery. The plot and raw data also expose the early exploration/exploitation trade-off rather than comparing shaped returns on incompatible scales.

Artifacts: baseline/intrinsic main models, per-seed evidence models, evidence PNG, raw comparison CSV, rollout CSV, and summary JSON.

## Creativity evidence

- One connected seven-level campaign and AI tour.
- Responsive level-select and algorithm-select scenes.
- Live Q-value inspector and policy-arrow lens.
- Animated movement, hazards, particles, trail, and result overlays.
- Pause, single-step, five playback speeds, reproducible greedy ties, and safe timeouts.
- Multi-seed plots with confidence bands, route overlays, hazard metrics, and exploration heatmap.
- One-command evidence builder and automated rubric acceptance suite.
- Entirely procedural original art; no external sprites or asset licences required.

## Verification commands

```bash
python -m pytest
python -m gridworld.build_evidence
python -m gridworld.benchmark
python main.py
```
