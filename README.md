# Reinforcement Learning & Deep RL Agents (Assignment 3)

This repository contains the complete implementation for **Assignment 3: Reinforcement Learning and Deep RL Agents**, covering both classical value-based methods in a Pygame Gridworld (Part I) and deep reinforcement learning in an Action Arena (Part II).

---

## Table of Contents
1. [Setup & Installation](#1-setup--installation)
2. [Interactive Manual Play (Gridworld)](#2-interactive-manual-play-gridworld)
3. [Task 1: Basic Q-Learning (Level 0)](#3-task-1-basic-q-learning-level-0)
4. [Task 2: SARSA & Hazard Comparison (Level 1)](#4-task-2-sarsa--hazard-comparison-level-1)
5. [Understanding Generated Files (`models/` & `logs/`)](#5-understanding-generated-files-models--logs)
6. [Command Reference Summary](#6-command-reference-summary)

---

## 1. Setup & Installation

Ensure you have **Python 3.10+** installed, then install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Interactive Manual Play (Gridworld)

You can play and test the mechanics of any level using your keyboard:

```bash
python -m gridworld.play --level 0
```

### Try Different Level Mechanics:
- `--level 0` : **Basic Apples** — Navigate around rocks to collect apples (+1 reward).
- `--level 1` : **Fire Hazards** — Fire tiles (`F`) along the bottom cause instant death on contact.
- `--level 2` : **Key & Chest** — Pick up yellow key (`K`) first to unlock the chest (`C`) for +2 reward.
- `--level 4` : **Monsters** — Red monsters (`M`) move with a 40% probability after every player move.

### Keyboard Controls:
| Key | Action |
| :--- | :--- |
| **Arrow Keys** (`↑` `↓` `←` `→`) | Move Agent |
| **R** | Reset current level |
| **ESC / Q** | Quit game window |

---

## 3. Task 1: Basic Q-Learning (Level 0)

Task 1 implements tabular **off-policy Q-Learning** with $\epsilon$-greedy exploration, linear decay, and random tie-breaking.

### A. Training Options

#### 1. Standard Fast Headless Training (Default):
Trains headlessly in the background (~1-2 seconds) using settings from `config.json`:
```bash
python -m gridworld.train --level 0 --agent qlearning
```

#### 2. Custom Episode Count:
Train for a specific number of episodes (e.g. 500 or 1000):
```bash
python -m gridworld.train --level 0 --agent qlearning --episodes 500
```

#### 3. Live Visual Training (Watch the AI learn in real-time):
Opens the Pygame window during training to visualize exploration and learning:
*(Tip: Keep episodes low, e.g. 50, since visual rendering is slower)*
```bash
python -m gridworld.train --level 0 --agent qlearning --episodes 50 --render
```

---

### B. Visual Evaluation Options

Evaluation loads the trained Q-table with exploration turned off ($\epsilon = 0.0$) to demonstrate the optimal greedy policy.

#### 1. Standard Evaluation (3 games at 6 FPS):
```bash
python -m gridworld.evaluate --level 0 --agent qlearning
```

#### 2. Slow-Motion Inspection (2 FPS):
Useful for observing every individual decision step-by-step:
```bash
python -m gridworld.evaluate --level 0 --agent qlearning --fps 2
```

#### 3. Fast Replay (5 games at 15 FPS):
```bash
python -m gridworld.evaluate --level 0 --agent qlearning --episodes 5 --fps 15
```

---

## 4. Task 2: SARSA & Hazard Comparison (Level 1)

Task 2 implements **on-policy SARSA** and compares its learning behavior with Q-Learning on a map with dangerous fire hazards.

### A. Training SARSA & Q-Learning on Level 1

#### 1. Train SARSA on Level 1 (1,000 episodes):
```bash
python -m gridworld.train --level 1 --agent sarsa --episodes 1000
```

#### 2. Train Q-Learning on Level 1 (for comparison):
```bash
python -m gridworld.train --level 1 --agent qlearning --episodes 1000
```

#### 3. Live Visual Training with SARSA:
```bash
python -m gridworld.train --level 1 --agent sarsa --episodes 50 --render
```

---

### B. Evaluating SARSA on Level 1

#### 1. Standard Evaluation:
```bash
python -m gridworld.evaluate --level 1 --agent sarsa --episodes 3 --fps 6
```

#### 2. Slow-Motion Evaluation:
```bash
python -m gridworld.evaluate --level 1 --agent sarsa --fps 2
```

---

### C. Automated Side-by-Side Comparison (Report Evidence)

Run this command to train both agents, trace their greedy trajectories, and generate a comparative learning curve chart:

```bash
python -m gridworld.compare --level 1 --episodes 1000
```

#### Optional Comparison Flags:
- `--episodes 1500` : Custom episode count.
- `--window 50` : Moving average smoothing window size for the plot.

```bash
python -m gridworld.compare --level 1 --episodes 1500 --window 100
```

---

## 5. Understanding Generated Files (`models/` & `logs/`)

Whenever you run training or comparison scripts, the outputs are automatically organized into `models/` and `logs/`:

```
GAI_Ass3/
├── models/
│   └── gridworld/
│       ├── level0_qlearning.pkl      <-- Task 1 Q-Learning model weights
│       ├── level1_qlearning.pkl      <-- Level 1 Q-Learning model weights
│       └── level1_sarsa.pkl          <-- Task 2 SARSA model weights
└── logs/
    └── gridworld/
        ├── level0_qlearning.png      <-- Task 1 training curve plot
        ├── level0_qlearning_rewards.csv
        ├── level1_qlearning.png
        ├── level1_qlearning_rewards.csv
        ├── level1_sarsa.png
        ├── level1_sarsa_rewards.csv
        └── level1_comparison_qlearning_vs_sarsa.png  <-- Task 2 comparison chart
```

### Detailed File Descriptions:

| File Type | Path / Format | Description & Usage |
| :--- | :--- | :--- |
| **Model Weights** | `models/gridworld/*.pkl` | **Serialized Q-Table:** Stores the learned dictionary of `(state, action) -> Q-value` pairs using Python's `pickle`. Loaded automatically by `evaluate.py` to replay greedy policies without retraining. Included in the final submission `.zip`. |
| **Training Curves** | `logs/gridworld/*.png` | **High-Resolution Learning Plot:** Shows raw episode rewards (light line) alongside a moving average (dark solid line) illustrating learning progression and stability over time. Directly insertable into your assignment PDF report. |
| **Raw Rewards CSV** | `logs/gridworld/*_rewards.csv` | **Raw Numerical Data:** Plain text CSV recording total reward obtained per episode (`episode_reward`). Useful if you wish to analyze training data in Pandas or plot custom graphs in Excel. |
| **Comparison Plot** | `logs/gridworld/*_comparison_*.png` | **Comparative Benchmark Chart:** Overlays Q-Learning vs SARSA rolling averages on the same figure, providing direct visual evidence for Task 2 and the written report. |

---

## 6. Command Reference Summary

| Purpose | Command |
| :--- | :--- |
| **Play Level 0 (Apples)** | `python -m gridworld.play --level 0` |
| **Play Level 1 (Fire)** | `python -m gridworld.play --level 1` |
| **Play Level 2 (Key/Chest)** | `python -m gridworld.play --level 2` |
| **Play Level 4 (Monsters)** | `python -m gridworld.play --level 4` |
| **Train Q-Learning (L0)** | `python -m gridworld.train --level 0 --agent qlearning --episodes 500` |
| **Evaluate Q-Learning (L0)** | `python -m gridworld.evaluate --level 0 --agent qlearning --episodes 3 --fps 6` |
| **Train SARSA (L1)** | `python -m gridworld.train --level 1 --agent sarsa --episodes 1000` |
| **Evaluate SARSA (L1)** | `python -m gridworld.evaluate --level 1 --agent sarsa --episodes 3 --fps 6` |
| **Run L1 Comparison** | `python -m gridworld.compare --level 1 --episodes 1000` |
