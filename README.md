# Reinforcement Learning & Deep RL Agents (Assignment 3)

This repository contains the complete implementation for **Assignment 3: Reinforcement Learning and Deep RL Agents**, covering both classical value-based methods in a Pygame Gridworld (Part I) and deep reinforcement learning in an Action Arena (Part II).

---

## Table of Contents
1. [Setup & Installation](#1-setup--installation)
2. [Interactive Manual Play (Gridworld)](#2-interactive-manual-play-gridworld)
3. [Task 1: Basic Q-Learning (Level 0)](#3-task-1-basic-q-learning-level-0)
4. [Task 2: SARSA & Hazard Comparison (Level 1)](#4-task-2-sarsa--hazard-comparison-level-1)
5. [Task 3: Key & Chest Mechanics (Levels 2 & 3)](#5-task-3-key--chest-mechanics-levels-2--3)
6. [Understanding Generated Files (`models/` & `logs/`)](#6-understanding-generated-files-models--logs)
7. [Command Reference Summary](#7-command-reference-summary)

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
- `--level 2` : **Key & Chest (Simple)** — Pick up yellow key (`K`) first to unlock chest (`C`) for +2 reward.
- `--level 3` : **Key & Chest (Labyrinth)** — Navigate complex rock corridors to get key, chest, and apples.
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

### Training & Evaluation Commands:
```bash
# Train Q-Learning on Level 0 (500 episodes)
python -m gridworld.train --level 0 --agent qlearning --episodes 500

# Visual Evaluation in Pygame (3 games at 6 FPS)
python -m gridworld.evaluate --level 0 --agent qlearning --episodes 3 --fps 6
```

---

## 4. Task 2: SARSA & Hazard Comparison (Level 1)

Task 2 implements **on-policy SARSA** and compares its learning behavior with Q-Learning on a map with dangerous fire hazards.

### Training & Comparison Commands:
```bash
# Train SARSA on Level 1 (1000 episodes)
python -m gridworld.train --level 1 --agent sarsa --episodes 1000

# Visually Evaluate SARSA on Level 1
python -m gridworld.evaluate --level 1 --agent sarsa --episodes 3 --fps 6

# Automated Side-by-Side Comparison (generates joint learning curve plot)
python -m gridworld.compare --level 1 --episodes 1000
```

---

## 5. Task 3: Key & Chest Mechanics (Levels 2 & 3)

Task 3 extends Q-Learning and SARSA to solve multi-step planning tasks where the agent must collect the key (`K`) before unlocking the chest (`C` for +2 reward) alongside collecting all apples (`A`).

### A. Level 2 (Simple Open Layout)
```bash
# Train Q-Learning on Level 2 (1000 episodes)
python -m gridworld.train --level 2 --agent qlearning --episodes 1000

# Train SARSA on Level 2 (1000 episodes)
python -m gridworld.train --level 2 --agent sarsa --episodes 1000

# Visually Evaluate Level 2 Agents:
python -m gridworld.evaluate --level 2 --agent qlearning --episodes 3 --fps 6
python -m gridworld.evaluate --level 2 --agent sarsa --episodes 3 --fps 6
```

### B. Level 3 (Complex Rock Corridors)
```bash
# Train Q-Learning on Level 3 (1500 episodes)
python -m gridworld.train --level 3 --agent qlearning --episodes 1500

# Train SARSA on Level 3 (1500 episodes)
python -m gridworld.train --level 3 --agent sarsa --episodes 1500

# Visually Evaluate Level 3 Agents:
python -m gridworld.evaluate --level 3 --agent qlearning --episodes 3 --fps 6
python -m gridworld.evaluate --level 3 --agent sarsa --episodes 3 --fps 6
```

---

## 6. Understanding Generated Files (`models/` & `logs/`)

```
GAI_Ass3/
├── models/
│   └── gridworld/
│       ├── level0_qlearning.pkl
│       ├── level1_qlearning.pkl
│       ├── level1_sarsa.pkl
│       ├── level2_qlearning.pkl
│       ├── level2_sarsa.pkl
│       ├── level3_qlearning.pkl
│       └── level3_sarsa.pkl
└── logs/
    └── gridworld/
        ├── level0_qlearning.png
        ├── level1_comparison_qlearning_vs_sarsa.png
        ├── level2_qlearning.png
        ├── level2_sarsa.png
        ├── level3_qlearning.png
        └── level3_sarsa.png
```

---

## 7. Command Reference Summary

| Level & Task | Algorithm | Train Command | Evaluate Command |
| :--- | :--- | :--- | :--- |
| **Level 0 (Apples)** | Q-Learning | `python -m gridworld.train --level 0 --agent qlearning --episodes 500` | `python -m gridworld.evaluate --level 0 --agent qlearning` |
| **Level 1 (Fire)** | SARSA | `python -m gridworld.train --level 1 --agent sarsa --episodes 1000` | `python -m gridworld.evaluate --level 1 --agent sarsa` |
| **Level 1 (Fire)** | Comparison | `python -m gridworld.compare --level 1 --episodes 1000` | N/A |
| **Level 2 (Key/Chest)** | Q-Learning | `python -m gridworld.train --level 2 --agent qlearning --episodes 1000` | `python -m gridworld.evaluate --level 2 --agent qlearning` |
| **Level 2 (Key/Chest)** | SARSA | `python -m gridworld.train --level 2 --agent sarsa --episodes 1000` | `python -m gridworld.evaluate --level 2 --agent sarsa` |
| **Level 3 (Corridors)** | Q-Learning | `python -m gridworld.train --level 3 --agent qlearning --episodes 1500` | `python -m gridworld.evaluate --level 3 --agent qlearning` |
| **Level 3 (Corridors)** | SARSA | `python -m gridworld.train --level 3 --agent sarsa --episodes 1500` | `python -m gridworld.evaluate --level 3 --agent sarsa` |
