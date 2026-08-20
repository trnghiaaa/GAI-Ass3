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

## 3. Training & Evaluating Agents (Part I)

### Task 1: Q-Learning (Level 0)
```bash
# Train Q-Learning on Level 0
python -m gridworld.train --level 0 --agent qlearning --episodes 500

# Visual Evaluation
python -m gridworld.evaluate --level 0 --agent qlearning --episodes 3 --fps 6
```

### Task 2: SARSA vs Q-Learning Comparison (Level 1 - Fire Corridor)
```bash
# Train SARSA on Level 1
python -m gridworld.train --level 1 --agent sarsa --episodes 1000

# Visually Evaluate SARSA
python -m gridworld.evaluate --level 1 --agent sarsa --episodes 3 --fps 6

# Run full side-by-side comparison with comparative plot generation:
python -m gridworld.compare --level 1 --episodes 1000
```
- Outputs comparison plot to `logs/gridworld/level1_comparison_qlearning_vs_sarsa.png`.
