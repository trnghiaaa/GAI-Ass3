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

## 3. Training & Evaluating Q-Learning (Part I - Task 1)

### Train Q-Learning Agent:
```bash
python -m gridworld.train --level 0 --agent qlearning --episodes 500
```
- Saved model weights are placed in `models/gridworld/level0_qlearning.pkl`
- Reward history and training plots are saved in `logs/gridworld/`

### Visually Evaluate Trained Agent:
```bash
python -m gridworld.evaluate --level 0 --agent qlearning --episodes 3 --fps 6
```
