# Part I video demonstration plan

Target length: **3–4 minutes** of the group's maximum 10-minute video, leaving room for Part II and all members.

## Recording setup

1. Double-click `run_part1.bat`, or run `python main.py`, at 1280×800 or larger.
2. Keep `logs/gridworld/level1_qlearning_vs_sarsa_evidence.png` and `level6_intrinsic_comparison_evidence.png` ready to show briefly.
3. Use the saved models included in `models/gridworld`; do not retrain during the recording.

## Suggested sequence

### 0:00–0:35 — Unified environment

- Show the main menu and seven-stage campaign track.
- Enter Free Play Level 2.
- Collect the key, demonstrate that a locked chest gives no reward, then open it for +2 and collect an apple for +1.
- Point out that the Pygame HUD displays only environment reward.

### 0:35–1:25 — Learned Level 1 behavior

- Open AI Showcase, Level 1, Q-learning.
- Toggle the policy lens and single-step the 7-step route beside fire.
- Replay SARSA and show the 9-step upper lane.
- Briefly display the evidence figure: 10 paired seeds, 6.0 versus 0.1 mean hazard-adjacent steps.
- State clearly that epsilon is zero for the shown learned-policy replay; exact ties are still randomly broken as required.

### 1:25–2:15 — Stochastic monsters

- Open AI Showcase Level 5 with SARSA.
- Let several episodes/actions run so both monsters visibly sample movement.
- Pause and single-step to make the post-action monster phase clear.
- Explain that each monster independently has a 40% chance to move and either collision direction kills the agent.

### 2:15–3:00 — Intrinsic reward

- Open AI Showcase Level 6 with the intrinsic Q-learning model.
- Show its policy completing the sparse maze.
- Briefly show the Level 6 comparison figure.
- Say the exact formula aloud and explain: environment reward remains unchanged; the per-episode destination counter affects only the learning update.
- Report the paired result: first success 10.2 → 8.0 episodes on average, while both final greedy policies complete in 22 steps.

### 3:00–3:20 — Originality and evidence

- Return to the menu or pause on the Q inspector.
- Mention procedural art, campaign mode, policy lens, Q inspector, seeded evaluation, raw CSV/JSON logs, and automated tests.

## Acceptability checklist

- Pygame window is visible, animated, and interactive.
- At least one learned policy is visibly consistent rather than random.
- A monster moves after an agent action.
- Item rewards and key/chest sequence are visible.
- Saved submission models—not temporary models—are used.
- Presenter names the algorithm and level being shown.
- Each group member appears and presents at least one project part across the full video.
