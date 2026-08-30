# Part II video demonstration plan

Recommended Part II segment: approximately 3.5–4 minutes of the ten-minute team
video.

1. Open `python part2.py` and briefly show the four-card launcher. Explain that
   manual play and both saved DQN policies use the same environment.
2. Open **Watch Direct DQN**. Point out continuous movement, the green reticle
   for close-range target assist (and heading-based fire outside lock range),
   enemy spawning/steering, health bars, projectile collisions, and the
   deterministic trained-policy footer. Continue until the phase and weapon
   upgrade banners appear; briefly point out that XP is separate from RL reward.
   In manual mode, show one three-card draft and choose an upgrade with the mouse.
   Press `Tab` once to show the composed build statistics, then close it. When a
   phase clears, point out the fresh 60-second phase clock and clean battlefield.
3. Return to the launcher and open **Watch Rotation DQN**. Explain that this
   agent must learn heading, signed turn direction, thrust, and forward shooting.
   The launcher's default seed 42 reaches phase 4 and visibly defeats the
   phase-3 boss rift in the submitted model.
   Pause with `P`, single-step with `.`, and identify the target reticle so the
   behavior is visibly a learned policy rather than keyboard control.
4. Briefly display `logs/arena/control_style_comparison.png` and the TensorBoard
   scalar page. State the final timesteps, network, seeded benchmark episode
   count, mean reward, phase progression, and the observed control-style tradeoff
   using values from the generated JSON.

Recording checklist:

- Both model ZIPs shown in `models/arena` are the models used in playback.
- At least one phase progression is visible.
- At least one XP level-up or upgraded weapon is visible.
- Show one support choice and, if time permits, the phase-3 boss rift.
- Enemies spawn and navigate, shots travel, collisions reduce health, and the
  player can take damage.
- Both control schemes receive a short learned-agent clip.
- Do not claim 100% success unless the final held-out JSON actually reports it.
