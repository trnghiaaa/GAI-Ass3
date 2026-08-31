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
   Hold a movement/rotation key with `Space` to demonstrate simultaneous manual
   steering and fire. Press `Tab` once to show the composed build statistics,
   permanent drone tiers, and support systems, then close it. When a
   phase clears, point out the animated deployment banner, refreshed phase
   clock, and clean battlefield. Show a draft card's current/next values and the
   result banner, then show the level-up ring animation.
3. Return to the launcher and open **Watch Rotation DQN**. Explain that this
   agent must learn heading, signed turn direction, thrust, and forward shooting.
   The launcher's disclosed deterministic demo seed 53006 reaches phase 4 and
   visibly defeats the phase-3 boss rift in the submitted model. Point out the
   special boss-phase warning and named multi-lane countdown, then show the policy
   repositioning before a cross/diagonal/trident/nova barrage.
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
- Show one support choice, one Rift Hunter if it appears, and the phase-3 boss
  rift with a clearly telegraphed multi-lane special attack.
- Enemies spawn and navigate, shots travel, collisions reduce health, and the
  player can take damage.
- Both control schemes receive a short learned-agent clip.
- Do not claim 100% success unless the final held-out JSON actually reports it.
