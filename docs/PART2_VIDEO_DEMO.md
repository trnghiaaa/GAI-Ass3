# Part II video demonstration plan

Aim for 3–4 minutes of the ten-minute group video.

1. Run `python main.py` and show the four-card launcher. Explain that manual
   play and both saved DQN policies use the same arena environment.
2. Open **Watch Direct DQN**. Show continuous movement/fire, spawning and
   steering enemies, health bars, collisions, an XP draft, and a phase
   transition. Use `P` to pause and `Tab` for the build panel.
3. Continue to a Phase-3 boss (or use a deterministic replay that reaches it).
   Point out the named warning/telegraph, multiple boss lanes, the policy
   moving out of danger, then the shield-break movement and low-health Aegis
   intermission. Show that fire redirects to sentries while the boss displays
   immunity, and that red missiles warn, guide briefly, then commit to a
   dodgeable path while cyan shots are friendly. State that the direct
   model was trained with real Phase-3 resets, not with a scripted dodge.
4. Return to the launcher and show **Watch Rotation DQN** briefly. Explain the
   harder coupled thrust/turn/forward-shoot action set and keep the comparison
   honest: it is a trained policy but is weaker against an immediate boss than
   direct control.
5. Show `logs/arena/evidence/direct_schema10_balanced_v2_boss.json` and the
   TensorBoard scalar page. Quote final values only from the JSON files.

Checklist:

- Both submitted model ZIPs are visible in `models/arena`.
- Enemies spawn and steer; shots, health and collisions are visible.
- At least one rift-clear phase transition and boss telegraph are shown.
- Show a pause and one manual upgrade choice.
- Show a short learned-policy clip for **both** control sets.
- Do not claim a perfect boss clear rate or equal performance between the
  control schemes; use the recorded metrics.
