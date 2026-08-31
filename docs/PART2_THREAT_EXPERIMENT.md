# Schema 6: harder progression and threat-aware DQN

Historical experiment: superseded by `PART2_SAFETY_EXPERIMENT.md`. Do not use
the metrics below as final schema-7 results.

Status: complete. Implementation, independent candidate training, matched
validation, one untouched holdout confirmation, model promotion, and regression
testing are complete. The selected rotation checkpoint is the default model;
the direct candidate was rejected on holdout and its proven frozen-prefix
baseline remains the default direct model.

Additional candidates: a second standard rotation seed (19100, 300k decisions)
and a cooldown-consistent rotation variant (20100, 200k decisions). The latter
uses `--cooldown-mask`; the other settings/rewards are unchanged. Masking excludes
unavailable SHOOT decisions and next-state Bellman targets, never selects a
hand-coded steering direction, and is preserved when loading checkpoints.
The four-frame action repeat still holds a ready SHOOT command across its
following cooldown frames; this is not a claim of zero cooldown frames.
Standard and masked candidates are explicitly distinguished in evaluation JSON.
One final conservative rotation candidate trains only appended observation
connections (`--new-inputs-only`, seed 21100, 100k decisions). The original
input columns, hidden/output weights and biases remain fixed; a gradient-mask
regression test verifies this. This tests whether broad network fine-tuning is
forgetting useful original behavior. It uses the same rewards and mechanics.

## Hypothesis and constraints

The previous 70-feature rotation policy could target bosses but lacked explicit
local-crowd, second-enemy, closing-speed and wall-clearance features. This
experiment appends those observations, exposes finite boss defenses, and removes
the incentive to keep approaching a spawner inside 160px center distance.
The five rotation actions and six direct actions are unchanged. There is no
automatic evasion controller, extra action, demonstration policy, or safety veto.
Manual simultaneous firing and the existing heuristic build choices remain
separate from learned DQN control and must be described as such in the report.

## Gameplay changes

- Cumulative XP threshold: `55 * (level - 1)^1.85`, formerly exponent 1.72.
  The first upgrade still costs 55 XP. Kill/objective XP amounts are unchanged.
- Boss health multiplier: 5.2 rather than 4.5, on top of phase health scaling.
- Bosses start with a separate, visible shield worth 20% of maximum health.
  Damage depletes shield before health; excess damage carries through. Shields
  do not regenerate, avoiding an unlimited shield-damage reward farm. Existing
  telegraph-channel damage reduction remains. Shield damage is counted once in
  the existing spawner-damage reward and telemetry.
- Each boss can summon at most two minibosses, unlocked at 70% and 35% health,
  separated by at least eight seconds. The global enemy cap and two-active-
  miniboss cap still apply. Summons wait during telegraphs or while the player
  is within 180px of the boss. A delayed threshold never grants extra summons.
- Existing difficulty caps, phase deadlines, phase cleanup, seeded drafts,
  boss telegraphs, and one-hit-per-barrage fairness remain.
- A late-phase bug in ordinary spawning was fixed: the configured absolute
  enemy cap (32) is now enforced, not merely used to normalize observations.
  The first transfer runs were already running when this correction was made;
  their in-memory training code retains the old cap behavior after Phase 15.
  Final candidate comparisons and playback use the corrected cap. The frozen
  validation baselines never exceeded Phase 15, so their recorded trajectories
  are unaffected. This distinction matters for exact training reproduction.

## Learning changes

- 89 fixed-size float32 features; original 70 input positions are preserved.
- Added inputs: second enemy, relative closing speeds, local crowd count and
  pressure, escape vector, wall/spawner clearance, boss shield/summon state,
  and body-relative boss-hazard escape direction.
- Small signed crowd-pressure reduction shaping, measured only for an unchanged
  enemy set. A very small penalty applies above pressure 0.5. No reward is
  granted for choosing THRUST/ROTATE or any other specific action.
- Existing hazard-risk reduction coefficient: 0.55 -> 0.8. Spawner approach
  potential becomes flat inside 160px. Enemy/spawner/phase/death rewards remain
  unchanged. Existing survival reward is not increased.
- Each transfer run: SB3 DQN, [256,256] MLP, 300,000 additional decisions,
  learning rate 0.00005, exploration 0.25 -> 0.04 over the first half, repeat 4.
  Replay and optimizer are fresh; both online/target network prefix weights
  are copied and appended input weights start at zero (unit-tested).
- This is transfer from existing schema-5 models, not 300k from scratch.
  The supplied PDF lists 100k-600k as a feasibility guide, not a strict maximum.

## Reproduction (PowerShell, repository root)

```powershell
$env:MPLCONFIGDIR = "$PWD/logs/.matplotlib"
.\.venv\Scripts\python.exe -m arena.train --control-style rotation --timesteps 300000 --profile threat_aware --seed 17100 --run-name threat_rotation_300k_s17100 --init-model models/arena/baseline_schema5/dqn_rotation.zip --benchmark-episodes 20
.\.venv\Scripts\python.exe -m arena.train --control-style direct --timesteps 300000 --profile threat_aware --seed 18100 --run-name threat_direct_300k_s18100 --init-model models/arena/baseline_schema5/dqn_direct.zip --benchmark-episodes 20
```

Use new run names to repeat; existing artifacts are never overwritten by train.
TensorBoard logs: `logs/arena/tensorboard/threat_*`.

## Evaluation protocol

All comparisons use the same schema-6 mechanics. Frozen schema-5 baselines
receive the original 70-feature prefix only; their actions are unchanged.
Their old scores in the easier environment are not a valid comparison.

1. Training callback and final/best checkpoint comparison use training-specific
   seeds. These are model-selection results, not an untouched test set.
2. Matched validation: 30 episodes, seeds 35000-35029.
3. Select using progression, boss-hit exposure, contacts/damage per 1000 frames,
   crowd/wall fractions, enemy clearance, and episode length. Do not reward
   passive phase timeouts as successful play.
4. Final confirmation: 30 different seeds, 36000-36029. Do not tune on them.

`arena.compare_candidate` writes per-episode CSV/JSON plus model/config hashes.
Crowd/wall/clearance metrics sample decision endpoints (four frames apart), not
every rendered frame. Contact counts include blocked contact attacks; damage
counts only actual hull damage. Boss dodge counts include any fully avoided
barrage, not necessarily an active evasive maneuver. Survival is reaching a
phase/safety time limit alive, not winning or proof of useful progression.

## Selection result

The rotation checkpoint at 200k decisions from `threat_rotation_300k_s19100`
was selected on validation seeds 35000-35029 and confirmed once on the unseen
holdout seeds 36000-36029. It is the default `models/arena/dqn_rotation.zip`.
On holdout, relative to the frozen prefix baseline, it lowered damage from
104.91 to 81.64 per 1,000 frames, contacts from 5.77 to 4.27, and increased
mean lifetime from 1,214 to 1,529 frames. Phase-2 and Phase-3 damage were
also lower (96.53 to 70.92 and 151.09 to 110.00). Boss dodges increased from
20 to 42 while boss hits increased from 23 to 30 because the candidate spent
more time in boss encounters. Near-wall samples increased from 54.3% to
65.1%, so this is a crowd/contact improvement, not a claim that every movement
measure improved.

The schema-6 direct training candidate progressed farther on validation but
lost lifetime/safety on holdout. It was rejected. The default direct model is
therefore a schema-6-compatible frozen 70-feature prefix of the original
trained direct DQN. It preserves the better holdout behavior and remains a
separate saved DQN model; the rejected schema-6 candidate and its evidence are
preserved under `models/arena/threat_direct_300k_s18100.zip` and `logs/arena`.

No changes were made after reading the holdout results. The final controls can
be watched with `python -m arena.evaluate_rotation` and
`python -m arena.evaluate_direct`.

## Rubric and limitations

Native Gymnasium plus legacy four-value API, continuous animated Pygame,
both exact action sets, fixed numeric observations, separate SB3 models,
TensorBoard, configurable rewards, and visual evaluators remain required.
The observation summarizes a partially observed scene; it does not expose all
projectile trajectories and should not be described as a proof of Markov state.
Highest marks also require the student's <=10-page report, contributions, and
<=10-minute video with both trained control sets. Code/tests alone do not
guarantee a grade. Use measured results, including failures, in that report.
