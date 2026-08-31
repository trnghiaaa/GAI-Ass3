# Schema 7: scalable bosses, safer navigation, and final model selection

Status: complete. All final metrics below use the submitted schema-7 mechanics,
the same 30 deterministic seeds (`43000`–`43029`), and action repeat four.

## Gameplay and UI changes

- Combat XP now grows more slowly: ordinary enemies grant 6 XP and the
  cumulative threshold is `60 * (level - 1)^1.95`. Ship levels remain uncapped.
- Boss health uses a 6.2 multiplier on top of phase scaling. A separate visible
  shield starts at 30% of boss health, grows by 2.5 percentage points per boss,
  and caps at 45%.
- Each boss can summon at most three Rift Hunters at 75%, 50%, and 25% health.
  At most two may be active, summons have a seven-second cooldown, respect the
  global enemy cap, and wait during telegraphs or when the player is too close.
- Boss skill damage and late skill cadence scale gradually. Existing readable
  telegraphs and one-hit-per-barrage fairness remain.
- Manual and learned-policy playback now have a visible PAUSE/RESUME button and
  `P` shortcut. Pausing freezes the simulation and phase clock; it is UI state,
  not an extra RL action.

## Learning changes

The fixed 89-value observation remains numeric and pixel-free. Its threat-aware
suffix exposes a second enemy, closing speeds, local crowd pressure and escape,
wall/spawner clearance, boss shield and summon budget, and body-relative hazard
escape. The reward still contains all rubric-required progression and damage
terms. Small auditable potentials now reward approaching a safe surface-distance
band instead of touching a spawner, separating from surviving nearby enemies,
and moving out of an unchanged boss barrage. No scripted evasion or action veto
is used.

The new `safety_aware`, `safety_exploration`, and `safety_consolidation` profiles
vary learning rate, discount, and exploration schedule. Checkpoint selection
uses reward and progression plus normalized damage, contacts, crowd exposure,
enemy clearance, lifetime, boss dodges, hits, and boss kills.

## Final holdout result

| Policy | Reward | Mean/max phase | Progression | Damage/1k frames | Contacts/1k | Boss kills | Dodges / hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct before | 266.89 | 4.13 / 9 | 100% | 62.35 | 3.63 | 0.47 | 2.27 / 0.97 |
| **Direct final** | **510.30** | **5.70 / 12** | **100%** | **35.99** | **1.94** | **1.10** | **4.77 / 1.37** |
| **Rotation final** | **89.48** | **2.70 / 3** | **100%** | **87.07** | **5.01** | **0.00** | **1.27 / 0.60** |
| Rejected fresh rotation | 84.43 | 2.50 / 3 | 96.67% | 70.72 | 4.08 | 0.00 | 0.67 / 0.67 |

The direct transfer checkpoint was promoted because it improved progression,
boss completion, normalized damage, contacts, and dodges on untouched seeds.
The fresh rotation candidate was safer per frame but regressed progression and
boss-dodge reliability; the subsequent low-rate consolidation also failed its
selection gate. The proven rotation model was therefore retained. Rejected
candidates are preserved as honest tuning evidence, not claimed successes.

Final learned-vs-random results are 510.30 versus -14.80 reward and 100% versus
3.33% progression for direct, and 89.48 versus -37.83 reward and 100% versus 0%
progression for rotation.

Code and tests alone cannot guarantee a grade. The final report must remain at
most ten pages, include contribution details and the video link, and the video
must visibly demonstrate both saved models and at least one phase transition.
