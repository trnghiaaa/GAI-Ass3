# v20 phase-transition survival diagnostics

The frozen v20 Control Style 1 model was evaluated headlessly for 60 episodes
using seeds 62000 through 62059. Cooldown-aware action masking was enabled and
the optional safety shield was disabled. No reward, environment, training, or
model behaviour was changed.

The run exactly reproduced the authoritative headline results: Phase 2+ in
85.0% of episodes, 53.3% survival, 33.2% near-wall time, 74.6% danger time,
45.6% shot hit rate, and zero invalid shots.

## Phase-entry health

| Metric | Result |
| --- | ---: |
| Episodes entering Phase 2 | 51 / 60 |
| Mean health on first Phase-2 entry | 71.45 |
| Median health on first Phase-2 entry | 72.00 |
| Phase-2 entry health range | 2–100 |
| Phase-2 entries below 35 health | 8 / 51 (15.7%) |
| Episodes entering Phase 3 | 2 / 60 |
| Mean Phase-3 entry health | 44.00 |
| Median Phase-3 entry health | 44.00 |
| Phase-3 entry health values | 58, 30 |

Among episodes that entered Phase 2, eventual deaths had mean entry health
72.0, while time-limit survivors had mean entry health 70.9. The death rate was
50.0% for entrants below 35 health and 48.8% for entrants at or above 35.
Low entry health therefore does not explain the dominant failure pattern.

## Before/after Phase 2

The transition step is classified as pre-Phase-2 because the environment
resolves collision damage before advancing the phase. The comparison below is
restricted to the 51 episodes that entered Phase 2.

| Metric | Before Phase 2 | After Phase 2 |
| --- | ---: | ---: |
| Mean segment steps | 772.37 | 789.04 |
| Mean damage taken | 28.55 | 60.12 |
| Damage per 1,000 segment steps | 36.96 | 76.19 |
| Mean contact events | 2.04 | 4.22 |
| Contacts per 1,000 segment steps | 2.64 | 5.34 |
| Danger-step fraction | 70.99% | 75.21% |
| Mean crowd pressure | 0.1235 | 0.2179 |
| Mean episodic minimum enemy distance | 37.98 px | 21.48 px |

After Phase 2 begins, damage rate rises by approximately 2.06 times and contact
rate by approximately 2.02 times. Crowd pressure rises by approximately 76.5%,
and the mean episodic minimum enemy distance becomes approximately 43.4% closer.

## Death phase

| Death location | Count | Rate across all episodes |
| --- | ---: | ---: |
| Phase 1 | 3 | 5.0% |
| Phase 2 | 25 | 41.7% |
| Phase 3 or higher | 0 | 0.0% |

The mean number of steps survived after first entering Phase 2 was 789.0. The
two Phase-3 episodes both reached the time limit rather than dying.

## Diagnosis

**Pattern B dominates.** The agent usually reaches Phase 2 with reasonable
health, then encounters substantially higher crowd pressure, closer enemies,
and roughly double the damage/contact rate. A future v21 experiment, if
authorized, should therefore target post-progression low-health separation,
repeated contacts, and crowd escape. It should not add generic survival rewards
or alter phase, spawner, or cooldown-mask behaviour.

The complete per-episode and aggregate telemetry is stored in
`v20_phase_diagnostics_seed62000_60.json`.
