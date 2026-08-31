# Part II upgrade-clarity and boss-balance audit

This is the final schema-7 balance record after the safety experiment. All
accepted policy figures use 30 deterministic episodes beginning at seed 53000
with action repeat four.

## Player-facing changes

- Draft cards use fitted two-line names and show `NEW` or the exact tier change.
- Each card separates the installed value from the result after selection.
- Wingman cards show Core tier, mastery tier, and live drone count before and
  after the choice. The confirmation banner repeats the applied result.
- Ordinary phases retain a 60-second limit. Boss 1 receives 90 seconds, an 18%
  shield, one health-gated Hunter, a seven-enemy cap, and slower/weaker minions.
- Bosses from Phase 6 receive 70 seconds and full scaling. Their finite Hunter
  budgets grow 2, 3, then 4, with no more than two active simultaneously.
- Non-boss Rift Hunter probability is 24% through Phase 5, then rises by six
  percentage points per phase to a 66% cap.
- Rotation shots receive a shared seven-degree correction only when already
  nearly aligned. The required rotate/thrust/shoot actions remain unchanged.

These rules are identical in manual play, headless training, and evaluation.
There is no hidden AI-only health, damage, timer, or boss modifier.

## Accepted final policies

| Metric | Direct | Rotation |
|---|---:|---:|
| Mean reward | 521.93 | 106.24 |
| Mean / maximum phase | 6.10 / 11 | 2.83 / 4 |
| Phase progression | 100% | 100% |
| Mean boss kills | 1.30 | 0.07 |
| Boss-skill dodges / hits | 3.23 / 0.97 | 1.60 / 1.67 |
| Damage per 1,000 frames | 34.96 | 77.00 |
| Contacts per 1,000 frames | 2.08 | 3.75 |

The direct policy clearly benefits from the readable Boss 1 opening and remains
the strongest video-demonstration policy. Rotation is deliberately retained as
the harder control comparison rather than made artificially equivalent.

## Rejected rotation transfer

`balance_rotation_600k_s27100` continued the rotation agent for 600,000
decisions using the safety-aware profile. Its internal 20-episode selection
holdout reached 25% boss clears, mean Phase 3.05, and maximum Phase 5. On the
common final 30-seed holdout, however, it cleared Boss 1 in 2/30 episodes—the
same as the retained model—while mean reward (100.24) and mean phase (2.77) were
lower. It was therefore rejected and the existing `dqn_rotation.zip` retained.

Raw accepted and candidate CSV/JSON files are under
`logs/arena/balance_experiment/`. The final model metadata contains the accepted
benchmark plus model/config hashes.
