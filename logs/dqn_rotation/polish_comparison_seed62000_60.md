# Final Control Style 1 polish comparison

Both policies were evaluated for 60 episodes beginning at seed 62000 with
cooldown masking enabled and the optional safety shield disabled.

| Metric | v19 baseline | v20 final/default |
| --- | ---: | ---: |
| Mean reward | 25.22 | 65.72 |
| Phase 2+ rate | 48.3% | 85.0% |
| Survival rate | 50.0% | 53.3% |
| Maximum phase reached | 2 | 3 |
| Mean spawners destroyed | 1.62 | 2.63 |
| Mean enemies destroyed | 4.60 | 4.30 |
| Near-wall time | 35.8% | 33.2% |
| Danger time | 78.1% | 74.6% |
| Danger NOOP | 29.6% | 16.9% |
| Danger THRUST | 32.1% | 36.6% |
| Mean crowd pressure | 0.164 | 0.177 |
| Mean damage taken | 82.37 | 86.57 |
| Mean contact events | 5.78 | 6.12 |
| Shot hit rate | 53.9% | 45.6% |
| Mean invalid shots | 0.0 | 0.0 |
| Cooldown replacement rate | 21.4% | 37.5% |

## Selection decision

The v20 policy is promoted as the final automatic Control Style 1 model. It
improves the assignment's primary phase-progression result by 36.7 percentage
points while also slightly improving survival and reducing both near-wall and
danger exposure. It did not reach the aspirational near-wall target of less
than 30%, and its shot accuracy declined, but neither trade-off outweighs the
large objective improvement. Invalid shots remained zero. The final model is
stored at `models/dqn_rotation/control_style_1_dqn_polished/final_model.zip`.

The v19 policy is frozen as the baseline and backup at
`models/dqn_rotation/control_style_1_dqn_progression/final_model.zip`.

No further Control Style 1 training is planned. The underlying evaluation data
is stored in `polish_comparison_v19_seed62000_60.json` and
`polish_comparison_v20_seed62000_60.json`.
