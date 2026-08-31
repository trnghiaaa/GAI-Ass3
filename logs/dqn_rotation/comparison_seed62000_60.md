# Control Style 1 fixed-seed comparison

All models were evaluated for 60 episodes beginning at seed 62000. Rotation
cooldown masking was enabled and the optional safety shield was disabled.

| Metric | v14 | v18 | v19 selected |
|---|---:|---:|---:|
| Mean reward | -13.41 | -14.15 | **9.54** |
| Survival rate | 46.7% | **51.7%** | 50.0% |
| Phase 2+ rate | 18.3% | 18.3% | **48.3%** |
| Maximum phase | 2 | 2 | 2 |
| Mean spawners destroyed | 0.85 | 0.75 | **1.62** |
| Mean enemies destroyed | 4.77 | **7.03** | 4.60 |
| Shot hit rate | 37.1% | 38.3% | **53.9%** |
| Raw cooldown SHOOT rate | **19.0%** | 34.0% | 21.0% |
| Cooldown replacement rate | **19.0%** | 34.0% | 21.0% |
| Danger-step fraction | 79.3% | 83.2% | **78.1%** |
| Danger NOOP fraction | **16.0%** | 21.0% | 30.0% |
| Danger THRUST fraction | **52.0%** | 40.0% | 32.0% |
| Near-wall fraction | **23.0%** | **23.0%** | 36.0% |
| Mean damage taken | 87.50 | 85.63 | **82.37** |

Selection prioritizes phase progression and spawner completion, then survival
and combat quality. v19 clearly improves the primary objective while retaining
comparable survival and combat. Its wall occupancy and danger-NOOP rate remain
documented targets for future improvement.

Raw JSON:

- `comparison_v14_seed62000_60.json`
- `comparison_v18_seed62000_60.json`
- `comparison_v19_seed62000_60.json`
