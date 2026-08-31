# Threat-awareness experiment: validation

All rows use the same harder schema-6 arena and fixed seeds. The frozen
baseline receives only its original 70 input features, with no action overrides.
These are transfer-learning comparisons, not from-scratch training budgets.

| Model | Phase 2+ | Mean phase | Boss kills | Frames alive | Damage/1000 | Contacts/1000 | Crowd % | Wall % | Boss hits / dodges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_direct_validation | 100.0% | 4.73 | 0.83 | 2555 | 64.10 | 3.77 | 17.6 | 16.3 | 1.00 / 1.70 |
| baseline_rotation_validation | 100.0% | 2.87 | 0.07 | 1348 | 87.54 | 5.00 | 28.9 | 48.0 | 0.83 / 0.60 |
| direct_200k_validation | 100.0% | 5.00 | 0.90 | 2503 | 70.74 | 4.27 | 19.7 | 20.4 | 1.23 / 2.30 |
| direct_selected_validation | 100.0% | 5.70 | 1.20 | 2942 | 74.19 | 4.45 | 22.2 | 23.8 | 1.30 / 3.00 |
| rotation_100k_validation | 100.0% | 2.53 | 0.00 | 1237 | 103.90 | 6.22 | 32.1 | 57.2 | 0.70 / 0.73 |
| rotation_150k_validation | 100.0% | 2.70 | 0.03 | 1404 | 85.06 | 5.22 | 29.8 | 58.8 | 0.70 / 1.33 |
| rotation_200k_validation | 100.0% | 2.63 | 0.07 | 1288 | 98.12 | 6.29 | 31.8 | 58.6 | 0.53 / 0.77 |
| rotation_250k_validation | 100.0% | 2.70 | 0.07 | 1373 | 93.53 | 5.46 | 29.3 | 59.2 | 0.87 / 0.97 |
| rotation_50k_validation | 100.0% | 2.40 | 0.00 | 1120 | 107.71 | 7.17 | 39.6 | 61.0 | 0.37 / 0.57 |
| rotation_adapter_50k_validation | 100.0% | 2.90 | 0.00 | 1368 | 91.94 | 5.05 | 28.4 | 62.4 | 1.07 / 0.87 |
| rotation_adapter_selected_validation | 100.0% | 2.67 | 0.00 | 1266 | 94.67 | 5.50 | 30.1 | 56.1 | 0.70 / 0.93 |
| rotation_cadence2_ablation_validation | 96.7% | 2.80 | 0.00 | 1329 | 89.84 | 4.74 | 26.6 | 55.7 | 0.97 / 1.03 |
| rotation_masked_50k_validation | 93.3% | 2.17 | 0.00 | 947 | 129.40 | 9.22 | 44.0 | 56.0 | 0.17 / 0.17 |
| rotation_masked_selected_validation | 100.0% | 2.30 | 0.00 | 1041 | 113.29 | 7.18 | 36.6 | 59.7 | 0.40 / 0.40 |
| rotation_seed17100_selected_validation | 100.0% | 2.70 | 0.07 | 1373 | 93.53 | 5.46 | 29.3 | 59.2 | 0.87 / 0.97 |
| rotation_seed19100_200k_validation | 100.0% | 2.83 | 0.00 | 1583 | 80.57 | 4.13 | 26.2 | 62.2 | 1.13 / 1.37 |
| rotation_seed19100_selected_validation | 100.0% | 2.67 | 0.03 | 1283 | 100.12 | 5.74 | 32.4 | 58.6 | 0.83 / 0.67 |

Crowd/wall fractions sample every decision (four frames). Damage and contacts
are normalized by actual simulation frames; longer runs can encounter more bosses.
A dodge means a fully avoided barrage, not necessarily a deliberate evasive action.
See per-episode JSON for deaths/timeouts, seeds, exact model hashes, and config hashes.
