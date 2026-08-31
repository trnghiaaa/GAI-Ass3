"""Generate comparisons from saved same-environment evaluations, not anecdotes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['validation', 'holdout'], default='validation')
    args = parser.parse_args()
    root = Path('logs/arena/threat_experiment')
    files = sorted(root.glob(f'*_{args.split}.json'))
    records = [(p.stem, json.loads(p.read_text(encoding='utf-8'))) for p in files]
    if not records:
        raise SystemExit('No evaluations found')
    contracts = {(r['aggregate']['config_sha256'], r['aggregate']['seed_start'], r['aggregate']['episodes']) for _, r in records}
    if len(contracts) != 1:
        raise ValueError('Do not combine different configurations or seed sets')
    report = [f'# Threat-awareness experiment: {args.split}', '',
              'All rows use the same harder schema-6 arena and fixed seeds. The frozen',
              'baseline receives only its original 70 input features, with no action overrides.',
              'These are transfer-learning comparisons, not from-scratch training budgets.', '',
              '| Model | Phase 2+ | Mean phase | Boss kills | Frames alive | Damage/1000 | Contacts/1000 | Crowd % | Wall % | Boss hits / dodges |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, record in records:
        a = record['aggregate']
        report.append(f"| {name} | {100*a['phase_progression_rate']:.1f}% | {a['mean_phase']:.2f} | {a['mean_bosses_destroyed']:.2f} | {a['mean_simulation_steps']:.0f} | {a['damage_per_1000_frames']:.2f} | {a['contacts_per_1000_frames']:.2f} | {100*a['mean_crowd_fraction']:.1f} | {100*a['mean_wall_fraction']:.1f} | {a['mean_boss_skill_hits']:.2f} / {a['mean_boss_skills_dodged']:.2f} |")
    report += ['', 'Crowd/wall fractions sample every decision (four frames). Damage and contacts',
               'are normalized by actual simulation frames; longer runs can encounter more bosses.',
               'A dodge means a fully avoided barrage, not necessarily a deliberate evasive action.',
               'See per-episode JSON for deaths/timeouts, seeds, exact model hashes, and config hashes.']
    (root / f'{args.split}_comparison.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    rotation = [(n.replace('_validation','').replace('_holdout',''), r['aggregate']) for n,r in records if r['aggregate']['control_style']=='rotation']
    fig, axes = plt.subplots(2, 2, figsize=(12,8), layout='constrained')
    for ax, (key, label) in zip(axes.flat, [('mean_phase','Mean phase (higher better)'), ('damage_per_1000_frames','Hull damage / 1000 frames (lower better)'), ('contacts_per_1000_frames','Contacts / 1000 frames (lower better)'), ('mean_simulation_steps','Frames alive (higher better)')]):
        ax.barh([n for n,_ in rotation], [a[key] for _,a in rotation], color=['#607d8b' if 'baseline' in n else '#2196f3' for n,_ in rotation])
        ax.set_title(label)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle(f'Rotation DQN: matched {args.split}, harder schema-6 arena')
    fig.savefig(root / f'{args.split}_comparison.png', dpi=150)
    plt.close(fig)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
