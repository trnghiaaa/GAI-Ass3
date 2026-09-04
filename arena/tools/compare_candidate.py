"""Evaluate a candidate or a frozen prefix baseline on the current mechanics.

The optional prefix adapter supplies exactly the original observation fields;
it does not steer, mask, or otherwise change the baseline's chosen actions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from arena.learning.cooldown import load_dqn

from arena.evaluation.benchmark import evaluate_model, write_benchmark
from arena.core.environment import ENVIRONMENT_SCHEMA_VERSION, OBSERVATION_NAMES


class PrefixPolicy:
    def __init__(self, model):
        self.model = model
        self.size = model.observation_space.shape[0]

    def predict(self, observation, deterministic=True):
        return self.model.predict(observation[..., :self.size], deterministic=deterministic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--control-style', choices=['rotation', 'direct'], required=True)
    parser.add_argument('--seed', type=int, default=35000)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-baseline', action='store_true')
    parser.add_argument('--action-repeat', type=int, default=None,
                        help='Explicit control-cadence ablation; defaults to saved metadata or 4')
    parser.add_argument('--start-phase', type=int, default=None,
                        help='Optional fixed initial phase for a focused benchmark')
    parser.add_argument(
        '--sentry-start-phase',
        type=int,
        default=None,
        help='Start at a bootstrapped boss sentry wave for a missile-dodge benchmark',
    )
    args = parser.parse_args()
    if args.start_phase is not None and args.sentry_start_phase is not None:
        parser.error('--start-phase and --sentry-start-phase are mutually exclusive')
    if args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new output name; existing evaluations are protected')
    torch.set_num_threads(1)
    model = load_dqn(str(args.model), device='cpu')
    size = model.observation_space.shape[0]
    meta_path = args.model.with_suffix('.metadata.json')
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if metadata['control_style'] != args.control_style:
            raise ValueError('Wrong control style')
        if metadata['observation_names'] != list(OBSERVATION_NAMES[:size]):
            raise ValueError('Observation semantics are incompatible')
    if size != len(OBSERVATION_NAMES) and not args.prefix_baseline:
        raise ValueError('Explicit --prefix-baseline required for a smaller frozen model')
    policy = PrefixPolicy(model) if args.prefix_baseline else model
    repeat = args.action_repeat if args.action_repeat is not None else int(metadata.get('action_repeat', 4))
    reset_options = (
        None if args.start_phase is None else {"start_phase": args.start_phase}
    )
    rows, aggregate = evaluate_model(
        policy,
        args.control_style,
        episodes=args.episodes,
        seed=args.seed,
        action_repeat=repeat,
        reset_options=reset_options,
        sentry_start_phase=args.sentry_start_phase,
    )
    aggregate['environment_schema'] = ENVIRONMENT_SCHEMA_VERSION
    aggregate['model'] = str(args.model)
    aggregate['model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    aggregate['config_sha256'] = hashlib.sha256(Path('arena/config.json').read_bytes()).hexdigest()
    aggregate['prefix_baseline'] = args.prefix_baseline
    aggregate['cooldown_mask'] = bool(getattr(model, 'cooldown_mask_enabled', False))
    write_benchmark(rows, aggregate, args.output)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
