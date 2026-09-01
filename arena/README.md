# Arena package layout

This directory contains Part II of the assignment. Public module commands,
model locations, observation contracts, and gameplay behavior remain unchanged.

```text
arena/
|-- environment.py       Gymnasium environment and arena simulation
|-- entities.py          Player, enemy, projectile, and spawner data
|-- renderer.py          Pygame presentation and animation
|-- app.py               Visual Part II launcher
|-- play.py              Manual-play entry point
|-- train.py             Stable-Baselines3 DQN training
|-- evaluate.py          Shared policy evaluation
|-- evaluate_direct.py   Direct-control visual evaluation
|-- evaluate_rotation.py Rotation-control visual evaluation
|-- benchmark.py         Deterministic benchmark metrics
|-- build_evidence.py    Report-evidence generation
|-- tune.py              Hyperparameter experiments
|-- settings.py          Project paths and model metadata helpers
|-- config.json          Environment, reward, and training configuration
|-- legacy_api.py        Four-value Gym compatibility adapter
|-- wrappers.py          Stable-Baselines3 environment wrappers
`-- __main__.py          `python -m arena` entry point
```

Related project directories:

- `tests/part2_arena/`: Part II regression tests
- `models/arena/`: selected DQN policies and metadata
- `logs/arena/`: training, evaluation, and evidence artefacts
- `docs/part2/`: report notes, rubric map, and video plan
- `scripts/part2/`: Windows launcher helpers

Common commands (run from the repository root):

```powershell
python -m arena
python -m arena.play --control-style rotation
python -m arena.evaluate_rotation
python -m arena.evaluate_direct
python -m pytest tests/part2_arena -q
```

Keep simulation mechanics in `environment.py`, presentation in `renderer.py`,
and training/evidence orchestration in their existing command modules. Avoid
moving public modules unless compatibility shims are retained.
