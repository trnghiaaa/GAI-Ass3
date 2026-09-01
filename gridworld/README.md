# Gridworld package layout

This directory contains Part I of the assignment. The public module commands
and saved-model locations remain unchanged.

```text
gridworld/
|-- agents/             Tabular agent implementations and shared base class
|-- levels/             Level definitions and layouts
|-- environment.py      Gridworld mechanics, state encoding, and transitions
|-- renderer.py         Pygame rendering
|-- app.py              Interactive visual application
|-- train.py            Q-learning and SARSA training entry point
|-- evaluate.py         Visual trained-policy evaluation
|-- compare.py          Algorithm and intrinsic-reward comparisons
|-- optimize.py         Multi-seed hyperparameter selection
|-- benchmark.py        Saved-policy benchmark generation
|-- build_evidence.py   Part I report-evidence orchestration
|-- config.json         Training and environment configuration
|-- __main__.py         `python -m gridworld` entry point
`-- __init__.py         Package marker
```

Related project directories:

- `tests/part1_gridworld/`: Gridworld regression tests
- `models/gridworld/`: trained tabular policies
- `logs/gridworld/`: curves, comparisons, and evaluation evidence
- `docs/PART1_RUBRIC_EVIDENCE.md`: report and rubric evidence guide

Common commands (run from the repository root):

```powershell
python -m gridworld
python -m gridworld.train --level 0 --agent qlearning
python -m gridworld.train --level 1 --agent sarsa
python -m gridworld.evaluate --level 0 --agent qlearning
python -m pytest tests/part1_gridworld -q
```

Keep reusable learning logic in `agents/`, map data in `levels/`, and visual
presentation in `renderer.py` or `app.py`. Training/evidence scripts should use
the environment and agents through their existing public imports.
