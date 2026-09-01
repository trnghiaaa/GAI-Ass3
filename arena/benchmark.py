"""Transparent compatibility alias for :mod:`arena.evaluation.benchmark`."""

import sys

from arena.evaluation import benchmark as _implementation


sys.modules[__name__] = _implementation
