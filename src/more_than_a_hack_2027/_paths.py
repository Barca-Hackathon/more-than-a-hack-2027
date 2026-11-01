"""Repository locations, and access to the organiser modules at the repository root.

The organiser code (`games.py`, `starter/`, `evaluation/`) is not a package, so it is reached the
same way `starter/*.py` reaches `games.py`: by putting its folders on sys.path. We never modify
those files -- `evaluation/possession_metrics.py` is the official metric and must stay byte-identical.

This package therefore has to run from a checkout of the repository (an editable install, which
is what `uv sync` gives you), not from a wheel.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, ".cache")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

_ORGANISER_DIRS = (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"))


def ensure_repo_on_path():
    """Make `games`, the starter modules and `possession_metrics` importable. Idempotent."""
    for folder in reversed(_ORGANISER_DIRS):
        if folder not in sys.path:
            sys.path.insert(0, folder)
