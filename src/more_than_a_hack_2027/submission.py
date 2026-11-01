"""Write a submission and check it with exactly the official validation."""

import os

import numpy as np

from ._paths import ensure_repo_on_path

ensure_repo_on_path()
import baseline as starter_baseline  # noqa: E402  starter/baseline.py
import possession_metrics as pm  # noqa: E402


def check_labels(labels, ctx):
    """Raise unless labels is a full-length H/A/D/X array with no X on a graded frame.

    X on a graded frame is never better than writing a label: spellF1 forward-fills it with the
    previous label anyway, and macroF1 counts it as a miss (EVALUATION.md).
    """
    labels = np.asarray(labels)
    if len(labels) != ctx.n_frames:
        raise ValueError("expected {} labels, got {}".format(ctx.n_frames, len(labels)))
    bad = sorted(set(np.unique(labels).tolist()) - set(pm.LABELS))
    if bad:
        raise ValueError("labels outside {}: {}".format(pm.LABELS, bad))
    abstained = int(np.sum((labels == "X") & (ctx.half_of_frame > 0)))
    if abstained:
        raise ValueError("{} graded frames are X -- write a label instead".format(abstained))


def write(path, labels, ctx):
    """Check, write and re-validate through the official reader. Returns a summary dict."""
    check_labels(labels, ctx)
    starter_baseline.write_submission(path, labels)

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, ctx.n_frames)
    if reason:
        raise ValueError("{} failed official validation: {}".format(path, reason))

    codes = pm.label_codes(table)
    runs = [(first, last + 1) for _, first, last in ctx.halves]
    spells = pm.possession_spells(codes, runs)
    lengths = [stop - start for start, stop, _ in spells]
    return {
        "path": os.path.abspath(path),
        "counts": {name: int(np.sum(codes == i)) for i, name in enumerate(pm.LABELS)},
        "n_possessions": len(spells),
        "mean_possession_s": float(np.mean(lengths)) / ctx.fps if lengths else 0.0,
    }
