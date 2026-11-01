"""The ablation table's bookkeeping: rows, feature groups and the paired bootstrap."""

import numpy as np

from more_than_a_hack_2027 import ablation, control, pipeline
from more_than_a_hack_2027._paths import ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402


def labels_with_spells(n, spells):
    codes = np.full(n, pm.D, dtype=np.int8)
    for start, stop, code in spells:
        codes[start:stop] = code
    return codes


def test_identity_draw_is_the_official_spell_score():
    gt = labels_with_spells(1000, [(10, 200, pm.H), (200, 260, pm.A), (300, 700, pm.H),
                                   (720, 760, pm.A), (760, 990, pm.H)])
    pred = labels_with_spells(1000, [(15, 190, pm.H), (190, 280, pm.A), (300, 980, pm.H)])
    flags, n_pred = ablation._flags(gt, np.array(["HADX"[c] for c in pred]))
    identity = np.arange(flags.shape[1])[None, :]
    official = pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]
    assert np.isclose(ablation._spell_f1_draws(flags, n_pred, identity)[0], official)
    assert np.isclose(ablation._spell_f1_draws(*ablation._flags(gt, np.array(
        ["HADX"[c] for c in gt])), identity)[0], 1.0)


def test_feature_groups_and_rows_are_well_formed():
    names = set(control.feature_names())
    groups = ablation.feature_groups()
    assert set().union(*map(set, groups.values())) == names        # every feature is in a group
    assert all(set(members) <= names and members for members in groups.values())
    keys = [row.key for row in ablation.ROWS]
    assert len(keys) == len(set(keys)) and ablation.REFERENCE in keys
    for row in ablation.ROWS:
        assert row.drop in ("",) + tuple(groups)
        if row.method == "v2":
            pipeline.split_cfg(row.cfg, pipeline.V2_DEFAULTS)         # no misspelled settings
