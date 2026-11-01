"""The external harness: official-metric scoring and leave-one-game-out selection."""

from types import SimpleNamespace

import numpy as np

from more_than_a_hack_2027.validation import harness as ee
from more_than_a_hack_2027._paths import ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402


def fake_game(game, n=500):
    labels = np.full(n, pm.D, dtype=np.int8)
    labels[50:200], labels[200:300], labels[350:480] = pm.H, pm.A, pm.H
    return SimpleNamespace(game=game, labels=labels, ctx=SimpleNamespace(game=game, n=n))


def method(ctx, cfg):
    """Perfect on the games listed in cfg["good"], one label shifted by `cfg["lag"]` elsewhere."""
    truth = fake_game(ctx.game).labels
    if ctx.game in cfg["good"]:
        return np.array(["HADX"[c] for c in truth])
    return np.array(["HADX"[c] for c in np.roll(truth, cfg["lag"])])


def test_perfect_prediction_scores_one():
    game = fake_game(1)
    row = ee.run_one(game, method, {"good": (1,), "lag": 0})
    assert row["spell_f1_mean"] == 1.0 and row["macro_f1"] == 1.0


def test_logo_never_scores_the_choice_on_the_held_out_game():
    games = {(g, "degraded"): fake_game(g) for g in (1, 2, 3)}
    configs = {
        "fits_1_and_2": {"good": (1, 2), "lag": 120},   # best on 1+2, bad on 3
        "steady": {"good": (), "lag": 5},               # decent everywhere
    }
    report = ee.logo_search(games, method, configs)
    held = report["held_out"].set_index("held_out")
    # Holding out 3, the choice is made on games 1 and 2 only, and "fits_1_and_2" wins there,
    # so its poor game-3 score is what gets reported -- the overfit is visible, not hidden.
    assert held.loc[3, "chosen"] == "fits_1_and_2"
    assert held.loc[3, "spell_f1_mean"] < 0.5
    assert held.loc[1, "chosen"] == "steady" and held.loc[2, "chosen"] == "steady"
    assert np.isclose(report["held_out_mean"], held["spell_f1_mean"].mean())
