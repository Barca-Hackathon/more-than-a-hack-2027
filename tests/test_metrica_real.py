"""The external harness on the downloaded Metrica games. Skipped until they are in .cache/.

Run `uv run more-than-a-hack-2027 eval-external` once to download them. These check the label
conversion and the harness end to end; they are plausibility bounds, not accuracy targets.
"""

import os
import warnings

import numpy as np
import pytest

from more_than_a_hack_2027.external import metrica

DATA_PRESENT = all(os.path.isfile(os.path.join(metrica.RAW_DIR, rel))
                   for game in metrica.GAMES for rel in metrica.FILES[game])
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="Metrica sample data not downloaded")


@pytest.mark.parametrize("game", metrica.GAMES)
def test_event_labels_are_plausible(game):
    eg = metrica.load_game(game)
    stats = metrica.possession_stats(eg.labels, eg.ctx.halves)
    assert 0.25 <= stats["dead_share"] <= 0.5
    assert 200 <= stats["n_spells"] <= 320
    assert 6.0 <= stats["spell_median_s"] <= 13.0
    assert 80 <= stats["n_stoppages"] <= 130
    assert 0.4 <= stats["home_share_alive"] <= 0.6


@pytest.mark.parametrize("game", metrica.GAMES)
def test_label_rules_only_change_who_has_the_ball(game):
    eg = metrica.load_game(game)
    old = metrica.load_game(game, rules=())
    alt = metrica.load_game(game, rules=("loss",))
    for other in (old, alt):
        assert np.array_equal(eg.labels == metrica.pm.D, other.labels == metrica.pm.D)
        assert np.array_equal(eg.labels == metrica.pm.X, other.labels == metrica.pm.X)
    spells = [metrica.possession_stats(x.labels, eg.ctx.halves)["n_spells"]
              for x in (old, alt, eg)]
    assert spells[0] > spells[1] > spells[2]            # each rule only removes possessions
    assert 300 <= spells[0] <= 500                      # the older labels, as first measured


def test_v1_beats_the_baseline_on_degraded_games():
    from more_than_a_hack_2027.validation import harness as ee
    warnings.simplefilter("ignore", RuntimeWarning)
    games = ee.load_games(variants=("degraded",), missing_counts=np.zeros(10, dtype=np.int16))
    methods = {m: ee.METHODS[m] for m in ("baseline", "v1")}
    rows = ee.evaluate(games, methods)
    mean = {m: np.mean([r["spell_f1_mean"] for r in rows if r["method"] == m]) for m in methods}
    assert mean["v1"] > mean["baseline"]


def test_v2_held_out_beats_v1_on_degraded_games():
    from more_than_a_hack_2027.validation import harness as ee
    warnings.simplefilter("ignore", RuntimeWarning)
    games = ee.load_games(variants=("degraded",), missing_counts=np.zeros(10, dtype=np.int16))
    models = ee.logo_models(games)
    for g, model in models.items():
        assert "METRICA-{}".format(g) not in model.meta["games"]       # never trained on itself
    rows = ee.evaluate(games, {"v1": ee.METHODS["v1"], "v2": ee.METHODS["v2"]}, models=models)
    mean = {m: np.mean([r["spell_f1_mean"] for r in rows if r["method"] == m]) for m in ("v1", "v2")}
    assert mean["v2"] > mean["v1"]
