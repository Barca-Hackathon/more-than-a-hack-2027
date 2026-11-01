"""tune.py's selection logic on a hand-built score table (no data needed)."""

import pandas as pd

from more_than_a_hack_2027.validation import tune


def _runs(scores):
    """scores: {(labels, config, game): spellF1}; two seeds with the same score."""
    rows = []
    for (labels, config, game), f1 in scores.items():
        for seed in (0, 1):
            rows.append({"labels": labels, "seed": seed, "config": config, "game": game,
                         "spell_f1_mean": f1, "macro_f1": 0.9, "n_pred_spells": 200,
                         "n_gt_spells": 240})
    return pd.DataFrame(rows)


def test_choice_averages_main_and_alt_and_ignores_old():
    scores = {}
    for game in (1, 2, 3):
        # "sharp" wins on main only, "robust" is second on both; "old" would pick "legacy".
        scores.update({("main", "sharp", game): 0.80, ("alt", "sharp", game): 0.60,
                       ("main", "robust", game): 0.76, ("alt", "robust", game): 0.70,
                       ("main", "legacy", game): 0.60, ("alt", "legacy", game): 0.60,
                       ("old", "sharp", game): 0.50, ("old", "robust", game): 0.55,
                       ("old", "legacy", game): 0.70})
    runs = _runs(scores)
    assert tune.choose(runs, [1, 2, 3]) == "robust"
    assert tune.choose(runs, [1, 2, 3], choose_on=("main",)) == "sharp"
    table = tune.summary(runs).set_index("config")
    assert abs(table.loc["robust", "criterion"] - 0.73) < 1e-12


def test_held_out_never_picks_on_the_game_it_scores():
    # Config "a" is best on games 1 and 2, config "b" only on game 3.
    scores = {}
    for labels in ("main", "alt", "old"):
        scores.update({(labels, "a", 1): 0.8, (labels, "a", 2): 0.8, (labels, "a", 3): 0.1,
                       (labels, "b", 1): 0.5, (labels, "b", 2): 0.5, (labels, "b", 3): 0.9})
    held = tune.held_out(_runs(scores)).set_index("held_out")
    assert held.loc[3, "chosen"] == "a"            # chosen on games 1 and 2 ...
    assert abs(held.loc[3, "main"] - 0.1) < 1e-12  # ... and scored on game 3 as it is
    # Without game 1, "b" averages (0.5 + 0.9) / 2 on games 2 and 3, against 0.45 for "a".
    assert held.loc[1, "chosen"] == "b" and held.loc[2, "chosen"] == "b"


def test_grid_names_map_to_their_settings():
    configs = tune.grid()
    assert len(configs) == len(tune.W_CONTROL) * len(tune.SWITCH)
    assert configs["w1.6_s025"] == {"w_control": 1.6, "switch": 25.0}


def test_shipped_decoder_settings_are_the_recorded_choice():
    import os
    import re

    from more_than_a_hack_2027 import pipeline
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "TUNING.md")
    with open(path) as handle:
        match = re.search(r"Chosen for the submission:\*\* w_control ([\d.]+), switch ([\d.]+)",
                          handle.read())
    assert match, "docs/TUNING.md has no 'Chosen' line"
    assert float(match.group(1)) == pipeline.V2_DEFAULTS["w_control"]
    assert float(match.group(2)) == pipeline.V2_DEFAULTS["switch"]
