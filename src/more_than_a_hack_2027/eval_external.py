"""Score methods on the Metrica games with the official metric, and choose settings fairly.

    from more_than_a_hack_2027 import eval_external as ee
    games = ee.load_games(variants=("clean", "degraded"))
    rows = ee.evaluate(games, {"baseline": ee.METHODS["baseline"], "v1": ee.METHODS["v1"]})
    print(ee.table(rows))

    # leave-one-game-out choice among configs: pick on two games, report the third
    report = ee.logo_search(games, ee.METHODS["v1"], {"a": {...}, "b": {...}}, variant="degraded")

This is the only place where variants are compared or settings chosen (docs/PLAN.md). The public
5 minutes of BAR-ATH are a correctness alarm and never feed a decision.

Scores here are relative: the labels are Metrica's event data turned into frames (external.py),
which is a different definition from the competition's corrected-tracking ownership, and
"degraded" only approximates the ATD (degrade.py).
"""

import warnings

import numpy as np
import pandas as pd

from . import degrade as degrade_mod
from . import external, pipeline
from ._paths import ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402

VARIANTS = ("clean", "degraded")
COLUMNS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "f1_D", "team_acc_both_alive",
           "n_pred_spells", "n_gt_spells", "seconds")


def _baseline(ctx, cfg):
    return pipeline.run_baseline(ctx, (cfg or {}).get("window", 50))


def _v1(ctx, cfg):
    return pipeline.run_v1(ctx, cfg, tags=())


def _v2(ctx, cfg):
    cfg = dict(cfg or {})
    model = cfg.pop("model", None)
    if model is None:
        raise ValueError("v2 on an external game needs a model trained without that game: "
                         "pass logo_models(games) to evaluate() / logo_search()")
    return pipeline.run_v2(ctx, cfg, tags=(), model=model)


METHODS = {"baseline": _baseline, "v1": _v1, "v2": _v2}
TRAINED = {"v2"}           # methods that need a per-held-out-game model


def logo_models(games, train_variants=("degraded",), **train_kwargs):
    """{game: control model trained on every OTHER game}, so no game is scored by a model that
    saw it. Training uses the `train_variants` copies of the other games."""
    from . import control
    ids = sorted({g for g, _ in games})
    models = {}
    for g in ids:
        train_games = [games[(h, v)] for h in ids if h != g for v in train_variants
                       if (h, v) in games]
        models[g] = control.train(train_games, **train_kwargs)
    return models


def _with_model(cfg, models, game):
    if models is None:
        return cfg
    return dict(cfg or {}, model=models[game])


def load_games(games=external.GAMES, variants=VARIANTS, degrade_cfg=None, missing_counts=None,
               label_rules=external.LABEL_RULES):
    """{(game, variant): ExternalGame}. Degraded games replay BAR-ATH's missing-player counts,
    read from the competition ATD unless `missing_counts` is given; without the data the
    degraded variant keeps every player and says so. label_rules: external.LABEL_RULES or a
    subset (external.labels_from_events), for checking a choice under another label rule."""
    if "degraded" in variants and missing_counts is None:
        missing_counts = _competition_missing_counts()
    out = {}
    for game in games:
        clean = external.load_game(game, rules=label_rules)
        if "clean" in variants:
            out[(game, "clean")] = clean
        if "degraded" in variants:
            live = np.isin(clean.labels, (pm.H, pm.A))    # the same under every label rule
            arrays = degrade_mod.degrade(external.load_arrays(game), missing_counts,
                                         degrade_cfg, game=game, live=live)
            out[(game, "degraded")] = external.load_game(game, arrays=arrays, rules=label_rules)
    return out


def _competition_missing_counts():
    try:
        from .context import load_context
        return degrade_mod.atd_missing_counts(load_context())
    except (FileNotFoundError, OSError) as error:
        warnings.warn("competition ATD not available ({}); degraded games keep every player"
                      .format(error))
        return None


def score(gt_codes, labels):
    """The official evaluate(), flattened to the numbers we compare on."""
    result = pm.evaluate(gt_codes, pm.label_codes(np.asarray(labels)))
    return {
        "spell_f1_mean": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "alive_dead_f1": result["alive_dead_f1"],
        "f1_D": result["per_class"]["D"]["f1"],
        "team_acc_both_alive": result["team_acc_both_alive"],
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
    }


def run_one(external_game, method, cfg=None, return_labels=False):
    import time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = time.perf_counter()
        labels = method(external_game.ctx, cfg)
        seconds = time.perf_counter() - start
    row = score(external_game.labels, labels)
    row["seconds"] = round(seconds, 1)
    return (row, labels) if return_labels else row


def evaluate(games, methods, cfg=None, models=None):
    """[row] with method, variant, game and the score columns, for every (method, game).

    models: logo_models(games), required by the trained methods (TRAINED)."""
    rows = []
    for name, method in methods.items():
        for (game, variant), eg in sorted(games.items()):
            row = {"method": name, "variant": variant, "game": game}
            run_cfg = _with_model(cfg, models, game) if name in TRAINED else cfg
            row.update(run_one(eg, method, run_cfg))
            rows.append(row)
    return rows


def table(rows):
    """Per-game rows plus a per-(method, variant) mean, as a DataFrame."""
    frame = pd.DataFrame(rows)
    means = frame.groupby(["method", "variant"], sort=False)[list(COLUMNS)].mean().reset_index()
    means["game"] = "mean"
    out = pd.concat([frame, means], ignore_index=True)
    out = out.sort_values(["variant", "method", "game"], key=lambda col: col.astype(str))
    return out[["method", "variant", "game", *COLUMNS]].reset_index(drop=True)


def logo_search(games, method, configs, variant="degraded", key="spell_f1_mean",
                tie_break="macro_f1", models=None):
    """Leave-one-game-out selection among named configs.

    For each held-out game: the config with the best mean `key` on the OTHER games (ties by
    `tie_break`, then by name) is chosen, and its held-out score is reported. The mean held-out
    score is the honest estimate of the selection procedure; the per-config means over all games
    are returned for inspection only.

    models: per-game models for a trained method (logo_models). Each game is then scored by a
    model that did not see it; the configs are still chosen on the other games' scores, whose
    models did see the held-out game -- a mild, documented approximation of full nesting, which
    with three games would leave one game to train on.
    """
    names = sorted(configs)
    game_ids = sorted(g for g, v in games if v == variant)
    scores = {(name, g): run_one(games[(g, variant)], method,
                                 _with_model(configs[name], models, g))
              for name in names for g in game_ids}
    held_out = []
    for g in game_ids:
        others = [h for h in game_ids if h != g]
        def rank(name):
            return (np.mean([scores[(name, h)][key] for h in others]),
                    np.mean([scores[(name, h)][tie_break] for h in others]))
        best = max(names, key=rank)          # names are sorted: ties go to the first
        held_out.append({"held_out": g, "chosen": best, **scores[(best, g)]})
    overall = pd.DataFrame([{"config": name, **{
        col: np.mean([scores[(name, g)][col] for g in game_ids]) for col in COLUMNS}}
        for name in names]).sort_values(key, ascending=False).reset_index(drop=True)
    held = pd.DataFrame(held_out)
    return {"held_out": held, "held_out_mean": float(held[key].mean()), "per_config": overall}
