"""Held-out evaluation and decoder choice for v3 (the sequence model), on the external games.

    uv run --group train more-than-a-hack-2027 tune-seq --out docs/TUNING_V3.md
    uv run more-than-a-hack-2027 tune-seq --logprobs lp_a.npz lp_b.npz ... --out docs/TUNING_V3.md

The second form scores held-out log-probabilities computed elsewhere (scripts/kaggle_seq.py
trains the held-out models on a GPU and writes them), averaging the files as an ensemble.

Protocol, as tune.py: for each Metrica game, a sequence model trained on the OTHER two games
(training.logo_models, all training degradation seeds) predicts it on the evaluation seeds; the
decoder settings are chosen on the mean spellF1 under both label readings (main: both label
rules; alt: "loss" only), and the honest estimate is leave-one-game-out (pick on two games, score
the third). No BAR-ATH labels are used.
"""

import itertools
import os

import numpy as np
import pandas as pd

from ..stages import decode
from ..validation import harness, tune
from ..external import metrica
from . import network, training

W_SEQ = (0.25, 0.5, 1.0)
SWITCH = (10.0, 20.0, 35.0, 50.0)
STOP = (6.0, 10.0, 15.0, 25.0)            # stop = restart penalty
TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
EVAL_SEEDS = (0, 1, 2)


def grid():
    return {"w{:.2f}_s{:03.0f}_d{:02.0f}".format(w, s, d):
            {"w_seq": w, "switch": s, "stop": d, "restart": d}
            for w, s, d in itertools.product(W_SEQ, SWITCH, STOP)}


def _labels(games):
    """{label set: {game: codes}} for the readings tune.py chooses on (and the old one)."""
    arrays = {g: metrica.load_arrays(g) for g in games}
    return {name: {g: metrica.load_game(g, arrays=arrays[g], rules=rules).labels for g in games}
            for name, rules in tune.LABELS.items()}


# The report's Model line for `tune-seq --logprobs` on scripts/kaggle_seq.py v3-heldout's files.
HELDOUT_NOTE = (
    "- **Model:** the v3 ensemble, eight temporal CNNs (seq/network.py with the touch inputs) "
    "trained by scripts/kaggle_seq.py's V3 recipe: the Metrica games plus the open DFL "
    "(control labels, 6 degradation seeds) and SkillCorner (one-touch labels, 3 seeds) "
    "matches, with the ball shown while dead as in the competition ATD's stoppages "
    "(external/degrade.py, dead_ball_view), training seeds 0-6 and one auxiliary-head "
    "variant, 3200 steps on a Kaggle T4. "
    "Each game is predicted by the eight networks trained on the other two games (Metrica "
    "degradation seeds 0-5), on degradation seeds 0-2, and their log-probabilities are "
    "averaged. No BAR-ATH labels are used anywhere."
)


def run(train_seeds=TRAIN_SEEDS, eval_seeds=EVAL_SEEDS, games=metrica.GAMES, cfg=None,
        configs=None, log=print):
    """Per-run DataFrame: one row per (labels, seed, config, game), as tune.run."""
    models = training.logo_models(train_seeds, games, cfg, log=log)
    lps = {}
    for s in eval_seeds:
        for g in games:
            X, sign, _, _, halves = training.game_inputs(s, g)
            lps[(s, g)] = network.logprobs_halves(models[g], X, sign, halves)
    return score_logprobs(lps, configs, log=log)


def load_logprobs(paths):
    """{(seed, game): log-probs} from held-out files (keys s{seed}_g{game}); several files are
    averaged as an ensemble (network.average_logprobs), over the keys they all have."""
    files = [np.load(p) for p in paths]
    keys = sorted(set.intersection(*(set(f.files) for f in files)))
    return {(int(k[1:k.index("_")]), int(k[k.index("_g") + 2:])):
            network.average_logprobs([f[k] for f in files]) for k in keys}


def score_logprobs(lps, configs=None, log=print):
    """Decode held-out log-probabilities {(seed, game): (n, 3)} under every config and score
    them under each label reading: one row per (labels, seed, config, game)."""
    configs = grid() if configs is None else configs
    games = sorted({g for _, g in lps})
    labels = _labels(games)
    rows = []
    for s, g in sorted(lps):
        with np.load(os.path.join(training.CACHE_DIR, training.input_name(
                "s{}_g{}".format(s, g)))) as z:
            graded = z["graded"]
            halves = tuple(tuple(int(v) for v in row) for row in z["halves"])
        lp = lps[(s, g)]
        for name, c in configs.items():
            ev = np.where(graded[:, None], c["w_seq"] * lp, 0.0)
            trans = decode.transition_matrix(c["switch"], c["stop"], c["restart"])
            pred = decode.decode_match(ev, halves, trans, min_dead=25)
            for label_name, by_game in labels.items():
                r = harness.score(by_game[g], pred)
                rows.append({"labels": label_name, "seed": s, "config": name, "game": g, **r})
        if log:
            log("seed {} game {} done".format(s, g))
    return pd.DataFrame(rows)


def load_named_logprobs(paths):
    """{input file base name: log-probs} from held-out files of other matches (kaggle_seq.py's
    lp_*_extra.npz); several files are averaged as an ensemble over the names they all have."""
    files = [np.load(p) for p in paths]
    keys = sorted(set.intersection(*(set(f.files) for f in files)))
    return {k: network.average_logprobs([f[k] for f in files]) for k in keys}


def score_named(lps, cfg=None):
    """Decode {input base name: log-probs} at v3's decoder settings (pipeline.V3_DEFAULTS plus
    cfg) and score each against the labels cached with its inputs. Adds the share of dead frames
    with the ball in view that are called D (`dead_seen_recall`): the failure the dead-ball view
    experiment targets."""
    from .. import pipeline
    top, _ = pipeline.split_cfg(cfg, pipeline.V3_DEFAULTS)
    trans = decode.transition_matrix(top["switch"], top["stop"], top["restart"])
    rows = []
    for name, lp in sorted(lps.items()):
        with np.load(os.path.join(training.CACHE_DIR, name + ".npz")) as z:
            labels, graded, X = z["labels"], z["graded"], z["X"]
            halves = tuple(tuple(int(v) for v in row) for row in z["halves"])
        pred = decode.decode_match(np.where(graded[:, None], top["w_seq"] * lp, 0.0), halves,
                                   trans, min_dead=int(round(top["min_dead_s"] * 25)))
        seen = X[:, network.BALL_SEEN] > 0.5
        dead_seen = graded & (labels == 2) & seen
        rows.append({"name": name, **harness.score(labels, pred),
                     "dead_seen_recall": float((pred[dead_seen] == "D").mean())
                     if dead_seen.any() else float("nan")})
    return pd.DataFrame(rows)


def markdown(runs, configs, train_seeds, eval_seeds, games, cfg, model_note=None):
    table = tune.summary(runs).sort_values(["criterion", "macro_f1", "config"],
                                           ascending=[False, False, True])
    chosen = tune.choose(runs, list(games))
    held = tune.held_out(runs)
    cfg = dict(training.TRAIN_DEFAULTS, **(cfg or {}))
    lines = [
        "# v3 decoder tuning (sequence model, external games, held out)",
        "",
        "Generated by `more-than-a-hack-2027 tune-seq` (src/more_than_a_hack_2027/seq/tuning.py). "
        "Do not edit by hand.",
        "",
        model_note or "- **Model:** network.py's temporal CNN, trained by training.py with {}. "
        "Each game is predicted by a model trained on the other two games (degradation seeds {}), "
        "on degradation seeds {}. No BAR-ATH labels are used anywhere.".format(
            ", ".join("{} {}".format(k, v) for k, v in sorted(cfg.items())),
            ", ".join(map(str, train_seeds)), ", ".join(map(str, eval_seeds))),
        "- **Choice:** as docs/TUNING.md, the best mean spellF1 under the *main* and *alt* label "
        "readings.",
        "",
        "| w_seq | switch | stop / restart | spellF1 main | spellF1 alt | criterion | spellF1 old "
        "| macroF1 main | seed sd main | possessions / game main |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.head(12).iterrows():
        c = configs[r["config"]]
        mark = " **(chosen)**" if r["config"] == chosen else ""
        lines.append("| {:.2f} | {:.0f} | {:.0f}{} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | "
                     "{:.3f} | {:.0f} |".format(c["w_seq"], c["switch"], c["stop"], mark,
                                                r["main"], r["alt"], r["criterion"], r["old"],
                                                r["macro_f1"], r["seed_sd"], r["n_pred"]))
    lines += [
        "",
        "(top 12 of {} configs)".format(len(configs)),
        "",
        "Leave-one-game-out estimate: the config is picked on the other two games and scored on "
        "the held-out one (whose model never saw it).",
        "",
        "| held-out game | picked | spellF1 main | spellF1 alt | spellF1 old |",
        "|---|---|---|---|---|",
    ]
    for _, r in held.iterrows():
        c = configs[r["chosen"]]
        lines.append("| {} | w_seq {:.2f}, switch {:.0f}, stop {:.0f} | {:.3f} | {:.3f} | {:.3f} |"
                     .format(r["held_out"], c["w_seq"], c["switch"], c["stop"], r["main"],
                             r["alt"], r["old"]))
    lines.append("| **mean** | | **{:.3f}** | **{:.3f}** | {:.3f} |".format(
        held["main"].mean(), held["alt"].mean(), held["old"].mean()))
    c = configs[chosen]
    lines += ["", "**Chosen for v3:** w_seq {:.2f}, switch {:.0f}, stop {:.0f}, restart {:.0f} "
              "(`pipeline.V3_DEFAULTS`).".format(c["w_seq"], c["switch"], c["stop"],
                                                 c["restart"]), ""]
    return "\n".join(lines)
