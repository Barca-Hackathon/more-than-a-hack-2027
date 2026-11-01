"""The decoder settings the submission ships with, chosen on the external games.

    uv run more-than-a-hack-2027 tune --seeds 0 1 2 --out docs/TUNING.md

What is chosen: v2's w_control (weight on the control model's log-probabilities) and its switch
penalty (H <-> A in open play). They decide how many possessions come out, the score depends on
them more than on anything else, and mostly through their ratio (docs/PLAN.md, "Control design").

Protocol. Every (config, label rules, seed, game) runs v2 on the degraded Metrica game with a
control model trained without that game (harness.logo_models), and is scored by the
official metric. Nothing here reads BAR-ATH labels of any kind.

Label rules. metrica.LABEL_RULES says which events do not change possession. "loss" follows
from the competition's definition alone; "one_touch" (a header or clearance straight back to the
opponent is not a possession) is the competition's own list of what does not count, but where the
organisers' tracking-derived ground truth draws that line cannot be checked from here, and the
best switch penalty moves with it. So a config is chosen on the MEAN spellF1 under both readings:
    main   LABEL_RULES ("loss", "one_touch")
    alt    ("loss",) only
and the older labels (no rule, the Nov 1 freeze) are shown for reference, never used to choose.

Held-out estimate. The choice itself is made on all three games. The honest estimate of the
procedure is leave-one-game-out: for each game, pick by the same criterion on the other two (mean
over seeds), and report that pick's score on the held-out game.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import harness
from ..external import metrica

W_CONTROL = (0.8, 1.6, 3.2)
SWITCH = (12.0, 25.0, 35.0, 50.0, 70.0, 100.0)
LABELS = {
    "main": metrica.LABEL_RULES,
    "alt": ("loss",),
    "old": (),
}
CHOOSE_ON = ("main", "alt")          # "old" is reported, never used to choose


def grid():
    """{name: v2 config}; names sort in grid order."""
    return {"w{:.1f}_s{:03.0f}".format(w, s): {"w_control": w, "switch": s}
            for w in W_CONTROL for s in SWITCH}


def _run_seed(seed, games, configs, labels):
    records = []
    for label_name, rules in labels.items():
        loaded = harness.load_games(games, ("degraded",), degrade_cfg={"seed": seed},
                                          label_rules=rules)
        models = harness.logo_models(loaded)
        for name, cfg in configs.items():
            for game in games:
                row = harness.run_one(loaded[(game, "degraded")],
                                            harness.METHODS["v2"],
                                            dict(cfg, model=models[game]))
                records.append({"labels": label_name, "seed": seed, "config": name,
                                "game": game, **row})
    return records


def run(seeds=(0, 1, 2), games=metrica.GAMES, configs=None, labels=None, jobs=1, log=print):
    """Per-run DataFrame: one row per (labels, seed, config, game)."""
    configs = grid() if configs is None else configs
    labels = LABELS if labels is None else labels
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(_run_seed, seeds, [games] * len(seeds),
                                  [configs] * len(seeds), [labels] * len(seeds)))
    else:
        parts = [_run_seed(seed, games, configs, labels) for seed in seeds]
    for seed, part in zip(seeds, parts):
        log("seed {}: {} runs".format(seed, len(part)))
    return pd.DataFrame([row for part in parts for row in part])


def _criterion(runs, games, choose_on=CHOOSE_ON):
    """(spellF1, macroF1) per config: mean over `games`, seeds and the choose_on label sets."""
    mine = runs[runs["game"].isin(games) & runs["labels"].isin(choose_on)]
    return mine.groupby("config")[["spell_f1_mean", "macro_f1"]].mean()


def choose(runs, games, choose_on=CHOOSE_ON):
    """Best config on `games`: spellF1, then macroF1, then name (deterministic)."""
    crit = _criterion(runs, games, choose_on).reset_index()
    crit = crit.sort_values(["spell_f1_mean", "macro_f1", "config"],
                            ascending=[False, False, True])
    return crit["config"].iloc[0]


def held_out(runs, choose_on=CHOOSE_ON):
    """Leave-one-game-out: [{held_out, chosen, spellF1 under each label set}]."""
    games = sorted(runs["game"].unique())
    out = []
    for g in games:
        chosen = choose(runs, [h for h in games if h != g], choose_on)
        mine = runs[(runs["game"] == g) & (runs["config"] == chosen)]
        row = {"held_out": g, "chosen": chosen}
        for label_name, group in mine.groupby("labels", sort=False):
            row[label_name] = group["spell_f1_mean"].mean()
        out.append(row)
    return pd.DataFrame(out)


def summary(runs):
    """One row per config: spellF1 per label set, the criterion, and main-label details."""
    table = runs.pivot_table(index="config", columns="labels", values="spell_f1_mean",
                             aggfunc="mean")
    crit = _criterion(runs, sorted(runs["game"].unique()))
    table["criterion"] = crit["spell_f1_mean"]
    main = runs[runs["labels"] == "main"].groupby("config")
    table["macro_f1"] = main["macro_f1"].mean()
    table["n_pred"] = main["n_pred_spells"].mean()
    table["n_gt"] = main["n_gt_spells"].mean()
    table["seed_sd"] = (runs[runs["labels"] == "main"].groupby(["config", "seed"])
                        ["spell_f1_mean"].mean().groupby("config").std(ddof=1))
    return table.reset_index()


def markdown(runs, configs, seeds, games):
    table = summary(runs).sort_values(["criterion", "macro_f1", "config"],
                                      ascending=[False, False, True])
    chosen = choose(runs, list(games))
    held = held_out(runs)
    gt = {name: runs[runs["labels"] == name].groupby("game")["n_gt_spells"].first().mean()
          for name in runs["labels"].unique()}
    lines = [
        "# Decoder tuning (external games, held out)",
        "",
        "Generated by `uv run more-than-a-hack-2027 tune --seeds {}` (src/more_than_a_hack_2027/"
        "validation/tune.py). Do not edit by hand.".format(" ".join(map(str, seeds))),
        "",
        "- **Data:** Metrica Sports sample games {} (github.com/metrica-sports/sample-data), "
        "tracking degraded to look like the competition ATD, {} degradation seeds. Each game is "
        "scored by a control model trained without it. No BAR-ATH labels are used anywhere."
        .format(", ".join(map(str, games)), len(seeds)),
        "- **Labels:** Metrica's event data turned into frames (metrica.py). *main* applies both "
        "label rules (a BALL LOST never starts a possession; a one-touch recovery is not one), "
        "true possessions per game {:.0f}. *alt* applies only the first ({:.0f}). *old* applies "
        "neither ({:.0f}); it is what the Nov 1 freeze was tuned on and is shown for reference "
        "only.".format(gt.get("main", np.nan), gt.get("alt", np.nan), gt.get("old", np.nan)),
        "- **Choice:** the best *criterion*, the mean spellF1 under *main* and *alt*, so the "
        "choice holds up whichever way the organisers' ground truth treats one-touch plays.",
        "",
        "| w_control | switch | spellF1 main | spellF1 alt | criterion | spellF1 old | macroF1 "
        "main | seed sd main | possessions / game main |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        cfg = configs[r["config"]]
        mark = " **(chosen)**" if r["config"] == chosen else ""
        lines.append("| {:.1f} | {:.0f}{} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | "
                     "{:.0f} |".format(cfg["w_control"], cfg["switch"], mark, r["main"],
                                       r["alt"], r["criterion"], r["old"], r["macro_f1"],
                                       r["seed_sd"], r["n_pred"]))
    lines += [
        "",
        "Leave-one-game-out estimate of this procedure: the config is picked by the same "
        "criterion on the other two games (all seeds), and scored on the held-out one.",
        "",
        "| held-out game | picked | spellF1 main | spellF1 alt | spellF1 old |",
        "|---|---|---|---|---|",
    ]
    for _, r in held.iterrows():
        cfg = configs[r["chosen"]]
        lines.append("| {} | w_control {:.1f}, switch {:.0f} | {:.3f} | {:.3f} | {:.3f} |".format(
            r["held_out"], cfg["w_control"], cfg["switch"], r["main"], r["alt"], r["old"]))
    lines.append("| **mean** | | **{:.3f}** | **{:.3f}** | {:.3f} |".format(
        held["main"].mean(), held["alt"].mean(), held["old"].mean()))
    cfg = configs[chosen]
    lines += ["", "**Chosen for the submission:** w_control {:.1f}, switch {:.0f} "
              "(`pipeline.V2_DEFAULTS`).".format(cfg["w_control"], cfg["switch"]), ""]
    return "\n".join(lines)
