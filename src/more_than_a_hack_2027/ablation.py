"""Ablation table for the technical document: what each part of the pipeline is worth.

    uv run more-than-a-hack-2027 ablate --seeds 0 1 2 --out docs/ABLATION.md

Every row is scored exactly like `eval-external`: on the degraded Metrica games, with each game
scored by a control model trained without it (eval_external.logo_models), and averaged over
several degradation seeds. Nothing here reads BAR-ATH labels of any kind.

The decoder settings stay at the v2 choice for every row. A removed part could win a little back
with re-tuned penalties, so read each drop as an upper bound on what that part is worth.

Is a gap real? The interval is a paired bootstrap of the difference from v2. It uses the official
scorer's resampling unit (whole ground-truth possessions, possession_metrics.bootstrap_spell_f1),
with the same draws for every row and seed, so the noise the rows share cancels out.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import control, eval_external
from ._paths import ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402

REFERENCE = "v2"


def feature_groups(cfg=None):
    """{group: [feature names]} for the control-model ablations."""
    names = control.feature_names(cfg)
    return {
        "proximity": [n for n in names if n.startswith("near")],
        "carry": [n for n in names if "carry" in n],
        "ball direction": ["ball_vx_attack", "ball_x_attack"],
        "team shape": ["team_push", "press"],
        "look-ahead": [n for n in names if "_future_" in n or n.startswith("next_")],
        "temporal context": [n for n in names
                             if "_past_" in n or "_future_" in n or n in ("last_carry", "next_carry")],
    }


@dataclass(frozen=True)
class Row:
    key: str
    label: str
    method: str
    cfg: dict = field(default_factory=dict)
    drop: str = ""                  # a feature_groups() key, retrained without it
    variant: str = "degraded"
    section: str = ""


ROWS = (
    Row("baseline", "nearest player, ±2 s majority vote (organiser baseline)", "baseline",
        section="Pipeline, one stage at a time"),
    Row("v1", "+ dead-ball and restart stages, segment decoder (nearest-player team evidence)",
        "v1", section="Pipeline, one stage at a time"),
    Row("v2", "+ learned control model (**current submission**)", "v2",
        section="Pipeline, one stage at a time"),
    Row("-restart", "no restart-team evidence", "v2", cfg={"w_restart": 0.0},
        section="v2 with one part removed"),
    Row("-min_dead", "no 1 s minimum stoppage in the decoder", "v2", cfg={"min_dead_s": 0.0},
        section="v2 with one part removed"),
    *(Row("-" + group.replace(" ", "_"), "control model without {} features".format(group),
          "v2", drop=group, section="v2 with one part removed")
      for group in feature_groups()),
    Row("clean", "v2 on clean tracking (the cost of ATD-like input)", "v2", variant="clean",
        section="Reference"),
    Row("nov1", "decoder as frozen on Nov 1 (w_control 1.6, switch 25, tuned on the older labels)",
        "v2", cfg={"w_control": 1.6, "switch": 25.0}, section="Reference"),
)


def _flags(gt_codes, labels):
    """(matched flags per IoU threshold [5, n_gt], n_pred), as the official spell score sees them."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pm.label_codes(np.asarray(labels)), runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), t) for t in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def _spell_f1_draws(flags, n_pred, picks):
    """spell_f1_mean under each resample of GT possessions: picks [draws, n_gt] -> [draws]."""
    n_gt = flags.shape[1]
    tp = flags[:, picks].sum(axis=2)                       # [thresholds, draws]
    fp = np.maximum(0, n_pred - tp)
    fn = n_gt - tp
    f1 = np.where(tp > 0, 2.0 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return f1.mean(axis=0)


def run(seeds=(0, 1, 2), games=(1, 2, 3), rows=ROWS, draws=1000, log=print):
    """(per-run DataFrame, summary DataFrame). One run = (row, seed, game)."""
    groups = feature_groups()
    records, flags = [], {}
    for seed in seeds:
        variants = tuple(sorted({row.variant for row in rows} | {"degraded"}))
        loaded = eval_external.load_games(games, variants, degrade_cfg={"seed": seed})
        models = {}
        for row in rows:
            if row.method in eval_external.TRAINED and row.drop not in models:
                models[row.drop] = eval_external.logo_models(
                    loaded, drop=tuple(groups[row.drop]) if row.drop else ())
            method = eval_external.METHODS[row.method]
            for game in games:
                eg = loaded[(game, row.variant)]
                cfg = dict(row.cfg)
                if row.method in eval_external.TRAINED:
                    cfg["model"] = models[row.drop][game]
                result, labels = eval_external.run_one(eg, method, cfg, return_labels=True)
                records.append({"row": row.key, "seed": seed, "game": game, **result})
                flags[(row.key, seed, game)] = _flags(eg.labels, labels)
            log("seed {}  {:<20s} spellF1 {:.4f}".format(
                seed, row.key, np.mean([r["spell_f1_mean"] for r in records[-len(games):]])))
    runs = pd.DataFrame(records)
    return runs, summarise(runs, flags, rows, seeds, games, draws)


def summarise(runs, flags, rows, seeds, games, draws):
    rng = np.random.RandomState(pm.BOOTSTRAP_SEED)
    picks = {}
    for game in games:                       # GT possessions do not depend on seed or row
        n_gt = flags[(REFERENCE, seeds[0], game)][0].shape[1]
        picks[game] = rng.randint(0, n_gt, size=(draws, n_gt))
    reference = {(s, g): _spell_f1_draws(*flags[(REFERENCE, s, g)], picks[g])
                 for s in seeds for g in games}
    out = []
    for row in rows:
        mine = runs[runs["row"] == row.key]
        per_seed = mine.groupby("seed")["spell_f1_mean"].mean()
        delta = np.mean([_spell_f1_draws(*flags[(row.key, s, g)], picks[g]) - reference[(s, g)]
                         for s in seeds for g in games], axis=0)
        low, high = np.percentile(delta, pm.BOOTSTRAP_PERCENTILES)
        ref_mean = runs[runs["row"] == REFERENCE]["spell_f1_mean"].mean()
        out.append({
            "section": row.section, "row": row.key, "label": row.label,
            "spell_f1": mine["spell_f1_mean"].mean(),
            "seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
            "delta": mine["spell_f1_mean"].mean() - ref_mean,
            "delta_low": float(low), "delta_high": float(high),
            "macro_f1": mine["macro_f1"].mean(),
            "team_acc": mine["team_acc_both_alive"].mean(),
            "n_pred": mine["n_pred_spells"].mean(), "n_gt": mine["n_gt_spells"].mean(),
        })
    return pd.DataFrame(out)


def markdown(summary, seeds, games, draws):
    """The table for docs/ and the technical document. Deterministic: no dates, no timings."""
    n_gt = summary["n_gt"].iloc[0]
    lines = [
        "# Ablation (external games, held out)",
        "",
        "Generated by `uv run more-than-a-hack-2027 ablate --seeds {}` (src/more_than_a_hack_2027/"
        "ablation.py). Do not edit by hand.".format(" ".join(map(str, seeds))),
        "",
        "- **Data:** Metrica Sports sample games {} (github.com/metrica-sports/sample-data), with "
        "Metrica's event data turned into frame labels (a BALL LOST never starts a possession and "
        "a one-touch recovery is not one; external.py). The tracking is degraded to look like the "
        "competition ATD (ball smoothing and noise, player dropout, track fragmentation), using "
        "{} degradation seeds. No BAR-ATH labels are used anywhere.".format(
            ", ".join(map(str, games)), len(seeds)),
        "- **Held out:** each game is scored by a control model trained without it. The decoder "
        "settings stay at the v2 choice for every row, so read a drop as an upper bound on what "
        "the removed part is worth.",
        "- **Numbers:** spellF1 and macroF1 are the official metric (evaluation/possession_metrics"
        ".py), averaged over {} games × {} seeds. *seed sd* is the spread of the 3-game mean "
        "across seeds. *90% interval* is a paired bootstrap of the difference from v2 ({} draws "
        "over ground-truth possessions, the scorer's resampling unit). If it excludes 0, the gap "
        "is more than possession-sampling noise.".format(len(games), len(seeds), draws),
        "",
        "| | spellF1 | seed sd | Δ vs v2 | 90% interval of Δ | macroF1 | team acc | possessions "
        "/ game (true {:.0f}) |".format(n_gt),
        "|---|---|---|---|---|---|---|---|",
    ]
    section = None
    for _, r in summary.iterrows():
        if r["section"] != section:
            section = r["section"]
            lines.append("| ***{}*** | | | | | | | |".format(section))
        is_ref = r["row"] == REFERENCE
        lines.append("| {} | {:.3f} | {:.3f} | {} | {} | {:.3f} | {:.3f} | {:.0f} |".format(
            r["label"], r["spell_f1"], r["seed_sd"],
            "—" if is_ref else "{:+.3f}".format(r["delta"]),
            "—" if is_ref else "[{:+.3f}, {:+.3f}]".format(r["delta_low"], r["delta_high"]),
            r["macro_f1"], r["team_acc"], r["n_pred"]))
    groups = feature_groups()
    lines += ["", "Control feature groups (one model is retrained without each, held out):", ""]
    lines += ["- **{}:** {}".format(name, ", ".join("`{}`".format(f) for f in members))
              for name, members in groups.items()]
    lines.append("")
    return "\n".join(lines)
