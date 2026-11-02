"""More than a Hack 2027, Phase 1: frame-level ball possession.

    uv run more-than-a-hack-2027 cache                 # parse the ATD once into .cache/
    uv run more-than-a-hack-2027 info                  # what the data layer sees
    uv run more-than-a-hack-2027 predict --out evaluation/results/<team>.csv   # the submission
    uv run more-than-a-hack-2027 predict --method baseline --window 50 \
        --out evaluation/results/our_baseline50.csv
    uv run more-than-a-hack-2027 eval-external         # score methods on the Metrica games
    uv run more-than-a-hack-2027 train-control         # refit models/control_v1.json
    uv run more-than-a-hack-2027 tune --out docs/TUNING.md       # choose the decoder settings
    uv run more-than-a-hack-2027 ablate --out docs/ABLATION.md   # held-out ablation table
"""

import argparse
import os
import sys


def _cmd_cache(args):
    from .data import atd
    atd.build_cache(force=args.force)
    return 0


def _cmd_info(args):
    import numpy as np
    from .data.context import AWAY, HOME, load_context

    ctx = load_context()
    graded = ctx.half_of_frame > 0
    print("game {}  frames {}  fps {}  pitch {} m".format(
        ctx.game, ctx.n_frames, ctx.fps, ctx.pitch))
    print("H = {}  A = {}".format(*ctx.team_names))
    print("track slots {}  (home {}, away {})".format(
        ctx.n_slots, len(ctx.team_slots(HOME)), len(ctx.team_slots(AWAY))))
    print("halves {}".format(ctx.halves))
    print("home attacks +x: {}".format(ctx.home_attacks_positive_x))
    seen = ctx.ball_seen
    print("ball seen  all {:.1%}  graded {:.1%}".format(seen.mean(), seen[graded].mean()))
    observed = (~np.isnan(ctx.xy[:, :, 0])).sum(axis=1)
    print("tracked objects per graded frame  mean {:.1f}".format(observed[graded].mean()))
    return 0


def _cmd_deadball(args):
    import json
    import warnings

    from .stages import deadball, restart
    from .data.context import load_context
    from .data.tagging import load_tags

    warnings.simplefilter("ignore", RuntimeWarning)
    ctx = load_context()
    dead = deadball.detect(ctx)
    tags = load_tags(ctx)
    calls = restart.decide(ctx, dead, tags)
    report = deadball.summary(ctx, dead, tags)
    report["restarts"] = restart.summary(calls)
    print(json.dumps(report, indent=2, default=str))
    if args.table:
        table = deadball.stoppage_table(dead, ctx.fps)
        table["restart_team"] = [{0: "H", 1: "A"}.get(c.team, "?") for c in calls]
        table["restart_logodds_home"] = [round(c.logodds_home, 2) for c in calls]
        table.to_csv(args.table, index=False)
        print("wrote {}".format(args.table))
    return 0


def _cmd_predict(args):
    from . import pipeline, submission
    from .data.context import load_context

    args.method = args.method or pipeline.SUBMISSION_METHOD
    ctx = load_context()
    labels = pipeline.run(args.method, ctx, window=args.window)
    if args.method == "baseline":
        # Byte-parity with the reference file matters more than the X rule here.
        from ._paths import ensure_repo_on_path
        ensure_repo_on_path()
        import baseline as starter_baseline
        starter_baseline.write_submission(args.out, labels)
        print("wrote {}".format(args.out))
        return 0
    summary = submission.write(args.out, labels, ctx)
    print("wrote {}".format(summary["path"]))
    print("  counts {}".format(summary["counts"]))
    print("  {} possessions, mean {:.1f} s".format(
        summary["n_possessions"], summary["mean_possession_s"]))
    return 0


def _cmd_eval_external(args):
    import warnings

    import pandas as pd

    from .external import degrade, metrica
    from .validation import harness

    warnings.simplefilter("ignore", RuntimeWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    games = harness.load_games(tuple(args.games), tuple(args.variants),
                                     degrade_cfg={"seed": args.seed})
    if args.describe:
        for (game, variant), eg in sorted(games.items()):
            graded = eg.ctx.half_of_frame > 0
            arrays = {"ball": eg.ctx.ball, "xy": eg.ctx.xy, "team_of_slot": eg.ctx.team_of_slot}
            print(game, variant, "labels", metrica.possession_stats(eg.labels, eg.ctx.halves))
            live = graded & (eg.labels != metrica.pm.D)
            print(game, variant, "input ", degrade.describe(arrays, graded, live))
    methods = {name: harness.METHODS[name] for name in args.methods}
    models = None
    if harness.TRAINED & set(methods):
        # Never score a game with a model that saw it: retrain without each game in turn.
        models = harness.logo_models(games)
    table = harness.table(harness.evaluate(games, methods, models=models))
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print("wrote {}".format(args.out))
    return 0


def _cmd_train_control(args):
    import warnings

    from .stages import control
    from .validation import harness

    warnings.simplefilter("ignore", RuntimeWarning)
    games = harness.load_games(variants=("degraded",), degrade_cfg={"seed": args.seed})
    model = control.train([games[key] for key in sorted(games)], l2=args.l2, stride=args.stride,
                          meta={"variant": "degraded", "degrade_seed": args.seed})
    model.to_json(args.out)
    print("trained on {} rows from {}".format(model.meta["n_rows"], model.meta["games"]))
    for name, weight in zip(model.names, model.weights):
        print("  {:<20s} {:+.3f}".format(name, weight))
    print("wrote {}".format(args.out))
    return 0


def _cmd_ablate(args):
    import warnings

    from .validation import ablation

    warnings.simplefilter("ignore", RuntimeWarning)
    games = tuple(args.games)
    runs, summary = ablation.run(tuple(args.seeds), games, draws=args.draws)
    text = ablation.markdown(summary, tuple(args.seeds), games, args.draws)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print("wrote {}".format(args.out))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        runs.to_csv(args.csv, index=False)
        print("wrote {}".format(args.csv))
    return 0


def _cmd_tune(args):
    import warnings

    from .validation import tune

    warnings.simplefilter("ignore", RuntimeWarning)
    seeds, games = tuple(args.seeds), tuple(args.games)
    configs = tune.grid()
    runs = tune.run(seeds, games, configs, jobs=args.jobs)
    text = tune.markdown(runs, configs, seeds, games)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print("wrote {}".format(args.out))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        runs.to_csv(args.csv, index=False)
        print("wrote {}".format(args.csv))
    return 0


def _cmd_train_seq(args):
    import warnings

    from .seq import network, training

    warnings.simplefilter("ignore", RuntimeWarning)
    blocks = [training.game_inputs(s, g, touch=args.touch) for s in args.seeds for g in args.games]
    model = training.train(blocks, meta={"games": list(args.games), "seeds": list(args.seeds),
                                         "variant": "degraded"})
    model.to_json(args.out)
    print("wrote {}".format(args.out))
    return 0


def _cmd_dfl_inputs(args):
    import warnings

    from .external import dfl
    from .validation import harness

    warnings.simplefilter("ignore", RuntimeWarning)
    missing = harness._competition_missing_counts()
    for match_id in args.matches or dfl.MATCHES:
        for seed in args.seeds:
            print(dfl.build_inputs(match_id, seed, missing_counts=missing, labels=args.labels,
                                   touch=args.touch))
    return 0


def _cmd_skillcorner_inputs(args):
    import warnings

    from .external import skillcorner
    from .validation import harness

    warnings.simplefilter("ignore", RuntimeWarning)
    missing = harness._competition_missing_counts()
    for match_id in args.matches or skillcorner.match_ids():
        for seed in args.seeds:
            print(skillcorner.build_inputs(match_id, seed, missing_counts=missing,
                                           labels=args.labels, touch=args.touch), flush=True)
    return 0


def _cmd_tune_seq(args):
    import warnings

    from .seq import tuning

    warnings.simplefilter("ignore", RuntimeWarning)
    configs = tuning.grid()
    note = None
    if args.logprobs:
        runs = tuning.score_logprobs(tuning.load_logprobs(args.logprobs), configs)
        note = args.note
    else:
        runs = tuning.run(tuple(args.train_seeds), tuple(args.seeds), tuple(args.games),
                           configs=configs)
    text = tuning.markdown(runs, configs, tuple(args.train_seeds), tuple(args.seeds),
                            tuple(args.games), None, model_note=note)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        print("wrote {}".format(args.out))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        runs.to_csv(args.csv, index=False)
        print("wrote {}".format(args.csv))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="more-than-a-hack-2027", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cache", help="parse the tactical ATD into .cache/")
    p.add_argument("--force", action="store_true", help="rebuild even if the cache is fresh")
    p.set_defaults(func=_cmd_cache)

    p = sub.add_parser("info", help="summarise the loaded context")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser("deadball", help="run dead-ball detection and print GT-free diagnostics")
    p.add_argument("--table", help="also write the stoppage table to this CSV (keep it in .cache/)")
    p.set_defaults(func=_cmd_deadball)

    p = sub.add_parser("predict", help="write a submission CSV")
    p.add_argument("--method", default=None, choices=("baseline", "hmm-nearest", "v1", "v2", "v3"),
                   help="default: the frozen submission method (pipeline.SUBMISSION_METHOD)")
    p.add_argument("--window", type=int, default=50, help="baseline smoothing half-width")
    p.add_argument("--out", required=True, help="output CSV path")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("eval-external",
                       help="score methods on the Metrica sample games (downloads them once)")
    p.add_argument("--methods", nargs="+", default=["baseline", "v1", "v2"],
                   choices=("baseline", "v1", "v2"))
    p.add_argument("--variants", nargs="+", default=["clean", "degraded"],
                   choices=("clean", "degraded"))
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    p.add_argument("--seed", type=int, default=0, help="degradation seed")
    p.add_argument("--describe", action="store_true",
                   help="also print label and input statistics per game")
    p.add_argument("--out", help="also write the score table to this CSV (keep it in .cache/)")
    p.set_defaults(func=_cmd_eval_external)

    p = sub.add_parser("train-control",
                       help="fit the control model on the degraded Metrica games")
    p.add_argument("--out", default=None, help="model JSON (default: the shipped model)")
    p.add_argument("--seed", type=int, default=0, help="degradation seed")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=3)
    p.set_defaults(func=_cmd_train_control)

    p = sub.add_parser("tune",
                       help="choose v2's decoder settings on the external games (for the tech doc)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="degradation seeds")
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    p.add_argument("--jobs", type=int, default=1, help="processes (one seed each)")
    p.add_argument("--out", help="write the Markdown table here (e.g. docs/TUNING.md)")
    p.add_argument("--csv", help="also write the per-run scores (keep it in .cache/)")
    p.set_defaults(func=_cmd_tune)

    p = sub.add_parser("ablate",
                       help="held-out ablation table on the external games (for the tech doc)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="degradation seeds")
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    p.add_argument("--draws", type=int, default=1000, help="paired bootstrap draws")
    p.add_argument("--out", help="write the Markdown table here (e.g. docs/ABLATION.md)")
    p.add_argument("--csv", help="also write the per-(row, seed, game) scores (keep it in .cache/)")
    p.set_defaults(func=_cmd_ablate)
    p = sub.add_parser("train-seq",
                       help="train the v3 sequence model on the degraded Metrica games (torch)")
    p.add_argument("--out", default=None, help="model JSON (default: .cache/seq_local.json; the shipped "
                   "ensemble is trained on Kaggle by scripts/kaggle_seq.py)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                   help="degradation seeds")
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    p.add_argument("--touch", action="store_true", help="add the touch inputs")
    p.set_defaults(func=_cmd_train_seq)

    p = sub.add_parser("dfl-inputs",
                       help="download the open DFL matches and cache their sequence-model inputs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="degradation seeds")
    p.add_argument("--labels", default="control", choices=("flag", "control"),
                   help="DFL's per-touch flag, or the flag filtered by control events")
    p.add_argument("--matches", nargs="+", default=None, help="default: all seven")
    p.add_argument("--touch", action="store_true", help="add the touch inputs (files *_t)")
    p.set_defaults(func=_cmd_dfl_inputs)

    p = sub.add_parser("skillcorner-inputs",
                       help="download the SkillCorner open matches and cache their "
                            "sequence-model inputs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="degradation seeds")
    p.add_argument("--labels", default="phase", choices=("phase", "control"),
                   help="SkillCorner's phases of play, or with the one-touch rule on top")
    p.add_argument("--matches", nargs="+", type=int, default=None, help="default: all twenty")
    p.add_argument("--touch", action="store_true", help="add the touch inputs (files *_t)")
    p.set_defaults(func=_cmd_skillcorner_inputs)

    p = sub.add_parser("tune-seq",
                       help="held-out decoder choice for v3 on the external games (torch)")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="evaluation seeds")
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3], choices=(1, 2, 3))
    p.add_argument("--out", help="write the Markdown table here (e.g. docs/TUNING_V3.md)")
    p.add_argument("--csv", help="also write the per-run scores (keep it in .cache/)")
    p.add_argument("--logprobs", nargs="+", default=None,
                   help="score these held-out log-prob files (scripts/kaggle_seq.py), averaged "
                        "as an ensemble, instead of training locally")
    p.add_argument("--note", default=None,
                   help="with --logprobs: the Model line of the report (what made the files)")
    p.set_defaults(func=_cmd_tune_seq)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train-control" and args.out is None:
        from .stages.control import DEFAULT_MODEL
        args.out = DEFAULT_MODEL
    if args.command == "train-seq" and args.out is None:
        from .seq.network import DEFAULT_MODEL
        args.out = DEFAULT_MODEL
    if getattr(args, "out", None):
        folder = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(folder, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
