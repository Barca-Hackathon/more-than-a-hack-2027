"""Robustness audits of v3 as frozen on Nov 12: where could it fail on BAR-ATH in ways the held-out score
does not show?

    uv run more-than-a-hack-2027 robust --audits bytes anchors --out docs/ROBUSTNESS.md

Every audit is GT-free on BAR-ATH: no label of the competition match is read, public or private.
Each BAR-ATH number comes with the same measurement on matches whose networks never saw them, so
it has something to be read against:

  Metrica   games 1-3, degradation seeds 0-2, with the held-out log-probabilities of the shipped
            recipe (.cache/seq_heldout/lp_seq_v3_*.npz) or, where the networks must be re-run on
            changed inputs, the held-out networks themselves
            (.cache/kaggle/out/v3-heldout-nov12-*/seq_v3_*_g*.json, eight per game);
  DFL       the seven matches, each predicted by the four `x-own` networks whose fold held it out
            (scripts/kaggle_seq.py, DFL_FOLDS), with the ball shown while dead as in the ATD.

The audits (AUDITS): bytes, margin, anchors, inputs, perturb, support, ridge. Each writes its rows
to .cache/robust/<audit>.json; `markdown()` renders whatever is there, so one audit can be re-run
without the others. Nothing here changes a prediction: fixes are only scored. The one that
cleared the rule, the kickoff-wait rule, ships in pipeline.run_v3 since Nov 13; the audits decode
without it (decode_lp), so the report keeps showing the failure and what the rule does to it.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace

import numpy as np

from .. import pipeline
from ..seq import network, training
from ..stages import control, decode
from . import harness
from .._paths import CACHE_DIR, ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402

ROBUST_DIR = os.path.join(CACHE_DIR, "robust")
KAGGLE_OUT = os.path.join(CACHE_DIR, "kaggle", "out")
HELDOUT_NETS = os.path.join(KAGGLE_OUT, "v3-heldout-nov12-*", "seq_v3_*_g{game}.json")
HELDOUT_LPS = os.path.join(CACHE_DIR, "seq_heldout", "lp_seq_v3_{member}.npz")
DFL_LPS = os.path.join(KAGGLE_OUT, "xown", "lp_x_own_s*_extra.npz")
MEMBERS = ("own",) + tuple("s{}".format(i) for i in range(7))     # the shipped files' path order
GAMES = (1, 2, 3)
SEEDS = (0, 1, 2)
FPS = 25

AUDITS = ("bytes", "margin", "anchors", "inputs", "perturb", "support", "ridge")

DEAD_NAMES = network.DEAD_NAMES
ANCHOR = network.dead_channel("anchor")
MASK = network.dead_channel("mask")             # the dead-ball stage's own live/dead call (+-1)
WAIT_MIN_S = pipeline.V3_DEFAULTS["wait_min_s"]


def channel_names():
    """The 123 input channels, in network.inputs' order."""
    ball = ["seen", "x", "y", "vx", "vy", "speed", "accel", "outside"]
    team = []
    for k in range(network.K_NEAREST):
        team += ["near{}_{}".format(k + 1, c) for c in "abcdef"]
    team += ["within_2m", "within_5m", "within_10m"] + ["shape_{}".format(c) for c in "abcde"]
    names = ["ball." + n for n in ball]
    names += ["home." + n for n in team] + ["away." + n for n in team]
    names += ["dead." + n for n in DEAD_NAMES]
    names += ["control." + n for n in control.feature_names()]
    for side in ("home", "away"):
        names += ["touch.{}_{}".format(side, i + 1) for i in range(network.TOUCH_BLOCK)]
    return names


def channel_group(name):
    head = name.split(".")[0]
    return "team blocks" if head in ("home", "away") else head


# -- matches and runs ----------------------------------------------------------------------------

@dataclass
class Run:
    """One match's inputs and member log-probabilities; truth is None on BAR-ATH."""
    name: str
    group: str
    X: np.ndarray
    graded: np.ndarray
    halves: tuple
    lps: list
    truth: object = None

    @property
    def lp(self):
        return average_halves(self.lps, self.halves)

    @property
    def seen(self):
        return self.X[:, network.BALL_SEEN] > 0.5


@dataclass
class Match:
    """A match with its tracking, for audits that change the input and re-run the networks."""
    name: str
    ctx: object
    models: list
    truth: object = None


def average_halves(lps, halves):
    """network.ensemble_logprobs without the forward pass: the members' mean inside the halves."""
    if len(lps) == 1:
        return lps[0]
    out = np.zeros_like(lps[0])
    for _, first, last in halves:
        part = slice(first, last + 1)
        out[part] = network.average_logprobs([lp[part] for lp in lps])
    return out


def decode_lp(lp, graded, halves, cfg=None, pin=None, pin_bonus=1000.0):
    """The networks' log-probabilities through v3's decoder, WITHOUT the kickoff-wait rule that
    pipeline.run_v3 applies since Nov 13: the audits measure the networks and the decoder, and
    the rule is one of the fixes they score. pin: (n,) bool frames forced to D (a fix under
    test; kickoff_wait_pin gives the shipped rule)."""
    top = dict(pipeline.V3_DEFAULTS, **(cfg or {}))    # decoder keys only, as seq/tuning.py
    evidence = np.where(graded[:, None], top["w_seq"] * lp, 0.0)
    if pin is not None:
        evidence[pin & graded, decode.D] += pin_bonus
    trans = decode.transition_matrix(top["switch"], top["stop"], top["restart"])
    return decode.decode_match(evidence, halves, trans,
                               min_dead=int(round(top["min_dead_s"] * FPS)))


def stage_inputs(ctx):
    """(dead-ball result, X, swap_sign) exactly as pipeline.run_v3 builds them."""
    from ..stages import deadball
    from ..stages.ball import clean_ball
    _, stages = pipeline.split_cfg(None, pipeline.V3_DEFAULTS)
    ball = clean_ball(ctx, stages["ball"])
    dead = deadball.detect(ctx, ball, stages["deadball"])
    X, sign = network.inputs(ctx, ball, dead, stages["control"], touch=True)
    return dead, X, sign


def member_logprobs(model, X, sign, halves, shift=0, mirror=False):
    """network.logprobs_halves with two changes of view: `shift` drops that many frames from the
    start of each half, so the 5 Hz grid falls on other frames (the dropped frames copy the first
    kept one); `mirror` flips the pitch left to right."""
    out = np.zeros((len(X), 3))
    Xs = network.swap(X, sign)
    flip = network.flip_sign(X.shape[1]).astype(X.dtype)
    for _, first, last in halves:
        part = slice(first + shift, last + 1)
        preds = []
        for V, swapped in ((X, False), (Xs, True)):
            p = model.forward(V[part] * flip if mirror else V[part])
            preds.append(p[:, [1, 0, 2]] if swapped else p)
        mean = np.mean(preds, axis=0)
        m = mean.max(axis=1, keepdims=True)
        out[part] = mean - m - np.log(np.exp(mean - m).sum(axis=1, keepdims=True))
        out[first:first + shift] = out[first + shift]
    return out


def _cached(key, compute):
    os.makedirs(ROBUST_DIR, exist_ok=True)
    path = os.path.join(ROBUST_DIR, "lp_{}.npy".format(key))
    if os.path.exists(path):
        return list(np.load(path))
    lps = compute()
    np.save(path, np.stack(lps))
    return lps


def predict_members(match, X, sign, key, shift=0, mirror=False):
    """The members' log-probabilities on (possibly changed) inputs, cached under `key`."""
    name = "{}_{}".format(match.name.replace(" ", "").replace("-", ""), key)
    return _cached(name, lambda: [member_logprobs(m, X, sign, match.ctx.halves, shift, mirror)
                                  for m in match.models])


def barath_match():
    from ..data.context import load_context
    return Match("BAR-ATH", load_context(), network.load_ensemble())


def heldout_models(game):
    paths = sorted(glob.glob(HELDOUT_NETS.format(game=game)), key=os.path.basename)
    if len(paths) != len(MEMBERS):
        raise FileNotFoundError("expected {} held-out networks for game {}, found {}".format(
            len(MEMBERS), game, len(paths)))
    return [network.SeqModel.from_json(p) for p in paths]


def metrica_match(game, seed=0):
    eg = harness.load_games((game,), ("degraded",), degrade_cfg={"seed": seed})[(game, "degraded")]
    return Match("Metrica {}".format(game), eg.ctx, heldout_models(game), eg.labels)


def barath_run(match=None):
    match = match or barath_match()
    _, X, sign = stage_inputs(match.ctx)
    lps = predict_members(match, X, sign, "base")
    return Run("BAR-ATH", "BAR-ATH", X, match.ctx.half_of_frame > 0, match.ctx.halves, lps)


def metrica_runs(games=GAMES, seeds=SEEDS):
    files = [np.load(HELDOUT_LPS.format(member=m)) for m in MEMBERS]
    for s in seeds:
        for g in games:
            key = "s{}_g{}".format(s, g)
            X, _, labels, graded, halves = training.load_cached(training.input_name(key, True))
            yield Run("Metrica {} seed {}".format(g, s), "Metrica", X, graded, halves,
                      [f[key] for f in files], labels)


def dfl_runs(seed="_s0_"):
    """Held-out DFL matches (ATD view, one degradation seed), four x-own networks each."""
    files = [np.load(p) for p in sorted(glob.glob(DFL_LPS))]
    if not files:
        return
    keys = sorted(set.intersection(*(set(f.files) for f in files)))
    for key in keys:
        if not key.startswith("dflcv_") or seed not in key + "_":
            continue
        X, _, labels, graded, halves = training.load_cached(key + ".npz")
        yield Run("DFL " + key.split("_")[1], "DFL", X, graded, halves, [f[key] for f in files],
                  labels)


def agreement(reference, labels):
    """spellF1 of `labels` scored against `reference` labels as if they were the truth."""
    return harness.score(pm.label_codes(np.asarray(reference)), labels)["spell_f1_mean"]


def describe(labels, graded, seen):
    live = np.isin(labels, ("H", "A"))
    dead = labels == "D"
    codes = pm.label_codes(labels)
    return {"possessions": len(pm.possession_spells(codes, pm.valid_runs(codes))),
            "dead_share": float(dead[graded].mean()),
            "dead_where_seen": float(dead[graded & seen].mean()),
            "live_unseen_frames": int((live & graded & ~seen).sum())}


def _save(audit, payload):
    os.makedirs(ROBUST_DIR, exist_ok=True)
    with open(os.path.join(ROBUST_DIR, audit + ".json"), "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return payload


def _load(audit):
    path = os.path.join(ROBUST_DIR, audit + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


# -- audit: bytes --------------------------------------------------------------------------------

BLAS_ENVS = (
    ("default", {}),
    ("1 thread", {"OPENBLAS_NUM_THREADS": "1"}),
    ("4 threads", {"OPENBLAS_NUM_THREADS": "4"}),
    ("Haswell kernels (AVX2)", {"OPENBLAS_CORETYPE": "Haswell"}),
    ("SkylakeX kernels (AVX-512)", {"OPENBLAS_CORETYPE": "SkylakeX"}),
    ("Zen kernels", {"OPENBLAS_CORETYPE": "Zen"}),
    ("Sandybridge kernels (AVX)", {"OPENBLAS_CORETYPE": "Sandybridge"}),
    ("Nehalem kernels (SSE)", {"OPENBLAS_CORETYPE": "Nehalem"}),
)


def audit_bytes(log=print):
    """`predict` in a fresh process under other BLAS thread counts and CPU kernels: does the
    submission come out byte for byte the same? OpenBLAS picks its kernels by CPU, so another
    machine may sum in another order."""
    out_dir = os.path.join(ROBUST_DIR, "blas")
    os.makedirs(out_dir, exist_ok=True)
    rows, reference = [], None
    for i, (name, env) in enumerate(BLAS_ENVS):
        path = os.path.join(out_dir, "run{}.csv".format(i))
        subprocess.run([sys.executable, "-c", "from more_than_a_hack_2027.cli import main; main()",
                        "predict", "--out", path],
                       env=dict(os.environ, **env), check=True, capture_output=True)
        with open(path, "rb") as handle:
            data = handle.read()
        labels = np.array([line.rsplit(b",", 1)[1] for line in data.split(b"\n")[1:] if line])
        reference = labels if reference is None else reference
        rows.append({"run": name, "sha256": hashlib.sha256(data).hexdigest(),
                     "frames_changed": int((labels != reference).sum())})
        log("bytes: {} {}".format(name, rows[-1]["sha256"][:12]))
    return _save("bytes", {"rows": rows})


# -- audit: margin -------------------------------------------------------------------------------

NOISE = (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2)
ROUND_DECIMALS = (6, 4, 3, 2)


def audit_margin(run=None, draws=20, log=print):
    """How far is the decoded path from a tie? Gaussian noise of a given size on the ensemble's
    log-probabilities, and the candidate fix (rounding them before decoding)."""
    run = run or barath_run()
    lp = run.lp
    base = decode_lp(lp, run.graded, run.halves)
    rng = np.random.RandomState(0)
    noise_rows = []
    for sd in NOISE:
        changed = [int((decode_lp(lp + rng.normal(0.0, sd, lp.shape), run.graded, run.halves)
                        != base).sum()) for _ in range(draws)]
        noise_rows.append({"sd": sd, "draws": draws, "draws_changed": int(np.sum(np.array(changed) > 0)),
                           "max_frames_changed": int(max(changed))})
        log("margin: sd {:g} -> {} of {} draws change".format(sd, noise_rows[-1]["draws_changed"],
                                                               draws))
    round_rows = []
    for d in ROUND_DECIMALS:
        rounded = decode_lp(np.round(lp, d), run.graded, run.halves)
        # after rounding, does last-bit noise still reach the path?
        changed = [int((decode_lp(np.round(lp + rng.normal(0.0, 1e-9, lp.shape), d), run.graded,
                                  run.halves) != rounded).sum()) for _ in range(draws)]
        round_rows.append({"decimals": d, "frames_changed_vs_frozen": int((rounded != base).sum()),
                           "draws_changed_under_1e-9": int(np.sum(np.array(changed) > 0)),
                           "agreement": agreement(base, rounded)})
    return _save("margin", {"noise": noise_rows, "rounding": round_rows})


# -- audit: anchors ------------------------------------------------------------------------------

def _runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def anchor_row(name, labels, anchors, graded, truth=None):
    """How v3 labels the frames the dead-ball stage holds certainly dead."""
    anchors = anchors & graded
    live = np.isin(labels, ("H", "A"))
    spans = _runs(anchors)
    row = {"match": name, "anchor_frames": int(anchors.sum()), "anchor_runs": len(spans),
           "live_share": float(live[anchors].mean()) if anchors.any() else 0.0,
           "runs_mostly_live": int(sum(live[a:b].mean() > 0.5 for a, b in spans)),
           "runs_any_live": int(sum(live[a:b].any() for a, b in spans))}
    if truth is not None:
        row["truly_dead_share"] = float((truth[anchors] == pm.D).mean())
        truly = anchors & (truth == pm.D)
        row["live_share_truly_dead"] = float(live[truly].mean()) if truly.any() else 0.0
    return row


def anchors_by_kind(dead, labels, graded):
    """BAR-ATH only: the anchor frames split by the kind of stoppage that holds them."""
    kind = np.full(len(labels), "outside any stoppage", dtype=object)
    for s in dead.stoppages:
        kind[s.start:s.restart] = s.kind
    anchors = dead.anchors & graded
    live = np.isin(labels, ("H", "A"))
    rows = []
    for k in sorted(set(kind[anchors])):
        m = anchors & (kind == k)
        rows.append({"kind": k, "anchor_frames": int(m.sum()), "live_frames": int(live[m].sum()),
                     "live_share": float(live[m].mean())})
    return rows


def stoppage_disagreement(run, labels):
    """Where v3 and the dead-ball stage disagree about a whole stoppage, and (with labels) who
    is right. A stage stoppage is `ridden through` when v3 calls most of it live; a v3 stoppage
    is `v3 only` when the stage calls most of it live."""
    stage = (run.X[:, MASK] > 0) & run.graded
    live = np.isin(labels, ("H", "A"))
    spans = [(a, b) for a, b in _runs(stage) if b - a >= FPS]
    ridden = [(a, b) for a, b in spans if live[a:b].mean() > 0.5]
    mine = [(a, b) for a, b in _runs((labels == "D") & run.graded)]
    v3_only = [(a, b) for a, b in mine if stage[a:b].mean() < 0.5]
    row = {"group": run.group, "match": run.name, "stage_stoppages": len(spans),
           "ridden": len(ridden), "v3_stoppages": len(mine), "v3_only": len(v3_only),
           "ridden_seen": int(sum(run.seen[a:b].mean() > 0.5 for a, b in ridden)),
           "ridden_seconds": float(np.median([(b - a) / FPS for a, b in ridden])) if ridden
           else 0.0}
    if run.truth is not None:
        def dead(spans_):
            return int(sum((run.truth[a:b] == pm.D).mean() > 0.5 for a, b in spans_))
        kept = [span for span in spans if span not in ridden]
        row.update(ridden_truly_dead=dead(ridden), kept_truly_dead=dead(kept),
                   v3_only_truly_dead=dead(v3_only))
    return row


def barath_checks(ctx, dead, labels):
    """BAR-ATH only: stoppages v3 rides through by kind, set-piece tags with a restart nearby
    (deadball.summary's window), and how each half begins."""
    from ..data.tagging import load_tags
    fps = ctx.fps
    live = np.isin(labels, ("H", "A"))
    seen = ctx.ball_seen
    kinds = {}
    for s in dead.stoppages:
        if s.half > 0 and s.restart - s.start >= fps:
            k = kinds.setdefault(s.kind, {"kind": s.kind, "stoppages": 0, "ridden": 0,
                                          "ridden_ball_seen": 0})
            k["stoppages"] += 1
            if live[s.start:s.restart].mean() > 0.5:
                k["ridden"] += 1
                k["ridden_ball_seen"] += int(seen[s.start:s.restart].mean() > 0.5)
    set_pieces = [t for t in load_tags(ctx) if t.code == "SET PIECES"]
    restarts = np.flatnonzero((labels[:-1] == "D") & live[1:]) + 1
    stage_restarts = np.array(sorted(s.restart for s in dead.stoppages))

    def covered(frames):
        return int(sum(np.searchsorted(frames, t.start + 15 * fps + 1)
                       > np.searchsorted(frames, t.start - 10 * fps) for t in set_pieces))
    halves = []
    for half, first, last in ctx.halves:
        first_live = first + int(np.argmax(live[first:last + 1]))
        # A kickoff long after the graded start is not flagged at_half_start; the half_start
        # stoppage that ends on it is.
        kick = ([k.frame for k in dead.kickoffs if k.half == half and k.at_half_start]
                or [s.restart for s in dead.stoppages
                    if s.half == half and s.kind == "half_start"])
        tail = last + 1 - (first + int(np.flatnonzero(live[first:last + 1])[-1]) + 1)
        halves.append({"half": int(half), "first_live_s": (first_live - first) / fps,
                       "kickoff_s": (kick[0] - first) / fps if kick else None,
                       "dead_tail_s": tail / fps})
    return {"kinds": [kinds[k] for k in sorted(kinds)],
            "set_pieces": len(set_pieces), "set_pieces_v3": covered(restarts),
            "set_pieces_stage": covered(stage_restarts), "halves": halves}


def kickoff_wait_pin(X, graded, halves, min_s=WAIT_MIN_S):
    """The frames pipeline.run_v3 pins to D (network.kickoff_wait)."""
    return network.kickoff_wait(X, graded, halves, FPS, min_s)


def training_only_runs():
    """SkillCorner matches (ATD view, seed 0) as inputs and labels only: no held-out networks
    exist for them, but a rule that reads only the inputs can still be checked against labels."""
    for path in sorted(glob.glob(os.path.join(training.CACHE_DIR, "sccv_*_s0_t.npz"))):
        X, _, labels, graded, halves = training.load_cached(os.path.basename(path))
        yield Run("SkillCorner " + os.path.basename(path).split("_")[1], "SkillCorner", X, graded,
                  halves, [], labels)


def audit_anchors(log=print):
    """Does v3 call live play where the ball is certainly dead (deadball.build_anchors: a
    validated ball-out, the wait before a kickoff, a goal until its kickoff)? And what would
    pinning those frames to D do, on BAR-ATH and held out?"""
    match = barath_match()
    dead, X, sign = stage_inputs(match.ctx)
    bar = Run("BAR-ATH", "BAR-ATH", X, match.ctx.half_of_frame > 0, match.ctx.halves,
              predict_members(match, X, sign, "base"))
    rows, fix, stoppages, waits, wait_fix = [], [], [], [], []
    kinds = checks = None

    def wait_rows(run):
        for first, stop, peak in network.kickoff_waits(run.X, run.graded, run.halves):
            row = {"group": run.group, "match": run.name, "seconds": (stop - first) / FPS,
                   "peak": peak, "ball_seen": float(run.seen[first:stop].mean()),
                   "pinned": bool(network.wait_is_pinned(first, stop, peak, FPS, WAIT_MIN_S))}
            if run.truth is not None:
                row["truly_dead"] = float((run.truth[first:stop] == pm.D).mean())
            if run.lps:
                labels = decode_lp(run.lp, run.graded, run.halves)
                row["v3_live"] = float(np.isin(labels[first:stop], ("H", "A")).mean())
                row["v3_possessions"] = len(spells(labels[first:stop]))
            waits.append(row)
    for run in [bar, *metrica_runs(), *dfl_runs()]:
        lp = run.lp
        anchors = run.X[:, ANCHOR] > 0.5
        base = decode_lp(lp, run.graded, run.halves)
        pinned = decode_lp(lp, run.graded, run.halves, pin=anchors)
        rows.append({"group": run.group, **anchor_row(run.name, base, anchors, run.graded,
                                                      run.truth)})
        entry = {"group": run.group, "match": run.name,
                 "frames_changed": int((pinned != base).sum()),
                 "agreement": agreement(base, pinned),
                 "base": describe(base, run.graded, run.seen),
                 "pinned": describe(pinned, run.graded, run.seen)}
        if run.truth is not None:
            entry["spell_f1_base"] = harness.score(run.truth, base)["spell_f1_mean"]
            entry["spell_f1_pinned"] = harness.score(run.truth, pinned)["spell_f1_mean"]
        else:
            kinds = anchors_by_kind(dead, base, run.graded)
            checks = barath_checks(match.ctx, dead, base)
        fix.append(entry)
        stoppages.append(stoppage_disagreement(run, base))
        wait_rows(run)
        waited = decode_lp(lp, run.graded, run.halves,
                           pin=kickoff_wait_pin(run.X, run.graded, run.halves))
        entry = {"group": run.group, "match": run.name,
                 "frames_changed": int((waited != base).sum()),
                 "possessions": [describe(base, run.graded, run.seen)["possessions"],
                                 describe(waited, run.graded, run.seen)["possessions"]]}
        if run.truth is not None:
            for key in ("spell_f1_mean", "macro_f1"):
                entry[key] = [harness.score(run.truth, base)[key],
                              harness.score(run.truth, waited)[key]]
            for rules, name in ((("loss",), "alt"),):
                if run.group == "Metrica":
                    game = int(run.name.split()[1])
                    alt = _alt_labels(game)
                    entry["spell_f1_alt"] = [harness.score(alt, base)["spell_f1_mean"],
                                             harness.score(alt, waited)["spell_f1_mean"]]
        wait_fix.append(entry)
        log("anchors: {} live share {:.3f}".format(run.name, rows[-1]["live_share"]))
    for run in training_only_runs():
        wait_rows(run)
    return _save("anchors", {"rows": rows, "kinds": kinds, "fix": fix, "stoppages": stoppages,
                             "barath": checks, "waits": waits, "wait_fix": wait_fix})


_ALT = {}


def _alt_labels(game):
    """A Metrica game's labels under the alt reading (one-touch recoveries count)."""
    from ..external import metrica
    if game not in _ALT:
        _ALT[game] = metrica.load_game(game, rules=("loss",)).labels
    return _ALT[game]


# -- audit: inputs -------------------------------------------------------------------------------

def training_samples(step=10):
    """{match: (source, graded rows of X, every `step`th)} for one degradation seed of every
    training match: Metrica, DFL and SkillCorner (both with the ATD's dead-ball view)."""
    out = {}
    patterns = (("Metrica", "s0_g[123]_t.npz"), ("DFL", "dflcv_*_s0_t.npz"),
                ("SkillCorner", "sccv_*_s0_t.npz"))
    for source, pattern in patterns:
        for path in sorted(glob.glob(os.path.join(training.CACHE_DIR, pattern))):
            with np.load(path) as z:
                out[os.path.basename(path)[:-len("_t.npz")]] = (
                    source, z["X"][z["graded"]][::step].astype(np.float64))
    return out


def _auc(sorted_pool, values):
    """P(value > pool draw) + half the ties, per column."""
    out = np.empty(values.shape[1])
    for c in range(values.shape[1]):
        lo = np.searchsorted(sorted_pool[:, c], values[:, c], side="left")
        hi = np.searchsorted(sorted_pool[:, c], values[:, c], side="right")
        out[c] = (lo + hi).mean() / (2.0 * len(sorted_pool))
    return out


def input_shift(target, samples, lo_hi=(0.1, 99.9)):
    """Per channel, `target` (rows, C) against the pooled training samples: the share of its
    rows outside the pool's central range, and how well the channel alone tells it from the pool
    (|AUC - 0.5|). The same for every training match against the others, as the baseline."""
    names = sorted(samples)
    pool = np.concatenate([samples[n][1] for n in names])
    lo, hi = np.percentile(pool, lo_hi, axis=0)
    out = {"lo": lo, "hi": hi,
           "outside": ((target < lo) | (target > hi)).mean(axis=0),
           "auc": np.abs(_auc(np.sort(pool, axis=0), target) - 0.5),
           "base_outside": {}, "base_auc": {}}
    for n in names:
        rest = np.concatenate([samples[m][1] for m in names if m != n])
        r_lo, r_hi = np.percentile(rest, lo_hi, axis=0)
        x = samples[n][1]
        out["base_outside"][n] = ((x < r_lo) | (x > r_hi)).mean(axis=0)
        out["base_auc"][n] = np.abs(_auc(np.sort(rest, axis=0), x) - 0.5)
    return out


def audit_inputs(step=10, log=print):
    """Which of the 123 input channels put BAR-ATH outside what the networks trained on?"""
    bar = barath_run()
    samples = training_samples(step)
    scale = network.load_ensemble()[0].scale
    samples = {k: (src, x / scale) for k, (src, x) in samples.items()}
    target = bar.X[bar.graded][::step].astype(np.float64) / scale
    shift = input_shift(target, samples)
    names = channel_names()
    base_out = np.array([shift["base_outside"][n] for n in sorted(samples)])
    base_auc = np.array([shift["base_auc"][n] for n in sorted(samples)])
    sources = [samples[n][0] for n in sorted(samples)]
    rows = []
    for c, name in enumerate(names):
        rows.append({"channel": name, "group": channel_group(name),
                     "outside": float(shift["outside"][c]),
                     "outside_train_max": float(base_out[:, c].max()),
                     "outside_train_median": float(np.median(base_out[:, c])),
                     "auc": float(shift["auc"][c]),
                     "auc_train_max": float(base_auc[:, c].max()),
                     "auc_train_median": float(np.median(base_auc[:, c])),
                     "lo": float(shift["lo"][c]), "hi": float(shift["hi"][c])})
    by_source = {}
    for source in sorted(set(sources)):
        pick = [i for i, s in enumerate(sources) if s == source]
        by_source[source] = {"matches": len(pick),
                             "outside_mean": float(base_out[pick].mean()),
                             "auc_mean": float(base_auc[pick].mean())}
    by_source["BAR-ATH"] = {"matches": 1, "outside_mean": float(shift["outside"].mean()),
                            "auc_mean": float(shift["auc"].mean())}
    log("inputs: BAR-ATH mean outside {:.4f}, mean |AUC-0.5| {:.3f}".format(
        by_source["BAR-ATH"]["outside_mean"], by_source["BAR-ATH"]["auc_mean"]))
    return _save("inputs", {"rows": rows, "by_source": by_source,
                            "training_matches": len(samples)})


# -- audit: perturb ------------------------------------------------------------------------------

def jitter(ctx, seed, ball_m=0.0, player_m=0.0):
    """Independent per-frame Gaussian noise on the ball and the player positions."""
    rng = np.random.RandomState(seed)
    ball = ctx.ball + rng.normal(0.0, ball_m, ctx.ball.shape) if ball_m else ctx.ball
    xy = ctx.xy + rng.normal(0.0, player_m, ctx.xy.shape) if player_m else ctx.xy
    return replace(ctx, ball=ball, xy=xy)


def permute_slots(ctx, seed):
    """The same tracks in another slot order: no input should depend on it."""
    perm = np.random.RandomState(seed).permutation(ctx.n_slots)
    return replace(ctx, xy=np.asarray(ctx.xy)[:, perm], speed=np.asarray(ctx.speed)[:, perm],
                   team_of_slot=np.asarray(ctx.team_of_slot)[perm])


def drop_more(ctx, seed, k):
    """k more of the visible players missing on every frame (degrade.drop_players' sticky choice)."""
    from ..external import degrade
    rng = np.random.RandomState(seed)
    xy, speed = degrade.drop_players(np.asarray(ctx.xy, dtype=np.float64),
                                     np.asarray(ctx.speed, dtype=np.float32),
                                     np.full(ctx.n_frames, k), rng, 35 * ctx.fps)
    return replace(ctx, xy=xy, speed=speed)


def flicker(ctx, seed, every_s=10.0, frames=3):
    """The ball lost for `frames` frames about once every `every_s` seconds."""
    rng = np.random.RandomState(seed)
    starts = np.flatnonzero(rng.uniform(size=ctx.n_frames) < 1.0 / (every_s * ctx.fps))
    ball = np.array(ctx.ball, dtype=np.float64)
    for s in starts:
        ball[s:s + frames] = np.nan
    return replace(ctx, ball=ball)


def cap_visible(X):
    """The two visible-player channels held at 11 players, the most a training match shows. The
    ATD sometimes tracks 12 or more for one team (the inputs audit)."""
    names = channel_names()
    X = X.copy()
    for side in ("home", "away"):
        c = names.index(side + ".shape_e")
        X[:, c] = np.minimum(X[:, c], 1.0)
    return X


# name -> (kind, argument): "ctx" changes the tracking, "view" the networks' view of the inputs,
# "inputs" the input channels themselves
PERTURBATIONS = (
    ("track slots shuffled", ("ctx", lambda ctx: permute_slots(ctx, 1))),
    ("pitch mirrored left to right", ("view", {"mirror": True})),
    ("5 Hz grid shifted by 1 frame", ("view", {"shift": 1})),
    ("5 Hz grid shifted by 2 frames", ("view", {"shift": 2})),
    ("5 Hz grid shifted by 3 frames", ("view", {"shift": 3})),
    ("5 Hz grid shifted by 4 frames", ("view", {"shift": 4})),
    ("ball jitter 0.05 m", ("ctx", lambda ctx: jitter(ctx, 2, ball_m=0.05))),
    ("ball jitter 0.1 m", ("ctx", lambda ctx: jitter(ctx, 2, ball_m=0.1))),
    ("ball jitter 0.2 m", ("ctx", lambda ctx: jitter(ctx, 2, ball_m=0.2))),
    ("player jitter 0.05 m", ("ctx", lambda ctx: jitter(ctx, 3, player_m=0.05))),
    ("player jitter 0.1 m", ("ctx", lambda ctx: jitter(ctx, 3, player_m=0.1))),
    ("1 more player missing", ("ctx", lambda ctx: drop_more(ctx, 4, 1))),
    ("2 more players missing", ("ctx", lambda ctx: drop_more(ctx, 4, 2))),
    ("ball flicker, 3 frames every 10 s", ("ctx", lambda ctx: flicker(ctx, 5))),
    ("visible players per team capped at 11", ("inputs", lambda X: cap_visible(X))),
)
PHASE_AVERAGE = "average of the 5 grid phases (candidate fix)"


def step_stats(ctx):
    """Frame-to-frame jitter of the tracking as it is: median step of the ball and the players,
    and the median second difference (what independent per-frame noise inflates most)."""
    graded = ctx.half_of_frame > 0
    ball = np.where(graded[:, None], ctx.ball, np.nan)
    xy = np.where(graded[:, None, None], ctx.xy, np.nan)

    def med(a, order):
        d = np.diff(a, n=order, axis=0)
        return float(np.nanmedian(np.hypot(d[..., 0], d[..., 1])))
    return {"ball_step": med(ball, 1), "ball_second_diff": med(ball, 2),
            "player_step": med(xy, 1), "player_second_diff": med(xy, 2)}


def perturb_match(match, names=None, log=print):
    """One match under every perturbation: agreement with its own unperturbed output, and the
    true score where there are labels."""
    ctx = match.ctx
    graded = ctx.half_of_frame > 0
    _, X, sign = stage_inputs(ctx)
    base_lps = predict_members(match, X, sign, "base")
    base = decode_lp(average_halves(base_lps, ctx.halves), graded, ctx.halves)
    seen = X[:, network.BALL_SEEN] > 0.5

    def row(name, labels, seen_now):
        out = {"match": match.name, "perturbation": name, "agreement": agreement(base, labels),
               "frames_changed": float((labels != base)[graded].mean()),
               **describe(labels, graded, seen_now)}
        if match.truth is not None:
            out["spell_f1"] = harness.score(match.truth, labels)["spell_f1_mean"]
        log("perturb: {} | {} | agreement {:.3f}".format(match.name, name, out["agreement"]))
        return out

    rows = [row("none", base, seen)]
    phases = [average_halves(base_lps, ctx.halves)]
    for i, (name, (kind, arg)) in enumerate(PERTURBATIONS):
        if names is not None and name not in names:
            continue
        if kind == "ctx":
            _, Xp, sign_p = stage_inputs(arg(ctx))
            lps = predict_members(match, Xp, sign_p, "p{}".format(i))
            seen_now = Xp[:, network.BALL_SEEN] > 0.5
        elif kind == "inputs":
            lps = predict_members(match, arg(X), sign, "p{}".format(i))
            seen_now = seen
        else:
            lps = predict_members(match, X, sign, "p{}".format(i), **arg)
            seen_now = seen
        lp = average_halves(lps, ctx.halves)
        if "shift" in (arg if kind == "view" else {}):
            phases.append(lp)
        rows.append(row(name, decode_lp(lp, graded, ctx.halves), seen_now))
    if len(phases) == 5:
        lp = average_halves(phases, ctx.halves)
        rows.append(row(PHASE_AVERAGE, decode_lp(lp, graded, ctx.halves), seen))
    return rows, step_stats(ctx)


def audit_perturb(games=GAMES, log=print):
    """Is BAR-ATH more fragile than a held-out game? The same input changes on both."""
    rows, steps = [], {}
    for make in [barath_match] + [lambda g=g: metrica_match(g) for g in games]:
        match = make()
        match_rows, steps[match.name] = perturb_match(match, log=log)
        rows += match_rows
        _save("perturb", {"rows": rows, "steps": steps})      # keep partial results
    return _save("perturb", {"rows": rows, "steps": steps})


# -- audit: support ------------------------------------------------------------------------------

def spells(labels):
    codes = pm.label_codes(np.asarray(labels))
    return pm.possession_spells(codes, pm.valid_runs(codes))


def matched(reference_spells, other_spells, threshold=pm.SPELL_IOU_HEADLINE):
    """Per reference spell: is it matched one-to-one by a spell of `other` at this IoU?"""
    overlaps = pm.spell_overlaps(reference_spells, other_spells)
    return pm.matched_gt_flags(overlaps, len(reference_spells), threshold)


def support(run):
    """(ensemble spells, per spell how many members reproduce it when decoded alone)."""
    mine = spells(decode_lp(run.lp, run.graded, run.halves))
    votes = np.zeros(len(mine), dtype=int)
    for lp in run.lps:
        votes += matched(mine, spells(decode_lp(lp, run.graded, run.halves)))
    return mine, votes


def audit_support(log=print):
    """How sure is each predicted possession? Member support, calibrated to precision on the
    held-out Metrica games, gives a GT-free expected precision for BAR-ATH."""
    n = len(MEMBERS)
    hits = np.zeros(n + 1)
    count = np.zeros(n + 1)
    for run in metrica_runs():
        mine, votes = support(run)
        truth = pm.possession_spells(run.truth, pm.valid_runs(run.truth))
        correct = matched(mine, truth)
        np.add.at(count, votes, 1)
        np.add.at(hits, votes, correct)
        log("support: {} done".format(run.name))
    bar = barath_run()
    mine, votes = support(bar)
    bar_count = np.bincount(votes, minlength=n + 1).astype(float)
    precision = hits / np.maximum(count, 1)
    # pool the thin low-support levels so each calibration cell has spells in it
    low = votes <= n // 2
    low_precision = hits[:n // 2 + 1].sum() / max(count[:n // 2 + 1].sum(), 1)
    expected = (low.sum() * low_precision + sum(
        bar_count[k] * precision[k] for k in range(n // 2 + 1, n + 1))) / len(mine)
    order = np.argsort(votes, kind="stable")
    weakest = [{"first_frame": int(mine[i][0]), "seconds": (mine[i][1] - mine[i][0]) / FPS,
                "team": "H" if mine[i][2] == pm.H else "A", "members": int(votes[i])}
               for i in order[:20]]
    longest = [{"first_frame": int(a), "seconds": (b - a) / FPS,
                "team": "H" if k == pm.H else "A", "members": int(votes[mine.index((a, b, k))])}
               for a, b, k in sorted(mine, key=lambda s: s[0] - s[1])[:10]]
    durations = np.array([(b - a) / FPS for a, b, _ in mine])
    return _save("support", {
        "levels": [{"members": k, "metrica_spells": int(count[k]),
                    "metrica_precision": float(precision[k]),
                    "metrica_share": float(count[k] / count.sum()),
                    "barath_spells": int(bar_count[k]),
                    "barath_share": float(bar_count[k] / bar_count.sum())}
                   for k in range(n + 1)],
        "metrica_precision": float(hits.sum() / count.sum()),
        "barath_expected_precision": float(expected),
        "barath_spells": len(mine),
        "barath_durations": {"under_2s": int((durations < 2).sum()),
                             "over_60s": int((durations > 60).sum()),
                             "median": float(np.median(durations))},
        "weakest": weakest, "longest": longest})


# -- audit: ridge --------------------------------------------------------------------------------

def ridge_configs(within=0.004):
    """The decoder settings whose held-out criterion is within `within` of the chosen one's
    (seq/tuning.py's runs, .cache/seq_heldout/tune_v3.csv)."""
    import pandas as pd
    from ..seq import tuning
    runs = pd.read_csv(os.path.join(CACHE_DIR, "seq_heldout", "tune_v3.csv"))
    runs = runs[runs["labels"].isin(("main", "alt"))]
    score = runs.groupby("config")["spell_f1_mean"].mean()
    grid = tuning.grid()
    top = pipeline.V3_DEFAULTS
    chosen = next(k for k, c in grid.items() if all(c[f] == top[f] for f in c))
    keep = score[score >= score[chosen] - within].sort_values(ascending=False)
    return chosen, [(k, grid[k], float(v)) for k, v in keep.items()]


def audit_ridge(log=print):
    """Does BAR-ATH's output depend on which of the equally good decoder settings was chosen,
    more than a held-out game's does?"""
    chosen, configs = ridge_configs()
    rows = []
    for run in [barath_run(), *metrica_runs(seeds=(0,))]:
        lp = run.lp
        base = decode_lp(lp, run.graded, run.halves)
        for name, cfg, criterion in configs:
            labels = decode_lp(lp, run.graded, run.halves, cfg)
            rows.append({"match": run.name, "config": name, "chosen": name == chosen,
                         "criterion": criterion, **{k: float(v) for k, v in cfg.items()},
                         "agreement": agreement(base, labels),
                         **describe(labels, run.graded, run.seen)})
        log("ridge: {} done".format(run.name))
    return _save("ridge", {"rows": rows})


# -- the report ----------------------------------------------------------------------------------

RUNNERS = {"bytes": audit_bytes, "margin": audit_margin, "anchors": audit_anchors,
           "inputs": audit_inputs, "perturb": audit_perturb, "support": audit_support,
           "ridge": audit_ridge}


def run(audits=AUDITS, log=print):
    for audit in audits:
        RUNNERS[audit](log=log)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _pct(x, digits=1):
    return "{:.{}f}%".format(100.0 * x, digits)


def _md_bytes(data):
    rows = data["rows"]
    same = len({r["sha256"] for r in rows}) == 1
    lines = ["## Does another machine write the same bytes?", "",
             "`predict` in a fresh process under other BLAS thread counts and CPU kernels "
             "(OpenBLAS chooses its kernels by CPU, so another machine can sum in another order).",
             ""]
    lines += _table(["run", "sha256", "frames changed"],
                    [(r["run"], "`{}…`".format(r["sha256"][:12]), r["frames_changed"])
                     for r in rows])
    lines += ["**All {} runs write the same file.**".format(len(rows)) if same else
              "**The runs do not all write the same file.**", ""]
    return lines


def _md_margin(data):
    lines = ["## How close is the decoded path to a tie?", "",
             "Gaussian noise on the ensemble's log-probabilities before decoding BAR-ATH, "
             "{} draws per size. BLAS differences are of the order 1e-12.".format(
                 data["noise"][0]["draws"]), ""]
    lines += _table(["noise sd", "draws that change a label", "most frames changed"],
                    [("{:g}".format(r["sd"]), "{} / {}".format(r["draws_changed"], r["draws"]),
                      r["max_frames_changed"]) for r in data["noise"]])
    lines += ["Rounding the log-probabilities before decoding (the candidate fix):", ""]
    lines += _table(["decimals", "frames changed against the frozen output",
                     "draws that change a label under 1e-9 noise", "agreement"],
                    [(r["decimals"], r["frames_changed_vs_frozen"],
                      r["draws_changed_under_1e-9"], "{:.4f}".format(r["agreement"]))
                     for r in data["rounding"]])
    return lines


def _md_anchors(data):
    lines = ["## Does v3 call live play where the ball is certainly dead?", "",
             "The dead-ball stage's anchors (a validated ball-out, the wait before a kickoff, a "
             "goal until its kickoff) are GT-free and near-certain. v3 reads them as one input "
             "channel and is free to overrule them.", ""]
    groups = {}
    for r in data["rows"]:
        groups.setdefault(r["group"], []).append(r)
    table = []
    for group, rows in groups.items():
        frames = sum(r["anchor_frames"] for r in rows)
        live = sum(r["live_share"] * r["anchor_frames"] for r in rows) / max(frames, 1)
        cells = [group, len(rows), frames, sum(r["anchor_runs"] for r in rows), _pct(live),
                 sum(r["runs_mostly_live"] for r in rows)]
        if "truly_dead_share" in rows[0]:
            truly = sum(r["truly_dead_share"] * r["anchor_frames"] for r in rows) / max(frames, 1)
            cells += [_pct(truly), _pct(np.mean([r["live_share_truly_dead"] for r in rows]))]
        else:
            cells += ["", ""]
        table.append(cells)
    lines += _table(["matches", "runs", "anchor frames", "anchor episodes",
                     "v3 calls them live", "episodes mostly live", "anchors truly dead",
                     "truly dead anchors v3 calls live"], table)
    if data.get("kinds"):
        lines += ["BAR-ATH by the kind of stoppage holding the anchor:", ""]
        lines += _table(["kind", "anchor frames", "v3 live frames", "share"],
                        [(k["kind"], k["anchor_frames"], k["live_frames"], _pct(k["live_share"]))
                         for k in data["kinds"]])
    lines += ["Candidate fix, pinning anchor frames to D in the decoder:", ""]
    table = []
    for group in groups:
        fix = [f for f in data["fix"] if f["group"] == group]
        cells = [group, int(np.mean([f["frames_changed"] for f in fix])),
                 "{:.3f}".format(np.mean([f["agreement"] for f in fix])),
                 "{:.1f} → {:.1f}".format(np.mean([f["base"]["possessions"] for f in fix]),
                                         np.mean([f["pinned"]["possessions"] for f in fix])),
                 "{} → {}".format(_pct(np.mean([f["base"]["dead_where_seen"] for f in fix])),
                                  _pct(np.mean([f["pinned"]["dead_where_seen"] for f in fix])))]
        if "spell_f1_base" in fix[0]:
            b = np.mean([f["spell_f1_base"] for f in fix])
            p = np.mean([f["spell_f1_pinned"] for f in fix])
            cells.append("{:.3f} → {:.3f} ({:+.3f})".format(b, p, p - b))
        else:
            cells.append("")
        table.append(cells)
    lines += _table(["matches", "frames changed per run", "agreement with unpinned",
                     "possessions", "D where the ball is seen", "spellF1"], table)
    lines += ["### Whole stoppages on which v3 and the dead-ball stage disagree", "",
              "A stoppage of the stage (≥ 1 s) is *ridden through* when v3 calls most of it live; "
              "a v3 stoppage is *v3 only* when the stage calls most of it live. Where there are "
              "labels, a stoppage is truly dead when most of its frames are.", ""]
    table = []
    for group in groups:
        sel = [r for r in data["stoppages"] if r["group"] == group]
        tot = {k: sum(r[k] for r in sel) for k in sel[0] if isinstance(sel[0][k], int)}
        cells = [group, tot["stage_stoppages"], tot["ridden"], tot["ridden_seen"]]
        if "ridden_truly_dead" in tot:
            kept = tot["stage_stoppages"] - tot["ridden"]
            cells += ["{} of {}".format(tot["ridden_truly_dead"], tot["ridden"]),
                      "{} of {}".format(tot["kept_truly_dead"], kept)]
        else:
            cells += ["", ""]
        cells += [tot["v3_stoppages"], tot["v3_only"]]
        cells.append("{} of {}".format(tot["v3_only_truly_dead"], tot["v3_only"])
                     if "v3_only_truly_dead" in tot else "")
        table.append(cells)
    lines += _table(["matches", "stage stoppages", "ridden through by v3",
                     "… with the ball mostly seen", "ridden through, truly dead",
                     "kept by v3, truly dead", "v3 stoppages", "v3 only",
                     "v3 only, truly dead"], table)
    if data.get("waits"):
        lines += _md_waits(data)
    bar = data.get("barath")
    if bar:
        lines += ["### BAR-ATH details", "", "The stage's stoppages by kind:", ""]
        lines += _table(["kind", "stoppages", "ridden through by v3", "… with the ball mostly seen"],
                        [(k["kind"], k["stoppages"], k["ridden"], k["ridden_ball_seen"])
                         for k in bar["kinds"]])
        lines += ["- Set-piece tags with a restart between 10 s before and 15 s after the tag "
                  "starts: v3 **{} of {}**, the dead-ball stage {} of {}.".format(
                      bar["set_pieces_v3"], bar["set_pieces"], bar["set_pieces_stage"],
                      bar["set_pieces"])]
        for h in bar["halves"]:
            lines.append("- Half {}: v3's first live frame {:.1f} s into the graded half (detected "
                         "kickoff {}); the half ends with {:.1f} s of D.".format(
                             h["half"], h["first_live_s"],
                             "{:.1f} s".format(h["kickoff_s"]) if h["kickoff_s"] is not None
                             else "not found", h["dead_tail_s"]))
        lines.append("")
    return lines


def _md_waits(data):
    lines = ["### The wait before a kickoff", "",
             "A half that starts inside an anchor is waiting for its kickoff, as the dead-ball "
             "stage sees it. FAQ.md says BAR-ATH's second half is graded from frame 110,808 but "
             "kicks off at 113,742, and that those frames are all `D`; the stage finds that "
             "kickoff by itself. *Peak speed* is the highest 8 s team speed in the wait, as a "
             "multiple of the match's typical live speed (the median over the frames the stage "
             "calls live). A wait that reaches live-play speed is a missed kickoff, where the "
             "stage tied the half to a later one. Waits of {:.0f} s or more:".format(WAIT_MIN_S),
             ""]
    long = [w for w in data["waits"] if w["seconds"] >= WAIT_MIN_S]
    lines += _table(["match", "wait", "peak speed", "ball seen", "truly dead",
                     "v3 calls it live", "v3 possessions inside", "pinned by the rule"],
                    [(w["match"], "{:.0f} s".format(w["seconds"]), "{:.2f}×".format(w["peak"]),
                      _pct(w["ball_seen"], 0),
                      _pct(w["truly_dead"], 0) if "truly_dead" in w else "(all, FAQ.md)",
                      _pct(w["v3_live"], 0) if "v3_live" in w else "",
                      w.get("v3_possessions", ""), "yes" if w["pinned"] else "no")
                     for w in long])
    short = [w for w in data["waits"] if w["seconds"] < WAIT_MIN_S]
    dead = [w["truly_dead"] for w in short if "truly_dead" in w]
    lines += ["{} more waits are shorter (median {:.1f} s). {} of them are mostly live play in "
              "the labels: a short anchor at the start of a half is often the kickoff itself. "
              "The rule leaves them to the networks.".format(
                  len(short), float(np.median([w["seconds"] for w in short])) if short else 0.0,
                  sum(d < 0.5 for d in dead)), "",
              "The fix, shipped since Nov 13 (`pipeline.V3_DEFAULTS[\"wait_min_s\"]`, "
              "`network.kickoff_wait`): pin a wait to D in the decoder when it lasts at least {:.0f} s and "
              "its peak speed stays under the match's typical live speed.".format(WAIT_MIN_S), ""]
    table = []
    for group in dict.fromkeys(f["group"] for f in data["wait_fix"]):
        fix = [f for f in data["wait_fix"] if f["group"] == group]

        def pair(key, digits=3):
            if key not in fix[0]:
                return ""
            a, b = (np.mean([f[key][i] for f in fix]) for i in (0, 1))
            return "{:.{d}f} → {:.{d}f} ({:+.{d}f})".format(a, b, b - a, d=digits)
        table.append((group, len(fix), int(sum(f["frames_changed"] for f in fix)),
                      "{:.1f} → {:.1f}".format(np.mean([f["possessions"][0] for f in fix]),
                                               np.mean([f["possessions"][1] for f in fix])),
                      pair("spell_f1_mean", 4), pair("spell_f1_alt", 4), pair("macro_f1", 4)))
    lines += _table(["matches", "runs", "frames changed in all", "possessions per run",
                     "spellF1 main", "spellF1 alt", "macroF1"], table)
    return lines


def _md_inputs(data):
    lines = ["## Which inputs are outside the training range?", "",
             "Per channel, after the networks' input scaling: the share of graded frames outside "
             "the central 99.8% of the pooled training frames ({} matches, one degradation seed "
             "each), and how well the channel alone tells the match from the pool (|AUC − 0.5|; 0 "
             "is indistinguishable). A training match is compared against the other training "
             "matches.".format(data["training_matches"]), ""]
    lines += _table(["match", "matches", "mean share outside the range", "mean |AUC − 0.5|"],
                    [(k, v["matches"], _pct(v["outside_mean"], 2), "{:.3f}".format(v["auc_mean"]))
                     for k, v in sorted(data["by_source"].items())])
    rows = data["rows"]
    flagged = [r for r in rows if r["outside"] > max(0.01, 2 * r["outside_train_max"])
               or r["auc"] > max(0.1, 1.25 * r["auc_train_max"])]
    lines += ["Channels where BAR-ATH is beyond every training match (outside share above 1% and "
              "twice the worst training match, or |AUC − 0.5| above 0.1 and 1.25× the worst "
              "training match): **{} of {}**.".format(len(flagged), len(rows)), ""]
    if flagged:
        lines += _table(["channel", "outside the range", "worst / median training match",
                         "|AUC − 0.5|", "worst / median training match"],
                        [(r["channel"], _pct(r["outside"], 2),
                          "{} / {}".format(_pct(r["outside_train_max"], 2),
                                           _pct(r["outside_train_median"], 2)),
                          "{:.3f}".format(r["auc"]),
                          "{:.3f} / {:.3f}".format(r["auc_train_max"], r["auc_train_median"]))
                         for r in sorted(flagged, key=lambda r: -r["auc"])])
    groups = sorted({r["group"] for r in rows})
    lines += ["By input group (means over the group's channels):", ""]
    lines += _table(["group", "channels", "BAR-ATH outside", "worst training match outside",
                     "BAR-ATH |AUC − 0.5|", "worst training match"],
                    [(g, len(sel), _pct(np.mean([r["outside"] for r in sel]), 2),
                      _pct(np.mean([r["outside_train_max"] for r in sel]), 2),
                      "{:.3f}".format(np.mean([r["auc"] for r in sel])),
                      "{:.3f}".format(np.mean([r["auc_train_max"] for r in sel])))
                     for g in groups for sel in [[r for r in rows if r["group"] == g]]])
    return lines


def _md_perturb(data):
    rows = data["rows"]
    lines = ["## Is BAR-ATH more fragile than a held-out game?", "",
             "The input is changed, the networks are re-run and the output is scored against the "
             "same match's unchanged output (agreement, as spellF1). The Metrica columns are the "
             "mean of the three games (degradation seed 0, held-out networks), where the true "
             "score change is also known.", ""]
    names = [r["perturbation"] for r in rows if r["match"] == rows[0]["match"]]
    by = {(r["match"], r["perturbation"]): r for r in rows}
    metrica = sorted({r["match"] for r in rows if r["match"].startswith("Metrica")})
    table = []
    for name in names:
        bar = by.get(("BAR-ATH", name))
        ms = [by[(m, name)] for m in metrica if (m, name) in by]
        if bar is None or not ms:
            continue
        base = [by[(m, "none")]["spell_f1"] for m in metrica if (m, name) in by]
        table.append((name, "{:.3f}".format(bar["agreement"]),
                      "{:.3f}".format(np.mean([m["agreement"] for m in ms])),
                      "{:+.3f}".format(np.mean([m["spell_f1"] for m in ms]) - np.mean(base)),
                      bar["possessions"], _pct(bar["dead_share"]), _pct(bar["dead_where_seen"]),
                      _pct(np.mean([m["dead_share"] for m in ms]))))
    lines += _table(["change", "BAR-ATH agreement", "Metrica agreement", "Metrica true spellF1",
                     "BAR-ATH possessions", "BAR-ATH dead share", "BAR-ATH D where ball seen",
                     "Metrica dead share"], table)
    if data.get("steps"):
        lines += ["Frame-to-frame movement of the tracking as it is (median, metres):", ""]
        lines += _table(["match", "ball step", "ball second difference", "player step",
                         "player second difference"],
                        [(k, "{:.3f}".format(v["ball_step"]),
                          "{:.4f}".format(v["ball_second_diff"]),
                          "{:.3f}".format(v["player_step"]),
                          "{:.4f}".format(v["player_second_diff"]))
                         for k, v in data["steps"].items()])
    return lines


def _md_support(data):
    lines = ["## How sure is each predicted possession?", "",
             "For every possession of the ensemble's output: how many of the 8 members, decoded "
             "alone, produce a possession that matches it at IoU ≥ 0.5. On the held-out Metrica "
             "games (3 games × 3 seeds) the same count is set against whether the possession is "
             "right.", ""]
    lines += _table(["members agreeing", "Metrica possessions", "share", "of them right",
                     "BAR-ATH possessions", "share"],
                    [(r["members"], r["metrica_spells"], _pct(r["metrica_share"]),
                      _pct(r["metrica_precision"]) if r["metrica_spells"] else "",
                      r["barath_spells"], _pct(r["barath_share"])) for r in data["levels"]])
    lines += ["- Held-out Metrica precision at IoU 0.5: **{}**.".format(
                  _pct(data["metrica_precision"])),
              "- BAR-ATH, expected from its own support counts: **{}** ({} possessions; levels "
              "0–4 pooled).".format(_pct(data["barath_expected_precision"]),
                                    data["barath_spells"]),
              "- BAR-ATH possessions: median {:.1f} s, {} under 2 s, {} over 60 s.".format(
                  data["barath_durations"]["median"], data["barath_durations"]["under_2s"],
                  data["barath_durations"]["over_60s"]), ""]
    lines += ["The 20 BAR-ATH possessions with the least support:", ""]
    lines += _table(["first frame", "seconds", "team", "members agreeing"],
                    [(w["first_frame"], "{:.1f}".format(w["seconds"]), w["team"], w["members"])
                     for w in data["weakest"]])
    lines += ["The 10 longest:", ""]
    lines += _table(["first frame", "seconds", "team", "members agreeing"],
                    [(w["first_frame"], "{:.1f}".format(w["seconds"]), w["team"], w["members"])
                     for w in data["longest"]])
    return lines


def _md_ridge(data):
    rows = data["rows"]
    lines = ["## Does BAR-ATH depend on which equally good decoder was chosen?", "",
             "The decoder settings within 0.004 of the chosen one on the held-out criterion, "
             "applied to the same log-probabilities. Agreement is with the chosen setting's "
             "output; the Metrica columns are the mean of the three games (seed 0).", ""]
    configs = [r["config"] for r in rows if r["match"] == "BAR-ATH"]
    table = []
    for c in configs:
        bar = next(r for r in rows if r["match"] == "BAR-ATH" and r["config"] == c)
        ms = [r for r in rows if r["match"] != "BAR-ATH" and r["config"] == c]
        table.append(("{:g} / {:g} / {:g}{}".format(bar["w_seq"], bar["switch"], bar["stop"],
                                                    " **(chosen)**" if bar["chosen"] else ""),
                      "{:.3f}".format(bar["criterion"]), "{:.3f}".format(bar["agreement"]),
                      "{:.3f}".format(np.mean([m["agreement"] for m in ms])),
                      bar["possessions"],
                      "{:.0f}".format(np.mean([m["possessions"] for m in ms])),
                      _pct(bar["dead_share"])))
    lines += _table(["w_seq / switch / stop", "held-out criterion", "BAR-ATH agreement",
                     "Metrica agreement", "BAR-ATH possessions", "Metrica possessions",
                     "BAR-ATH dead share"], table)
    return lines


SECTIONS = {"bytes": _md_bytes, "margin": _md_margin, "anchors": _md_anchors,
            "inputs": _md_inputs, "perturb": _md_perturb, "support": _md_support,
            "ridge": _md_ridge}


def markdown():
    lines = ["# Robustness audits of v3", "",
             "Generated by `more-than-a-hack-2027 robust` "
             "(src/more_than_a_hack_2027/validation/robustness.py). Do not edit by hand.", "",
             "Every BAR-ATH number here is GT-free: no label of the competition match is read. "
             "Held-out Metrica and DFL matches give each one a baseline. The audits are of v3 as "
             "frozen on Nov 12 (sha256 `2725f912…`, 218 possessions): the networks and the "
             "decoder. Fixes are scored here, not applied. One cleared the rule in docs/PLAN.md "
             "and ships since Nov 13, the kickoff-wait rule; only the byte-stability runs, "
             "which call `predict`, include it.", ""]
    for audit in AUDITS:
        data = _load(audit)
        if data is not None:
            lines += SECTIONS[audit](data)
    return "\n".join(lines).rstrip("\n") + "\n"
