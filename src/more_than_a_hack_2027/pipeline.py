"""Methods that turn a Context into a full-length label array.

    baseline     starter/baseline.py, ported and bit-exact. The regression anchor.
    hmm-nearest  nearest-player evidence through the Viterbi decoder. A SMOKE TEST of the
                 evidence -> decode -> write path with untuned placeholder numbers; it is here so
                 the stage contract has a working example, not as a candidate submission. The
                 real evidence comes from deadball / control / shape / tag_prior.
"""

import numpy as np

from .stages import decode
from ._paths import ensure_repo_on_path
from .stages.nearest_player import NOBODY, nearest_player

ensure_repo_on_path()
import baseline as starter_baseline  # noqa: E402  starter/baseline.py
import games  # noqa: E402

# Untuned placeholders for the smoke test. Real values are chosen on the external games.
SMOKE_DEFAULTS = {
    "near_m": 3.0,            # ball within this of the nearest player counts as strong evidence
    "w_near": 1.0,
    "w_far": 0.3,
    "w_alive_seen": 1.0,      # a visible ball is evidence the ball is alive
    "switch": 25.0,
    "stop": 40.0,
    "restart": 40.0,
}


def run_baseline(ctx, window=50):
    codes, _, _ = nearest_player(ctx)
    return starter_baseline.smooth(codes, window)


def nearest_evidence(ctx, cfg):
    """(n, 3) log-scores for H, A, D from the nearest player alone."""
    codes, distance, _ = nearest_player(ctx)
    evidence = np.zeros((ctx.n_frames, decode.N_STATES))
    seen = codes != NOBODY
    weight = np.where(distance <= cfg["near_m"], cfg["w_near"], cfg["w_far"])
    for team in (decode.H, decode.A):
        mine = seen & (codes == team)
        evidence[mine, team] += weight[mine]
    evidence[seen, decode.D] -= cfg["w_alive_seen"]
    return evidence


def run_hmm_nearest(ctx, cfg=None):
    cfg = dict(SMOKE_DEFAULTS, **(cfg or {}))
    trans = decode.transition_matrix(cfg["switch"], cfg["stop"], cfg["restart"])
    return decode.decode_match(nearest_evidence(ctx, cfg), ctx.halves, trans)


# v1: dead-ball stage (deadball + restart) with nearest-player team evidence standing in for the
# control stage until P3 delivers it. Penalties are untuned priors.
V1_DEFAULTS = {
    "near_m": 3.0,
    "w_near": 1.0,
    "w_far": 0.3,
    "switch": 25.0,     # H <-> A in open play: needs sustained team evidence
    "stop": 2.0,        # the dead-ball evidence already decides where stoppages are
    "restart": 2.0,
}


def team_evidence_nearest(ctx, codes, distance, cfg):
    """(n, 2) H/A evidence from the nearest player. Placeholder for control.py."""
    evidence = np.zeros((ctx.n_frames, 2))
    seen = codes != NOBODY
    weight = np.where(np.nan_to_num(distance, nan=np.inf) <= cfg["near_m"],
                      cfg["w_near"], cfg["w_far"])
    for team in (decode.H, decode.A):
        mine = seen & (codes == team)
        evidence[mine, team] += weight[mine]
    return evidence


# Keys of a v1/v2 config that hold a whole stage's overrides rather than a decoder number.
STAGE_KEYS = ("ball", "deadball", "restart", "control")


def split_cfg(cfg, defaults=None):
    """(decoder settings with defaults, {stage: overrides}) from one nested config."""
    defaults = V1_DEFAULTS if defaults is None else defaults
    cfg = dict(cfg or {})
    stages = {key: dict(cfg.pop(key, None) or {}) for key in STAGE_KEYS}
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise KeyError("unknown settings {}".format(sorted(unknown)))
    return dict(defaults, **cfg), stages


def run_v1(ctx, cfg=None, return_parts=False, tags=None):
    """cfg: V1_DEFAULTS keys, plus optional "ball" / "deadball" / "restart" override dicts.

    tags: Smart Tagging intervals; None loads them for the competition match, and no other game
    has any (the external games run without them).
    """
    from .stages import deadball, restart
    from .stages.ball import clean_ball
    from .data.tagging import load_tags

    top, stages = split_cfg(cfg)
    ball = clean_ball(ctx, stages["ball"])
    dead = deadball.detect(ctx, ball, stages["deadball"])
    if tags is None:
        tags = load_tags(ctx) if ctx.game == games.GAME else ()
    calls = restart.decide(ctx, dead, tags, stages["restart"])

    codes, distance, _ = dead.nearest
    evidence = np.zeros((ctx.n_frames, decode.N_STATES))
    evidence[:, :2] = team_evidence_nearest(ctx, codes, distance, top)
    evidence[:, :2] += restart.frame_evidence(ctx, calls, stages["restart"])
    evidence[:, decode.D] = dead.evidence

    trans = decode.transition_matrix(top["switch"], top["stop"], top["restart"])
    labels = decode.decode_match(evidence, ctx.halves, trans)
    if return_parts:
        return labels, {"dead": dead, "calls": calls, "tags": tags, "evidence": evidence}
    return labels


# v2: v1 with the learned control stage (control.py) as the team evidence. w_control and the
# penalties are chosen on the external games (validation/harness.py), not here.
V2_DEFAULTS = {
    # Chosen on the degraded external games by tune.py (docs/TUNING.md): best mean spellF1
    # under both label readings, 3 seeds, each game scored by a model that did not see it.
    # History: switch 25 (Nov 1 freeze, labels that counted one-touch plays), then w 1.6 /
    # switch 50 (corrected labels), then this once the games also lose the ball in live play
    # (degrade.LIVE_GAPS). The last step is a tie (+0.0002 over w 1.6 / switch 50), kept
    # because the rule fixed before the run picks the top of the table.
    "w_control": 3.2,
    "switch": 70.0,
    "stop": 2.0,
    "restart": 2.0,
    "min_dead_s": 1.0,     # no D run shorter than this (decode.viterbi_min_dead)
    "w_restart": 1.0,      # scale on the restart-team evidence (0 = ablate it)
}


def run_v2(ctx, cfg=None, return_parts=False, tags=None, model=None):
    """cfg: V2_DEFAULTS keys plus optional stage overrides ("ball", "deadball", "restart",
    "control"). model: a control.ControlModel; None loads the shipped models/control_v1.json."""
    from .stages import control, deadball, restart
    from .stages.ball import clean_ball
    from .data.tagging import load_tags

    top, stages = split_cfg(cfg, V2_DEFAULTS)
    model = model if model is not None else control.ControlModel.from_json()
    ball = clean_ball(ctx, stages["ball"])
    dead = deadball.detect(ctx, ball, stages["deadball"])
    if tags is None:
        tags = load_tags(ctx) if ctx.game == games.GAME else ()
    calls = restart.decide(ctx, dead, tags, stages["restart"])
    z = control.logodds(ctx, model, ball, stages["control"])

    evidence = np.zeros((ctx.n_frames, decode.N_STATES))
    evidence[:, :2] = control.evidence(z, top["w_control"])
    evidence[:, :2] += top["w_restart"] * restart.frame_evidence(ctx, calls, stages["restart"])
    evidence[:, decode.D] = dead.evidence

    trans = decode.transition_matrix(top["switch"], top["stop"], top["restart"])
    labels = decode.decode_match(evidence, ctx.halves, trans,
                                 min_dead=int(round(top["min_dead_s"] * ctx.fps)))
    if return_parts:
        return labels, {"dead": dead, "calls": calls, "tags": tags, "evidence": evidence,
                        "logodds": z}
    return labels


# The frozen submission method (docs/PLAN.md, M3). `predict` without --method runs this, and
# scripts/check_reproducibility.sh checks it from a clean copy of the repository.
SUBMISSION_METHOD = "v2"


def run(method, ctx, **kwargs):
    if method == "baseline":
        labels = run_baseline(ctx, kwargs.get("window", 50))
        # The reference writes D on every frame it has no answer for, including ungraded ones.
        # Keep that, so the file is byte-identical to evaluation/results/baseline_*.csv.
        return labels
    if method == "hmm-nearest":
        return run_hmm_nearest(ctx)
    if method == "v1":
        return run_v1(ctx, kwargs.get("cfg"), tags=kwargs.get("tags"))
    if method == "v2":
        return run_v2(ctx, kwargs.get("cfg"), tags=kwargs.get("tags"), model=kwargs.get("model"))
    raise ValueError("unknown method {!r}".format(method))
