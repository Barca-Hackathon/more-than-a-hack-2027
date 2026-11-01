"""Which team takes each restart, as summed log-odds evidence (positive = HOME).

The first possession after a stoppage belongs to the restarting team, and a wrong call there costs
a whole possession on spellF1. Sources, each optional:

  taker    the team of the player nearest the ball at the restart frame
  tag      a Smart Tagging SET PIECES tag with a team, near the restart (throw-ins are N/A)
  out      for a ball-out: the OPPONENT of the last player to touch it -- true for throw-ins,
           corners and goal kicks alike
  kickoff  the team with the player nearest the centre spot at a kickoff; after a goal that is
           the conceding team

Weights are priors, to be tuned on the external games.
"""

from dataclasses import dataclass, field

import numpy as np

from .context import AWAY, HOME
from .nearest_player import NOBODY

RESTART_DEFAULTS = {
    "touch_m": 1.5,             # ball this close to a player = a touch, for "last toucher"
    "touch_lookback_s": 3.0,
    "taker_before": 5,          # frames around the restart searched for the taker
    "taker_after": 15,
    "taker_scale_m": 2.0,       # taker weight decays as exp(-distance / scale)
    "tag_before_s": 10.0,       # a tag counts if the restart is within [start - before, start + after]
    "tag_after_s": 15.0,
    "w_taker": 2.0,
    "w_tag": 2.5,
    "w_out": 2.0,
    "w_kickoff": 3.0,
    "hold_s": 1.0,              # frames after the restart that receive the team evidence
    "w_frame": 3.0,
}


@dataclass
class RestartCall:
    frame: int
    team: object                # HOME, AWAY or None
    logodds_home: float
    sources: dict = field(default_factory=dict)


def _sign(team):
    return 1.0 if team == HOME else -1.0


def last_toucher(codes, distance, frame, lookback, touch_m):
    """Team of the player at the ball in the latest touch before `frame`, or None."""
    lo = max(0, frame - lookback)
    window = np.nan_to_num(distance[lo:frame], nan=np.inf) <= touch_m
    hits = np.flatnonzero(window & (codes[lo:frame] != NOBODY))
    if not len(hits):
        return None
    return int(codes[lo + hits[-1]])


def taker(codes, distance, frame, before, after):
    """(team, distance) of the nearest player at the seen frame closest to `frame`."""
    lo, hi = max(0, frame - before), min(len(codes), frame + after + 1)
    frames = np.arange(lo, hi)
    seen = frames[codes[lo:hi] != NOBODY]
    if not len(seen):
        return None, None
    best = int(seen[np.argmin(np.abs(seen - frame))])
    return int(codes[best]), float(distance[best])


def decide(ctx, dead, tags=(), cfg=None):
    """One RestartCall per stoppage in dead.stoppages, in order."""
    cfg = dict(RESTART_DEFAULTS, **(cfg or {}))
    fps = ctx.fps
    codes, distance, _ = dead.nearest
    set_pieces = [t for t in tags if t.code == "SET PIECES" and t.team is not None]
    lookback = int(cfg["touch_lookback_s"] * fps)

    calls = []
    for stoppage in dead.stoppages:
        r = stoppage.restart
        sources = {}

        team, dist = taker(codes, distance, r, cfg["taker_before"], cfg["taker_after"])
        if team is not None:
            sources["taker"] = _sign(team) * cfg["w_taker"] * float(np.exp(-dist / cfg["taker_scale_m"]))

        near = [t for t in set_pieces
                if t.start - cfg["tag_before_s"] * fps <= r <= t.start + cfg["tag_after_s"] * fps]
        if near:
            tag = min(near, key=lambda t: abs(t.start - r))
            sources["tag"] = _sign(tag.team) * cfg["w_tag"]

        if stoppage.out is not None and stoppage.kind != "goal":
            touched = last_toucher(codes, distance, stoppage.out.first, lookback, cfg["touch_m"])
            if touched is not None:
                sources["out"] = -_sign(touched) * cfg["w_out"]

        if stoppage.kickoff is not None and stoppage.kickoff.kicker_team is not None:
            sources["kickoff"] = _sign(stoppage.kickoff.kicker_team) * cfg["w_kickoff"]

        total = float(sum(sources.values()))
        decided = HOME if total > 0 else AWAY if total < 0 else None
        calls.append(RestartCall(frame=int(r), team=decided, logodds_home=total, sources=sources))
    return calls


def frame_evidence(ctx, calls, cfg=None):
    """(n, 2) evidence for H and A on the frames just after each restart."""
    cfg = dict(RESTART_DEFAULTS, **(cfg or {}))
    hold = max(1, int(cfg["hold_s"] * ctx.fps))
    evidence = np.zeros((ctx.n_frames, 2))
    for call in calls:
        if call.team is None:
            continue
        weight = cfg["w_frame"] * float(np.tanh(abs(call.logodds_home) / 2.0))
        evidence[call.frame:call.frame + hold, call.team] += weight
    return evidence


def summary(calls):
    decided = [c for c in calls if c.team is not None]
    by_source = {}
    for call in calls:
        for name in call.sources:
            by_source[name] = by_source.get(name, 0) + 1
    disagreements = sum(1 for c in calls
                        if len({np.sign(v) for v in c.sources.values() if v}) > 1)
    return {
        "n_restarts": len(calls),
        "decided": len(decided),
        "home": sum(1 for c in decided if c.team == HOME),
        "away": sum(1 for c in decided if c.team == AWAY),
        "sources_used": dict(sorted(by_source.items())),
        "sources_disagree": disagreements,
    }
