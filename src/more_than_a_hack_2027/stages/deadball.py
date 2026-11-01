"""Dead-ball detection: per-frame evidence for D, and a list of refined stoppages.

Three layers, each usable on its own:

1. Anchors -- frames that are dead for certain, found by geometry alone:
     * the ball beyond a pitch line (touchline or goal line), and the first seconds after it;
     * every frame of a half before its first kickoff (the second half is graded for ~2 minutes
       before it kicks off, all of it D) -- only when that kickoff is within pre_kickoff_max_s
       of the half start, so a missed opening kickoff cannot anchor a whole stretch as dead;
     * a goal (goal-line out between the posts) until the next kickoff;
     * kickoff formations: >= 90% of players in their own half with a kicker at the centre spot.
2. A two-state live/dead HMM fitted by EM on this match's own tracking (hmm2.py), with the
   anchors clamped to dead. No labels. Emissions are what distinguishes a stoppage on the
   tracking: a low and *steady* collective speed (players walking), a ball that is often
   missing, and a ball that is rarely moving fast. Output: P(dead) per frame.
3. Stoppage refinement -- posterior runs become intervals whose edges are snapped to events,
   because spellF1 rewards sharp boundaries:
     * start -> the ball-out crossing frame, or the steepest team deceleration (the whistle)
       minus a reaction lag;
     * end   -> the restart: a kickoff, a kick from a still ball with a player at it, a throw-in
       re-entering the pitch, or the ball reappearing.

Every number in DEADBALL_DEFAULTS is a physical or football prior, not a fit to the public ground
truth. Tune them on the external validation games only (docs/PLAN.md).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import hmm2
from .ball import clean_ball
from ..data.context import AWAY, HOME
from .nearest_player import NOBODY, nearest_player

DEADBALL_DEFAULTS = {
    # --- ball out of play -------------------------------------------------------------------
    "out_margin_m": 0.3,            # beyond a line by at least this somewhere in the episode
    "out_min_frames": 5,            # ... and for this many frames, or
    "out_min_excursion_m": 0.5,     # ... by at least this far
    "out_merge_gap": 12,            # out runs closer than this (frames) are one episode
    "out_reentry_s": 2.0,           # back in play sooner than this = not a real out
    "min_dead_after_out_s": 2.0,    # no restart happens faster than this after the ball goes out
    "goal_half_width_m": 3.66 + 0.5,  # half the goal mouth, plus tracking tolerance
    "goal_kickoff_max_s": 300.0,    # a goal is confirmed by a kickoff within this window
    # --- kickoff formation ------------------------------------------------------------------
    "kickoff_own_half_frac": 0.9,
    "kickoff_tol_m": 1.0,           # players this close past halfway still count as own half
    "kickoff_min_players": 16,
    "kickoff_kicker_m": 3.0,        # someone stands this close to the centre spot
    "kickoff_min_s": 1.0,           # a formation must hold this long (except at a half's start)
    "kickoff_start_grace_s": 2.0,   # a formation this close to the half's first frame is kickoff
    "kickoff_lag_frames": 10,       # the formation breaks this many frames after the kick
    "pre_kickoff_max_s": 300.0,     # "dead until the first kickoff" only if it is this early: a
                                    # missed opening kickoff must not anchor half a half as dead
    # --- HMM features -----------------------------------------------------------------------
    "speed_min_players": 10,
    "level_window_s": 1.0,          # rolling mean of team speed
    "spread_window_s": 3.0,         # rolling std of team speed
    "ball_moving_mps": 4.0,
    "init_dead_mps": 1.2,           # initial guess: slower than this looks dead
    "init_live_s": 40.0,            # initial mean live / dead durations
    "init_dead_s": 15.0,
    "em_stride": 5,
    "em_iter": 60,
    # --- stoppage extraction ----------------------------------------------------------------
    "posterior_threshold": 0.5,
    "min_stoppage_s": 2.0,
    "merge_gap_s": 1.0,
    # --- edge refinement --------------------------------------------------------------------
    "snap_out_before_s": 5.0,
    "snap_out_after_s": 3.0,
    "whistle_before_s": 4.0,
    "whistle_after_s": 1.0,
    "whistle_lag_frames": 12,
    "restart_before_s": 5.0,
    "restart_after_s": 2.0,
    "restart_lag_frames": 25,       # the posterior lags the restart by about this much
    "kick_still_mps": 1.5,          # the ball was nearly still ...
    "kick_still_window_s": 1.5,     # ... somewhere in this window before the kick
    "kick_fast_mps": 5.0,           # the kick is the upward crossing of this speed
    "kick_player_m": 2.0,           # with a player this close ...
    "kick_player_frames": 15,       # ... at some point in these preceding frames
    # --- decoder evidence -------------------------------------------------------------------
    "evidence_strength": 6.0,
}

LIVE, DEAD = 0, 1


@dataclass(frozen=True)
class OutEpisode:
    first: int              # first frame beyond the line
    last: int               # last frame of the (merged) episode, inclusive
    line: str               # "touchline" or "goal_line"
    end: str                # "low" (x<0 or y<0) or "high"
    between_posts: bool
    excursion: float        # metres beyond the line at most
    goal: bool = False      # confirmed by a following kickoff


@dataclass(frozen=True)
class Kickoff:
    frame: int              # estimated kick frame
    half: int
    formation_start: int
    formation_stop: int     # exclusive
    at_half_start: bool
    kicker_team: object     # team of the player nearest the centre spot, or None


@dataclass
class Stoppage:
    start: int              # first dead frame
    restart: int            # first live frame (exclusive end)
    half: int
    kind: str               # out_touchline | out_goal_line | goal | half_start | whistle
    restart_kind: str       # kickoff | kick | throw_in | reappear | posterior
    confidence: float       # mean P(dead) over the interval
    out: object = None      # OutEpisode or None
    kickoff: object = None  # Kickoff or None

    @property
    def duration(self):
        return self.restart - self.start


@dataclass
class DeadBallResult:
    posterior: np.ndarray
    llr: np.ndarray
    anchors: np.ndarray
    mask: np.ndarray
    evidence: np.ndarray
    stoppages: list
    out_episodes: list
    kickoffs: list
    params: object
    history: list
    features: dict = field(repr=False)
    ball: object = field(repr=False)
    nearest: tuple = field(repr=False)


# -- helpers -------------------------------------------------------------------------------------

def runs(mask):
    """(starts, stops) of True runs, stops exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def merge_runs(starts, stops, gap):
    out = []
    for a, b in zip(starts.tolist(), stops.tolist()):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def graded_sequences(ctx):
    return [(first, last + 1) for _, first, last in ctx.halves]


# -- signals -------------------------------------------------------------------------------------

def team_speed(ctx, min_players):
    """Mean speed of all observed players per frame; NaN when too few are observed."""
    speed = np.asarray(ctx.speed, dtype=np.float64)
    finite = np.isfinite(speed)
    count = finite.sum(axis=1)
    total = np.where(finite, speed, 0.0).sum(axis=1)
    return np.where(count >= min_players, total / np.maximum(count, 1), np.nan)


def features(ctx, ball, cfg):
    fps = ctx.fps
    v = pd.Series(team_speed(ctx, cfg["speed_min_players"]))
    w_level = max(1, int(round(cfg["level_window_s"] * fps)))
    w_spread = max(2, int(round(cfg["spread_window_s"] * fps)))
    level = v.rolling(w_level, center=True, min_periods=max(1, w_level // 2)).mean().to_numpy()
    spread = v.rolling(w_spread, center=True, min_periods=max(2, w_spread // 2)).std().to_numpy()

    moving = np.where(ball.seen & np.isfinite(ball.speed),
                      (ball.speed > cfg["ball_moving_mps"]).astype(float), np.nan)
    G = np.column_stack([np.log(np.maximum(level, 0.05)), np.log(spread + 0.02)])
    B = np.column_stack([ball.seen.astype(float), moving])
    return {"speed": v.to_numpy(), "level": level, "spread": spread, "G": G, "B": B,
            "G_names": ("log_speed_level", "log_speed_spread"),
            "B_names": ("ball_seen", "ball_moving")}


def _back_in_play_soon(ball, last, limit, margin):
    """True if the first seen frame after `last` is inside the pitch and comes before `limit`."""
    after = np.flatnonzero(ball.seen[last + 1:limit + 1])
    if not len(after):
        return False
    first_seen = last + 1 + int(after[0])
    return bool(ball.outside[first_seen] < -margin)


def out_episodes(ctx, ball, cfg):
    """Validated ball-out episodes.

    Two artefacts look like a ball out and are rejected:
      * line noise -- tiny or brief excursions (out_margin_m / out_min_frames / excursion);
      * lofted balls near a touchline -- the ground projection of a high ball lands outside the
        pitch for a few frames and then comes back. A real out is never followed by the ball back
        in play within out_reentry_s (even a quick throw-in takes longer), and the projection
        artefact usually starts at an implausible apparent speed (ball.aerial).
    """
    length, width = ctx.pitch
    beyond = (np.nan_to_num(ball.outside, nan=-1.0) > 0) & (ctx.half_of_frame > 0)
    starts, stops = runs(beyond)
    reentry = int(cfg["out_reentry_s"] * ctx.fps)
    episodes = []
    for a, b in merge_runs(starts, stops, cfg["out_merge_gap"]):
        window = ball.outside[a:b]
        excursion = float(np.nanmax(window))
        n_out = int(np.sum(window > 0))
        if excursion < cfg["out_margin_m"]:
            continue
        if n_out < cfg["out_min_frames"] and excursion < cfg["out_min_excursion_m"]:
            continue
        if _back_in_play_soon(ball, b - 1, a + reentry, cfg["out_margin_m"]):
            continue
        if ball.aerial[max(0, a - 3):a + 4].any():
            continue
        x, y = ball.pos[a]
        if x < 0 or x > length:
            line, end = "goal_line", ("low" if x < 0 else "high")
            between = abs(y - width / 2.0) <= cfg["goal_half_width_m"]
        else:
            line, end, between = "touchline", ("low" if y < 0 else "high"), False
        episodes.append(OutEpisode(a, b - 1, line, end, bool(between), excursion))
    return episodes


def kickoff_formation(ctx, cfg):
    """(formation mask, own-half fraction, distance of nearest player to the centre spot)."""
    length, width = ctx.pitch
    x = np.asarray(ctx.xy[:, :, 0], dtype=np.float64)
    y = np.asarray(ctx.xy[:, :, 1], dtype=np.float64)
    observed = ~np.isnan(x)
    own = np.zeros(x.shape, dtype=bool)
    for half, frames in ctx.half_slices():
        for team in (HOME, AWAY):
            slots = ctx.team_slots(team)
            sign = ctx.attack_sign(half, team)
            # attacking +x means the own half is x < length / 2
            own[frames, slots] = (length / 2.0 - x[frames][:, slots]) * sign > -cfg["kickoff_tol_m"]
    n_obs = observed.sum(axis=1)
    frac = (own & observed).sum(axis=1) / np.maximum(n_obs, 1)
    frac = pd.Series(frac).rolling(ctx.fps, center=True, min_periods=1).median().to_numpy()
    to_centre = np.hypot(x - length / 2.0, y - width / 2.0)
    nearest = np.where(observed, to_centre, np.inf).min(axis=1)
    formation = ((frac >= cfg["kickoff_own_half_frac"]) & (n_obs >= cfg["kickoff_min_players"])
                 & (nearest <= cfg["kickoff_kicker_m"]) & (ctx.half_of_frame > 0))
    return formation, frac, nearest


def _team_nearest_centre(ctx, frame):
    length, width = ctx.pitch
    d = np.hypot(ctx.xy[frame, :, 0] - length / 2.0, ctx.xy[frame, :, 1] - width / 2.0)
    if np.all(np.isnan(d)):
        return None
    return int(ctx.team_of_slot[int(np.nanargmin(d))])


def detect_kickoffs(ctx, formation, cfg):
    fps = ctx.fps
    starts, stops = runs(formation)
    kickoffs = []
    for half, first, last in ctx.halves:
        for a, b in zip(starts.tolist(), stops.tolist()):
            if not first <= a <= last:
                continue
            at_start = a <= first + cfg["kickoff_start_grace_s"] * fps
            if b - a < cfg["kickoff_min_s"] * fps and not at_start:
                continue
            frame = max(a, b - int(cfg["kickoff_lag_frames"]))
            kickoffs.append(Kickoff(frame, half, a, b, bool(at_start),
                                    _team_nearest_centre(ctx, frame)))
    return kickoffs


def candidate_restarts(ctx, ball, distance, cfg):
    """Sorted [(frame, kind)] of events that can end a stoppage.

    The provided ball track is already smoothed by the tracker, so a kick is a ~1.5 s ramp in
    speed rather than a step. A kick is therefore the upward crossing of kick_fast_mps by a ball
    that was nearly still somewhere in the preceding window, with a player close to it in the
    frames before. A centred smoother puts the true kick near the middle of the ramp, which is
    roughly where the crossing lands.
    """
    speed = pd.Series(np.where(ball.seen, ball.speed, np.nan))
    window = max(2, int(round(cfg["kick_still_window_s"] * ctx.fps)))
    slowest_before = speed.shift(1).rolling(window, min_periods=window // 2).min().to_numpy()
    values = speed.to_numpy()
    previous = speed.shift(1).to_numpy()
    crossing = (values > cfg["kick_fast_mps"]) & (previous <= cfg["kick_fast_mps"])
    inside = np.nan_to_num(ball.outside, nan=1.0) <= cfg["out_margin_m"]
    near_recent = (pd.Series(np.nan_to_num(distance, nan=np.inf))
                   .rolling(int(cfg["kick_player_frames"]), min_periods=1).min().to_numpy()
                   <= cfg["kick_player_m"])
    kick = crossing & (slowest_before < cfg["kick_still_mps"]) & inside & near_recent

    outside = ball.outside
    reenter = np.zeros(len(outside), dtype=bool)
    reenter[1:] = (outside[:-1] > 0) & (outside[1:] <= 0)

    reappear = np.zeros(len(outside), dtype=bool)
    reappear[1:] = ball.seen[1:] & ~ball.seen[:-1] & inside[1:]

    graded = ctx.half_of_frame > 0
    events = [(int(f), "kick") for f in np.flatnonzero(kick & graded)]
    events += [(int(f), "throw_in") for f in np.flatnonzero(reenter & graded)]
    events += [(int(f), "reappear") for f in np.flatnonzero(reappear & graded)]
    events.sort()
    return events


# -- the model -----------------------------------------------------------------------------------

def build_anchors(ctx, cfg, episodes, kickoffs):
    """(anchor mask, episodes with goals confirmed)."""
    fps = ctx.fps
    anchors = np.zeros(ctx.n_frames, dtype=bool)
    hold = int(round(cfg["min_dead_after_out_s"] * fps))
    for episode in episodes:
        anchors[episode.first:episode.last + 1] = True
        anchors[episode.first:episode.first + hold] = True

    for half, first, last in ctx.halves:
        mine = [k.frame for k in kickoffs if k.half == half]
        if mine and min(mine) - first <= cfg["pre_kickoff_max_s"] * fps:
            anchors[first:min(mine)] = True
    for kickoff in kickoffs:
        anchors[kickoff.formation_start:kickoff.frame] = True

    confirmed = []
    window = int(cfg["goal_kickoff_max_s"] * fps)
    half_of = ctx.half_of_frame
    for k, episode in enumerate(episodes):
        goal = False
        if episode.line == "goal_line" and episode.between_posts:
            # A goal is followed by a mid-half kickoff in the same half, with no other ball-out
            # in between; anything else (a shot just wide, then the half-time kickoff) is not.
            after = [kick for kick in kickoffs
                     if not kick.at_half_start and half_of[kick.frame] == half_of[episode.first]
                     and episode.first < kick.frame <= episode.first + window]
            if after:
                kick = min(after, key=lambda kick: kick.frame)
                later_outs = [e for e in episodes[k + 1:] if e.first < kick.frame]
                if not later_outs:
                    anchors[episode.first:kick.frame] = True
                    goal = True
        confirmed.append(OutEpisode(episode.first, episode.last, episode.line, episode.end,
                                    episode.between_posts, episode.excursion, goal))
    anchors &= ctx.half_of_frame > 0
    return anchors, confirmed


def initial_params(feat, anchors, graded, cfg, fps):
    """One M-step from a heuristic soft labelling: anchors and slow frames start as dead."""
    G, B = feat["G"], feat["B"]
    level = np.nan_to_num(feat["level"], nan=np.inf)
    gamma = ((level < cfg["init_dead_mps"]) | anchors).astype(np.float64)
    idx = np.flatnonzero(graded)
    stride = cfg["em_stride"]
    base = hmm2.Params(mu=np.zeros((2, G.shape[1])), sd=np.ones((2, G.shape[1])),
                       p=np.full((2, B.shape[1]), 0.5), p01=0.0, p10=0.0)
    params = hmm2.m_step(G[idx], B[idx], gamma[idx], (1.0, 0.0, 0.0, 1.0), base)
    return hmm2.Params(mu=params.mu, sd=params.sd, p=params.p,
                       p01=min(0.5, stride / (cfg["init_live_s"] * fps)),
                       p10=min(0.5, stride / (cfg["init_dead_s"] * fps)),
                       init1=0.5)


def _snap_start(ctx, s, floor, episodes_by_first, feat, cfg, half_first):
    fps = ctx.fps
    lo, hi = s - int(cfg["snap_out_before_s"] * fps), s + int(cfg["snap_out_after_s"] * fps)
    candidates = [e for f, e in episodes_by_first if lo <= f <= hi and f >= floor]
    if candidates:
        episode = candidates[0]
        kind = "goal" if episode.goal else "out_" + episode.line
        return episode.first, kind, episode
    if s <= half_first:
        return half_first, "half_start", None

    lo = max(floor, half_first, s - int(cfg["whistle_before_s"] * fps))
    hi = min(s + int(cfg["whistle_after_s"] * fps), len(feat["level"]) - 1)
    level = feat["level"][lo:hi + 1]
    if hi - lo >= 2 and np.isfinite(level).sum() >= 3:
        slope = np.gradient(np.where(np.isfinite(level), level, np.nanmean(level)))
        start = lo + int(np.argmin(slope)) - int(cfg["whistle_lag_frames"])
        return max(start, floor, half_first), "whistle", None
    return max(s, floor), "whistle", None


# Which restart events to prefer, by how the stoppage began.
PREFERRED_RESTART = {"out_touchline": ("throw_in",), "out_goal_line": ("kick",), "whistle": ("kick",)}


def _snap_restart(ctx, s_start, run_start, e, half_last, kickoffs, events, event_frames, cfg,
                  kind, last_anchor):
    fps = ctx.fps
    # Never restart before the evidence said the ball was dead, nor inside a certain-dead anchor.
    lo = max(s_start + 1, run_start, e - int(cfg["restart_before_s"] * fps))
    if last_anchor is not None:
        lo = max(lo, last_anchor + 1)
    hi = min(half_last + 1, e + int(cfg["restart_after_s"] * fps))
    kicks = [k for k in kickoffs if s_start < k.frame <= hi]
    if kicks:
        kick = max(kicks, key=lambda k: k.frame)
        return kick.frame, "kickoff", kick
    i, j = np.searchsorted(event_frames, [lo, hi + 1])
    if j > i:
        target = e - int(cfg["restart_lag_frames"])
        window = events[i:j]
        preferred = [ev for ev in window if ev[1] in PREFERRED_RESTART.get(kind, ())]
        frame, event_kind = min(preferred or window, key=lambda ev: (abs(ev[0] - target), ev[0]))
        return frame, event_kind, None
    return min(e, half_last + 1), "posterior", None


def detect(ctx, ball=None, cfg=None):
    cfg = dict(DEADBALL_DEFAULTS, **(cfg or {}))
    fps = ctx.fps
    ball = ball if ball is not None else clean_ball(ctx)
    graded = ctx.half_of_frame > 0
    sequences = graded_sequences(ctx)

    feat = features(ctx, ball, cfg)
    formation, _, _ = kickoff_formation(ctx, cfg)
    kickoffs = detect_kickoffs(ctx, formation, cfg)
    anchors, episodes = build_anchors(ctx, cfg, out_episodes(ctx, ball, cfg), kickoffs)

    params0 = initial_params(feat, anchors, graded, cfg, fps)
    params, history = hmm2.fit(feat["G"], feat["B"], sequences, params0, clamp1=anchors,
                               n_iter=cfg["em_iter"], stride=cfg["em_stride"])
    if params.mu[DEAD, 0] >= params.mu[LIVE, 0]:
        raise RuntimeError("EM converged with the 'dead' state faster than the 'live' one; "
                           "the anchors did not pin the states -- inspect the features")
    # Fitted on every em_stride-th frame, evaluated on every frame: temper by the stride so each
    # observation counts once (see hmm2.posterior).
    temperature = float(cfg["em_stride"])
    posterior, _, _ = hmm2.posterior(params, feat["G"], feat["B"], sequences, clamp1=anchors,
                                     temperature=temperature)
    loglik = hmm2.emission_loglik(params, feat["G"], feat["B"]) / temperature
    llr = loglik[:, DEAD] - loglik[:, LIVE]

    # Posterior runs -> raw intervals.
    dead = (np.nan_to_num(posterior) >= cfg["posterior_threshold"]) | anchors
    dead &= graded
    starts, stops = runs(dead)
    raw = merge_runs(starts, stops, int(cfg["merge_gap_s"] * fps))
    min_len = int(cfg["min_stoppage_s"] * fps)
    raw = [(a, b) for a, b in raw if b - a >= min_len or anchors[a:b].any()]

    codes, distance, slot = nearest_player(ctx)
    events = candidate_restarts(ctx, ball, distance, cfg)
    event_frames = np.array([f for f, _ in events], dtype=np.int64)
    by_first = sorted(((e.first, e) for e in episodes), key=lambda item: item[0])

    stoppages = []
    for half, first, last in ctx.halves:
        floor = first
        for a, b in raw:
            if not first <= a <= last:
                continue
            start, kind, episode = _snap_start(ctx, a, floor, by_first, feat, cfg, first)
            anchored = np.flatnonzero(anchors[a:b])
            last_anchor = a + int(anchored[-1]) if len(anchored) else None
            restart, restart_kind, kick = _snap_restart(ctx, start, a, b, last, kickoffs, events,
                                                        event_frames, cfg, kind, last_anchor)
            if restart <= start:
                continue
            if restart - start < min_len and kick is None and kind != "half_start":
                continue    # refinement left too little to be a real stoppage
            if kick is not None and not kick.at_half_start and kind != "half_start":
                # A kickoff mid-half only follows a goal. The ball is rarely tracked into the
                # net, so the kickoff, not the ball-out, is what identifies most goals.
                kind = "goal"
            stoppages.append(Stoppage(
                start=int(start), restart=int(restart), half=half, kind=kind,
                restart_kind=restart_kind,
                confidence=float(np.nanmean(posterior[start:restart])),
                out=episode, kickoff=kick))
            floor = restart

    mask = np.zeros(ctx.n_frames, dtype=bool)
    for stoppage in stoppages:
        mask[stoppage.start:stoppage.restart] = True
    strength = cfg["evidence_strength"]
    evidence = np.where(graded, np.where(mask, strength, -strength), 0.0)

    return DeadBallResult(
        posterior=posterior, llr=llr, anchors=anchors, mask=mask, evidence=evidence,
        stoppages=stoppages, out_episodes=episodes, kickoffs=kickoffs, params=params,
        history=history, features=feat, ball=ball, nearest=(codes, distance, slot))


# -- diagnostics (no ground truth) ---------------------------------------------------------------

def summary(ctx, result, tags=()):
    fps = ctx.fps
    graded = ctx.half_of_frame > 0
    durations = np.array([s.duration for s in result.stoppages]) / fps
    kinds, restart_kinds = {}, {}
    for s in result.stoppages:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
        restart_kinds[s.restart_kind] = restart_kinds.get(s.restart_kind, 0) + 1

    set_pieces = [t for t in tags if t.code == "SET PIECES"]
    restarts = np.array(sorted(s.restart for s in result.stoppages))
    covered = 0
    for tag in set_pieces:
        lo, hi = tag.start - 10 * fps, tag.start + 15 * fps
        i, j = np.searchsorted(restarts, [lo, hi + 1])
        covered += int(j > i)
    outs_inside = sum(bool(result.mask[e.first]) for e in result.out_episodes)

    p = result.params
    return {
        "n_stoppages": len(result.stoppages),
        "by_kind": dict(sorted(kinds.items())),
        "by_restart_kind": dict(sorted(restart_kinds.items())),
        "dead_share": float(result.mask[graded].mean()),
        "anchor_share": float(result.anchors[graded].mean()),
        "posterior_dead_share": float(np.nanmean(result.posterior[graded] >= 0.5)),
        "duration_s": {"p10": float(np.percentile(durations, 10)) if len(durations) else 0.0,
                       "median": float(np.median(durations)) if len(durations) else 0.0,
                       "p90": float(np.percentile(durations, 90)) if len(durations) else 0.0},
        "set_pieces_with_restart_nearby": "{}/{}".format(covered, len(set_pieces)),
        "out_episodes_inside_stoppage": "{}/{}".format(outs_inside, len(result.out_episodes)),
        "goals": [(s.start, s.restart) for s in result.stoppages if s.kind == "goal"],
        "kickoffs": [(k.frame, k.half, "start" if k.at_half_start else "",
                      {HOME: "H", AWAY: "A"}.get(k.kicker_team, "?")) for k in result.kickoffs],
        "hmm": {
            "live_speed_mps": float(np.exp(p.mu[LIVE, 0])),
            "dead_speed_mps": float(np.exp(p.mu[DEAD, 0])),
            "live_spread_mps": float(np.exp(p.mu[LIVE, 1]) - 0.02),
            "dead_spread_mps": float(np.exp(p.mu[DEAD, 1]) - 0.02),
            "p_ball_seen_live_dead": [float(p.p[LIVE, 0]), float(p.p[DEAD, 0])],
            "p_ball_moving_live_dead": [float(p.p[LIVE, 1]), float(p.p[DEAD, 1])],
            "mean_live_s": float(1.0 / max(p.p01, 1e-12) / fps),
            "mean_dead_s": float(1.0 / max(p.p10, 1e-12) / fps),
            "em_iterations": len(result.history),
        },
    }


def stoppage_table(result, fps=25):
    """A DataFrame for inspection (frames only -- safe to look at, not to publish wholesale)."""
    return pd.DataFrame([{
        "start": s.start, "restart": s.restart, "duration_s": s.duration / fps, "half": s.half,
        "kind": s.kind, "restart_kind": s.restart_kind, "confidence": round(s.confidence, 3),
    } for s in result.stoppages])
