"""Make Metrica's clean tracking look like the competition ATD, so external scores transfer.

Every number here is calibrated on the competition INPUT (the ATD tracking), never on the ground
truth. What was measured, BAR-ATH graded frames vs the three Metrica games:

  ball, missing         40.7% in 150 gaps, 99.7% of it in gaps > 5 s  |  32-41%, 97%: already
                        alike, and in both the ball is mostly missing while it is dead. Kept.
  ball, smoothness      acceleration autocorrelation 0.989 at lag 1, 0.859 at lag 5; |accel|
                        p50 5.7, p90 18, p99 37 m/s^2  |  Metrica is piecewise linear (lag-1
                        -0.45, instantaneous kicks). A grid fit over (Gaussian sigma, white noise)
                        picked sigma = 7 frames, 0.5 m: 0.989 / 0.822 and 4.2 / 17.6 / 31.4.
                        Peak speeds come out lower (p99 19 vs 25 m/s): a known residual.
  players per frame     20.5 (p5 14); dropout is not spatial (the visible x-range does not shrink
                        when players go missing)  |  22.0 in Metrica. We replay the ATD's own
                        per-frame missing count from a seeded offset, and choose who is missing
                        by a per-player priority that is redrawn every ~35 s (the ATD's median
                        track run), so the same players stay missing for a while.
  track identity        1219 track runs on 46 slots, no IDs  |  one slot per player. After dropout,
                        every visible run gets a fresh slot from its team's pool, so a slot is a
                        track fragment as in the ATD, not a player.
  player smoothness     player |accel| p50 1.1 m/s^2 vs 1.4-1.8: Metrica is already the noisier
                        one, so player positions are left as they are.
  ball, in live play    6.7% of the frames v2 calls live have no ball (Nov 1, 50.7 live
                        minutes)  |  0.5-1.2% in Metrica, almost all 2-3 frame flickers. BAR-ATH's
                        come in three kinds, replayed by live_gaps() (LIVE_GAPS):
                          - inside live play, 10 losses of 5.3-11.2 s (and 13 flickers under
                            0.5 s, which Metrica already has);
                          - before a stoppage: 38 of 143 stoppages lose the ball first, 24 of
                            them within 1.4 s and the rest 2.6-9.7 s;
                          - after a restart: 8 of 143 restarts pick it up 2.1-10.9 s late.
                        These are measured against v2's own live/dead calls, so they are GT-free;
                        "before a stoppage" also holds some of v2's own late stop calls.
                        Placing them needs to know where play is live on the Metrica game, which
                        only its labels tell: degrade(live=...) takes that mask from the labels.
                        Gaps only hide the ball; they never add information.

The missing-count series is read from the competition ATD at run time (it is match-derived, so it
is never written into the repository). Without the data, tests pass their own series.
"""

import heapq

import numpy as np

DEGRADE_DEFAULTS = {
    "seed": 0,
    "ball_sigma_frames": 7.0,
    "ball_noise_m": 0.5,
    "dropout": True,
    "priority_renewal_s": 35.0,
    "fragment": True,
    "full_squad": 22,
    "live_gaps": True,          # hide the ball in live play as BAR-ATH does (needs live=)
    "live_gap_params": None,    # overrides of LIVE_GAPS, for experiments
}

# Rounded summaries of the BAR-ATH measurement in the module docstring.
LIVE_GAPS = {
    "interior_per_live_min": 0.20,      # 10 in 50.7 live minutes
    "interior_s": (5.0, 11.0),
    "before_stop_p": 0.27,              # 38 / 143
    "before_stop_short_p": 0.63,        # 24 / 38
    "before_stop_short_s": (0.04, 1.4),
    "before_stop_long_s": (2.5, 10.0),
    "after_restart_p": 0.056,           # 8 / 143
    "after_restart_s": (2.0, 11.0),
}


def _runs(seen):
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def gaussian_smooth(values, sigma):
    """Gaussian smoothing of an (n, d) track inside each visible run; NaN stays NaN.

    Each run is padded by odd reflection (linear extrapolation) before convolving, so the ends
    of a track keep their trend instead of being pulled back towards earlier positions -- what a
    tracker's Kalman/RTS smoother does too. That matters: Metrica's ball tracks often end right
    after crossing a line, and a plain normalised convolution would smooth the out away.
    """
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0:
        return values.copy()
    radius = int(np.ceil(3 * sigma))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.full_like(values, np.nan)
    for a, b in _runs(~np.isnan(values[:, 0])):
        run = values[a:b]
        if b - a < 3:
            out[a:b] = run
            continue
        pad = min(radius, b - a - 1)
        left = 2 * run[0] - run[pad:0:-1]
        right = 2 * run[-1] - run[-2:-pad - 2:-1]
        padded = np.concatenate([left, run, right])
        if pad < radius:                       # short run: extend flat beyond the reflection
            extra = radius - pad
            padded = np.concatenate([np.repeat(padded[:1], extra, axis=0), padded,
                                     np.repeat(padded[-1:], extra, axis=0)])
        for c in range(values.shape[1]):
            out[a:b, c] = np.convolve(padded[:, c], kernel, mode="valid")
    return out


def live_gaps(live, rng, fps=25, params=None):
    """(n,) bool: frames whose ball to hide, given the (n,) bool mask of live play.

    Per live run, in this order: a Poisson number of losses strictly inside it, then possibly
    one that ends at its stoppage, then possibly one that starts at its restart. A run that
    starts at frame 0 has no restart, and one that ends at the last frame has no stoppage.
    """
    params = dict(LIVE_GAPS, **(params or {}))
    live = np.asarray(live, dtype=bool)
    n = len(live)
    hide = np.zeros(n, dtype=bool)

    def frames(bounds):
        return max(1, int(round(rng.uniform(*bounds) * fps)))

    for a, b in _runs(live):
        a, b = int(a), int(b)
        for _ in range(rng.poisson(params["interior_per_live_min"] * (b - a) / (fps * 60.0))):
            m = frames(params["interior_s"])
            if b - a - m - 2 > 0:                       # seen on both sides
                start = a + 1 + rng.randint(b - a - m - 1)
                hide[start:start + m] = True
        if b < n and rng.uniform() < params["before_stop_p"]:
            short = rng.uniform() < params["before_stop_short_p"]
            m = frames(params["before_stop_short_s" if short else "before_stop_long_s"])
            hide[max(a, b - m):b] = True
        if a > 0 and rng.uniform() < params["after_restart_p"]:
            m = frames(params["after_restart_s"])
            hide[a:min(b, a + m)] = True
    return hide


def atd_missing_counts(ctx):
    """Per graded frame of the competition match: how many of 22 players the ATD is missing."""
    graded = ctx.half_of_frame > 0
    observed = (~np.isnan(np.asarray(ctx.xy)[graded, :, 0])).sum(axis=1)
    return np.clip(22 - observed, 0, None).astype(np.int16)


def _priorities(rng, n, n_players, renewal_frames):
    """(n, P) float32: a uniform priority per player, redrawn at Poisson renewal times."""
    out = np.empty((n, n_players), dtype=np.float32)
    for p in range(n_players):
        t = 0
        while t < n:
            length = max(1, int(round(rng.exponential(renewal_frames))))
            out[t:t + length, p] = rng.uniform()
            t += length
    return out


def drop_players(xy, speed, missing, rng, renewal_frames):
    """NaN out `missing[t]` of the visible players at each frame, chosen by sticky priority."""
    n, n_players = xy.shape[:2]
    visible = ~np.isnan(xy[:, :, 0])
    priority = _priorities(rng, n, n_players, renewal_frames)
    priority[~visible] = -1.0
    rank = np.argsort(np.argsort(-priority, axis=1, kind="stable"), axis=1, kind="stable")
    k = np.minimum(missing, visible.sum(axis=1))
    drop = visible & (rank < k[:, None])
    xy, speed = xy.copy(), speed.copy()
    xy[drop] = np.nan
    speed[drop] = np.nan
    return xy, speed


def fragment(xy, speed, team_of_slot):
    """Reassign every visible run of every player to the lowest free slot of its team."""
    n = xy.shape[0]
    visible = ~np.isnan(xy[:, :, 0])
    new_xy, new_speed, new_team = [], [], []
    for team in np.unique(team_of_slot):
        runs = []
        for p in np.flatnonzero(team_of_slot == team):
            edges = np.diff(np.concatenate(([0], visible[:, p].astype(np.int8), [0])))
            runs += [(a, b, p) for a, b in zip(np.flatnonzero(edges == 1),
                                               np.flatnonzero(edges == -1))]
        runs.sort()
        free, busy, assignment, n_slots = [], [], [], 0
        for a, b, p in runs:
            while busy and busy[0][0] <= a:
                heapq.heappush(free, heapq.heappop(busy)[1])
            if free:
                slot = heapq.heappop(free)
            else:
                slot, n_slots = n_slots, n_slots + 1
            heapq.heappush(busy, (b, slot))
            assignment.append((a, b, p, slot))
        team_xy = np.full((n, n_slots, 2), np.nan)
        team_speed = np.full((n, n_slots), np.nan, dtype=np.float32)
        for a, b, p, slot in assignment:
            team_xy[a:b, slot] = xy[a:b, p]
            team_speed[a:b, slot] = speed[a:b, p]
        new_xy.append(team_xy)
        new_speed.append(team_speed)
        new_team.append(np.full(n_slots, team, dtype=np.int8))
    return (np.concatenate(new_xy, axis=1), np.concatenate(new_speed, axis=1),
            np.concatenate(new_team))


def degrade(arrays, missing_counts=None, cfg=None, game=0, live=None):
    """A degraded copy of external.load_arrays() output. Deterministic for a given seed + game.

    missing_counts: the ATD's per-frame missing-player series (atd_missing_counts); it is replayed
    cyclically from a seeded offset. None skips player dropout.
    live: (n,) bool, where play is live (from the game's labels); None skips the live-play ball
    gaps. They draw from their own random stream, so switching them off leaves every other
    degradation identical and on/off comparisons are paired.
    """
    cfg = dict(DEGRADE_DEFAULTS, **(cfg or {}))
    rng = np.random.RandomState(int(cfg["seed"]) * 1000 + int(game))
    out = dict(arrays)

    ball = np.asarray(arrays["ball"], dtype=np.float64)
    if cfg["live_gaps"] and live is not None:
        gap_rng = np.random.RandomState([int(cfg["seed"]), int(game), 1])
        ball = ball.copy()
        ball[live_gaps(live, gap_rng, params=cfg["live_gap_params"])] = np.nan
    noisy = ball + rng.normal(0.0, cfg["ball_noise_m"], size=ball.shape)
    out["ball"] = gaussian_smooth(noisy, cfg["ball_sigma_frames"])

    xy = np.asarray(arrays["xy"], dtype=np.float64)
    speed = np.asarray(arrays["speed"], dtype=np.float32)
    n = xy.shape[0]
    if cfg["dropout"] and missing_counts is not None and len(missing_counts):
        offset = rng.randint(len(missing_counts))
        missing = np.take(missing_counts, (offset + np.arange(n)) % len(missing_counts))
        xy, speed = drop_players(xy, speed, missing, rng, cfg["priority_renewal_s"] * 25)
    team_of_slot = np.asarray(arrays["team_of_slot"])
    if cfg["fragment"]:
        xy, speed, team_of_slot = fragment(xy, speed, team_of_slot)
    out.update(xy=xy, speed=speed, team_of_slot=team_of_slot)
    return out


def describe(arrays, graded, live=None):
    """The input statistics the calibration targets, for a before/after printout."""
    ball = np.where(graded[:, None], np.asarray(arrays["ball"]), np.nan)
    v = np.diff(ball, axis=0) * 25
    a = np.diff(v, axis=0)
    def autocorr(lag):
        m = ~np.isnan(a[:-lag, 0]) & ~np.isnan(a[lag:, 0])
        return float(np.corrcoef(a[:-lag, 0][m], a[lag:, 0][m])[0, 1])
    accel = np.hypot(a[:, 0], a[:, 1]) * 25
    observed = (~np.isnan(np.asarray(arrays["xy"])[graded, :, 0])).sum(axis=1)
    return {
        "ball_missing": float(np.isnan(ball[graded, 0]).mean()),
        "ball_accel_autocorr_1": autocorr(1),
        "ball_accel_autocorr_5": autocorr(5),
        "ball_accel_p50_p90_p99": [float(x) for x in np.nanpercentile(accel, [50, 90, 99])],
        "ball_missing_live": (float(np.isnan(np.asarray(arrays["ball"])[live, 0]).mean())
                              if live is not None else None),
        "players_per_frame": float(observed.mean()),
        "n_slots": int(np.asarray(arrays["team_of_slot"]).shape[0]),
    }
