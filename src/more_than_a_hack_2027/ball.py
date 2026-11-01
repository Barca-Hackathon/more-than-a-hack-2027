"""Ball track cleaning and kinematics, shared by the dead-ball and control stages.

What the provided ball track looks like (measured on the tactical ATD):
  * When present it is smooth -- only ~20 frames sit more than 1 m off a 5-frame median -- so
    cleaning is a light spike filter, not a tracker.
  * When absent it is absent for a long time (median gap 11 s), so there is nothing to
    interpolate. Gaps are left as NaN on purpose.
  * Apparent speeds above ~40 m/s are balls in the air: the track is the ground-plane projection
    of an elevated ball, which moves faster than the ball itself. Flagged as `aerial`.
  * The ball is reported beyond the pitch lines on ~4k graded frames -- a direct out-of-play
    signal, kept as a signed distance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BALL_DEFAULTS = {
    "spike_window": 5,       # frames, centred rolling median
    "spike_m": 1.0,          # a detection this far from the median is dropped
    "smooth_window": 3,      # frames, centred rolling mean after spike removal
    "aerial_mps": 40.0,      # apparent speed above this = ball in the air
}


@dataclass(frozen=True)
class BallTrack:
    pos: np.ndarray        # (n, 2) metres, NaN where unseen
    speed: np.ndarray      # (n,) m/s, NaN where it cannot be computed
    seen: np.ndarray       # (n,) bool
    outside: np.ndarray    # (n,) metres beyond the nearest pitch line (>0 outside), NaN unseen
    aerial: np.ndarray     # (n,) bool


def outside_distance(pos, pitch):
    """Signed distance beyond the pitch boundary: >0 outside, <0 inside (distance to nearest line)."""
    length, width = pitch
    x, y = pos[:, 0], pos[:, 1]
    return np.maximum.reduce([-x, x - length, -y, y - width])


def _speed(pos, fps):
    """Central-difference speed, falling back to one-sided at the edges of a seen run."""
    forward = np.full(len(pos), np.nan)
    forward[:-1] = np.hypot(*(pos[1:] - pos[:-1]).T) * fps
    backward = np.full(len(pos), np.nan)
    backward[1:] = forward[:-1]
    central = np.full(len(pos), np.nan)
    central[1:-1] = np.hypot(*(pos[2:] - pos[:-2]).T) * fps / 2.0
    return np.where(np.isnan(central), np.where(np.isnan(forward), backward, forward), central)


def clean_ball(ctx, cfg=None):
    cfg = dict(BALL_DEFAULTS, **(cfg or {}))
    raw = np.asarray(ctx.ball, dtype=np.float64)
    frame = pd.DataFrame(raw, columns=["x", "y"])

    window = int(cfg["spike_window"])
    median = frame.rolling(window, center=True, min_periods=window // 2 + 1).median()
    deviation = np.hypot(frame["x"] - median["x"], frame["y"] - median["y"]).to_numpy()
    spike = deviation > cfg["spike_m"]
    frame[spike] = np.nan

    smooth = frame.rolling(int(cfg["smooth_window"]), center=True, min_periods=1).mean()
    pos = smooth.to_numpy(copy=True)
    pos[np.isnan(frame["x"].to_numpy())] = np.nan    # never invent positions inside a gap

    seen = ~np.isnan(pos[:, 0])
    speed = _speed(pos, ctx.fps)
    return BallTrack(
        pos=pos,
        speed=speed,
        seen=seen,
        outside=np.where(seen, outside_distance(pos, ctx.pitch), np.nan),
        aerial=np.nan_to_num(speed) > cfg["aerial_mps"],
    )
