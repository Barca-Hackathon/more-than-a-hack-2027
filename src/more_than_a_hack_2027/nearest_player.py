"""Vectorised port of starter/baseline.py: the team of the tracked player nearest the ball.

Bit-exact with the reference implementation (tests/test_baseline_parity.py checks this against
the committed evaluation/results/baseline_*.csv), so it doubles as the regression anchor for the
data layer: if this ever stops matching, the parse or the alignment has changed.

Also returns the distance, which the evidence stages use and the baseline throws away.
"""

import numpy as np

NOBODY = -1


def nearest_player(ctx):
    """(codes, distance_m, slot) per frame.

    codes is HOME=0 / AWAY=1 / NOBODY=-1 (ball unseen, or no tracked player). distance_m and slot
    are NaN / -1 where codes is NOBODY. Ties go to the lowest slot index, as in the reference.
    """
    ball = ctx.ball
    dx = ctx.xy[:, :, 0] - ball[:, 0, None]
    dy = ctx.xy[:, :, 1] - ball[:, 1, None]
    squared = dx * dx + dy * dy
    del dx, dy

    observed = ~np.isnan(squared)
    answer = observed.any(axis=1)
    best = np.argmin(np.where(observed, squared, np.inf), axis=1)

    codes = np.full(ctx.n_frames, NOBODY, dtype=np.int8)
    codes[answer] = ctx.team_of_slot[best[answer]]
    slot = np.where(answer, best, -1).astype(np.int16)
    distance = np.full(ctx.n_frames, np.nan)
    distance[answer] = np.sqrt(squared[np.flatnonzero(answer), best[answer]])
    return codes, distance, slot
