"""A temporal CNN that reads the match as a sequence: per-frame log P(H), log P(A), log P(D).

Why. control.py scores each frame from hand-made windowed features with a linear model, and
deadball.py calls stoppages GT-free. On the held-out external games both stall: the control
model's team accuracy is ~0.92 whatever the model class (gradient boosting on the same features
gains < 0.01), and a missing ball reads as a stoppage. The information is in the order of events
-- who was at the ball before, who is next, how the players move around a gap -- which a network
with ~+-10 s of receptive field learns directly from the geometry.

Inputs (`inputs`), every frame, all computed inside one half:
    ball      seen flag, position and velocity in the home attack frame, speed, acceleration,
              metres beyond the nearest line
    home/away the 3 players nearest the ball (distance, offset, speed relative to the ball, own
              speed), how many are within 2 / 5 / 10 m, centroid position and velocity, mean
              speed, players visible
    dead      deadball.py's GT-free outputs (HMM posterior, stoppages, anchors, ball-out
              episodes) and windowed team-speed / ball-visibility statistics
    control   control.py's 18 antisymmetric features (0 where the ball is unseen)
    touch     optional (`touch=True`, models with arch["touch"]): per team, the spells in which
              one of its players is the ball's owner -- the player closest to the ball, within
              2 m -- followed by track identity, so a player who keeps the ball is told apart
              from a string of different players or a deflection: the current spell's length,
              time since the last spell and until the next, their lengths, and how many of the
              team's players owned the ball within +-3 s. Two blocks (home, away) at the end.

Team symmetry. `swap` maps the inputs to the same frame with the teams exchanged (and the pitch
rotated 180 degrees). Training sees both copies, and prediction averages the model on the frame and
on its swap (with H and A exchanged back), so the output is exactly team-symmetric: the model
cannot learn "home has the ball more".

Architecture (numpy forward pass here, the same graph in training.py for training):
    down   conv, kernel = stride = 5          full rate -> 5 Hz
    blocks residual: x + conv1x1(gelu(dilated conv k=3 (gelu(layernorm(x)))))
    up     transposed conv, kernel = stride = 5
    skip   conv k=5 on the full-rate inputs, added to `up`
    head   gelu, conv k=5, gelu, conv 1x1 -> 3 logits
The shipped model is plain JSON; prediction needs numpy only, and torch is only needed to train.
"""

import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._paths import CACHE_DIR, MODEL_DIR
from ..data.context import AWAY, HOME

# The shipped v3 ensemble: every models/seq_v3_*.json (trained on a Kaggle GPU by
# scripts/kaggle_seq.py v3-final), averaged by ensemble_logprobs.
ENSEMBLE_GLOB = os.path.join(MODEL_DIR, "seq_v3_*.json")
# Where `train-seq` writes a locally trained model unless told otherwise (not shipped).
DEFAULT_MODEL = os.path.join(CACHE_DIR, "seq_local.json")

K_NEAREST = 3
DEAD_WINDOWS = (25, 75, 200)


# -- inputs ---------------------------------------------------------------------------------------

def _per_half(ctx, values, fn):
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for _, first, last in ctx.halves:
        out[first:last + 1] = fn(values[first:last + 1])
    return out


def _gradient(seg, fps):
    if len(seg) < 3:
        return np.zeros_like(seg, dtype=np.float64)
    return np.gradient(seg, axis=0) * fps


def _roll(ctx, x, w, how, future=False):
    def fn(seg):
        s = pd.Series(seg[::-1] if future else seg)
        r = getattr(s.rolling(w, min_periods=1), how)().to_numpy()
        return r[::-1] if future else r
    return _per_half(ctx, x, fn)


def _since(ctx, mask, future=False, cap=750):
    """Frames since (until) the last (next) True inside each half, capped."""
    def fn(seg):
        m = seg[::-1] if future else seg
        idx = np.arange(len(m))
        last = np.maximum.accumulate(np.where(m, idx, -10 ** 9))
        d = np.minimum(idx - last, cap).astype(np.float64)
        return d[::-1] if future else d
    return _per_half(ctx, mask.astype(bool), fn)


def _attack_sign(ctx):
    sign = np.zeros(ctx.n_frames)
    for half, first, last in ctx.halves:
        sign[first:last + 1] = 1.0 if ctx.home_attacks_positive_x[half] else -1.0
    return sign


def dead_inputs(ctx, ball, dead):
    """(names, (n, F)) from deadball.detect's result: posterior, stoppages and windowed stats."""
    f = dead.features
    level = np.log(np.maximum(np.nan_to_num(f["speed"], nan=0.05), 0.05))
    seen = ball.seen.astype(np.float64)
    moving = np.where(ball.seen & np.isfinite(ball.speed), (ball.speed > 4.0).astype(np.float64),
                      0.0)
    post = np.clip(np.nan_to_num(dead.posterior, nan=0.5), 1e-4, 1 - 1e-4)
    cols = {
        "post_logit": np.log(post / (1 - post)),
        "llr": np.clip(np.nan_to_num(dead.llr), -20, 20),
        "mask": dead.mask.astype(np.float64) * 2 - 1,
        "anchor": dead.anchors.astype(np.float64),
        "lvl_c": np.log(np.maximum(np.nan_to_num(f["level"], nan=0.05), 0.05)),
        "spread": np.log(np.nan_to_num(f["spread"], nan=0.0) + 0.02),
    }
    for w in DEAD_WINDOWS:
        cols["lvl_p{}".format(w)] = _roll(ctx, level, w, "mean")
        cols["lvl_f{}".format(w)] = _roll(ctx, level, w, "mean", future=True)
        cols["seen_p{}".format(w)] = _roll(ctx, seen, w, "mean")
        cols["seen_f{}".format(w)] = _roll(ctx, seen, w, "mean", future=True)
        cols["mov_p{}".format(w)] = _roll(ctx, moving, w, "mean")
        cols["mov_f{}".format(w)] = _roll(ctx, moving, w, "mean", future=True)
    cols["acc_burst"] = cols["lvl_f25"] - cols["lvl_p25"]
    cols["acc_burst75"] = cols["lvl_f75"] - cols["lvl_p75"]
    cols["since_seen"] = np.log1p(_since(ctx, ball.seen))
    cols["until_seen"] = np.log1p(_since(ctx, ball.seen, future=True))
    out_mask = np.zeros(ctx.n_frames, dtype=bool)
    for episode in dead.out_episodes:
        out_mask[episode.first:episode.last + 1] = True
    cols["since_out"] = np.log1p(_since(ctx, out_mask))
    cols["until_out"] = np.log1p(_since(ctx, out_mask, future=True))
    outside = pd.Series(np.where(ball.seen, ball.outside, np.nan)).ffill().to_numpy()
    cols["last_outside"] = np.clip(np.nan_to_num(outside, nan=-10.0), -15, 5)
    starts = np.zeros(ctx.n_frames, dtype=bool)
    ends = np.zeros(ctx.n_frames, dtype=bool)
    for stoppage in dead.stoppages:
        starts[stoppage.start] = True
        ends[min(stoppage.restart, ctx.n_frames - 1)] = True
    cols["since_stop_start"] = np.log1p(_since(ctx, starts))
    cols["until_restart"] = np.log1p(_since(ctx, ends, future=True))
    names = tuple(sorted(cols))
    return names, np.column_stack([cols[k] for k in names])


def _team_block(ctx, team, bx, by, bvx, bvy, sign, pvx, pvy):
    slots = ctx.team_slots(team)
    x = ctx.xy[:, slots, 0]
    y = ctx.xy[:, slots, 1]
    dx = (x - bx[:, None]) * sign[:, None]
    dy = y - by[:, None]
    dist = np.hypot(dx, dy)
    finite = np.where(np.isnan(dist), np.inf, dist)
    order = np.argsort(finite, axis=1, kind="stable")[:, :K_NEAREST]
    rows = np.arange(len(x))
    vx = pvx[:, slots]
    vy = pvy[:, slots]
    cols = []
    for k in range(K_NEAREST):
        j = order[:, k]
        d = finite[rows, j]
        ok = np.isfinite(d)
        rvx = (vx[rows, j] - bvx) * sign
        rvy = vy[rows, j] - bvy
        vok = ok & np.isfinite(rvx) & np.isfinite(rvy)
        own = np.hypot(vx[rows, j], vy[rows, j])
        cols += [
            np.where(ok, np.minimum(d, 30.0) / 10.0, 3.0),
            np.where(ok, np.clip(dx[rows, j], -30, 30) / 10.0, 0.0),
            np.where(ok, np.clip(dy[rows, j], -30, 30) / 10.0, 0.0),
            np.where(vok, np.clip(np.hypot(rvx, rvy), 0, 30) / 10.0, 1.0),
            np.where(vok, np.clip(rvx, -30, 30) / 10.0, 0.0),
            np.where(ok & np.isfinite(own), np.clip(own, 0, 12) / 5.0, 0.0),
        ]
    for radius in (2.0, 5.0, 10.0):
        cols.append((finite < radius).sum(axis=1) / 5.0)
    visible = (~np.isnan(x)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        cx = np.where(visible > 0, np.nansum(x, axis=1) / np.maximum(visible, 1), np.nan)
        cvx = np.where(visible > 0, np.nansum(np.nan_to_num(vx), axis=1) / np.maximum(visible, 1),
                       np.nan)
        mean_speed = np.where(visible > 0, np.nansum(np.nan_to_num(np.hypot(vx, vy)), axis=1)
                              / np.maximum(visible, 1), np.nan)
    length = ctx.pitch[0]
    cols += [
        np.nan_to_num((cx - length / 2.0) * sign / (length / 2.0)),
        np.nan_to_num((cx - bx) * sign / 30.0),
        np.nan_to_num(cvx * sign / 5.0),
        np.nan_to_num(mean_speed / 5.0),
        visible / 11.0,
    ]
    return np.column_stack(cols)


TEAM_BLOCK = 6 * K_NEAREST + 3 + 5
BALL_BLOCK = 8
# Under a team swap (teams exchanged, pitch rotated 180 degrees): per-channel sign, and the two
# team blocks trade places. Ball: seen, x, y, vx, vy, speed, accel, outside.
BALL_SWAP = np.array([1, -1, -1, -1, -1, 1, 1, 1], dtype=np.float64)
TEAM_SWAP = np.array([1, -1, -1, 1, -1, 1] * K_NEAREST + [1, 1, 1] + [-1, -1, -1, 1, 1],
                     dtype=np.float64)
# Under a left-right mirror (y -> width - y), which changes nothing about who has the ball: only
# the y components flip. The dead-ball and control inputs do not depend on y's sign.
BALL_FLIP = np.array([1, 1, -1, 1, -1, 1, 1, 1], dtype=np.float64)
TEAM_FLIP = np.array([1, 1, -1, 1, 1, 1] * K_NEAREST + [1, 1, 1] + [1, 1, 1, 1, 1],
                     dtype=np.float64)


def geometry_inputs(ctx, ball):
    """(n, BALL_BLOCK + 2 * TEAM_BLOCK): the ball, then the home block, then the away block."""
    length, width = ctx.pitch
    sign = _attack_sign(ctx)
    pos = ball.pos
    seen = ball.seen
    bx = pd.Series(pos[:, 0]).ffill().bfill().to_numpy()
    by = pd.Series(pos[:, 1]).ffill().bfill().to_numpy()
    bxy = np.column_stack([bx, by])
    bv = np.zeros_like(bxy)
    acc = np.zeros_like(bxy)
    pv = np.full(ctx.xy.shape, np.nan)
    for _, first, last in ctx.halves:
        part = slice(first, last + 1)
        bv[part] = _gradient(bxy[part], ctx.fps)
        acc[part] = _gradient(bv[part], ctx.fps)
        pv[part] = _gradient(ctx.xy[part], ctx.fps)
    bvx, bvy = bv[:, 0], bv[:, 1]
    ball_cols = np.column_stack([
        seen.astype(np.float64),
        np.clip((bx - length / 2.0) * sign / (length / 2.0), -1.3, 1.3),
        np.clip((by - width / 2.0) / (width / 2.0), -1.3, 1.3),
        np.clip(bvx * sign / 10.0, -3, 3),
        np.clip(bvy / 10.0, -3, 3),
        np.clip(np.hypot(bvx, bvy) / 10.0, 0, 3),
        np.clip(np.hypot(acc[:, 0], acc[:, 1]) / 20.0, 0, 3),
        np.clip(np.nan_to_num(ball.outside, nan=0.0) / 5.0, -3, 2),
    ])
    home = _team_block(ctx, HOME, bx, by, bvx, bvy, sign, pv[..., 0], pv[..., 1])
    away = _team_block(ctx, AWAY, bx, by, bvx, bvy, sign, pv[..., 0], pv[..., 1])
    return np.column_stack([ball_cols, home, away])


def inputs(ctx, ball=None, dead=None, control_cfg=None, touch=False):
    """(n, C) float32 model inputs, and the per-channel swap signs (C,).

    Layout: ball | home | away | dead (team-free) | control (antisymmetric), then with
    touch=True: home touch | away touch.
    """
    from ..stages import control, deadball
    from ..stages.ball import clean_ball

    ball = ball if ball is not None else clean_ball(ctx)
    dead = dead if dead is not None else deadball.detect(ctx, ball)
    geo = geometry_inputs(ctx, ball)
    _, dead_x = dead_inputs(ctx, ball, dead)
    ctrl = np.nan_to_num(control.features(ctx, ball, control_cfg))
    X = np.column_stack([geo, np.nan_to_num(dead_x), ctrl]).astype(np.float32)
    swap_sign = np.concatenate([BALL_SWAP, TEAM_SWAP, TEAM_SWAP,
                                np.ones(dead_x.shape[1]), -np.ones(ctrl.shape[1])])
    if touch:
        if X.shape[1] != N_BASE:
            raise ValueError("touch inputs need the {}-channel base layout, got {}"
                             .format(N_BASE, X.shape[1]))
        X = np.column_stack([X, touch_inputs(ctx, ball)]).astype(np.float32)
        swap_sign = np.concatenate([swap_sign, np.ones(2 * TOUCH_BLOCK)])
    return X, swap_sign


def has_touch(n_channels):
    return n_channels == N_BASE + 2 * TOUCH_BLOCK


# -- touch inputs -------------------------------------------------------------------------------

N_BASE = 111               # channels without the touch blocks (checked in `inputs`)
TOUCH_BLOCK = 6
TOUCH_RADIUS_M = 2.0       # the ball's owner is the closest player within this distance
TOUCH_GAP = 5              # frames an owner may drop out (a ball flicker) inside one spell
TOUCH_WINDOW_S = 3.0       # distinct owners counted within +- this


def ball_owner(ctx, ball, radius=TOUCH_RADIUS_M, gap=TOUCH_GAP):
    """(n,) slot of the player closest to the ball when within `radius`, else -1. A gap of at
    most `gap` frames between two frames owned by the same slot is filled."""
    n = ctx.n_frames
    pos = ball.pos
    with np.errstate(invalid="ignore"):
        d = np.hypot(ctx.xy[..., 0] - pos[:, 0:1], ctx.xy[..., 1] - pos[:, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    owner = np.where(d[np.arange(n), j] <= radius, j, -1)
    graded = np.zeros(n, bool)
    for _, first, last in ctx.halves:
        graded[first:last + 1] = True
        seg = owner[first:last + 1]
        idx = np.flatnonzero(seg >= 0)
        if len(idx) < 2:
            continue
        holes = np.flatnonzero((np.diff(idx) > 1) & (np.diff(idx) <= gap + 1)
                               & (seg[idx[:-1]] == seg[idx[1:]]))
        for k in holes:
            seg[idx[k] + 1:idx[k + 1]] = seg[idx[k]]
    owner[~graded] = -1
    return owner


def _owner_runs(owner):
    """[(start, stop, slot)] maximal runs of one owner (slot >= 0)."""
    edges = np.flatnonzero(np.diff(owner) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(owner)]))
    return [(int(a), int(b), int(owner[a])) for a, b in zip(bounds[:-1], bounds[1:])
            if owner[a] >= 0]


def _team_touch(owner_seg, team_of_slot, team, fps):
    """(m, TOUCH_BLOCK) for one team inside one half."""
    m = len(owner_seg)
    runs = [r for r in _owner_runs(owner_seg) if team_of_slot[r[2]] == team]
    cur = np.zeros(m)
    end_len = np.full(m, np.nan)        # run length placed at the frame its run ends
    start_len = np.full(m, np.nan)      # ... and at the frame it starts
    ends = np.zeros(m, bool)
    starts = np.zeros(m, bool)
    inside = np.zeros(m, bool)
    for a, b, _ in runs:
        length = (b - a) / fps
        cur[a:b] = length
        inside[a:b] = True
        if b < m:
            end_len[b] = length
            ends[b] = True
        start_len[a] = length
        starts[a] = True
    idx = np.arange(m)
    last_end = np.maximum.accumulate(np.where(ends, idx, -10 ** 9))
    since = np.where(inside, 0.0, (idx - last_end) / fps)
    nxt = np.minimum.accumulate(np.where(starts, idx, 10 ** 9)[::-1])[::-1]
    until = np.where(inside, 0.0, (nxt - idx) / fps)
    last_len = pd.Series(end_len).ffill().fillna(0.0).to_numpy()
    following = pd.Series(start_len).bfill().to_numpy()
    next_len = np.r_[following[1:], np.nan]
    next_len = np.where(np.isnan(next_len), 0.0, next_len)
    # distinct owners of this team within +-TOUCH_WINDOW_S
    w = int(TOUCH_WINDOW_S * fps)
    distinct = np.zeros(m)
    for slot in sorted({r[2] for r in runs}):
        own = np.zeros(m + 1)
        for a, b, s in runs:
            if s == slot:
                own[a + 1:b + 1] = 1.0
        c = np.cumsum(own)
        lo = np.clip(idx - w, 0, m)
        hi = np.clip(idx + w + 1, 0, m)
        distinct += (c[hi] - c[lo]) > 0
    return np.column_stack([
        np.minimum(cur, 6.0) / 4.0,
        np.minimum(since, 5.0) / 5.0,
        np.minimum(until, 5.0) / 5.0,
        np.minimum(last_len, 6.0) / 4.0,
        np.minimum(next_len, 6.0) / 4.0,
        np.minimum(distinct, 6.0) / 3.0,
    ])


def touch_inputs(ctx, ball):
    """(n, 2 * TOUCH_BLOCK): the home touch block, then the away one (0 outside the halves)."""
    owner = ball_owner(ctx, ball)
    out = np.zeros((ctx.n_frames, 2 * TOUCH_BLOCK))
    for _, first, last in ctx.halves:
        seg = owner[first:last + 1]
        for k, team in enumerate((HOME, AWAY)):
            out[first:last + 1, k * TOUCH_BLOCK:(k + 1) * TOUCH_BLOCK] = _team_touch(
                seg, ctx.team_of_slot, team, ctx.fps)
    return out


def flip_sign(n_channels):
    """(C,) signs that mirror the inputs left-right (network.inputs' layout)."""
    geo = np.concatenate([BALL_FLIP, TEAM_FLIP, TEAM_FLIP])
    return np.concatenate([geo, np.ones(n_channels - len(geo))])


def swap(X, swap_sign):
    """The same frames with the teams exchanged: home and away blocks trade places."""
    out = X * swap_sign.astype(X.dtype)
    a = BALL_BLOCK
    b = a + TEAM_BLOCK
    c = b + TEAM_BLOCK
    out[:, a:b], out[:, b:c] = out[:, b:c].copy(), out[:, a:b].copy()
    if has_touch(X.shape[1]):
        d = N_BASE + TOUCH_BLOCK
        out[:, N_BASE:d], out[:, d:] = out[:, d:].copy(), out[:, N_BASE:d].copy()
    return out


# -- the network, numpy forward ------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _conv(x, w, b, dilation=1):
    """x (T, Cin), w (Cout, Cin, k), 'same' zero padding (k odd)."""
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    out = np.zeros((x.shape[0], w.shape[0]), dtype=x.dtype) + b
    for j in range(k):
        out += xp[j * dilation:j * dilation + x.shape[0]] @ w[:, :, j].T
    return out


def _layernorm(x, g, b, eps=1e-5):
    m = x.mean(axis=1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


@dataclass
class SeqModel:
    arch: dict                      # stride, channels, dilations
    scale: np.ndarray               # (C,) input divisor
    weights: dict                   # name -> array (torch layout)
    info: dict = field(default_factory=dict)

    def forward(self, X):
        """(T, C) inputs of one half -> (T, 3) log-probabilities for H, A, D."""
        W = self.weights
        stride = int(self.arch["stride"])
        T = len(X)
        pad = (-T) % stride
        x = np.pad((X / self.scale).astype(np.float64), ((0, pad), (0, 0)))
        n = len(x) // stride
        xr = x.reshape(n, stride, x.shape[1])
        h = np.einsum("tjc,ocj->to", xr, W["down.weight"]) + W["down.bias"]
        for i, d in enumerate(self.arch["dilations"]):
            p = "blocks.{}.".format(i)
            z = _layernorm(h, W[p + "n.weight"], W[p + "n.bias"])
            z = _conv(gelu(z), W[p + "c1.weight"], W[p + "c1.bias"], dilation=d)
            z = gelu(z) @ W[p + "c2.weight"][:, :, 0].T + W[p + "c2.bias"]
            h = h + z
        up = np.einsum("tc,coj->tjo", h, W["up.weight"]).reshape(n * stride, -1) + W["up.bias"]
        u = up + _conv(x, W["skip.weight"], W["skip.bias"])
        z = _conv(gelu(u), W["head.1.weight"], W["head.1.bias"])
        logits = gelu(z) @ W["head.3.weight"][:3, :, 0].T + W["head.3.bias"][:3]
        logits = logits[:T]                        # head 0; any auxiliary head is training-only
        m = logits.max(axis=1, keepdims=True)
        return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))

    def to_json(self, path):
        payload = {"arch": self.arch, "scale": [float(v) for v in self.scale],
                   "weights": {k: {"shape": list(v.shape),
                                   "data": [float(x) for x in np.asarray(v, np.float32).ravel()]}
                               for k, v in sorted(self.weights.items())},
                   "info": self.info}
        with open(path, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")

    @classmethod
    def from_json(cls, path):
        with open(path) as handle:
            payload = json.load(handle)
        weights = {k: np.asarray(v["data"], dtype=np.float32).astype(np.float64).reshape(v["shape"])
                   for k, v in payload["weights"].items()}
        return cls(payload["arch"], np.asarray(payload["scale"], dtype=np.float64), weights,
                   payload.get("info", {}))


def average_logprobs(logprobs_list):
    """An ensemble's (n, 3) log-probabilities: the mean of its members', renormalised (a
    normalised geometric mean of the probabilities)."""
    mean = np.mean([np.asarray(lp, dtype=np.float64) for lp in logprobs_list], axis=0)
    m = mean.max(axis=1, keepdims=True)
    return mean - m - np.log(np.exp(mean - m).sum(axis=1, keepdims=True))


def load_ensemble(pattern=ENSEMBLE_GLOB):
    """The models matching `pattern`, in path order (the averaging order is fixed)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no sequence model matches {}".format(pattern))
    return [SeqModel.from_json(p) for p in paths]


def ensemble_logprobs(ctx, models, X, swap_sign):
    """logprobs() of each model, averaged (average_logprobs) inside the halves, 0 elsewhere."""
    lps = [logprobs(ctx, m, X, swap_sign) for m in models]
    if len(lps) == 1:
        return lps[0]
    out = np.zeros_like(lps[0])
    for _, first, last in ctx.halves:
        part = slice(first, last + 1)
        out[part] = average_logprobs([lp[part] for lp in lps])
    return out


def logprobs(ctx, model, X, swap_sign):
    """(n, 3) team-symmetric log P(H), log P(A), log P(D) inside the halves, 0 elsewhere."""
    return logprobs_halves(model, X, swap_sign, ctx.halves)


def logprobs_halves(model, X, swap_sign, halves):
    out = np.zeros((len(X), 3))
    Xs = swap(X, swap_sign)
    views = [(X, False), (Xs, True)]
    if model.arch.get("flip_tta"):
        mirror = flip_sign(X.shape[1]).astype(X.dtype)
        views += [(X * mirror, False), (Xs * mirror, True)]
    for _, first, last in halves:
        part = slice(first, last + 1)
        preds = []
        for V, swapped in views:
            p = model.forward(V[part])
            preds.append(p[:, [1, 0, 2]] if swapped else p)
        mean = np.mean(preds, axis=0)              # geometric mean of the predictions ...
        m = mean.max(axis=1, keepdims=True)        # ... renormalised
        out[part] = mean - m - np.log(np.exp(mean - m).sum(axis=1, keepdims=True))
    return out
