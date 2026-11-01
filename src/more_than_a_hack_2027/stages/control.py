"""Which team controls the ball, while it is visible: per-frame H/A evidence for the decoder.

Why a learned model. The nearest player is the right team on only ~75% of live frames on the
degraded external games: during a pass, a cross or a clearance the ball flies past opponents, and
the competition ground truth is "last team in control", which does not switch on a duel or a
deflection. With the true team on every live frame, v1 would gain ~0.15 spellF1 on those games
while perfect dead-ball calls would add ~0.01, so this stage is where the score is.

Model. A logistic regression on windowed features of the ball and both teams, trained on the
Metrica games' event-derived labels (metrica.py) on degraded tracking (degrade.py), never on
BAR-ATH. Every feature is written from the home team's side and is ANTISYMMETRIC: swapping the
two teams negates it. With no intercept the model is therefore exactly team-symmetric -- it
cannot learn "home has the ball more", which would not transfer between games.

    closeness       log-ratio of the two teams' nearest distances to the ball, now and as the
                    minimum over past / future windows (who was / will be at the ball)
    carrying        a player within carry_m whose velocity matches the ball's (relative speed
                    under carry_rel_mps): the ball is moving WITH him, not past him. Window
                    fractions, past and future.
    last / next     the team of the most recent (and of the next) carrying frame, decayed with
                    the time since (until) it: "last team in control" directly
    direction       ball velocity and position along each team's attack direction
    team shape      centroid velocity along the attack direction (the team in possession moves
                    up), and players within press_m of the ball

The evidence handed to the decoder is w_control * log P(team) per frame, and 0 for both teams
where the ball is unseen (the decoder then carries the last team, which is what the ground truth
does too). Frames are highly correlated at 25 fps, so w_control is well below 1; it is chosen on
the external games together with the decoder's switch penalty.

Every window is computed inside one half (ctx.half_slices) and never across a boundary.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._paths import MODEL_DIR
from ..data.context import AWAY, HOME
from .ball import clean_ball

CONTROL_DEFAULTS = {
    "far_m": 30.0,              # distance used when a team has no tracked player
    "eps_m": 0.5,               # added inside the log-ratios
    "carry_m": 2.0,
    "carry_rel_mps": 3.0,
    "press_m": 5.0,
    "velocity_half_window": 2,  # frames, central differences for ball and player velocity
    "push_window_s": 2.0,
    "decay_s": 5.0,             # time constant of the last / next control features
    "windows_s": (0.5, 1.0, 2.0),
    "carry_windows_s": (1.0, 3.0),
    "zmax": 8.0,                # clip on the log-odds before it becomes evidence
}

DEFAULT_MODEL = os.path.join(MODEL_DIR, "control_v1.json")


@dataclass(frozen=True)
class ControlModel:
    names: tuple
    scale: np.ndarray       # per-feature divisor (RMS on the training set); no centring
    weights: np.ndarray     # on the scaled features; no intercept, by design
    meta: dict

    def logodds(self, X):
        """log P(H) - log P(A) per row; 0 where any feature is NaN (ball unseen)."""
        Z = np.asarray(X, dtype=np.float64) / self.scale
        z = Z @ self.weights
        return np.where(np.isnan(z), 0.0, z)

    def to_json(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {"names": list(self.names), "scale": [float(v) for v in self.scale],
                   "weights": [float(v) for v in self.weights], "meta": self.meta}
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_json(cls, path=DEFAULT_MODEL):
        with open(path) as handle:
            payload = json.load(handle)
        return cls(tuple(payload["names"]), np.array(payload["scale"]),
                   np.array(payload["weights"]), payload.get("meta", {}))


# -- features ------------------------------------------------------------------------------------

def _velocity(pos, k, fps):
    """(n, 2) central-difference velocity over +-k frames; NaN at the edges of a visible run."""
    v = np.full(pos.shape, np.nan)
    if len(pos) > 2 * k:
        v[k:-k] = (pos[2 * k:] - pos[:-2 * k]) * fps / (2.0 * k)
    return v


def _team_nearest(xy, ball, slots):
    """(distance, index into `slots`) of the team's player nearest the ball; NaN / -1 if none."""
    d = np.hypot(xy[:, slots, 0] - ball[:, None, 0], xy[:, slots, 1] - ball[:, None, 1])
    observed = ~np.isnan(d)
    any_obs = observed.any(axis=1)
    best = np.argmin(np.where(observed, d, np.inf), axis=1)
    dist = np.where(any_obs, d[np.arange(len(d)), best], np.nan)
    return dist, np.where(any_obs, best, -1), d


def _rolling(values, window, how, future=False):
    s = pd.Series(values[::-1] if future else values)
    r = getattr(s.rolling(window, min_periods=1), how)().to_numpy()
    return r[::-1] if future else r


def _decayed_last(team_sign, tau, future=False):
    """+-1 decayed by the frames since the latest nonzero entry of team_sign (or until the next)."""
    sign = team_sign[::-1] if future else team_sign
    idx = np.where(sign != 0, np.arange(len(sign)), -1)
    last = np.maximum.accumulate(idx)
    value = np.where(last >= 0, sign[np.maximum(last, 0)], 0.0)
    age = np.arange(len(sign)) - last
    out = np.where(last >= 0, value * np.exp(-age / tau), 0.0)
    return out[::-1] if future else out


def feature_names(cfg=None):
    cfg = dict(CONTROL_DEFAULTS, **(cfg or {}))
    names = ["near"]
    for w in cfg["windows_s"]:
        names += ["near_past_{}s".format(w), "near_future_{}s".format(w)]
    names.append("carry")
    for w in cfg["carry_windows_s"]:
        names += ["carry_past_{}s".format(w), "carry_future_{}s".format(w)]
    names += ["last_carry", "next_carry", "ball_vx_attack", "ball_x_attack", "team_push",
              "press"]
    return tuple(names)


def _half_features(ctx, xy, ball, half, cfg):
    fps = ctx.fps
    length = ctx.pitch[0]
    k = int(cfg["velocity_half_window"])
    seen = ~np.isnan(ball[:, 0])
    vball = _velocity(ball, k, fps)
    far, eps = cfg["far_m"], cfg["eps_m"]

    near, carry, centroid_v, press = {}, {}, {}, {}
    for team in (HOME, AWAY):
        slots = ctx.team_slots(team)
        dist, best, all_d = _team_nearest(xy, ball, slots)
        near[team] = np.clip(np.nan_to_num(dist, nan=far), 0.0, far)
        # velocity of that nearest player, from his own slot's positions
        vxy = _velocity(xy[:, slots, :], k, fps)                        # (n, S_team, 2)
        rows = np.arange(len(ball))
        vplayer = np.where((best >= 0)[:, None], vxy[rows, np.maximum(best, 0)], np.nan)
        rel = np.hypot(*(vball - vplayer).T)
        carry[team] = (near[team] <= cfg["carry_m"]) & (rel <= cfg["carry_rel_mps"])
        team_x = xy[:, slots, 0]
        count = (~np.isnan(team_x)).sum(axis=1)
        centre_x = np.where(count > 0, np.nansum(team_x, axis=1) / np.maximum(count, 1), np.nan)
        cv = np.full(len(ball), np.nan)
        cv[1:] = np.diff(centre_x) * fps * ctx.attack_sign(half, team)
        centroid_v[team] = _rolling(np.nan_to_num(cv), max(1, int(cfg["push_window_s"] * fps)),
                                    "mean")
        press[team] = (np.nan_to_num(all_d, nan=np.inf) <= cfg["press_m"]).sum(axis=1)

    def ratio(a, b):
        return np.log(b + eps) - np.log(a + eps)

    cols = [ratio(near[HOME], near[AWAY])]
    for w in cfg["windows_s"]:
        frames = max(1, int(round(w * fps)))
        for future in (False, True):
            h = _rolling(np.where(seen, near[HOME], np.nan), frames, "min", future)
            a = _rolling(np.where(seen, near[AWAY], np.nan), frames, "min", future)
            cols.append(ratio(np.nan_to_num(h, nan=far), np.nan_to_num(a, nan=far)))
    carry_sign = carry[HOME].astype(np.float64) - carry[AWAY].astype(np.float64)
    cols.append(carry_sign)
    for w in cfg["carry_windows_s"]:
        frames = max(1, int(round(w * fps)))
        cols += [_rolling(carry_sign, frames, "mean"), _rolling(carry_sign, frames, "mean", True)]
    tau = cfg["decay_s"] * fps
    cols += [_decayed_last(carry_sign, tau), _decayed_last(carry_sign, tau, future=True)]

    sign_home = ctx.attack_sign(half, HOME)
    cols.append(np.clip(vball[:, 0] * sign_home / 10.0, -3.0, 3.0))
    cols.append((ball[:, 0] - length / 2.0) * sign_home / (length / 2.0))
    cols.append(centroid_v[HOME] - centroid_v[AWAY])
    cols.append((press[HOME] - press[AWAY]).astype(np.float64))

    X = np.stack(cols, axis=1)
    X[~seen] = np.nan
    return X


def features(ctx, ball=None, cfg=None):
    """(n_frames, F) antisymmetric features; NaN rows where the ball is unseen or ungraded."""
    cfg = dict(CONTROL_DEFAULTS, **(cfg or {}))
    ball = ball if ball is not None else clean_ball(ctx)
    pos = ball.pos if hasattr(ball, "pos") else np.asarray(ball)
    names = feature_names(cfg)
    X = np.full((ctx.n_frames, len(names)), np.nan)
    xy = np.asarray(ctx.xy)
    for half, frames in ctx.half_slices():
        X[frames] = _half_features(ctx, xy[frames], pos[frames], half, cfg)
    return X


# -- fitting -------------------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, n_iter=50, tol=1e-10):
    """Newton's method for an L2-penalised logistic regression without intercept. Deterministic."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        z = X @ w
        p = 0.5 * (1.0 + np.tanh(0.5 * z))               # stable sigmoid
        grad = X.T @ (p - y) + l2 * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w


def training_rows(external_games, stride=3, cfg=None):
    """(X, y) over the graded, live, ball-seen frames of the given ExternalGames (y = 1 for H)."""
    from .._paths import ensure_repo_on_path
    ensure_repo_on_path()
    import possession_metrics as pm

    blocks_x, blocks_y = [], []
    for eg in external_games:
        X = features(eg.ctx, cfg=cfg)
        live = (eg.ctx.half_of_frame > 0) & np.isin(eg.labels, (pm.H, pm.A))
        rows = np.flatnonzero(live & ~np.isnan(X).any(axis=1))[::stride]
        blocks_x.append(X[rows])
        blocks_y.append((eg.labels[rows] == pm.H).astype(np.float64))
    return np.concatenate(blocks_x), np.concatenate(blocks_y)


def train(external_games, l2=1.0, stride=3, cfg=None, meta=None, drop=()):
    """Fit the model. drop: feature names left out of the fit (weight fixed at 0), for
    ablations; the model keeps the full feature list so it runs through the same code."""
    names = feature_names(cfg)
    unknown = set(drop) - set(names)
    if unknown:
        raise KeyError("unknown features {}".format(sorted(unknown)))
    X, y = training_rows(external_games, stride, cfg)
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    scale[scale == 0] = 1.0
    keep = np.array([name not in drop for name in names])
    weights = np.zeros(len(names))
    # Contiguous like X itself: a strided copy changes the BLAS summation order, and with it the
    # last bits of the weights, so the shipped model would no longer regenerate byte for byte.
    weights[keep] = fit_logistic(np.ascontiguousarray(X[:, keep]) / scale[keep], y, l2=l2)
    info = {"n_rows": int(len(y)), "l2": l2, "stride": stride,
            "games": [eg.ctx.game for eg in external_games]}
    if drop:
        info["dropped"] = sorted(drop)
    info.update(meta or {})
    return ControlModel(names, scale, weights, info)


# -- evidence ------------------------------------------------------------------------------------

def logodds(ctx, model, ball=None, cfg=None):
    cfg = dict(CONTROL_DEFAULTS, **(cfg or {}))
    if tuple(model.names) != feature_names(cfg):
        raise ValueError("model features {} do not match the configured features {}".format(
            model.names, feature_names(cfg)))
    z = model.logodds(features(ctx, ball, cfg))
    return np.clip(z, -cfg["zmax"], cfg["zmax"])


def evidence(z, weight):
    """(n, 2) decoder log-scores w * log P(H), w * log P(A) from log-odds z (0 where z == 0)."""
    out = np.zeros((len(z), 2))
    out[:, HOME] = -np.logaddexp(0.0, -z) * weight
    out[:, AWAY] = -np.logaddexp(0.0, z) * weight
    out[z == 0] = 0.0
    return out
