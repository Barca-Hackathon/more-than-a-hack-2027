"""Control stage: team symmetry, the fit, the model file and the evidence it hands the decoder."""

import dataclasses

import numpy as np
import pytest

from more_than_a_hack_2027.stages import control
from more_than_a_hack_2027 import pipeline


def moving_match(make_ctx, n=400, seed=0):
    """Players wandering, a ball that a home player carries, then an away player."""
    rng = np.random.RandomState(seed)
    ctx = make_ctx(n, n_per_team=5)
    xy = np.array(ctx.xy) + np.cumsum(rng.normal(0, 0.05, size=ctx.xy.shape), axis=0)
    ball = np.full((n, 2), np.nan)
    ball[:180] = xy[:180, 0] + 0.5             # home slot 0 carries it
    ball[220:] = xy[220:, 7] - 0.5             # away slot 7 carries it
    return make_ctx(n, n_per_team=5, xy=xy, ball=ball)


def swap_teams(ctx):
    return dataclasses.replace(
        ctx, team_of_slot=(1 - ctx.team_of_slot).astype(np.int8),
        home_attacks_positive_x={h: not v for h, v in ctx.home_attacks_positive_x.items()})


def test_features_are_antisymmetric_under_a_team_swap(synthetic):
    ctx = moving_match(synthetic)
    X = control.features(ctx)
    Y = control.features(swap_teams(ctx))
    seen = ~np.isnan(X).any(axis=1)
    assert seen.sum() > 300
    assert np.allclose(X[seen], -Y[seen])
    assert np.isnan(X[~ctx.ball_seen]).all()


def test_carrying_team_gets_the_evidence(synthetic):
    ctx = moving_match(synthetic)
    X = control.features(ctx)
    names = control.feature_names()
    carry = X[:, names.index("carry")]
    assert np.nanmean(carry[10:170]) > 0.9 and np.nanmean(carry[230:390]) < -0.9


def test_fit_logistic_recovers_weights():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20000, 3))
    w = np.array([1.5, -0.7, 0.0])
    y = (rng.uniform(size=20000) < 1 / (1 + np.exp(-X @ w))).astype(float)
    fitted = control.fit_logistic(X, y, l2=1e-3)
    assert np.allclose(fitted, w, atol=0.06)


def test_model_round_trips_and_abstains_without_the_ball(tmp_path):
    names = control.feature_names()
    model = control.ControlModel(names, np.ones(len(names)), np.linspace(-1, 1, len(names)),
                                 {"note": "test"})
    path = str(tmp_path / "m.json")
    model.to_json(path)
    loaded = control.ControlModel.from_json(path)
    X = np.ones((3, len(names)))
    X[1] = np.nan
    assert np.array_equal(loaded.logodds(X), model.logodds(X))
    assert loaded.logodds(X)[1] == 0.0


def test_shipped_model_matches_the_feature_set():
    model = control.ControlModel.from_json()
    assert model.names == control.feature_names()
    assert np.all(np.isfinite(model.weights)) and np.all(model.scale > 0)


def test_evidence_is_log_probability_scaled():
    z = np.array([0.0, 4.0, -4.0])
    ev = control.evidence(z, 2.0)
    assert np.array_equal(ev[0], [0.0, 0.0])                       # unseen ball: no opinion
    assert np.isclose(ev[1, 0], 2.0 * np.log(1 / (1 + np.exp(-4.0))))
    assert np.isclose(ev[1, 1], 2.0 * np.log(1 / (1 + np.exp(4.0))))
    assert np.allclose(ev[2], ev[1][::-1])


def test_decayed_last_carries_the_latest_team():
    sign = np.array([0, 1, 0, 0, -1, 0], dtype=float)
    past = control._decayed_last(sign, tau=1.0)
    assert past[0] == 0 and past[1] == 1 and np.isclose(past[3], np.exp(-2)) and past[5] < 0
    future = control._decayed_last(sign, tau=1.0, future=True)
    assert np.isclose(future[0], np.exp(-1)) and future[5] == 0


def test_v2_config_rejects_unknown_settings():
    with pytest.raises(KeyError):
        pipeline.split_cfg({"w_contrl": 1.0}, pipeline.V2_DEFAULTS)
    top, stages = pipeline.split_cfg({"w_control": 0.5, "control": {"carry_m": 1.5}},
                                     pipeline.V2_DEFAULTS)
    assert top["w_control"] == 0.5 and stages["control"] == {"carry_m": 1.5}


def test_training_can_leave_features_out(synthetic):
    from types import SimpleNamespace
    ctx = moving_match(synthetic)
    labels = np.full(ctx.n_frames, pipeline.decode.H, dtype=np.int8)
    labels[200:] = pipeline.decode.A
    game = SimpleNamespace(ctx=ctx, labels=labels)
    dropped = ("near", "carry")
    model = control.train([game], stride=1, drop=dropped)
    names = control.feature_names()
    assert model.names == names and model.meta["dropped"] == sorted(dropped)
    assert all(model.weights[names.index(n)] == 0.0 for n in dropped)
    assert np.count_nonzero(model.weights) == len(names) - len(dropped)
    with pytest.raises(KeyError):
        control.train([game], drop=("nearr",))


def test_predict_defaults_to_the_frozen_submission_method(monkeypatch, tmp_path):
    from more_than_a_hack_2027 import cli, submission
    from more_than_a_hack_2027.data import context
    seen = []
    monkeypatch.setattr(context, "load_context", lambda: None)
    monkeypatch.setattr(pipeline, "run", lambda method, ctx, **kw: seen.append(method))
    monkeypatch.setattr(submission, "write", lambda path, labels, ctx: {
        "path": path, "counts": {}, "n_possessions": 0, "mean_possession_s": 0.0})
    cli.main(["predict", "--out", str(tmp_path / "x.csv")])
    assert seen == [pipeline.SUBMISSION_METHOD]
