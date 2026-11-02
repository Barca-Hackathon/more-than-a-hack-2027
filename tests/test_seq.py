"""network.py: inputs, team symmetry and the numpy forward pass (no gated data needed)."""

import numpy as np
import pytest

from more_than_a_hack_2027 import pipeline
from more_than_a_hack_2027.seq import network
from more_than_a_hack_2027.stages.ball import clean_ball

PITCH = (105.0, 68.0)   # as in conftest.make_ctx


def _random_model(cin, ch=8, dilations=(1, 2), stride=5, seed=0):
    rng = np.random.RandomState(seed)
    w = {"down.weight": rng.randn(ch, cin, stride) * 0.2, "down.bias": rng.randn(ch) * 0.1,
         "up.weight": rng.randn(ch, ch, stride) * 0.2, "up.bias": rng.randn(ch) * 0.1,
         "skip.weight": rng.randn(ch, cin, 5) * 0.2, "skip.bias": rng.randn(ch) * 0.1,
         "head.1.weight": rng.randn(ch, ch, 5) * 0.2, "head.1.bias": rng.randn(ch) * 0.1,
         "head.3.weight": rng.randn(3, ch, 1) * 0.2, "head.3.bias": rng.randn(3) * 0.1}
    for i, _ in enumerate(dilations):
        p = "blocks.{}.".format(i)
        w.update({p + "n.weight": 1 + rng.randn(ch) * 0.1, p + "n.bias": rng.randn(ch) * 0.1,
                  p + "c1.weight": rng.randn(ch, ch, 3) * 0.2, p + "c1.bias": rng.randn(ch) * 0.1,
                  p + "c2.weight": rng.randn(ch, ch, 1) * 0.2, p + "c2.bias": rng.randn(ch) * 0.1})
    arch = {"stride": stride, "channels": ch, "dilations": list(dilations)}
    return network.SeqModel(arch, np.ones(cin), w)


def _moving_match(synthetic, n=300):
    ball = np.column_stack([np.linspace(20, 80, n), 30 + 5 * np.sin(np.arange(n) / 20.0)])
    ball[120:140] = np.nan                       # a short gap
    ctx = synthetic(n, ball=ball)
    xy = np.array(ctx.xy)
    xy[:, 0, :] = ball + 1.0                     # a home player near the ball
    xy[:, 12, :] = ball - 2.0                    # an away player a bit further
    xy[200:, 3, :] = np.nan                      # someone leaves the tracking
    return ctx, xy


def _rotated_and_swapped(ctx, xy):
    """The same match with the pitch rotated 180 degrees and the team labels exchanged."""
    from more_than_a_hack_2027.data.context import Context
    length, width = PITCH
    rot = xy.copy()
    rot[..., 0] = length - xy[..., 0]
    rot[..., 1] = width - xy[..., 1]
    ball = ctx.ball.copy()
    ball[:, 0] = length - ball[:, 0]
    ball[:, 1] = width - ball[:, 1]
    return Context(game=ctx.game, n_frames=ctx.n_frames, fps=ctx.fps, pitch=ctx.pitch, xy=rot,
                   speed=ctx.speed, ball=ball, team_of_slot=1 - ctx.team_of_slot,
                   team_names=ctx.team_names[::-1], halves=ctx.halves,
                   home_attacks_positive_x=ctx.home_attacks_positive_x,
                   half_of_frame=ctx.half_of_frame)


def test_geometry_swap_matches_the_rotated_match_with_teams_exchanged(synthetic):
    ctx, xy = _moving_match(synthetic)
    from more_than_a_hack_2027.data.context import Context
    ctx = Context(**{**ctx.__dict__, "xy": xy})
    other = _rotated_and_swapped(ctx, xy)
    geo = network.geometry_inputs(ctx, clean_ball(ctx))
    geo_other = network.geometry_inputs(other, clean_ball(other))
    sign = np.concatenate([network.BALL_SWAP, network.TEAM_SWAP, network.TEAM_SWAP])
    np.testing.assert_allclose(network.swap(geo, sign), geo_other, atol=1e-9)


def test_swap_is_an_involution():
    rng = np.random.RandomState(1)
    n_geo = network.BALL_BLOCK + 2 * network.TEAM_BLOCK
    X = rng.randn(50, n_geo + 4)
    sign = np.concatenate([network.BALL_SWAP, network.TEAM_SWAP, network.TEAM_SWAP,
                           [1, 1, -1, -1]])
    np.testing.assert_array_equal(network.swap(network.swap(X, sign), sign), X)


def test_logprobs_are_normalised_and_exactly_team_symmetric(synthetic):
    n_geo = network.BALL_BLOCK + 2 * network.TEAM_BLOCK
    cin = n_geo + 3
    sign = np.concatenate([network.BALL_SWAP, network.TEAM_SWAP, network.TEAM_SWAP,
                           [1, -1, -1]])
    model = _random_model(cin)
    ctx = synthetic(203, halves=((1, 0, 100), (2, 101, 202)))
    X = np.random.RandomState(2).randn(203, cin)
    lp = network.logprobs(ctx, model, X, sign)
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-9)
    lp_swapped = network.logprobs(ctx, model, network.swap(X, sign), sign)
    np.testing.assert_allclose(lp_swapped[:, [1, 0, 2]], lp, atol=1e-12)


def test_model_round_trips_through_json(tmp_path):
    model = _random_model(7)
    path = tmp_path / "m.json"
    model.to_json(str(path))
    back = network.SeqModel.from_json(str(path))
    X = np.random.RandomState(3).randn(40, 7)
    # weights are stored as float32
    np.testing.assert_allclose(back.forward(X), model.forward(X), atol=1e-5)


def test_numpy_forward_matches_torch():
    torch = pytest.importorskip("torch")
    from more_than_a_hack_2027.seq import training
    cfg = dict(training.TRAIN_DEFAULTS, channels=16, dilations=(1, 2, 4))
    torch.manual_seed(0)
    net = training._net(11, cfg)
    net.eval()
    X = np.random.RandomState(4).randn(203, 11).astype(np.float32)
    with torch.no_grad():
        x = torch.nn.functional.pad(torch.from_numpy(X.T[None]), (0, (-len(X)) % 5))
        ref = torch.log_softmax(net(x)[0, :, :len(X)], 0).numpy().T
    weights = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}
    model = network.SeqModel({"stride": 5, "channels": 16, "dilations": [1, 2, 4]},
                              np.ones(11), weights)
    np.testing.assert_allclose(model.forward(X), ref, atol=1e-5)


def test_training_weights_short_possessions_and_turnovers():
    from more_than_a_hack_2027.seq import training
    y = np.array([2] * 10 + [0] * 50 + [1] * 200 + [2] * 10)
    cfg = dict(training.TRAIN_DEFAULTS, short_weight=2.0, short_s=4.0, turn_weight=1.0,
               turn_frames=5)
    w = training._weights(y, cfg)
    assert w[30] == 3.0                  # inside the 2 s home spell
    assert w[200] == 1.0                 # inside the 8 s away spell
    assert w[58] == 4.0 and w[62] == 2.0  # near the H -> A change (and inside the short spell)
    assert w[5] == 1.0                   # stoppages are never up-weighted


def test_v3_config_rejects_unknown_settings(synthetic):
    with pytest.raises(KeyError):
        pipeline.split_cfg({"w_control": 1.0}, pipeline.V3_DEFAULTS)


def test_mirror_sign_matches_the_left_right_mirrored_match(synthetic):
    from more_than_a_hack_2027.data.context import Context
    ctx, xy = _moving_match(synthetic)
    ctx = Context(**{**ctx.__dict__, "xy": xy})
    mirrored_xy = xy.copy()
    mirrored_xy[..., 1] = PITCH[1] - xy[..., 1]
    ball = ctx.ball.copy()
    ball[:, 1] = PITCH[1] - ball[:, 1]
    other = Context(**{**ctx.__dict__, "xy": mirrored_xy, "ball": ball})
    geo = network.geometry_inputs(ctx, clean_ball(ctx))
    geo_other = network.geometry_inputs(other, clean_ball(other))
    np.testing.assert_allclose(geo * network.flip_sign(geo.shape[1]), geo_other, atol=1e-9)


def test_an_auxiliary_training_head_does_not_change_predictions():
    model = _random_model(7)
    two = network.SeqModel(dict(model.arch, heads=2), model.scale, dict(model.weights))
    rng = np.random.RandomState(5)
    two.weights["head.3.weight"] = np.concatenate([model.weights["head.3.weight"],
                                                   rng.randn(3, 8, 1)])
    two.weights["head.3.bias"] = np.concatenate([model.weights["head.3.bias"], rng.randn(3)])
    X = rng.randn(60, 7)
    np.testing.assert_allclose(two.forward(X), model.forward(X), atol=1e-12)


def _training_block(seed, n=300, extra=10):
    rng = np.random.RandomState(seed)
    sign = np.concatenate([network.BALL_SWAP, network.TEAM_SWAP, network.TEAM_SWAP,
                           np.ones(extra)])
    X = rng.randn(n, len(sign)).astype(np.float32)
    labels = rng.randint(0, 3, n).astype(np.int8)
    return X, sign, labels, np.ones(n, bool), ((1, 0, n // 2 - 1), (2, n // 2, n - 1))


@pytest.mark.parametrize("aux_head", [None, "own", "main"])
def test_training_runs_with_and_without_an_auxiliary_dataset(aux_head, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from more_than_a_hack_2027.seq import training

    aux = ()
    if aux_head:
        monkeypatch.setattr(training, "CACHE_DIR", str(tmp_path))
        X, sign, labels, graded, halves = _training_block(1)
        np.savez(tmp_path / "aux.npz", X=X, swap_sign=sign, labels=labels, graded=graded,
                 halves=np.array(halves))
        aux = ("aux.npz",)
    cfg = {"channels": 8, "dilations": (1, 2), "steps": 2, "batch": 2, "crop": 100,
           "threads": 1, "aux": aux, "aux_head": aux_head or "own"}
    model = training.train([_training_block(0)], cfg, log=None)
    assert model.arch["heads"] == (2 if aux_head == "own" else 1)
    lp = model.forward(_training_block(2)[0])
    assert lp.shape == (300, 3) and np.isfinite(lp).all()


def test_with_aux_until_zero_the_auxiliary_rows_are_never_drawn(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from more_than_a_hack_2027.seq import training

    monkeypatch.setattr(training, "CACHE_DIR", str(tmp_path))
    for name, seed in (("a.npz", 1), ("b.npz", 3)):
        X, sign, labels, graded, halves = _training_block(seed)
        np.savez(tmp_path / name, X=X, swap_sign=sign, labels=labels, graded=graded,
                 halves=np.array(halves))
    cfg = {"channels": 8, "dilations": (1, 2), "steps": 3, "batch": 2, "crop": 100,
           "threads": 1, "aux_head": "main", "aux_until": 0.0}
    one = training.train([_training_block(0)], dict(cfg, aux=("a.npz",)), log=None)
    two = training.train([_training_block(0)], dict(cfg, aux=("b.npz",)), log=None)
    for key in one.weights:
        np.testing.assert_array_equal(one.weights[key], two.weights[key])
    mixed = training.train([_training_block(0)], dict(cfg, aux=("b.npz",), aux_until=1.0),
                           log=None)
    assert any(not np.array_equal(one.weights[k], mixed.weights[k]) for k in one.weights)
