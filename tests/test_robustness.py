"""The robustness audits' helpers: perturbations, spell support and the input-shift statistics."""

import numpy as np

from more_than_a_hack_2027.seq import network
from more_than_a_hack_2027.stages import decode
from more_than_a_hack_2027.validation import robustness
from more_than_a_hack_2027._paths import ensure_repo_on_path

from conftest import make_ctx

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402


def moving_ctx(n=400):
    rng = np.random.RandomState(0)
    ctx = make_ctx(n, halves=((1, 0, 199), (2, 200, n - 1)))
    xy = ctx.xy + np.cumsum(rng.normal(0, 0.05, ctx.xy.shape), axis=0)
    ball = np.column_stack([np.linspace(20, 80, n), np.linspace(10, 50, n)])
    ball[150:170] = np.nan
    return make_ctx(n, halves=ctx.halves, xy=xy, ball=ball)


def test_channel_names_match_the_network_inputs():
    names = robustness.channel_names()
    assert len(names) == network.N_BASE + 2 * network.TOUCH_BLOCK == len(set(names))
    assert names[robustness.ANCHOR] == "dead.anchor"
    assert names[network.BALL_SEEN] == "ball.seen"
    assert {robustness.channel_group(n) for n in names} == {
        "ball", "team blocks", "dead", "control", "touch"}


def test_dead_names_are_the_stage_inputs_names():
    from more_than_a_hack_2027.stages import deadball
    from more_than_a_hack_2027.stages.ball import clean_ball
    ctx = moving_ctx()
    ball = clean_ball(ctx)
    names, _ = network.dead_inputs(ctx, ball, deadball.detect(ctx, ball))
    assert names == robustness.DEAD_NAMES


def test_perturbations_are_seeded_and_keep_the_shape():
    ctx = moving_ctx()
    a, b = robustness.jitter(ctx, 2, ball_m=0.1), robustness.jitter(ctx, 2, ball_m=0.1)
    assert np.array_equal(a.ball, b.ball, equal_nan=True) and a.xy is ctx.xy
    assert np.isnan(a.ball[150:170]).all() and not np.allclose(a.ball[:150], ctx.ball[:150])
    p = robustness.jitter(ctx, 3, player_m=0.1)
    assert p.ball is ctx.ball and 0.05 < np.abs(p.xy - ctx.xy).mean() < 0.15

    dropped = robustness.drop_more(ctx, 4, 2)
    assert (np.isnan(dropped.xy[:, :, 0]).sum(axis=1) == 2).all()

    flick = robustness.flicker(ctx, 5, every_s=2.0)
    hidden = np.isnan(flick.ball[:, 0]) & ~np.isnan(ctx.ball[:, 0])
    assert 0 < hidden.sum() < 100


def test_slot_order_does_not_change_the_slot_free_inputs():
    ctx = moving_ctx()
    shuffled = robustness.permute_slots(ctx, 1)
    assert sorted(shuffled.team_of_slot) == sorted(ctx.team_of_slot)
    assert not np.array_equal(shuffled.team_of_slot, ctx.team_of_slot)
    _, X, _ = robustness.stage_inputs(ctx)
    _, Xs, _ = robustness.stage_inputs(shuffled)
    assert np.allclose(X, Xs, atol=1e-5)


def test_member_logprobs_is_the_network_view_and_shift_keeps_the_length():
    rng = np.random.RandomState(0)
    ctx = moving_ctx()
    _, X, sign = robustness.stage_inputs(ctx)
    arch = {"stride": 5, "channels": 8, "dilations": [1, 2], "touch": True}
    c = X.shape[1]
    weights = {"down.weight": rng.normal(0, 0.1, (8, c, 5)), "down.bias": np.zeros(8),
               "up.weight": rng.normal(0, 0.1, (8, 8, 5)), "up.bias": np.zeros(8),
               "skip.weight": rng.normal(0, 0.1, (8, c, 1)), "skip.bias": np.zeros(8),
               "head.1.weight": rng.normal(0, 0.1, (8, 8, 3)), "head.1.bias": np.zeros(8),
               "head.3.weight": rng.normal(0, 0.1, (3, 8, 1)), "head.3.bias": np.zeros(3)}
    for i in range(2):
        p = "blocks.{}.".format(i)
        weights.update({p + "n.weight": np.ones(8), p + "n.bias": np.zeros(8),
                        p + "c1.weight": rng.normal(0, 0.1, (8, 8, 3)), p + "c1.bias": np.zeros(8),
                        p + "c2.weight": rng.normal(0, 0.1, (8, 8, 1)), p + "c2.bias": np.zeros(8)})
    model = network.SeqModel(arch, np.ones(c), weights)
    base = robustness.member_logprobs(model, X, sign, ctx.halves)
    assert np.array_equal(base, network.logprobs_halves(model, X, sign, ctx.halves))
    shifted = robustness.member_logprobs(model, X, sign, ctx.halves, shift=2)
    assert shifted.shape == base.shape and np.allclose(np.exp(shifted).sum(axis=1), 1.0)
    assert np.array_equal(shifted[200], shifted[202]) and not np.allclose(shifted, base)
    mirrored = robustness.member_logprobs(model, X, sign, ctx.halves, mirror=True)
    assert not np.allclose(mirrored, base)


def test_pinned_frames_decode_as_dead():
    n = 300
    lp = np.log(np.tile([0.8, 0.15, 0.05], (n, 1)))
    graded = np.ones(n, dtype=bool)
    halves = ((1, 0, n - 1),)
    assert (robustness.decode_lp(lp, graded, halves) == "H").all()
    pin = np.zeros(n, dtype=bool)
    pin[100:110] = True
    labels = robustness.decode_lp(lp, graded, halves, pin=pin)
    assert (labels[100:110] == "D").all()
    assert (labels == "D").sum() >= 25 and labels[0] == "H" and labels[-1] == "H"


def test_support_counts_members_that_reproduce_a_spell():
    n = 600
    truth = np.array(["H"] * 200 + ["D"] * 100 + ["A"] * 300)
    strong = np.where(truth[:, None] == np.array(list("HAD"))[None, :], 0.9, 0.05)
    other = strong.copy()
    other[300:] = [0.9, 0.05, 0.05]                    # this member calls the second spell H
    run = robustness.Run("toy", "toy", np.zeros((n, 1), dtype=np.float32), np.ones(n, bool),
                         ((1, 0, n - 1),), [np.log(strong), np.log(strong), np.log(other)])
    mine, votes = robustness.support(run)
    assert [(a, b) for a, b, _ in mine] == [(0, 200), (300, 600)]
    assert list(votes) == [3, 2]
    assert robustness.agreement(truth, robustness.decode_lp(run.lp, run.graded, run.halves)) == 1.0


def test_input_shift_flags_only_the_shifted_channel():
    rng = np.random.RandomState(0)
    samples = {"m{}".format(i): ("toy", rng.normal(0, 1, (4000, 3))) for i in range(4)}
    target = rng.normal(0, 1, (4000, 3))
    target[:, 2] += 3.0
    shift = robustness.input_shift(target, samples)
    assert shift["outside"][2] > 0.3 and shift["outside"][:2].max() < 0.01
    assert shift["auc"][2] > 0.4 and shift["auc"][:2].max() < 0.03
    base = np.array(list(shift["base_auc"].values()))
    assert base.max() < 0.03


LEVEL = network.dead_channel("lvl_p200")


def test_kickoff_wait_pins_a_long_slow_wait_only():
    n = 3000
    X = np.zeros((n, len(robustness.channel_names())), dtype=np.float32)
    halves = ((1, 0, 1499), (2, 1500, n - 1))
    graded = np.ones(n, dtype=bool)
    X[:, robustness.MASK] = -1.0
    X[:, LEVEL] = np.log(2.0)                 # live play at 2 m/s
    for a, b in ((0, 400), (1500, 1900)):                # both halves wait 16 s for a kickoff
        X[a:b, robustness.ANCHOR] = 1.0
        X[a:b, robustness.MASK] = 1.0
        X[a:b, LEVEL] = np.log(1.2)
    X[1600:1700, LEVEL] = np.log(3.0)         # ... but the second holds live play
    waits = network.kickoff_waits(X, graded, halves)
    assert [(a, b) for a, b, _ in waits] == [(0, 400), (1500, 1900)]
    assert waits[0][2] < 1.0 < waits[1][2]
    pin = robustness.kickoff_wait_pin(X, graded, halves)
    assert pin[:400].all() and not pin[400:].any()
    assert not robustness.kickoff_wait_pin(X, graded, halves, min_s=20.0).any()


def test_run_v3_and_saved_logprobs_pin_the_same_wait():
    from more_than_a_hack_2027 import pipeline
    n = 1000
    X = np.zeros((n, len(robustness.channel_names())), dtype=np.float32)
    graded = np.ones(n, dtype=bool)
    halves = ((1, 0, n - 1),)
    X[:, robustness.MASK] = -1.0
    X[:, LEVEL] = np.log(2.0)
    X[:300, robustness.ANCHOR] = 1.0
    X[:300, robustness.MASK] = 1.0
    X[:300, LEVEL] = np.log(1.0)
    lp = np.log(np.tile([0.9, 0.05, 0.05], (n, 1)))      # the networks say H throughout
    top = dict(pipeline.V3_DEFAULTS)
    evidence = pipeline.pin_waits(lp.copy(), X, graded, halves, 25, top)
    trans = decode.transition_matrix(top["switch"], top["stop"], top["restart"])
    labels = decode.decode_match(evidence, halves, trans, min_dead=25)
    assert (labels[:300] == "D").all() and (labels[300:] == "H").all()
    off = pipeline.pin_waits(lp.copy(), X, graded, halves, 25, dict(top, wait_min_s=0.0))
    np.testing.assert_array_equal(off, lp)
