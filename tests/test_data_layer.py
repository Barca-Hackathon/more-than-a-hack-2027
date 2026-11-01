"""The data layer against the real match files. Skipped on a clone without the gated data."""

import os
import warnings

import numpy as np
import pytest

from more_than_a_hack_2027._paths import ROOT, ensure_repo_on_path

ensure_repo_on_path()
import games  # noqa: E402

DATA_PRESENT = all(os.path.isfile(games.asset_paths()[name]) for name in ("atd_meta", "atd_raw"))
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="match data not in data/ (gated)")


@pytest.fixture(scope="module")
def ctx():
    from more_than_a_hack_2027.context import load_context
    return load_context()


def test_alignment_and_shapes(ctx):
    assert ctx.n_frames == 199319
    assert ctx.xy.shape == (199319, 46, 2)
    assert ctx.ball.shape == (199319, 2)
    assert ctx.halves == tuple(games.half_video_blocks(games.GAME))
    assert int((ctx.half_of_frame > 0).sum()) == 158857


def test_published_coverage_numbers(ctx):
    # DATA.md: ball on 113,488 of all frames and 94,214 of graded ones; 20.5 tracked objects.
    seen = ctx.ball_seen
    assert int(seen.sum()) == 113488
    assert int(seen[ctx.half_of_frame > 0].sum()) == 94214


def test_attack_direction_matches_goalkeeper_positions(ctx):
    # The team attacking +x defends x=0, so its goalkeeper-side extreme is the lower one.
    from more_than_a_hack_2027.context import AWAY, HOME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for half, frames in ctx.half_slices():
            x = ctx.xy[frames][:, :, 0]
            home_x, away_x = x[:, ctx.team_slots(HOME)], x[:, ctx.team_slots(AWAY)]
            if ctx.attack_sign(half, HOME) == 1:
                assert np.nanmedian(np.nanmin(home_x, 1)) < np.nanmedian(np.nanmin(away_x, 1))
            else:
                assert np.nanmedian(np.nanmax(home_x, 1)) > np.nanmedian(np.nanmax(away_x, 1))


@pytest.mark.parametrize("name,window", [("baseline_raw", 0), ("baseline_smooth25", 25),
                                         ("baseline_smooth50", 50), ("baseline_smooth125", 125)])
def test_baseline_port_is_bit_exact(ctx, name, window):
    from more_than_a_hack_2027 import pipeline
    import possession_metrics as pm
    reference = pm.read_label_csv(os.path.join(ROOT, "evaluation", "results", name + ".csv"))
    labels = pipeline.run("baseline", ctx, window=window)
    assert np.array_equal(labels, reference["label"].values.astype(str))


def test_writer_refuses_abstention_on_graded_frames(ctx, tmp_path):
    from more_than_a_hack_2027 import submission
    labels = np.full(ctx.n_frames, "D", dtype="<U1")
    labels[ctx.halves[0][1]] = "X"
    with pytest.raises(ValueError, match="graded frames are X"):
        submission.write(str(tmp_path / "bad.csv"), labels, ctx)
