"""Dead-ball stage on the real match. Skipped on a clone without the gated data.

These are correctness and plausibility checks, not accuracy targets: nothing here reads the
ground truth. The two kickoff frames are the ones published in FAQ.md / starter code.
"""

import os
import warnings

import numpy as np
import pytest

from more_than_a_hack_2027._paths import ensure_repo_on_path

ensure_repo_on_path()
import games  # noqa: E402

DATA_PRESENT = all(os.path.isfile(games.asset_paths()[name])
                   for name in ("atd_meta", "atd_raw", "smart_tagging"))
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="match data not in data/ (gated)")


@pytest.fixture(scope="module")
def run():
    from more_than_a_hack_2027 import pipeline
    from more_than_a_hack_2027.data.context import load_context
    warnings.simplefilter("ignore", RuntimeWarning)
    ctx = load_context()
    labels, parts = pipeline.run_v1(ctx, return_parts=True)
    return ctx, labels, parts


def test_kickoffs_found_in_both_halves(run):
    ctx, _, parts = run
    by_half = {}
    for kick in parts["dead"].kickoffs:
        by_half.setdefault(kick.half, []).append(kick.frame)
    assert abs(min(by_half[1]) - 11185) <= 2 * ctx.fps      # published: half 1 starts at kickoff
    assert abs(min(by_half[2]) - 113742) <= ctx.fps         # published in FAQ.md


def test_second_half_is_dead_until_kickoff(run):
    ctx, labels, parts = run
    first = ctx.halves[1][1]
    kick = min(k.frame for k in parts["dead"].kickoffs if k.half == 2)
    assert parts["dead"].mask[first:kick].all()
    assert set(labels[first:kick]) == {"D"}


def test_every_ball_out_is_inside_a_stoppage(run):
    _, _, parts = run
    dead = parts["dead"]
    assert all(dead.mask[e.first] for e in dead.out_episodes)


def test_plausible_totals(run):
    ctx, _, parts = run
    dead = parts["dead"]
    graded = ctx.half_of_frame > 0
    assert 80 <= len(dead.stoppages) <= 250
    assert 0.3 <= dead.mask[graded].mean() <= 0.65
    assert all(s.restart > s.start for s in dead.stoppages)
    starts = np.array([s.start for s in dead.stoppages])
    restarts = np.array([s.restart for s in dead.stoppages])
    assert np.all(starts[1:] >= restarts[:-1])                 # no overlaps, in order


def test_v1_submission_is_valid(run, tmp_path):
    from more_than_a_hack_2027 import submission
    ctx, labels, _ = run
    summary = submission.write(str(tmp_path / "v1.csv"), labels, ctx)
    assert summary["counts"]["X"] == ctx.n_frames - int((ctx.half_of_frame > 0).sum())


def test_v2_submission_is_valid_and_plausible(run, tmp_path):
    from more_than_a_hack_2027 import pipeline, submission
    ctx, _, _ = run
    labels = pipeline.run_v2(ctx)
    summary = submission.write(str(tmp_path / "v2.csv"), labels, ctx)
    assert 150 <= summary["n_possessions"] <= 600
    assert np.array_equal(labels, pipeline.run_v2(ctx))            # deterministic
