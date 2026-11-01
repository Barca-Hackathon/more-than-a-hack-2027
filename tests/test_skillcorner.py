"""external/skillcorner.py: phases of play -> labels, 10 Hz -> 25 Hz, and the pitch frame."""

import numpy as np

from more_than_a_hack_2027.external import skillcorner as sc


def test_phases_give_the_team_in_possession_and_dead_ball_between_them():
    frames = np.arange(100, 120)
    phases = [(102, 106, 7), (106, 110, 9), (114, 118, 7)]
    codes = sc.phase_labels(frames, phases, home_id=7)
    np.testing.assert_array_equal(
        codes, [2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2])


def test_resampling_hits_the_source_frames_and_interpolates_between_them():
    src = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    out = sc.resample(src, 11)                       # 0.4 s of source per 25 Hz frame
    np.testing.assert_allclose(out, np.arange(11) * 4.0)


def test_a_25hz_frame_next_to_a_missing_source_frame_is_missing():
    out = sc.resample(np.array([0.0, 10.0, np.nan, 30.0]), 8)
    np.testing.assert_allclose(out[:3], [0.0, 4.0, 8.0])
    assert np.isnan(out[3:]).all()                  # 1.2 s onwards touches the hole


def test_a_25hz_frame_on_a_source_frame_keeps_it_even_next_to_a_hole():
    out = sc.resample(np.array([0.0, 10.0, 20.0, np.nan]), 6)
    assert out[5] == 20.0                           # t = 2.0 s exactly


def test_labels_take_the_nearest_source_frame():
    np.testing.assert_array_equal(sc.nearest(np.array([0, 1, 2]), 6), [0, 0, 1, 1, 2, 2])


def test_pitch_frame_moves_the_origin_to_the_top_left_corner_with_y_down():
    x, y = sc.to_pitch([0.0, 52.5, -52.5], [0.0, 34.0, -34.0], 105.0, 68.0)
    np.testing.assert_allclose(x, [52.5, 105.0, 0.0])
    np.testing.assert_allclose(y, [34.0, 0.0, 68.0])
    x, _ = sc.to_pitch([50.0], [0.0], 100.0, 64.0)  # scaled from the match's own pitch size
    np.testing.assert_allclose(x, [105.0])
