"""external/dfl.py: the control filter on DFL's possession flag, and the event-clock fit."""

import numpy as np

from more_than_a_hack_2027.external import dfl


def test_a_touch_without_a_control_event_goes_back_to_the_team_in_possession():
    flag = np.array([0] * 20 + [1] * 5 + [0] * 20)
    out = dfl.filter_switches(flag, [5, 40], [0, 0], tol=2)
    assert (out == 0).all()


def test_a_change_with_a_control_event_of_the_new_team_counts():
    flag = np.array([0] * 20 + [1] * 5 + [0] * 20)
    out = dfl.filter_switches(flag, [5, 22, 40], [0, 1, 0], tol=2)
    np.testing.assert_array_equal(out, flag)


def test_a_control_event_counts_within_the_tolerance_of_the_spell():
    flag = np.array([0] * 20 + [1] * 5 + [0] * 20)
    team_events = ([5, 18, 40], [0, 1, 0])          # team 1's event 2 frames before its spell
    np.testing.assert_array_equal(dfl.filter_switches(flag, *team_events, tol=2), flag)
    assert (dfl.filter_switches(flag, *team_events, tol=1) == 0).all()


def test_the_first_spell_after_a_dead_ball_is_kept():
    flag = np.array([0] * 10 + [2] * 5 + [1] * 10)
    np.testing.assert_array_equal(dfl.filter_switches(flag, [], [], tol=2), flag)


def test_the_clock_offset_is_the_one_that_puts_the_ball_on_the_events():
    n = 5000
    ftime = 1000.0 + np.arange(n) / 25
    k = np.arange(n)
    ball = np.column_stack([52 + 30 * np.sin(k / 50), 34 + 20 * np.cos(k / 37)])
    true_offset = 0.48                               # the event clock runs 0.48 s behind
    evs = [(ftime[i] - true_offset, "Play", "T1", ball[i, 0], dfl.PITCH[1] - ball[i, 1], {})
           for i in range(200, 4800, 200)]
    offset, distance = dfl.clock_offset(ball, ftime, evs)
    assert abs(offset - true_offset) < 1e-9
    assert distance < 1e-6
