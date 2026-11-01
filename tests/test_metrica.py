"""Event -> frame label conversion and the external loaders' helpers, on hand-built inputs."""

import dataclasses

import numpy as np

from more_than_a_hack_2027.external import metrica
from more_than_a_hack_2027.data.context import AWAY, HOME
from more_than_a_hack_2027.external.metrica import Event
from more_than_a_hack_2027._paths import ensure_repo_on_path

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def ev(k, team, kind, start, end=None, subtype="", period=1):
    return Event(k, team, kind, subtype, period, start, start if end is None else end)


def codes(*runs):
    """codes((H, 10), (D, 5)) -> array of 15 codes."""
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def test_kickoff_turnover_ball_out_and_throw_in():
    events = [
        ev(0, HOME, "SET PIECE", 10, subtype="KICK OFF"),
        ev(1, HOME, "PASS", 10, 30),
        ev(2, HOME, "BALL LOST", 40, 45, "INTERCEPTION"),
        ev(3, AWAY, "RECOVERY", 45),
        ev(4, AWAY, "CHALLENGE", 50, subtype="GROUND-WON"),     # a duel never switches
        ev(5, HOME, "CHALLENGE", 50, subtype="GROUND-LOST"),
        ev(6, AWAY, "BALL OUT", 60, 70),
        ev(7, HOME, "RECOVERY", 75),                            # collecting a dead ball
        ev(8, HOME, "SET PIECE", 90, subtype="THROW IN"),
        ev(9, HOME, "PASS", 90, 99),
    ]
    labels = metrica.labels_from_events(events, 100, ((1, 0, 99),))
    expected = codes((D, 10), (H, 35), (A, 25), (D, 20), (H, 10))
    assert np.array_equal(labels, expected)


def test_after_the_whistle_actions_do_not_restart_but_untagged_play_does():
    events = [
        ev(0, HOME, "SET PIECE", 0, subtype="KICK OFF"),
        ev(1, HOME, "FAULT RECEIVED", 100),
        ev(2, AWAY, "BALL LOST", 105, 110, "FORCED"),           # kicked away after the whistle
        ev(3, HOME, "SET PIECE", 300, subtype="FREE KICK"),
        ev(4, HOME, "PASS", 400, 420, "OFFSIDE"),
        ev(5, HOME, "SHOT", 420, 425, "ON TARGET-SAVED"),       # shot after the flag
        ev(6, AWAY, "SET PIECE", 600, subtype="FREE KICK"),
        ev(7, AWAY, "BALL OUT", 700, 710),
        ev(8, HOME, "PASS", 800, 830),                          # no throw-in tagged: but play is
        ev(9, HOME, "PASS", 850, 880),                          # clearly on, 3 events over > 5 s
        ev(10, HOME, "PASS", 950, 990),
    ]
    labels = metrica.labels_from_events(events, 1000, ((1, 0, 999),), fps=25)
    expected = codes((H, 100), (D, 200), (H, 120), (D, 180), (A, 110), (D, 90), (H, 200))
    assert np.array_equal(labels, expected)


def test_retaken_set_piece_keeps_the_ball_dead():
    events = [
        ev(0, HOME, "SET PIECE", 0, subtype="KICK OFF"),
        ev(1, AWAY, "FAULT RECEIVED", 50),
        ev(2, AWAY, "SET PIECE", 100, subtype="FREE KICK-RETAKEN"),
        ev(3, AWAY, "BALL LOST", 100, 120, "FORCED"),
        ev(4, AWAY, "SET PIECE", 200, subtype="FREE KICK"),
    ]
    labels = metrica.labels_from_events(events, 250, ((1, 0, 249),))
    assert np.array_equal(labels, codes((H, 50), (D, 150), (A, 50)))


def test_goal_and_end_of_half():
    events = [
        ev(0, AWAY, "SET PIECE", 0, subtype="KICK OFF"),
        ev(1, AWAY, "SHOT", 20, 30, "ON TARGET-GOAL"),
        ev(2, HOME, "SET PIECE", 80, subtype="KICK OFF"),
        ev(3, HOME, "BALL LOST", 90, 95, "END HALF"),
    ]
    labels = metrica.labels_from_events(events, 120, ((1, 0, 119),))
    assert np.array_equal(labels, codes((A, 30), (D, 50), (H, 15), (D, 25)))


def test_contiguous_periods_get_an_ungraded_frame_between_them():
    period = np.array([1] * 5 + [2] * 4, dtype=np.int8)
    halves = metrica.graded_halves(period)
    assert halves == ((1, 0, 3), (2, 5, 8))
    events = [ev(0, HOME, "SET PIECE", 0, subtype="KICK OFF"),
              ev(1, AWAY, "SET PIECE", 6, subtype="KICK OFF", period=2)]
    labels = metrica.labels_from_events(events, 9, halves)
    assert np.array_equal(labels, np.array([H, H, H, H, X, D, A, A, A]))
    assert pm.valid_runs(labels) == [(0, 4), (5, 9)]          # the metric sees two halves


def test_player_speed_is_metres_per_second_and_stays_inside_periods():
    n = 40
    xy = np.zeros((n, 1, 2))
    xy[:, 0, 0] = np.arange(n) * 0.2                   # 0.2 m per frame = 5 m/s at 25 fps
    period = np.array([1] * 20 + [2] * 20)
    speed = metrica.player_speed(xy, period)
    assert np.allclose(speed[2:18, 0], 5.0) and np.allclose(speed[22:38, 0], 5.0)
    assert np.isnan(speed[[0, 1, 18, 19, 20, 21], 0]).all()


# -- what does not change possession (metrica.LABEL_RULES) ------------------------------------

def _open_play(*events):
    """HOME kicks off at 0 and the half runs to frame 199; events follow."""
    return [ev(0, HOME, "SET PIECE", 0, subtype="KICK OFF"), ev(1, HOME, "PASS", 0, 20),
            *[dataclasses.replace(e, index=k + 2) for k, e in enumerate(events)]]


def test_a_ball_lost_never_starts_a_possession():
    # AWAY wins the tackle at 50; HOME's BALL LOST for that same tackle is tagged 3 frames later.
    events = _open_play(ev(0, AWAY, "CHALLENGE", 50, subtype="TACKLE-WON"),
                        ev(0, AWAY, "RECOVERY", 50, subtype="THEFT"),
                        ev(0, HOME, "BALL LOST", 53, subtype="THEFT"),
                        ev(0, AWAY, "PASS", 90, 110))
    labels = metrica.labels_from_events(events, 200, ((1, 0, 199),))
    assert np.array_equal(labels, codes((H, 50), (A, 150)))
    old = metrica.labels_from_events(events, 200, ((1, 0, 199),), rules=())
    assert np.array_equal(old, codes((H, 50), (A, 3), (H, 37), (A, 110)))


def test_a_one_touch_recovery_is_not_a_possession():
    # AWAY heads the ball straight back to HOME: recovery and loss on the same frame.
    events = _open_play(ev(0, AWAY, "CHALLENGE", 60, subtype="AERIAL-WON"),
                        ev(0, AWAY, "RECOVERY", 60, subtype="INTERCEPTION"),
                        ev(0, AWAY, "BALL LOST", 60, 75, "HEAD"),
                        ev(0, HOME, "RECOVERY", 75),
                        ev(0, HOME, "PASS", 80, 100))
    halves = ((1, 0, 199),)
    assert np.array_equal(metrica.labels_from_events(events, 200, halves), codes((H, 200)))
    loss_only = metrica.labels_from_events(events, 200, halves, rules=("loss",))
    assert np.array_equal(loss_only, codes((H, 60), (A, 15), (H, 125)))


def test_a_one_frame_carry_does_not_rescue_a_one_touch_recovery():
    # Game 3's export adds a 1-frame CARRY to a one-touch recovery; a real carry is control.
    touch = _open_play(ev(0, AWAY, "RECOVERY", 60), ev(0, AWAY, "BALL LOST", 60, 80),
                       ev(0, AWAY, "CARRY", 60, 61), ev(0, HOME, "RECOVERY", 80))
    carried = _open_play(ev(0, AWAY, "RECOVERY", 60), ev(0, AWAY, "CARRY", 60, 85),
                         ev(0, AWAY, "BALL LOST", 85, 90), ev(0, HOME, "RECOVERY", 90))
    halves = ((1, 0, 199),)
    assert np.array_equal(metrica.labels_from_events(touch, 200, halves), codes((H, 200)))
    assert np.array_equal(metrica.labels_from_events(carried, 200, halves),
                          codes((H, 60), (A, 30), (H, 110)))


def test_a_recovery_kept_for_a_moment_is_a_possession():
    # Lost 3 frames after the recovery: past ONE_TOUCH_FRAMES, so it counts.
    events = _open_play(ev(0, AWAY, "RECOVERY", 60), ev(0, AWAY, "BALL LOST", 63, 80),
                        ev(0, HOME, "RECOVERY", 80))
    assert metrica.ONE_TOUCH_FRAMES < 3
    labels = metrica.labels_from_events(events, 200, ((1, 0, 199),))
    assert np.array_equal(labels, codes((H, 60), (A, 20), (H, 120)))


def test_one_touch_out_of_play_keeps_the_attacking_team_until_the_stop():
    # A clearance out for a corner: the defending team never had the ball.
    events = _open_play(ev(0, AWAY, "RECOVERY", 60), ev(0, AWAY, "BALL OUT", 60, 70),
                        ev(0, HOME, "SET PIECE", 150, subtype="CORNER KICK"))
    labels = metrica.labels_from_events(events, 200, ((1, 0, 199),))
    assert np.array_equal(labels, codes((H, 70), (D, 80), (H, 50)))


def test_unknown_label_rules_are_refused():
    import pytest
    with pytest.raises(ValueError):
        metrica.labels_from_events(_open_play(ev(0, AWAY, "RECOVERY", 60)), 200,
                                    ((1, 0, 199),), rules=("nonsense",))
