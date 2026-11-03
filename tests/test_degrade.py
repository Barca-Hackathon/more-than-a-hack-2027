"""Degradation invariants on synthetic tracking."""

import numpy as np

from more_than_a_hack_2027.external import degrade


def arrays(n=600, per_team=11, seed=0):
    rng = np.random.RandomState(seed)
    players = 2 * per_team
    xy = np.cumsum(rng.normal(0, 0.1, size=(n, players, 2)), axis=0) + rng.uniform(10, 60, (players, 2))
    ball = np.cumsum(rng.normal(0, 0.3, size=(n, 2)), axis=0) + 50
    ball[200:260] = np.nan
    return {
        "xy": xy, "ball": ball, "speed": rng.uniform(0, 5, (n, players)).astype(np.float32),
        "team_of_slot": np.array([0] * per_team + [1] * per_team, dtype=np.int8),
        "period": np.ones(n, dtype=np.int8), "team_names": np.array(["Home", "Away"]),
    }


def test_smoothing_keeps_lines_exact_and_gaps_missing():
    track = np.full((80, 2), np.nan)
    track[10:70, 0] = np.linspace(0, 30, 60)            # a ball rolling up to a line, then gone
    track[10:70, 1] = 3.0
    out = degrade.gaussian_smooth(track, 7.0)
    assert np.allclose(out[10:70], track[10:70])
    assert np.isnan(out[:10]).all() and np.isnan(out[70:]).all()


def test_dropout_removes_exactly_the_requested_count():
    data = arrays()
    n = len(data["ball"])
    missing = np.tile(np.array([0, 1, 2, 5, 30], dtype=np.int16), n // 5)
    xy, speed = degrade.drop_players(data["xy"], data["speed"], missing,
                                     np.random.RandomState(0), renewal_frames=50)
    visible = (~np.isnan(xy[:, :, 0])).sum(axis=1)
    assert np.array_equal(visible, np.maximum(22 - missing, 0))
    assert np.array_equal(np.isnan(speed), np.isnan(xy[:, :, 0]))


def test_fragmentation_keeps_every_observation_and_its_team():
    data = arrays()
    xy, speed = degrade.drop_players(data["xy"], data["speed"],
                                     np.full(len(data["ball"]), 3, dtype=np.int16),
                                     np.random.RandomState(1), renewal_frames=40)
    new_xy, new_speed, new_team = degrade.fragment(xy, speed, data["team_of_slot"])
    for team in (0, 1):
        before = xy[:, data["team_of_slot"] == team]
        after = new_xy[:, new_team == team]
        assert np.array_equal((~np.isnan(before[:, :, 0])).sum(1), (~np.isnan(after[:, :, 0])).sum(1))
        for t in (0, 123, 599):                              # same positions, possibly other slots
            a = np.sort(before[t][~np.isnan(before[t, :, 0])], axis=0)
            b = np.sort(after[t][~np.isnan(after[t, :, 0])], axis=0)
            assert np.array_equal(a, b)
    assert new_xy.shape[1] >= 16


def test_degrade_is_deterministic_and_seeded():
    data = arrays()
    counts = np.random.RandomState(3).randint(0, 4, size=1000).astype(np.int16)
    first = degrade.degrade(data, counts, game=1)
    second = degrade.degrade(data, counts, game=1)
    other = degrade.degrade(data, counts, {"seed": 7}, game=1)
    assert np.array_equal(first["xy"], second["xy"], equal_nan=True)
    assert np.array_equal(first["ball"], second["ball"], equal_nan=True)
    assert not np.array_equal(first["ball"], other["ball"], equal_nan=True)
    assert np.isnan(first["ball"][200:260]).all()


def _live_mask():
    live = np.zeros(30000, dtype=bool)
    for a, b in ((100, 9000), (9500, 20000), (20400, 29990)):     # three live runs
        live[a:b] = True
    return live


def test_live_gaps_stay_inside_live_play_and_hit_the_target_kinds():
    live = _live_mask()
    hide = degrade.live_gaps(live, np.random.RandomState(0),
                             params={"interior_per_live_min": 2.0, "before_stop_p": 1.0,
                                     "after_restart_p": 1.0})
    assert not (hide & ~live).any()                       # never outside live play
    for a, b in ((100, 9000), (9500, 20000), (20400, 29990)):
        assert hide[b - 1] and hide[a]                   # every run: lost before the stop ...
    assert hide[1000:8000].any()                         # ... after the restart, and inside


def test_live_gaps_can_be_switched_off_per_kind():
    live = _live_mask()
    none = degrade.live_gaps(live, np.random.RandomState(0),
                             params={"interior_per_live_min": 0.0, "before_stop_p": 0.0,
                                     "after_restart_p": 0.0})
    assert not none.any()


def test_live_gaps_leave_every_other_degradation_unchanged():
    data = arrays()
    counts = np.random.RandomState(3).randint(0, 4, size=1000).astype(np.int16)
    live = np.zeros(len(data["ball"]), dtype=bool)
    live[20:180] = live[300:580] = True
    params = {"interior_per_live_min": 30.0, "interior_s": (0.4, 1.0)}
    on = degrade.degrade(data, counts, {"live_gap_params": params}, game=1, live=live)
    off = degrade.degrade(data, counts, {"live_gaps": False}, game=1, live=live)
    assert np.array_equal(on["xy"], off["xy"], equal_nan=True)
    hidden = np.isnan(on["ball"][:, 0]) & ~np.isnan(off["ball"][:, 0])
    assert hidden.any() and not (hidden & ~live).any()
    assert not (~np.isnan(on["ball"][:, 0]) & np.isnan(off["ball"][:, 0])).any()
    again = degrade.degrade(data, counts, {"live_gap_params": params}, game=1, live=live)
    assert np.array_equal(on["ball"], again["ball"], equal_nan=True)


def test_dead_ball_view_replays_the_pattern_of_a_stoppage_of_similar_length():
    short_seen = np.ones(50, dtype=bool)                  # a 2 s stoppage, ball in view
    long_hidden = np.zeros(1000, dtype=bool)              # a 40 s stoppage, ball gone
    half = np.r_[np.ones(100, bool), np.zeros(100, bool)]  # 8 s: seen, then gone
    views = [short_seen, long_hidden, half]
    dead = np.zeros(3000, dtype=bool)
    dead[100:160] = True                                  # 2.4 s: nearest is the 2 s view
    dead[500:1400] = True                                 # 36 s: nearest is the 40 s view
    dead[2000:2400] = True                                # 16 s: nearest is the 8 s view
    hide = degrade.dead_ball_view(dead, views, np.random.RandomState(0), nearest=1)
    assert not hide[~dead].any()
    assert not hide[100:160].any() and hide[500:1400].all()
    np.testing.assert_array_equal(hide[2000:2400], np.r_[np.zeros(200, bool), np.ones(200, bool)])


def test_dead_ball_views_hide_only_dead_frames_and_leave_the_rest_paired():
    data = arrays()
    counts = np.random.RandomState(3).randint(0, 4, size=1000).astype(np.int16)
    live = np.ones(len(data["ball"]), dtype=bool)
    live[300:420] = False                                 # one stoppage, ball tracked through it
    views = [np.r_[np.ones(30, bool), np.zeros(60, bool), np.ones(30, bool)]]
    on = degrade.degrade(data, counts, {}, game=1, live=live, dead_views=views)
    off = degrade.degrade(data, counts, {}, game=1, live=live)
    assert np.array_equal(on["xy"], off["xy"], equal_nan=True)
    hidden = np.isnan(on["ball"][:, 0]) & ~np.isnan(off["ball"][:, 0])
    assert hidden.any() and not (hidden & live).any()
    assert not (~np.isnan(on["ball"][:, 0]) & np.isnan(off["ball"][:, 0])).any()
    np.testing.assert_array_equal(on["ball"][:250], off["ball"][:250])   # far from the stoppage
