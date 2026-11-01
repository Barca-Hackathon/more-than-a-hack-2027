"""Degradation invariants on synthetic tracking."""

import numpy as np

from more_than_a_hack_2027 import degrade


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
