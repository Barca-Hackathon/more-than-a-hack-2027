"""Restart-team evidence on hand-built inputs."""

from types import SimpleNamespace

import numpy as np

from more_than_a_hack_2027 import restart
from more_than_a_hack_2027.context import AWAY, HOME
from more_than_a_hack_2027.deadball import Kickoff, OutEpisode, Stoppage
from more_than_a_hack_2027.io_tagging import Tag, team_code
from more_than_a_hack_2027.nearest_player import NOBODY


def test_last_toucher_is_latest_close_player():
    codes = np.array([HOME, HOME, AWAY, AWAY, NOBODY, NOBODY], dtype=np.int8)
    distance = np.array([0.5, 0.8, 3.0, 1.0, np.nan, np.nan])
    assert restart.last_toucher(codes, distance, 6, 10, 1.5) == AWAY
    assert restart.last_toucher(codes, distance, 3, 10, 1.5) == HOME
    assert restart.last_toucher(codes, distance, 1, 1, 0.1) is None


def make_dead(n, stoppages, codes, distance):
    return SimpleNamespace(stoppages=stoppages, nearest=(codes, distance, None))


def test_ball_out_goes_to_the_opponent_of_the_last_toucher():
    n = 400
    codes = np.full(n, NOBODY, dtype=np.int8)
    distance = np.full(n, np.nan)
    codes[90:100], distance[90:100] = HOME, 0.5        # home touched it last, then it went out
    episode = OutEpisode(100, 120, "touchline", "low", False, 3.0)
    stop = Stoppage(start=100, restart=300, half=1, kind="out_touchline", restart_kind="throw_in",
                    confidence=1.0, out=episode)
    ctx = SimpleNamespace(fps=25, n_frames=n)
    call = restart.decide(ctx, make_dead(n, [stop], codes, distance))[0]
    assert call.team == AWAY and call.sources["out"] < 0


def test_tag_and_kickoff_sources_combine():
    n = 2000
    codes = np.full(n, NOBODY, dtype=np.int8)
    distance = np.full(n, np.nan)
    kick = Kickoff(frame=1000, half=1, formation_start=900, formation_stop=1010,
                   at_half_start=False, kicker_team=AWAY)
    stop = Stoppage(start=500, restart=1000, half=1, kind="goal", restart_kind="kickoff",
                    confidence=1.0, kickoff=kick)
    tag = Tag(code="SET PIECES", start=990, stop=1115, team=AWAY, kind="kick off (after goal)")
    ctx = SimpleNamespace(fps=25, n_frames=n)
    call = restart.decide(ctx, make_dead(n, [stop], codes, distance), [tag])[0]
    assert call.team == AWAY
    assert set(call.sources) == {"tag", "kickoff"}

    evidence = restart.frame_evidence(SimpleNamespace(fps=25, n_frames=n), [call])
    assert evidence[1000, AWAY] > 0 and evidence[1000, HOME] == 0 and evidence[999].sum() == 0


def test_team_code_mapping():
    names = ("Barcelona", "Atletic Club")
    assert team_code("barcelona ", names) == HOME
    assert team_code("Atletic Club", names) == AWAY
    assert team_code("N/A", names) is None and team_code(None, names) is None
