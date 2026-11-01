"""Synthetic match builder, so stage logic can be tested without the gated data."""

import numpy as np
import pytest

from more_than_a_hack_2027.context import Context

PITCH = (105.0, 68.0)


def make_ctx(n, halves=None, n_per_team=10, xy=None, speed=None, ball=None, fps=25):
    """A Context with 2*n_per_team slots (home first). Defaults: everyone static in their own half
    (home attacks +x in half 1), nobody near the centre spot, speed 2 m/s, no ball."""
    halves = halves or ((1, 0, n - 1),)
    slots = 2 * n_per_team
    team_of_slot = np.array([0] * n_per_team + [1] * n_per_team, dtype=np.int8)
    if xy is None:
        xy = np.zeros((n, slots, 2))
        xy[:, :n_per_team, 0] = np.linspace(10, 44, n_per_team)
        xy[:, n_per_team:, 0] = np.linspace(61, 95, n_per_team)
        xy[:, :, 1] = np.tile(np.linspace(5, 63, n_per_team), 2)
    if speed is None:
        speed = np.full((n, slots), 2.0, dtype=np.float32)
    if ball is None:
        ball = np.full((n, 2), np.nan)
    half_of_frame = np.zeros(n, dtype=np.int8)
    for half, first, last in halves:
        half_of_frame[first:last + 1] = half
    return Context(
        game="SYN", n_frames=n, fps=fps, pitch=PITCH, xy=xy, speed=speed, ball=ball,
        team_of_slot=team_of_slot, team_names=("Home FC", "Away FC"), halves=tuple(halves),
        home_attacks_positive_x={1: True, 2: False}, half_of_frame=half_of_frame)


@pytest.fixture
def synthetic():
    return make_ctx
