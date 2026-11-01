"""The shared, read-only view of one match that every pipeline stage receives.

Stage contract (so the workstreams can be built in parallel):

    def stage(ctx: Context, cfg: dict) -> np.ndarray      # leading axis is ctx.n_frames

  * Index 0 of every per-frame array is video frame 0 -- i.e. submission row 0.
  * Arrays cover the whole video; stages may leave ungraded frames as NaN / defaults.
  * Never compute across a half boundary: use ctx.halves (or ctx.half_of_frame).
  * Coordinates are metres, origin at a pitch corner, x along the length. Use
    ctx.attack_sign(half, team) to normalise "towards the opponent goal" to +x.
  * Stages must be deterministic (fixed seeds), because the submission must be reproducible
    byte-for-byte from the repository.
"""

from dataclasses import dataclass, field

import numpy as np

from . import atd
from .._paths import ensure_repo_on_path

ensure_repo_on_path()
import games  # noqa: E402

HOME, AWAY = 0, 1


@dataclass(frozen=True)
class Context:
    game: str
    n_frames: int
    fps: int
    pitch: tuple                 # (length, width) in metres
    xy: np.ndarray               # (n, S, 2) float64 metres; NaN where the slot is unobserved
    speed: np.ndarray            # (n, S) float32 m/s; NaN where unknown
    ball: np.ndarray             # (n, 2) float64 metres; NaN where the ball is unseen
    team_of_slot: np.ndarray     # (S,) int8, HOME=0 / AWAY=1
    team_names: tuple            # (home, away)
    halves: tuple                # ((half, first_video_frame, last_video_frame), ...) inclusive
    home_attacks_positive_x: dict  # {half: bool}
    half_of_frame: np.ndarray = field(repr=False)  # (n,) int8, 0 outside the graded halves

    @property
    def n_slots(self):
        return int(self.team_of_slot.shape[0])

    @property
    def ball_seen(self):
        return ~np.isnan(self.ball[:, 0])

    def team_slots(self, team):
        return np.flatnonzero(self.team_of_slot == team)

    def attack_sign(self, half, team):
        """+1 if `team` attacks towards +x in `half`, else -1."""
        home_positive = self.home_attacks_positive_x[half]
        return 1 if (home_positive == (team == HOME)) else -1

    def half_slices(self):
        """[(half, slice)] over video frames, half-open, in play order."""
        return [(half, slice(first, last + 1)) for half, first, last in self.halves]


def load_context(game=games.GAME, data_dir=None, source="tactical"):
    if source != "tactical":
        raise ValueError("the submission is indexed to the tactical recording only")
    data = atd.load_cached(source, game, data_dir)

    halves = tuple(games.half_video_blocks(game))
    half_of_frame = np.zeros(data["n_frames"], dtype=np.int8)
    for half, first, last in halves:
        half_of_frame[first:last + 1] = half

    home_dir = data["attack_direction_first_half"][0]
    if home_dir not in ("left_to_right", "right_to_left"):
        raise ValueError("unexpected attack_direction_first_half {!r}".format(home_dir))
    first_half_positive = home_dir == "left_to_right"
    home_attacks_positive_x = {1: first_half_positive, 2: not first_half_positive}

    return Context(
        game=game,
        n_frames=int(data["n_frames"]),
        fps=int(games.GAMES[game]["fps"]),
        pitch=tuple(data["pitch"]),
        xy=data["xy"],
        speed=data["speed"],
        ball=data["ball"],
        team_of_slot=data["team_of_slot"],
        team_names=tuple(data["team_names"]),
        halves=halves,
        home_attacks_positive_x=home_attacks_positive_x,
        half_of_frame=half_of_frame,
    )
