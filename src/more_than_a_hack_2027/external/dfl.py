"""The open DFL matches (IDSSE dataset), as extra training data for the sequence model.

Source: Bassek, Rein, Weber et al., "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y,
CC BY 4.0. Seven Bundesliga matches with 25 Hz positions; files from
huggingface.co/datasets/pysport/idsse-data (the URLs kloppy uses). The tech doc must cite them.

Storage: the raw XML files are downloaded once into data/other_matches/dfl/ and kept there
(never deleted; data/ is never committed). Parsed arrays are cached in .cache/dfl/, and the
sequence-model inputs of each degraded copy in .cache/seq_inputs/.

Labels. Each position frame carries DFL's ball status (alive / dead) and ball-possessing team.
That possession flag changes on every touch (400-510 possessions per match, median 3-4 s, against
~246 and ~9 s for the Metrica event-derived labels), so it is NOT the label the submission is
scored on. It is used only through a second output head of the sequence model (training "aux"),
so the shared network learns who touches the ball from seven more matches while the scored head
is trained on the Metrica labels alone.
"""

import os
import urllib.request

import numpy as np

from .._paths import CACHE_DIR, OTHER_MATCHES_DIR

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
RAW_DIR = os.path.join(OTHER_MATCHES_DIR, "dfl")
ARRAY_DIR = os.path.join(CACHE_DIR, "dfl")
PITCH = (105.0, 68.0)
FILES = {"meta": "{}_matchinfo.xml", "tracking": "{}_positions.xml"}


def raw_path(match_id, kind):
    return os.path.join(RAW_DIR, FILES[kind].format(match_id))


def download(match_ids=MATCHES, verbose=True):
    """Fetch any missing raw file into data/other_matches/dfl/. Existing files are kept as they
    are; a partial download is written to *.part and only renamed once complete."""
    from kloppy._providers.sportec import get_IDSSE_url

    os.makedirs(RAW_DIR, exist_ok=True)
    for match_id in match_ids:
        for kind in FILES:
            path = raw_path(match_id, kind)
            if os.path.exists(path):
                continue
            url = get_IDSSE_url(match_id, kind)
            if verbose:
                print("downloading {} -> {}".format(url, path))
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)


def load_arrays(match_id):
    """dict of arrays for one match (parsed once from the raw XML, then cached)."""
    os.makedirs(ARRAY_DIR, exist_ok=True)
    path = os.path.join(ARRAY_DIR, match_id + ".npz")
    if not os.path.exists(path):
        download((match_id,))
        _parse(match_id, path)
    return dict(np.load(path))


def _parse(match_id, path):
    from kloppy import sportec

    ds = sportec.load_tracking(meta_data=raw_path(match_id, "meta"),
                               raw_data=raw_path(match_id, "tracking"), coordinates="kloppy")
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    players = [(p, 0) for p in home.players] + [(p, 1) for p in away.players]
    index = {p.player_id: i for i, (p, _) in enumerate(players)}
    n = len(ds.frames)
    length, width = PITCH
    xy = np.full((n, len(players), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    period = np.zeros(n, np.int8)
    alive = np.zeros(n, bool)
    owner = np.full(n, -1, np.int8)
    frame_id = np.zeros(n, np.int64)
    for i, frame in enumerate(ds.frames):
        period[i] = frame.period.id
        frame_id[i] = frame.frame_id
        alive[i] = str(frame.ball_state) == "BallState.ALIVE"
        if frame.ball_owning_team is not None:
            owner[i] = 0 if frame.ball_owning_team.team_id == home.team_id else 1
        if frame.ball_coordinates is not None:
            c = frame.ball_coordinates
            ball[i] = (c.x * length, c.y * width, c.z or 0)
        for player, data in frame.players_data.items():
            if data.coordinates is not None and player.player_id in index:
                xy[i, index[player.player_id]] = (data.coordinates.x * length,
                                                  data.coordinates.y * width)
    np.savez_compressed(path, xy=xy, ball=ball, period=period, alive=alive, owner=owner,
                        frame_id=frame_id, team=np.array([t for _, t in players], np.int8),
                        fps=ds.metadata.frame_rate, names=np.array([home.name, away.name]))


def arrays_and_labels(match_id):
    """(metrica.load_arrays-style dict, H/A/D codes from DFL's ball status and possession)."""
    z = load_arrays(match_id)
    alive, owner, period = z["alive"], z["owner"].astype(int), z["period"].astype(int)
    xy = z["xy"].astype(np.float64)
    speed = np.full(xy.shape[:2], np.nan, np.float32)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        v = np.gradient(xy[rows], axis=0) * 25.0
        speed[rows] = np.hypot(v[..., 0], v[..., 1])
    ball = z["ball"][:, :2].astype(np.float64)
    # DFL tracks the ball while it is dead; BAR-ATH and Metrica mostly do not. Keep it only for
    # the first 0.5 s after a stop and the last 1 s before the restart.
    edges = np.diff(np.r_[0, (~alive).astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ball[a + 12:max(a + 12, b - 25)] = np.nan
    labels = np.where(alive, owner, 2).astype(np.int8)
    arrays = {"xy": xy, "speed": speed, "ball": ball, "team_of_slot": z["team"].astype(np.int8),
              "team_names": z["names"], "period": period}
    return arrays, labels


def input_file(match_id, seed):
    """Basename of the cached sequence-model inputs (training CACHE_DIR), as training "aux"."""
    return "dfl_{}_s{}.npz".format(match_id, seed)


def build_inputs(match_id, seed, missing_counts=None):
    """Degrade one match like the external games and cache network inputs + labels."""
    from . import degrade, metrica
    from ..validation import harness
    from ..seq import network
    from ..seq.training import CACHE_DIR as INPUT_DIR

    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, input_file(match_id, seed))
    if os.path.exists(path):
        return path
    if missing_counts is None:
        missing_counts = harness._competition_missing_counts()
    arrays, labels = arrays_and_labels(match_id)
    game = 100 + MATCHES.index(match_id)
    degraded = degrade.degrade(arrays, missing_counts, {"seed": seed}, game=game,
                               live=labels < 2)
    ctx = metrica.make_context(game, degraded)
    graded = ctx.half_of_frame > 0
    labels = np.where(graded, labels, 3).astype(np.int8)
    X, swap_sign = network.inputs(ctx)
    np.savez(path, X=X, swap_sign=swap_sign, labels=labels, graded=graded,
             halves=np.array(ctx.halves))
    return path
