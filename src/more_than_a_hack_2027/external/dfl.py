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
~246 and ~9 s for the Metrica event-derived labels), so it is not the scored kind of label
(labels="flag", files dfl_*). The "control" labels (files dflc_*) follow the Metrica rule
instead -- possession changes when the other team GAINS control -- by filtering the flag with
DFL's own event data: a change counts only if the new team has a control event of its own (a
pass or cross, shot, ball claim, set piece, or a tackle won with the ball) within its flag spell,
+-1.5 s; otherwise the touch is given back to the team that had the ball. The flag keeps the
frame-exact timing, the events only decide which changes count. This gives 214-272 possessions
per match (median 9-11 s), like the Metrica labels.

The raw event times are logged live and run up to ~2 s off the tracking clock, so each half's
offset is fitted first: the one (-2 s ... +2 s) that puts the tracked ball closest to the event
positions of passes, shots and other ball actions (median 4-6 m at the best offset).
"""

import datetime
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from .._paths import CACHE_DIR, OTHER_MATCHES_DIR

MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
RAW_DIR = os.path.join(OTHER_MATCHES_DIR, "dfl")
ARRAY_DIR = os.path.join(CACHE_DIR, "dfl")
PITCH = (105.0, 68.0)
FILES = {"meta": "{}_matchinfo.xml", "tracking": "{}_positions.xml", "event": "{}_events.xml"}
FPS = 25
FIRST_FRAMES = (10000, 100000)          # DFL frame number of the first frame of each half
SET_PIECES = {"KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty"}
CONTROL_EVENTS = {"Play", "ShotAtGoal", "BallClaiming"} | SET_PIECES
LOCATED_EVENTS = ("Play", "ShotAtGoal", "OtherBallAction")     # used to fit the clock offset
CONTROL_TOL_FRAMES = 38                 # 1.5 s either side of a flag spell
SYNC_OFFSETS_S = np.arange(-2.0, 2.01, 0.04)
LABEL_KINDS = {"flag": "dfl", "control": "dflc"}


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


def _timestamp(text):
    return datetime.datetime.fromisoformat(text).timestamp()


def first_frame_times(match_id):
    """{DFL frame number: wall-clock time (s)} for the first frame of each half, read from the
    positions XML once and cached."""
    path = os.path.join(ARRAY_DIR, match_id + "_t0.json")
    if os.path.exists(path):
        with open(path) as handle:
            return {int(k): v for k, v in json.load(handle).items()}
    patterns = {n: re.compile(rb'N="%d" T="([^"]+)"' % n) for n in FIRST_FRAMES}
    found, tail = {}, b""
    with open(raw_path(match_id, "tracking"), "rb") as handle:
        while len(found) < len(patterns):
            chunk = handle.read(1 << 24)
            if not chunk:
                raise ValueError("{}: first frames {} not found".format(
                    match_id, sorted(set(patterns) - set(found))))
            buf = tail + chunk
            for n, pattern in patterns.items():
                m = pattern.search(buf) if n not in found else None
                if m:
                    found[n] = _timestamp(m.group(1).decode())
            tail = buf[-256:]
    os.makedirs(ARRAY_DIR, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(found, handle)
    return found


def frame_times(match_id, frame_id, period):
    """Wall-clock time (s) of every frame."""
    t0 = first_frame_times(match_id)
    base = np.where(period == 1, FIRST_FRAMES[0], FIRST_FRAMES[1])
    return np.array([t0[b] for b in base]) + (frame_id - base) / FPS


def home_team_id(match_id):
    return ET.parse(raw_path(match_id, "meta")).getroot().find(".//General").get("HomeTeamId")


def events(match_id):
    """[(time s, tag, acting team id or None, x, y, attributes)] in file order. The acting team
    of a tackle is its winner; of a set piece, the team of the play inside it."""
    out = []
    for event in ET.parse(raw_path(match_id, "event")).getroot():
        t = _timestamp(event.get("EventTime"))
        x, y = event.get("X-Position"), event.get("Y-Position")
        x = float(x) if x else np.nan
        y = float(y) if y else np.nan
        for child in event:
            team = child.get("Team") or child.get("WinnerTeam")
            if child.tag in SET_PIECES:
                inner = next(iter(child), None)
                if inner is not None:
                    team = team or inner.get("Team")
            out.append((t, child.tag, team, x, y, dict(child.attrib)))
    return out


def is_control_event(tag, attrs):
    return tag in CONTROL_EVENTS or (tag == "TacklingGame"
                                     and attrs.get("WinnerRole") == "withBallControl")


def clock_offset(ball, ftime, evs):
    """The event-clock offset (s) of one half: the one that puts the ball closest (median) to
    the located events. ball (n, 2+) and ftime (n,) are that half's frames."""
    located = [e for e in evs if e[1] in LOCATED_EVENTS and np.isfinite(e[3])
               and ftime[0] - 5 <= e[0] <= ftime[-1] + 5]
    times = np.array([e[0] for e in located])
    xy = np.array([(e[3], PITCH[1] - e[4]) for e in located])    # DFL's y runs the other way
    best = None
    for offset in SYNC_OFFSETS_S:
        idx = np.round((times + offset - ftime[0]) / 0.04).astype(int)
        ok = (idx >= 0) & (idx < len(ftime))
        b = ball[idx[ok]]
        d = np.nanmedian(np.hypot(b[:, 0] - xy[ok, 0], b[:, 1] - xy[ok, 1]))
        if best is None or d < best[0]:
            best = (d, offset)
    return best[1], best[0]


def _runs(codes):
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    starts, stops = np.r_[0, change], np.r_[change, len(codes)]
    return [(a, b, codes[a]) for a, b in zip(starts, stops)]


def filter_switches(flag, event_frames, event_teams, tol=CONTROL_TOL_FRAMES):
    """Control-filtered labels from per-frame flag codes (0/1 team, 2 dead): a flag spell of a
    team with no control event of its own within [start - tol, stop + tol) is given to the team
    whose spell it interrupts (the preceding one, in the same stretch of live play).

    event_frames must be sorted; event_teams are their team codes."""
    event_frames, event_teams = np.asarray(event_frames), np.asarray(event_teams)
    out = np.asarray(flag).copy()
    prev, prev_end = None, -1
    for a, b, team in _runs(np.asarray(flag)):
        if team == 2:
            prev, prev_end = None, b
            continue
        lo, hi = np.searchsorted(event_frames, a - tol), np.searchsorted(event_frames, b + tol)
        own = bool((event_teams[lo:hi] == team).any())
        if not own and prev is not None and prev_end == a:
            out[a:b] = prev
        else:
            prev = out[a]
        prev_end = b
    return out


def control_labels(match_id, log=None):
    """H/A/D codes (0/1/2) per frame: the flag filtered by DFL's control events."""
    z = load_arrays(match_id)
    period, alive, owner = z["period"].astype(int), z["alive"], z["owner"].astype(int)
    ftime = frame_times(match_id, z["frame_id"], period)
    home = home_team_id(match_id)
    evs = events(match_id)
    marks = []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        offset, dist = clock_offset(z["ball"][rows], ftime[rows], evs)
        if log:
            log("{} half {}: event clock offset {:+.2f} s, median ball distance {:.1f} m".format(
                match_id, half, offset, dist))
        start, stop = ftime[rows[0]], ftime[rows[-1]]
        for t, tag, team, _, _, attrs in evs:
            if team is None or not start - 5 <= t <= stop + 5 or not is_control_event(tag, attrs):
                continue
            i = int(round((t + offset - start) / 0.04))
            marks.append((rows[0] + min(max(i, 0), len(rows) - 1), 0 if team == home else 1))
    marks.sort()
    flag = np.where(alive & (owner >= 0), owner, 2)
    return filter_switches(flag, [f for f, _ in marks], [k for _, k in marks]).astype(np.int8)


def input_file(match_id, seed, labels="flag"):
    """Basename of the cached sequence-model inputs (training CACHE_DIR), as training "aux":
    dfl_* (flag labels) or dflc_* (control labels)."""
    return "{}_{}_s{}.npz".format(LABEL_KINDS[labels], match_id, seed)


def build_inputs(match_id, seed, missing_counts=None, labels="flag"):
    """Degrade one match like the external games and cache network inputs + labels
    (labels: "flag" or "control", see the module docstring)."""
    from . import degrade, metrica
    from ..validation import harness
    from ..seq import network
    from ..seq.training import CACHE_DIR as INPUT_DIR

    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, input_file(match_id, seed, labels))
    if os.path.exists(path):
        return path
    if missing_counts is None:
        missing_counts = harness._competition_missing_counts()
    arrays, flag = arrays_and_labels(match_id)
    game = 100 + MATCHES.index(match_id)
    degraded = degrade.degrade(arrays, missing_counts, {"seed": seed}, game=game,
                               live=flag < 2)
    ctx = metrica.make_context(game, degraded)
    graded = ctx.half_of_frame > 0
    codes = flag if labels == "flag" else control_labels(match_id)
    labels = np.where(graded, codes, 3).astype(np.int8)
    X, swap_sign = network.inputs(ctx)
    np.savez(path, X=X, swap_sign=swap_sign, labels=labels, graded=graded,
             halves=np.array(ctx.halves))
    return path
