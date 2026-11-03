"""The SkillCorner open matches, as extra training data for the sequence model.

Source: SkillCorner open data, github.com/SkillCorner/opendata (MIT licence; SkillCorner asks to
be credited). Twenty Australian A-League 2024/25 matches of broadcast tracking at 10 Hz, with
every player's position extrapolated when off camera (`is_detected` says which were seen), plus
SkillCorner's "phases of play" and dynamic events. The tech doc must cite them.

Storage: the raw files are downloaded once into data/other_matches/skillcorner/{id}/ and kept
there (never deleted; data/ is never committed). Parsed 25 Hz arrays are cached in
.cache/skillcorner/, and the sequence-model inputs of each degraded copy in .cache/seq_inputs/
(files sc_{id}_s{seed}.npz).

Labels. A phase of play is a stretch of live ball with one team in possession, so the ball is
dead between phases and a possession is a run of consecutive phases of one team. That gives
~280 possessions per match with a ~6 s median, near the ~246 and ~9 s of the Metrica
event-derived labels (more short contested spells: SkillCorner's "chaotic" phases). Those are
labels="phase" (files sc_*). labels="control" (files scc_*) applies the Metrica one-touch rule on
top: a spell that is a single zero-length touch straight after the other team's spell (a
first-time clearance, a deflection) is not a possession, ~20-35 per match.

Resampling. The 10 Hz positions are interpolated linearly to the 25 Hz the pipeline assumes,
inside each period; a gap stays a gap. Labels take the phase of the nearest 10 Hz frame.
Coordinates move from SkillCorner's centre origin (y up) to the 105 x 68 m, top-left origin
(y down) frame of the Metrica and DFL arrays, scaled from each match's pitch size.
"""

import json
import os
import urllib.request

import numpy as np

from .._paths import CACHE_DIR, OTHER_MATCHES_DIR

RAW_DIR = os.path.join(OTHER_MATCHES_DIR, "skillcorner")
ARRAY_DIR = os.path.join(CACHE_DIR, "skillcorner")
BASE_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/"
LFS_URL = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/"
FILES = {"match": "{}_match.json", "tracking": "{}_tracking_extrapolated.jsonl",
         "phases": "{}_phases_of_play.csv", "events": "{}_dynamic_events.csv"}
SOURCE_FPS = 10
FPS = 25
PITCH = (105.0, 68.0)
GAME_OFFSET = 200          # Context game numbers 200, 201, ... (Metrica 1-3, DFL 100+)
LABEL_KINDS = {"phase": "sc", "control": "scc"}
UNTRACKED = 3              # label code of frames with no tracked player (possession_metrics X)


def _fetch(url, path):
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)


def match_ids():
    """The ids listed in the repository's matches.json, oldest first."""
    path = os.path.join(RAW_DIR, "matches.json")
    if not os.path.exists(path):
        os.makedirs(RAW_DIR, exist_ok=True)
        _fetch(BASE_URL + "matches.json", path)
    with open(path) as handle:
        return tuple(sorted(int(m["id"]) for m in json.load(handle)))


def raw_path(match_id, kind):
    return os.path.join(RAW_DIR, str(match_id), FILES[kind].format(match_id))


def download(ids=None, verbose=True):
    """Fetch any missing raw file into data/other_matches/skillcorner/. Existing files are kept;
    a partial download is written to *.part and only renamed once complete."""
    for match_id in ids or match_ids():
        os.makedirs(os.path.join(RAW_DIR, str(match_id)), exist_ok=True)
        for kind, name in FILES.items():
            path = raw_path(match_id, kind)
            if os.path.exists(path):
                continue
            base = LFS_URL if kind == "tracking" else BASE_URL
            url = "{}matches/{}/{}".format(base, match_id, name.format(match_id))
            if verbose:
                print("downloading {} -> {}".format(url, path))
            _fetch(url, path)


# -- tracking ------------------------------------------------------------------------------------

def to_pitch(x, y, length, width):
    """SkillCorner metres (centre origin, y up) -> 105 x 68 m, top-left origin, y down."""
    return ((np.asarray(x, float) / length + 0.5) * PITCH[0],
            (0.5 - np.asarray(y, float) / width) * PITCH[1])


def resample(values, n_out, ratio=FPS / SOURCE_FPS):
    """Linear interpolation of a 10 Hz array (time on axis 0) at 25 Hz. A 25 Hz frame between
    two source frames is NaN if either of them is."""
    t = np.arange(n_out) / ratio
    lo = np.clip(np.floor(t).astype(int), 0, len(values) - 1)
    hi = np.clip(lo + 1, 0, len(values) - 1)
    w = (t - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        return np.where(w == 0, values[lo], values[lo] * (1 - w) + values[hi] * w)


def nearest(values, n_out, ratio=FPS / SOURCE_FPS):
    """The 10 Hz value nearest each 25 Hz frame (for labels)."""
    idx = np.clip(np.round(np.arange(n_out) / ratio).astype(int), 0, len(values) - 1)
    return values[idx]


def _parse(match_id, path):
    with open(raw_path(match_id, "match")) as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: 0 if p["team_id"] == home_id else 1 for p in meta["players"]}
    slots = sorted(team_of, key=lambda pid: (team_of[pid], pid))
    slot_of = {pid: i for i, pid in enumerate(slots)}

    frames, periods, balls, xys = [], [], [], []
    with open(raw_path(match_id, "tracking")) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("period") is None:
                continue
            xy = np.full((len(slots), 2), np.nan, np.float32)
            for p in row.get("player_data") or ():
                i = slot_of.get(p["player_id"])
                if i is not None and p.get("x") is not None:
                    xy[i] = to_pitch(p["x"], p["y"], length, width)
            b = row.get("ball_data") or {}
            ball = (to_pitch(b["x"], b["y"], length, width) if b.get("x") is not None
                    else (np.nan, np.nan))
            frames.append(row["frame"])
            periods.append(row["period"])
            balls.append(ball)
            xys.append(xy)
    np.savez_compressed(path, frame=np.array(frames, np.int64), period=np.array(periods, np.int8),
                        ball=np.array(balls, np.float32), xy=np.array(xys, np.float32),
                        team=np.array([team_of[pid] for pid in slots], np.int8),
                        names=np.array([meta["home_team"]["short_name"],
                                        meta["away_team"]["short_name"]]),
                        home_id=home_id)


def load_source(match_id):
    """10 Hz arrays of one match (parsed once from the raw files, then cached)."""
    os.makedirs(ARRAY_DIR, exist_ok=True)
    path = os.path.join(ARRAY_DIR, "{}.npz".format(match_id))
    if not os.path.exists(path):
        download((match_id,))
        _parse(match_id, path)
    return dict(np.load(path))


# -- labels --------------------------------------------------------------------------------------

def phase_labels(frames, phases, home_id):
    """H/A/D codes (0/1/2) per 10 Hz frame: the team in possession inside each phase of play
    [frame_start, frame_end), dead ball between phases."""
    codes = np.full(len(frames), 2, np.int8)
    for start, end, team in phases:
        codes[(frames >= start) & (frames < end)] = 0 if team == home_id else 1
    return codes


def read_phases(match_id):
    import pandas as pd
    ph = pd.read_csv(raw_path(match_id, "phases"))
    return list(zip(ph.frame_start.astype(int), ph.frame_end.astype(int),
                    ph.team_in_possession_id.astype(int)))


def read_touches(match_id):
    """[(frame_start, frame_end, team_id)] of SkillCorner's player possessions."""
    import pandas as pd
    de = pd.read_csv(raw_path(match_id, "events"), low_memory=False)
    pp = de[de.event_type == "player_possession"]
    return list(zip(pp.frame_start.astype(int), pp.frame_end.astype(int), pp.team_id.astype(int)))


def team_spells(phases):
    """[(start, end, team)]: consecutive phases of one team with no dead ball between, merged."""
    spells = []
    for start, end, team in sorted(phases):
        if spells and spells[-1][2] == team and start - spells[-1][1] <= 1:
            spells[-1][1] = end
        else:
            spells.append([start, end, team])
    return [tuple(x) for x in spells]


def control_spells(phases, touches):
    """team_spells under the Metrica one-touch rule: a spell straight after the other team's,
    made of a single touch of zero duration (a first-time clearance, a deflection, a first-time
    pass to nobody), is not a possession; its frames stay with the team that had the ball."""
    out = []
    for start, end, team in team_spells(phases):
        own = [(a, b) for a, b, t in touches if t == team and start <= a < end]
        one_touch = len(own) == 1 and own[0][0] == own[0][1]
        if out and one_touch and out[-1][2] != team and start - out[-1][1] <= 1:
            team = out[-1][2]
        if out and out[-1][2] == team and start - out[-1][1] <= 1:
            out[-1] = (out[-1][0], end, team)
        else:
            out.append((start, end, team))
    return out


def arrays_and_labels(match_id, labels="phase", hide_dead=True):
    """(metrica.load_arrays-style dict at 25 Hz, H/A/D codes from the phases of play; X where
    nobody is tracked). labels: "phase" or "control" (control_spells). hide_dead: hide the ball
    while it is dead, as below; False keeps SkillCorner's (extrapolated) dead-ball positions, for
    degrade(dead_views=...)."""
    z = load_source(match_id)
    spells = read_phases(match_id)
    if labels == "control":
        spells = control_spells(spells, read_touches(match_id))
    codes10 = phase_labels(z["frame"], spells, int(z["home_id"]))
    parts = {"xy": [], "ball": [], "period": [], "labels": []}
    for p in np.unique(z["period"]):
        rows = np.flatnonzero(z["period"] == p)
        # a period's frames are consecutive; a hole would be resampled as if it were not there
        assert (np.diff(z["frame"][rows]) == 1).all(), "frame gap in period {}".format(p)
        n_out = int((len(rows) - 1) * FPS / SOURCE_FPS) + 1
        parts["xy"].append(resample(z["xy"][rows].astype(np.float64), n_out))
        parts["ball"].append(resample(z["ball"][rows].astype(np.float64), n_out))
        parts["period"].append(np.full(n_out, p, np.int8))
        parts["labels"].append(nearest(codes10[rows], n_out))
    xy = np.concatenate(parts["xy"])
    ball = np.concatenate(parts["ball"])
    period = np.concatenate(parts["period"])
    labels = np.concatenate(parts["labels"]).astype(np.int8)

    speed = np.full(xy.shape[:2], np.nan, np.float32)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        v = np.gradient(xy[rows], axis=0) * FPS
        speed[rows] = np.hypot(v[..., 0], v[..., 1])
    # As for DFL: BAR-ATH and Metrica mostly lose the ball while it is dead. Keep it only for
    # the first 0.5 s after a stop and the last 1 s before the restart.
    if hide_dead:
        dead = labels == 2
        edges = np.diff(np.r_[0, dead.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ball[a + 12:max(a + 12, b - 25)] = np.nan
    # Broadcast replays leave stretches with nobody tracked (~30 min a match, mostly while the
    # ball is dead). BAR-ATH is tracked throughout, so these frames are not scored (X): the
    # model must not learn "no players = dead ball".
    labels[np.isnan(xy[..., 0]).all(axis=1)] = UNTRACKED
    arrays = {"xy": xy, "speed": speed, "ball": ball, "team_of_slot": z["team"].astype(np.int8),
              "team_names": z["names"], "period": period}
    return arrays, labels


def input_file(match_id, seed, labels="phase", touch=False, view="edges"):
    """Basename of the cached sequence-model inputs (training CACHE_DIR), as training "aux":
    sc_* (phase labels) or scc_* (control labels); scv_* / sccv_* with the ATD's dead-ball view;
    *_t with the touch inputs."""
    from ..seq.training import input_name
    kind = LABEL_KINDS[labels] + ("v" if view == "atd" else "")
    return input_name("{}_{}_s{}".format(kind, match_id, seed), touch)


def build_inputs(match_id, seed, missing_counts=None, labels="phase", touch=False, view="edges",
                 dead_views=None):
    """Degrade one match like the external games and cache network inputs + labels. view: as
    dfl.build_inputs ("edges", or "atd" for the competition ATD's dead-ball view)."""
    from . import degrade, metrica
    from ..validation import harness
    from ..seq import network
    from ..seq.training import CACHE_DIR as INPUT_DIR

    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, input_file(match_id, seed, labels, touch, view))
    if os.path.exists(path):
        return path
    if missing_counts is None:
        missing_counts = harness._competition_missing_counts()
    if view == "atd" and dead_views is None:
        from .dfl import _competition_dead_views
        dead_views = _competition_dead_views()
    arrays, codes = arrays_and_labels(match_id, labels, hide_dead=view != "atd")
    game = GAME_OFFSET + match_ids().index(match_id)
    degraded = degrade.degrade(arrays, missing_counts, {"seed": seed}, game=game,
                               live=codes < 2, dead_views=dead_views if view == "atd" else None)
    ctx = metrica.make_context(game, degraded)
    graded = ctx.half_of_frame > 0
    labels = np.where(graded, codes, 3).astype(np.int8)
    X, swap_sign = network.inputs(ctx, touch=touch)
    np.savez(path, X=X, swap_sign=swap_sign, labels=labels, graded=graded,
             halves=np.array(ctx.halves))
    return path
