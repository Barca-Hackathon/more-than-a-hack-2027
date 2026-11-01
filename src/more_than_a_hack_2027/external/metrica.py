"""Metrica Sports' public sample games as Contexts with event-derived H/A/D labels.

These games are where hyperparameters are chosen and variants are compared (docs/PLAN.md,
"External validation"). They are NOT the provided match, and the labels below come from
Metrica analysts' event data rather than the corrected-tracking ownership behind the competition
ground truth, so use the scores for relative decisions only.

Source: https://github.com/metrica-sports/sample-data. Their README asks that public use
acknowledges the source; the tech doc must do so. The raw files are downloaded once into
data/other_matches/metrica/ and kept there (never deleted, never committed); parsed arrays are
cached in .cache/external/.

    Games 1, 2   CSV tracking (one file per team, x/y only) + RawEventsData.csv
    Game 3       EPTS tracking (frame:players:ball, x/y only) + metadata XML + events JSON

Everything is 25 fps on a 105 x 68 m pitch; coordinates are normalised with (0, 0) at a corner,
which maps straight to the competition's metres-from-a-corner convention. video_frame is
frame - 1, as on the competition data.

Labels, from the event stream:
  * stop events end play: BALL OUT and shots that go out or in (end frame), fouls (FAULT RECEIVED,
    CHALLENGE ...-FAULT-..., at the start frame), offside and hand ball (end frame), END HALF.
  * the ball stays dead until the next tagged restart: a SET PIECE (not one marked RETAKEN) or a
    goal kick. Between a whistle and the restart the stream often has 1-3 "after the whistle"
    actions within ~3 s at walking pace (a player kicks it away, a shot after the offside flag),
    so those do not restart play. A few restarts are not tagged at all, which shows as a run of
    ordinary play before the next tag: at least UNTAGGED_MIN_EVENTS play events spanning
    UNTAGGED_MIN_SPAN_S or more. Then play resumed at the first of them. RECOVERY, CHALLENGE and
    CARD events never restart play (a keeper collecting a ball that is out, a booking).
  * in play, possession changes when the other team GAINS control (CHALLENGE.md): at its PASS,
    CARRY, SET PIECE, SHOT or RECOVERY. CHALLENGE never changes it (a duel is not control), and
    neither do the two LABEL_RULES:
      - "loss": a BALL LOST ends a team's control and never starts it. Metrica tags both sides of
        a tackle or an interception, and the loser's BALL LOST often lands 1-35 frames AFTER the
        winner's RECOVERY; counting it handed the ball back to the team that had just lost it.
      - "one_touch": a one-touch RECOVERY, where the same team's next action is a loss (BALL LOST, BALL OUT) within
        ONE_TOUCH_FRAMES, i.e. a header or clearance straight back to the opponent or out ("a
        simple touch of the ball without control", "an aerial duel won without subsequent
        control"). In the three games 232 of 233 such losses are tagged on the very same frame
        as the recovery; the next shortest gap is 3 frames. The tracking agrees: after such a
        recovery the ball stays within 1.5 m of that team for a median 6 of the next 25
        frames, against 16-25 when the team keeps it. A CARRY of ONE_TOUCH_FRAMES or less is
        not a carry (game 3's export adds a 1-frame CARRY to each of these).
    "loss" follows from the definition alone. "one_touch" is the competition's own list of what
    does not count, but where the organisers' tracking-derived ground truth draws that line is
    not knowable from here, so tune.py checks every decoder choice under rules=("loss",) too.
    rules=() is the older rule (any control event of the other team switches); both rules were
    added on Nov 1 and together cut the spells per game from ~375 to ~245.
  * each period is graded from its first tracking frame to its last, dead until the kickoff.
    The last frame of the first period is left ungraded (X), because the periods are contiguous
    in the frame numbering and the metric must see two separate runs, as it does on BAR-ATH.
"""

import io
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._paths import CACHE_DIR, OTHER_MATCHES_DIR, ensure_repo_on_path
from ..data.context import AWAY, HOME, Context

ensure_repo_on_path()
import possession_metrics as pm  # noqa: E402

GAMES = (1, 2, 3)
FPS = 25
PITCH = (105.0, 68.0)
SOURCE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ("Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"),
    2: ("Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"),
    3: ("Sample_Game_3/Sample_Game_3_events.json",
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt"),
}
RAW_DIR = os.path.join(OTHER_MATCHES_DIR, "metrica")    # raw downloads, kept for good
CACHE_VERSION = 1
SPEED_HALF_WINDOW = 2          # central difference over +-2 frames for player speed

# Event vocabulary (Metrica's).
CONTROL_TYPES = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST"}
LOSS_TYPES = {"BALL LOST", "BALL OUT"}
OWN_ACTIONS = (CONTROL_TYPES - {"RECOVERY"}) | LOSS_TYPES   # what a player does with the ball
ONE_TOUCH_FRAMES = 2
LABEL_RULES = ("loss", "one_touch")      # what does NOT change possession; see the docstring
RESTART_TYPES = {"SET PIECE", "PASS", "CARRY", "SHOT", "BALL LOST"}   # can resume play untagged
UNTAGGED_MIN_EVENTS = 3
UNTAGGED_MIN_SPAN_S = 5.0


@dataclass(frozen=True)
class Event:
    index: int
    team: int            # HOME / AWAY
    type: str
    subtype: str
    period: int
    start: int           # video frame (frame - 1)
    end: int             # video frame; == start when the source has no end frame


def game_name(game):
    return "METRICA-{}".format(game)


def download(games=GAMES, raw_dir=RAW_DIR, verbose=True):
    """Fetch the sample files that are not on disk yet. Idempotent."""
    for game in games:
        for rel in FILES[game]:
            path = os.path.join(raw_dir, rel)
            if os.path.isfile(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = "{}/{}".format(SOURCE_URL, rel)
            if verbose:
                print("downloading {}".format(url))
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# Tracking

def _read_csv_team(path):
    """(frames, period, xy (n, P, 2) normalised, ball (n, 2) normalised) from one team file."""
    names = pd.read_csv(path, skiprows=2, nrows=0).columns
    data = pd.read_csv(path, skiprows=3, header=None, engine="c").to_numpy(dtype=np.float64)
    period, frames = data[:, 0].astype(int), data[:, 1].astype(int)
    player_cols = [i for i, name in enumerate(names) if str(name).startswith("Player")]
    xy = np.stack([data[:, [i, i + 1]] for i in player_cols], axis=1)
    ball_col = list(names).index("Ball")
    ball = data[:, [ball_col, ball_col + 1]]
    return frames, period, xy, ball


def _load_csv_game(game, raw_dir):
    _, home_path, away_path = (os.path.join(raw_dir, rel) for rel in FILES[game])
    frames, period, home_xy, ball = _read_csv_team(home_path)
    frames_a, period_a, away_xy, ball_a = _read_csv_team(away_path)
    if not (np.array_equal(frames, frames_a) and np.array_equal(period, period_a)):
        raise ValueError("game {}: home and away tracking files are not aligned".format(game))
    xy = np.concatenate([home_xy, away_xy], axis=1)
    team_of_slot = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], dtype=np.int8)
    return frames, period, xy, ball, team_of_slot, ("Home", "Away")


def _load_epts_game(game, raw_dir):
    _, meta_path, track_path = (os.path.join(raw_dir, rel) for rel in FILES[game])
    root = ET.parse(meta_path).getroot()
    teams = [(t.attrib["id"], t.findtext("Name")) for t in root.iter("Team")]
    team_index = {team_id: k for k, (team_id, _) in enumerate(teams)}     # first listed = home
    player_team = {p.attrib["id"]: team_index[p.attrib["teamId"]] for p in root.iter("Player")}
    channel_player = {c.attrib["id"]: c.attrib["playerId"] for c in root.iter("PlayerChannel")}

    # The player order changes with every substitution; give every player one fixed slot.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.attrib["playerChannelId"] for r in spec.iter("PlayerChannelRef")]
        specs.append((int(spec.attrib["startFrame"]), int(spec.attrib["endFrame"]),
                      [channel_player[ref] for ref in refs[::2]]))
    players = sorted({p for _, _, ids in specs for p in ids},
                     key=lambda p: (player_team[p], int(p.lstrip("P"))))
    slot_of = {p: k for k, p in enumerate(players)}
    team_of_slot = np.array([player_team[p] for p in players], dtype=np.int8)

    with open(track_path) as handle:
        text = handle.read()
    table = pd.read_csv(io.StringIO(text.replace(":", ",").replace(";", ",")),
                        header=None, engine="c", na_values=["NaN"]).to_numpy(dtype=np.float64)
    frames = table[:, 0].astype(int)
    n = len(frames)
    groups = table[:, 1:-2].reshape(n, -1, 2)
    ball = table[:, -2:]

    xy = np.full((n, len(players), 2), np.nan)
    row_of_frame = frames - frames[0]
    for first, last, ids in specs:
        rows = row_of_frame[(frames >= first) & (frames <= last)]
        cols = [slot_of[p] for p in ids]
        xy[rows[:, None], np.array(cols)[None, :]] = groups[rows][:, :len(ids)]

    period = np.ones(n, dtype=int)
    params = {pp.findtext("Name"): pp.findtext("Value") for pp in root.iter("ProviderParameter")}
    period[frames >= int(params["second_half_start"])] = 2
    return frames, period, xy, ball, team_of_slot, tuple(name for _, name in teams)


def player_speed(xy, period, fps=FPS, k=SPEED_HALF_WINDOW):
    """(n, S) float32 m/s by a central difference over +-k frames, never across periods."""
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        a, b = rows[0], rows[-1] + 1
        if b - a <= 2 * k:
            continue
        delta = xy[a + 2 * k:b] - xy[a:b - 2 * k]
        speed[a + k:b - k] = np.hypot(delta[..., 0], delta[..., 1]) * fps / (2 * k)
    return speed


def _home_attacks_positive_x(xy, team_of_slot, period, fps=FPS):
    """{period: bool} from where the teams line up over the first 5 s of each period."""
    out = {}
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)[:5 * fps]
        x = xy[rows, :, 0]
        home = np.nanmean(x[:, team_of_slot == HOME])
        away = np.nanmean(x[:, team_of_slot == AWAY])
        out[int(p)] = bool(home < away)
    return out


def _cache_path(game):
    return os.path.join(CACHE_DIR, "external", "{}_v{}.npz".format(game_name(game), CACHE_VERSION))


def _build_arrays(game, raw_dir):
    if game == 3:
        frames, period, xy, ball, team_of_slot, names = _load_epts_game(game, raw_dir)
    else:
        frames, period, xy, ball, team_of_slot, names = _load_csv_game(game, raw_dir)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("game {}: frame ids are not 1..n".format(game))
    scale = np.array(PITCH)
    xy_m = xy * scale
    ball_m = ball * scale
    return {
        "xy": xy_m, "ball": ball_m, "period": period.astype(np.int8),
        "speed": player_speed(xy_m, period),
        "team_of_slot": team_of_slot, "team_names": np.array(names),
    }


def load_arrays(game, raw_dir=RAW_DIR, force=False):
    path = _cache_path(game)
    if not force and os.path.isfile(path):
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    download([game], raw_dir, verbose=True)
    arrays = _build_arrays(game, raw_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def make_context(game, arrays, halves=None):
    """A Context over one external game. `arrays` may be degraded (see degrade.py)."""
    period = arrays["period"]
    n = len(period)
    if halves is None:
        halves = graded_halves(period)
    half_of_frame = np.zeros(n, dtype=np.int8)
    for half, first, last in halves:
        half_of_frame[first:last + 1] = half
    return Context(
        game=game_name(game), n_frames=n, fps=FPS, pitch=PITCH,
        xy=arrays["xy"], speed=arrays["speed"], ball=arrays["ball"],
        team_of_slot=arrays["team_of_slot"], team_names=tuple(str(s) for s in arrays["team_names"]),
        halves=tuple(halves),
        home_attacks_positive_x=_home_attacks_positive_x(arrays["xy"], arrays["team_of_slot"],
                                                         period),
        half_of_frame=half_of_frame)


def graded_halves(period):
    """((half, first, last), ...) inclusive video frames; one ungraded frame between periods."""
    halves = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        halves.append((int(p), int(rows[0]), int(rows[-1])))
    return tuple((h, a, b - 1) if k < len(halves) - 1 else (h, a, b)
                 for k, (h, a, b) in enumerate(halves))


# --------------------------------------------------------------------------------------------
# Events and labels

def _subtype_text(subtypes):
    if subtypes is None:
        return ""
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return "-".join(s["name"] for s in subtypes)


def load_events(game, raw_dir=RAW_DIR, team_names=None):
    """[Event] in source order, frames converted to video frames."""
    path = os.path.join(raw_dir, FILES[game][0])
    events = []
    if game == 3:
        with open(path) as handle:
            rows = json.load(handle)["data"]
        names = list(team_names) if team_names is not None else ["Team A", "Team B"]
        for k, row in enumerate(rows):
            start = int(row["start"]["frame"])
            end = int(row["end"]["frame"] or start) if row.get("end") else start
            events.append(Event(k, names.index(row["team"]["name"]), row["type"]["name"],
                                _subtype_text(row.get("subtypes")), int(row["period"]),
                                start - 1, max(end, start) - 1))
    else:
        table = pd.read_csv(path, keep_default_na=False)
        for k, row in enumerate(table.itertuples(index=False)):
            start, end = int(row[4]), int(row[6])
            team = {"Home": HOME, "Away": AWAY}[row.Team]
            events.append(Event(k, team, row.Type, row.Subtype, int(row.Period),
                                start - 1, max(end, start) - 1))
    return events


def stop_frame(event):
    """Video frame at which `event` stops play, or None if it does not."""
    kind, sub = event.type, event.subtype
    if kind == "BALL OUT":
        return event.end
    if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return event.end
    if kind == "FAULT RECEIVED":
        return event.start
    if kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return event.start
    if kind in ("BALL LOST", "PASS") and ("OFFSIDE" in sub or "HAND BALL" in sub):
        return event.end
    if "END HALF" in sub:
        return event.end
    return None


def is_play(event):
    """An event that shows the ball in play (if enough of them follow each other)."""
    return event.type in RESTART_TYPES and "END HALF" not in event.subtype


def is_tagged_restart(event):
    if event.type == "SET PIECE":
        return "RETAKEN" not in event.subtype
    return "GOAL KICK" in event.subtype and event.type in ("PASS", "BALL LOST", "SHOT")


def _restart_after(events, frame, fps):
    """The event that restarts play after a stop at `frame`, or None for the rest of the half."""
    later = [e for e in events if e.start >= frame]
    tagged = next((k for k, e in enumerate(later) if is_tagged_restart(e)), len(later))
    play = [e for e in later[:tagged] if e.start > frame and is_play(e)]
    for k, e in enumerate(play):
        rest = play[k:]
        if (len(rest) >= UNTAGGED_MIN_EVENTS
                and rest[-1].start - e.start >= UNTAGGED_MIN_SPAN_S * fps):
            return e
    return later[tagged] if tagged < len(later) else None


def _is_trivial_carry(event):
    """A CARRY no longer than a touch. Game 3's export pairs every one-touch recovery with a
    1-frame CARRY (88 of 88; other carries last a median 25 frames): it is not a carry."""
    return event.type == "CARRY" and event.end - event.start <= ONE_TOUCH_FRAMES


def is_one_touch(events, k):
    """events[k] is a RECOVERY whose team's next own action is a loss within ONE_TOUCH_FRAMES.

    events: one period, sorted by (start, index)."""
    e = events[k]
    if e.type != "RECOVERY":
        return False
    nxt = next((x for x in events[k + 1:] if x.team == e.team and x.type in OWN_ACTIONS
                and not _is_trivial_carry(x)), None)
    return (nxt is not None and nxt.type in LOSS_TYPES
            and nxt.start - e.start <= ONE_TOUCH_FRAMES)


def gains_control(events, k, rules=LABEL_RULES):
    """Does events[k] give its team the ball in open play? (module docstring)"""
    e = events[k]
    unknown = set(rules) - set(LABEL_RULES)
    if unknown:
        raise ValueError("unknown label rules {}".format(sorted(unknown)))
    if e.type not in CONTROL_TYPES:
        return False
    if "loss" in rules and e.type == "BALL LOST":
        return False
    if "one_touch" in rules and (_is_trivial_carry(e) or is_one_touch(events, k)):
        return False
    return True


def labels_from_events(events, n_frames, halves, fps=FPS, rules=LABEL_RULES):
    """H/A/D/X codes (possession_metrics' H, A, D, X) for every video frame."""
    labels = np.full(n_frames, pm.X, dtype=np.int8)
    for half, first, last in halves:
        mine = sorted((e for e in events if e.period == half), key=lambda e: (e.start, e.index))
        stops = sorted(f for f in (stop_frame(e) for e in mine) if f is not None)
        labels[first:last + 1] = pm.D
        dead_from = first
        while True:
            restart = _restart_after(mine, dead_from, fps)
            if restart is None or restart.start > last:
                break
            begin = max(restart.start, first)
            end = next((f for f in stops if f > begin), last + 1)
            end = min(end, last + 1)
            owner = restart.team
            cursor = begin
            for k, e in enumerate(mine):
                if (begin < e.start < end and e.team != owner
                        and gains_control(mine, k, rules)):
                    labels[cursor:e.start] = owner
                    owner, cursor = e.team, e.start
            labels[cursor:end] = owner
            if end > last:
                break
            dead_from = end
    return labels


def possession_stats(labels, halves, fps=FPS):
    """GT-side sanity numbers for one label array."""
    runs = [(first, last + 1) for _, first, last in halves]
    segs = pm.segments(labels, runs)
    dead = [(b - a) / fps for a, b, c in segs if c == pm.D]
    spells = [(b - a) / fps for a, b, c in segs if c in (pm.H, pm.A)]
    graded = labels != pm.X
    return {
        "dead_share": float(np.mean(labels[graded] == pm.D)),
        "home_share_alive": float(np.mean(labels[graded & (labels != pm.D)] == pm.H)),
        "n_spells": len(spells),
        "spell_median_s": float(np.median(spells)) if spells else 0.0,
        "n_stoppages": len(dead),
        "stoppage_median_s": float(np.median(dead)) if dead else 0.0,
        "stoppage_max_s": float(np.max(dead)) if dead else 0.0,
    }


@dataclass(frozen=True)
class ExternalGame:
    game: int
    ctx: Context
    labels: np.ndarray      # possession_metrics codes
    events: list


def load_game(game, raw_dir=RAW_DIR, arrays=None, rules=LABEL_RULES):
    """Context (clean tracking unless `arrays` is given) + event-derived labels."""
    clean = load_arrays(game, raw_dir)
    ctx = make_context(game, arrays if arrays is not None else clean)
    events = load_events(game, raw_dir, ctx.team_names)
    labels = labels_from_events(events, ctx.n_frames, ctx.halves, rules=rules)
    return ExternalGame(game, ctx, labels, events)
