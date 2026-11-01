"""Parse the raw ATD once, cache it as .npy arrays, and load it back memory-mapped.

Raw line format (FIFA/EPTS, see the *_atd.xml DataFormatSpecification):

    <frame_id>:<x,y,s>;<x,y,s>;...(one group per track slot)...:<x,y,s of the ball>

Facts checked on the delivered file and relied on here:
  * frame_id runs 1..199319 with no gaps, so video_frame = frame_id - 1 (starter/load_atd.py).
  * Player speed is in m/s even though the XML labels the channel kph (it matches finite
    differences of the positions to within 1-2%).
  * The ball speed channel is NaN on ~91% of ball frames; it is dropped, compute velocity instead.
  * The ball is reported outside [0, 1] on ~5k frames -- that is signal (ball out), so it is kept.

Positions are converted to metres with the same float64 arithmetic as starter/baseline.py
(normalised * pitch length/width), which is what makes the baseline port bit-exact.
"""

import io
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ._paths import CACHE_DIR, ensure_repo_on_path

ensure_repo_on_path()
import games  # noqa: E402

# Bump when the parse or the cached layout changes, so stale caches are rebuilt automatically.
CACHE_VERSION = 1
FIELDS_PER_GROUP = 3


def cache_dir(source="tactical", game=games.GAME):
    return os.path.join(CACHE_DIR, "atd_{}_{}".format(game, source))


def parse_meta(meta_path):
    """Team names (home first), team of every track slot (0 home, 1 away) and attack directions.

    The first <Team> is home, the FIFA/EPTS convention starter/baseline.py also relies on.
    attack_direction_first_half is 'left_to_right' or 'right_to_left' in the normalised x axis.
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")

    directions = []
    for team in teams:
        value = None
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                value = param.findtext("Value")
        directions.append(value)

    slots = [player.get("teamId") for player in root.iter("Player")]
    team_of_slot = np.array([0 if team_id == home_id else 1 for team_id in slots], dtype=np.int8)
    return {
        "team_names": [team.findtext("Name") for team in teams],
        "team_of_slot": team_of_slot,
        "attack_direction_first_half": directions,
    }


def parse_raw(raw_path, n_slots, pitch):
    """(frame_id, xy[n, S, 2] metres float64, speed[n, S] float32, ball[n, 2] metres float64)."""
    with open(raw_path, "rb") as handle:
        text = handle.read().replace(b":", b",").replace(b";", b",")
    n_cols = 1 + FIELDS_PER_GROUP * (n_slots + 1)
    table = pd.read_csv(io.BytesIO(text), header=None, dtype=np.float64, engine="c")
    del text
    if table.shape[1] != n_cols:
        raise ValueError("{}: expected {} fields per line, found {}".format(
            raw_path, n_cols, table.shape[1]))

    values = table.to_numpy()
    del table
    frame_id = values[:, 0].astype(np.int64)
    groups = values[:, 1:].reshape(len(values), n_slots + 1, FIELDS_PER_GROUP)

    length, width = pitch
    scale = np.array([length, width], dtype=np.float64)
    xy = groups[:, :n_slots, :2] * scale
    speed = groups[:, :n_slots, 2].astype(np.float32)
    ball = groups[:, n_slots, :2] * scale
    return frame_id, xy, speed, ball


def _source_signature(paths):
    return {os.path.basename(p): [os.path.getsize(p), os.stat(p).st_mtime_ns] for p in paths}


def build_cache(source="tactical", game=games.GAME, data_dir=None, force=False, verbose=True):
    """Parse the ATD into cache_dir(). Returns the directory. Skips work if the cache is fresh."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    folder = cache_dir(source, game)
    meta_file = os.path.join(folder, "meta.json")
    signature = _source_signature([meta_path, raw_path])

    if not force and os.path.isfile(meta_file):
        with open(meta_file) as handle:
            cached = json.load(handle)
        if cached.get("version") == CACHE_VERSION and cached.get("sources") == signature:
            return folder

    if verbose:
        print("parsing {} (one-off, ~10-30 s) ...".format(os.path.basename(raw_path)))
    meta = parse_meta(meta_path)
    pitch = games.GAMES[game]["pitch"]
    frame_id, xy, speed, ball = parse_raw(raw_path, len(meta["team_of_slot"]), pitch)

    n_expected = games.source_frames(source, game)
    if not np.array_equal(frame_id, np.arange(1, n_expected + 1)):
        raise ValueError("{}: frame ids are not 1..{} -- alignment assumption broken".format(
            raw_path, n_expected))

    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "xy.npy"), xy)
    np.save(os.path.join(folder, "speed.npy"), speed)
    np.save(os.path.join(folder, "ball.npy"), ball)
    np.save(os.path.join(folder, "team_of_slot.npy"), meta["team_of_slot"])
    with open(meta_file, "w") as handle:
        json.dump({
            "version": CACHE_VERSION,
            "sources": signature,
            "game": game,
            "source": source,
            "n_frames": int(n_expected),
            "pitch": list(pitch),
            "team_names": meta["team_names"],
            "attack_direction_first_half": meta["attack_direction_first_half"],
        }, handle, indent=2)
    if verbose:
        print("  cached to {}".format(os.path.relpath(folder)))
    return folder


def load_cached(source="tactical", game=games.GAME, data_dir=None, mmap=True):
    """dict of arrays + metadata, building the cache first if needed. Arrays are read-only."""
    folder = build_cache(source, game, data_dir)
    mode = "r" if mmap else None
    with open(os.path.join(folder, "meta.json")) as handle:
        meta = json.load(handle)
    meta["xy"] = np.load(os.path.join(folder, "xy.npy"), mmap_mode=mode)
    meta["speed"] = np.load(os.path.join(folder, "speed.npy"), mmap_mode=mode)
    meta["ball"] = np.load(os.path.join(folder, "ball.npy"), mmap_mode=mode)
    meta["team_of_slot"] = np.load(os.path.join(folder, "team_of_slot.npy"))
    return meta
