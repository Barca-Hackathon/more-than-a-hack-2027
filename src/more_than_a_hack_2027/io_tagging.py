"""Smart Tagging as video-frame intervals with the team mapped to HOME / AWAY / None.

The tagging is automatic (not ground truth) and coarse in time: set-piece windows are 35 s long
(5 s for kickoffs) and the actual restart can sit anywhere from shortly before the window starts
to ~10 s into it. Only a handful of free kicks are tagged, and every throw-in has Team = N/A.
Treat a tag as a weak prior, never as a label (DATA.md).
"""

from dataclasses import dataclass, field

from ._paths import ensure_repo_on_path
from .context import AWAY, HOME

ensure_repo_on_path()
import load_smart_tagging  # noqa: E402  starter/load_smart_tagging.py

SET_PIECES = "SET PIECES"
GOALS = "GOALS"


@dataclass(frozen=True)
class Tag:
    code: str                 # upper case, e.g. "SET PIECES"
    start: int                # first video frame, inclusive
    stop: int                 # first video frame after the interval
    team: object              # HOME, AWAY or None (N/A or unrecognised)
    kind: str                 # Type tag in lower case ("throw-in", ...) or "" if none
    tags: dict = field(default_factory=dict, compare=False, repr=False)


def team_code(text, team_names):
    """Map a tag's Team text to HOME / AWAY by comparing with the tracking's team names."""
    if not text:
        return None
    value = text.strip().lower()
    home, away = (name.strip().lower() for name in team_names)
    if value == home:
        return HOME
    if value == away:
        return AWAY
    return None


def load_tags(ctx, data_dir=None):
    """All tags, sorted by start frame."""
    out = []
    for event in load_smart_tagging.load(ctx.game, data_dir):
        tags = event["tags"]
        out.append(Tag(
            code=(event["code"] or "").strip().upper(),
            start=int(event["start_frame"]),
            stop=int(event["end_frame"]),
            team=team_code(tags.get("Team"), ctx.team_names),
            kind=(tags.get("Type") or "").strip().lower(),
            tags=dict(tags),
        ))
    out.sort(key=lambda tag: (tag.start, tag.code))
    return out


def by_code(tags, code):
    code = code.upper()
    return [tag for tag in tags if tag.code == code]
