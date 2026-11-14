# Era check (judge fixtures; see docs/data_collection.md, "Replay eras"; beatmap 1262832, score 7613193930):
# - The API served a stable-era score (legacy_score_id 5083462447) in the lazer shape: a list of
#   mods including CL (lazer's "classic" marker), and great/ok/meh/miss statistics. Other calls can
#   still return the legacy shape (one Mod bitfield, count_300/100/50/miss); handle both.
# - The .osr header's legacy mod bitfield (88 = HD 8 + HR 16 + DT 64) agrees with the API mods.
#   CL has no legacy bit, so drop it before comparing.
# - Frame stream starts with osu!'s usual header frames: time_delta 0, then -1 at (256, -500),
#   then a large negative skip (-1243 here). The resampler must handle negative deltas.
# - A lazer-set score (beatmap 1262832, score 3133639837, no legacy_score_id) starts differently:
#   no header frames, first frame time_delta -1484 at an off-playfield position (-171, -56).
#   Don't assume a fixed header; drop leading frames until time is non-decreasing.
"""Load .osr replays as absolute-time arrays for the judge, resampler and dataset builder."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osrparse import Replay

RNG_SEED_DELTA = -12345        # stable writes a fake last frame with this delta to store the RNG seed
# Key bits in a frame: M1 = 1, M2 = 2, K1 = 4, K2 = 8, smoke = 16. Pressing K1 records 4 + 1 = 5
# and K2 records 8 + 2 = 10 (osu! wiki, osr file format), and lazer replays never set K1/K2 at all.
# So M1 / M2 alone say whether "left" / "right" is held, for mouse buttons and keys alike. Checked on
# all 1,321 v1 replays (11.7M frames): K1 never without M1, K2 never without M2; K bits appear in
# 504 of 507 stable replays and in none of 814 lazer replays.
M1, M2 = 1, 2


@dataclass
class ReplayFrames:
    t: np.ndarray              # absolute time in ms (beatmap clock), non-decreasing
    x: np.ndarray              # cursor x, osu! pixels
    y: np.ndarray              # cursor y, osu! pixels
    keys: np.ndarray           # osrparse Key bits per frame
    mods: int                  # legacy mod bitfield from the .osr header
    beatmap_hash: str = ""      # MD5 of the .osu file the replay was played on


def load_replay(path: str | Path) -> ReplayFrames:
    r = Replay.from_path(str(path))
    if r.mode.value != 0:
        raise ValueError("only osu!standard replays are supported")
    events = [e for e in r.replay_data if e.time_delta != RNG_SEED_DELTA]
    t = np.cumsum([e.time_delta for e in events])
    x = np.array([e.x for e in events], dtype=float)
    y = np.array([e.y for e in events], dtype=float)
    keys = np.array([int(e.keys) for e in events], dtype=int)

    # Header frames make time jump backwards at the start (see the notes above), and some replays
    # jump back again a few frames later while the cursor is still idle (seen: -61 ms at frame 7 of
    # score 6035774631). Keep everything after the last backwards jump that comes before the first
    # key press; after that, drop any frame that would go back in time.
    pressed = np.flatnonzero(keys & (M1 | M2))
    first_press = pressed[0] if len(pressed) else len(t)
    back = np.flatnonzero(np.diff(t) < 0)
    back = back[back < first_press]
    start = back[-1] + 1 if len(back) else 0
    t, x, y, keys = t[start:], x[start:], y[start:], keys[start:]
    keep = t >= np.maximum.accumulate(t)
    return ReplayFrames(t[keep], x[keep], y[keep], keys[keep], int(r.mods), r.beatmap_hash)


def presses(frames: ReplayFrames) -> np.ndarray:
    """Indices of frames where M1 or M2 goes from up to down (one click per frame at most)."""
    held = frames.keys & (M1 | M2)
    before = np.concatenate([[0], held[:-1]])
    return np.flatnonzero(held & ~before)
