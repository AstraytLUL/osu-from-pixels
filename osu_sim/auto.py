"""Auto: a scripted player that reads the beatmap directly. The oracle upper bound and the DAgger expert.

It clicks every object exactly on time, alternating keys, holds through sliders with the cursor on
the ball, and spins spinners just under the speed cap the judge enforces. Between objects the cursor
moves in a straight line at constant speed. Its output is a ReplayFrames, so the judge scores it
exactly like a human replay.
"""
import bisect
import math

import numpy as np

from data.replay import M1, M2, ReplayFrames
from osu_sim.judge import SPINNER_MAX_RAD_PER_MS
from osu_sim.mods import rate as mod_rate
from osu_sim.playable import SPINNER_CENTRE, Playable, PlayObject

CIRCLE_HOLD = 40               # ms a circle's key is held (shortened if the next object is closer)
SPINNER_RADIUS = 50            # osu! pixels from the centre
SPIN_FRACTION = 0.9            # spin at 90% of the counted-rotation cap


class AutoCursor:
    """Auto's cursor position at any map time."""

    def __init__(self, pl: Playable):
        self.objs = pl.objects
        self.starts = [o.time for o in self.objs]
        self.omega = SPIN_FRACTION * SPINNER_MAX_RAD_PER_MS / mod_rate(pl.mods)   # rad per map ms

    def _in_object(self, o: PlayObject, t: float) -> tuple[float, float]:
        if o.kind == "slider":
            return o.ball_position(t)
        if o.kind == "spinner":
            a = self.omega * (t - o.time)
            return SPINNER_CENTRE[0] + SPINNER_RADIUS * math.cos(a), SPINNER_CENTRE[1] + SPINNER_RADIUS * math.sin(a)
        return o.x, o.y

    def __call__(self, t: float) -> tuple[float, float]:
        k = bisect.bisect_right(self.starts, t) - 1
        if k < 0:
            return self._in_object(self.objs[0], self.objs[0].time)
        o = self.objs[k]
        if t <= o.end_time or k + 1 == len(self.objs):
            return self._in_object(o, min(t, o.end_time))
        # between objects: straight line from where this one ended to where the next one starts
        nxt = self.objs[k + 1]
        x0, y0 = self._in_object(o, o.end_time)
        x1, y1 = self._in_object(nxt, nxt.time)
        w = (t - o.end_time) / (nxt.time - o.end_time)
        return x0 + (x1 - x0) * w, y0 + (y1 - y0) * w


def key_schedule(pl: Playable) -> list[tuple[float, float, int]]:
    """(press time, release time, key) per object, alternating keys; a key is never pressed while held."""
    out = []
    free_at = {M1: -math.inf, M2: -math.inf}
    last = M2
    objs = pl.objects
    for i, o in enumerate(objs):
        key = M1 if last == M2 else M2
        if free_at[key] >= o.time:         # still held (e.g. through a slider): use the other key
            key = M2 if key == M1 else M1
        if o.kind == "circle":
            nxt = objs[i + 1].time if i + 1 < len(objs) else math.inf
            release = o.time + max(1.0, min(CIRCLE_HOLD, (nxt - o.time) / 2))
        else:
            release = o.end_time + 1       # hold just past the end of a slider / spinner
        out.append((float(o.time), release, key))
        free_at[key] = release
        last = key
    return out


def auto_replay(pl: Playable, fps: float = 60.0) -> ReplayFrames:
    """Auto's play as replay frames: a regular grid at `fps` in real time, plus a frame at every key
    event and slider checkpoint so that nothing falls between frames."""
    keys = key_schedule(pl)
    step = 1000.0 / fps * mod_rate(pl.mods)           # map ms between frames
    first, last = pl.objects[0].time - 1000, pl.objects[-1].end_time + 500
    times = set(np.arange(first, last, step).round().tolist())
    for press, release, _ in keys:
        times.update((press, release))
    for o in pl.objects:
        if o.kind == "slider":
            times.update(round(t) for t, _ in o.timing.ticks + o.timing.repeats)
            times.add(math.ceil(o.timing.legacy_last_tick))
    t = np.array(sorted(times), dtype=float)

    cursor = AutoCursor(pl)
    xy = np.array([cursor(ti) for ti in t])
    k = np.zeros(len(t), dtype=int)
    for press, release, key in keys:
        k[(t >= press) & (t < release)] |= key
    return ReplayFrames(t=t, x=xy[:, 0], y=xy[:, 1], keys=k, mods=pl.mods)
