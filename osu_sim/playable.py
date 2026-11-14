"""A beatmap made ready to play: mods applied, slider paths and timing computed, stacking applied.

The judge, renderer and Auto all read this, so they agree on every position and time.

Sources (osu! lazer): osu.Game.Rulesets.Osu/Objects/OsuHitObject.cs (preempt, fade-in, scale, stack
offset), osu.Game.Rulesets.Osu/Beatmaps/OsuBeatmapProcessor.cs (stacking), osu.Game/Beatmaps/
IBeatmapDifficultyInfo.cs (DifficultyRange).
"""
import math
from dataclasses import dataclass, field

from osu_sim.beatmap import Beatmap, HitObject
from osu_sim.mods import apply_mods
from osu_sim.slider import ControlPoints, SliderPath, SliderTiming, build_path, slider_timing

OBJECT_RADIUS = 64
STACK_DISTANCE = 3
PREEMPT_MIN, PREEMPT_MID, PREEMPT_MAX = 450, 1200, 1800
FOLLOW_AREA = 2.4              # slider follow circle radius, in object radii, while tracking
SPINNER_CENTRE = (256.0, 192.0)


def difficulty_range(d: float, lo_end: float, mid: float, hi_end: float) -> float:
    """Map a 0..10 difficulty value piecewise-linearly: 0 -> lo_end, 5 -> mid, 10 -> hi_end."""
    if d > 5:
        return mid + (hi_end - mid) * (d - 5) / 5
    if d < 5:
        return mid + (mid - lo_end) * (d - 5) / 5
    return mid


def preempt_ms(ar: float) -> int:
    """How long before its time an object appears (ms). AR 9 -> 600, AR 10 -> 450."""
    return int(difficulty_range(ar, PREEMPT_MAX, PREEMPT_MID, PREEMPT_MIN))


def scale_from_cs(cs: float) -> float:
    # LegacyRulesetExtensions.CalculateScaleFromCircleSize with the 1.00041 fudge
    return (1.0 - 0.7 * difficulty_range(cs, -1, 0, 1)) / 2 * 1.00041


@dataclass
class PlayObject:
    index: int                 # position in the beatmap's hit_objects
    kind: str                  # "circle" | "slider" | "spinner"
    time: float                # start time, ms (map time)
    end_time: float            # = time for circles
    x: float                   # stacked position (what the player sees and must click)
    y: float
    stack_height: int = 0
    path: SliderPath | None = None          # slider path relative to the unstacked start
    timing: SliderTiming | None = None
    raw_x: float = 0.0         # position before stacking
    raw_y: float = 0.0

    def stack_offset(self, scale: float) -> float:
        return self.stack_height * scale * -6.4

    def ball_position(self, t: float) -> tuple[float, float]:
        """Slider ball (stacked) position at time t; clamps to the head before and the end after."""
        if self.kind != "slider":
            return self.x, self.y
        assert self.timing is not None and self.path is not None
        duration = self.end_time - self.time
        progress = 0.0 if duration <= 0 else min(max((t - self.time) / duration, 0.0), 1.0)
        return self.position_at_progress(progress)

    def position_at_progress(self, progress: float) -> tuple[float, float]:
        """Stacked position at overall slider progress 0..1 (repeats go back and forth)."""
        assert self.timing is not None and self.path is not None
        spans = self.timing.spans
        p = progress * spans % 1.0
        if progress >= 1.0:
            p = 1.0 if spans % 2 == 1 else 0.0
        elif int(progress * spans) % 2 == 1:
            p = 1 - p
        px, py = self.path.position_at(p)
        dx, dy = self.x - self.raw_x, self.y - self.raw_y          # stack offset
        return px + dx, py + dy

    def path_point(self, path_progress: float) -> tuple[float, float]:
        """Stacked position at fraction `path_progress` of the path (no span folding)."""
        assert self.path is not None
        px, py = self.path.position_at(path_progress)
        return px + self.x - self.raw_x, py + self.y - self.raw_y

    @property
    def end_position(self) -> tuple[float, float]:
        return self.ball_position(self.end_time)


@dataclass
class Playable:
    mods: int
    od: float
    cs: float
    ar: float
    radius: float              # object radius, osu! pixels
    scale: float
    preempt: int               # ms
    fade_in: float             # ms
    objects: list[PlayObject] = field(default_factory=list)
    beatmap: Beatmap | None = None


def make_playable(beatmap: Beatmap, mods: int = 0) -> Playable:
    bm = apply_mods(beatmap, mods)
    d = bm.difficulty
    scale = scale_from_cs(d.cs)
    preempt = preempt_ms(d.ar)
    pl = Playable(mods=mods, od=d.od, cs=d.cs, ar=d.ar, radius=OBJECT_RADIUS * scale, scale=scale,
                  preempt=preempt, fade_in=400 * min(1.0, preempt / PREEMPT_MIN), beatmap=bm)
    cps = ControlPoints(bm.timing_points)
    for i, o in enumerate(bm.hit_objects):
        pl.objects.append(_play_object(i, o, bm, cps))
    _apply_stacking(pl, bm.stack_leniency, bm.format_version)
    return pl


def _play_object(i: int, o: HitObject, bm: Beatmap, cps: ControlPoints) -> PlayObject:
    if o.kind == "slider":
        # stable truncates slider control points to whole pixels (ConvertHitObjectParser.readPoint)
        points = [(float(int(px)), float(int(py))) for px, py in o.curve_points]
        points[0] = (o.x, o.y)
        path = build_path(o.curve_type, points, o.length)
        timing = slider_timing(o, bm, path, cps)
        return PlayObject(i, "slider", o.time, timing.end, o.x, o.y, path=path, timing=timing, raw_x=o.x, raw_y=o.y)
    if o.kind == "spinner":
        return PlayObject(i, "spinner", o.time, o.end_time, *SPINNER_CENTRE, raw_x=SPINNER_CENTRE[0], raw_y=SPINNER_CENTRE[1])
    return PlayObject(i, "circle", o.time, o.time, o.x, o.y, raw_x=o.x, raw_y=o.y)


# ---------------------------------------------------------------- stacking (OsuBeatmapProcessor)

def _raw_end(o: PlayObject) -> tuple[float, float]:
    if o.kind != "slider":
        return o.raw_x, o.raw_y
    assert o.path is not None and o.timing is not None
    return o.path.position_at(1.0 if o.timing.spans % 2 == 1 else 0.0)


def _apply_stacking(pl: Playable, leniency: float, version: int) -> None:
    objs = pl.objects
    if version >= 6:
        _stacking_new(objs, pl.preempt * leniency)
    else:
        _stacking_old(objs, pl.preempt * leniency)
    for o in objs:
        if o.kind == "spinner":
            o.stack_height = 0
        off = o.stack_offset(pl.scale)
        o.x, o.y = o.raw_x + off, o.raw_y + off


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _stacking_new(objs: list[PlayObject], threshold: float) -> None:
    # Full-range version of applyStacking(beatmap, hitObjects, 0, count - 1): the extension pass
    # that grows endIndex is a no-op when the range already covers every object.
    for i in range(len(objs) - 1, 0, -1):
        n = i
        obj_i = objs[i]
        if obj_i.stack_height != 0 or obj_i.kind == "spinner":
            continue
        if obj_i.kind == "circle":
            while True:
                n -= 1
                if n < 0:
                    break
                obj_n = objs[n]
                if obj_n.kind == "spinner":
                    continue
                # stable compares whole-ms times here
                if int(obj_i.time) - int(obj_n.end_time) > threshold:
                    break
                if obj_n.kind == "slider" and _dist(_raw_end(obj_n), (obj_i.raw_x, obj_i.raw_y)) < STACK_DISTANCE:
                    offset = obj_i.stack_height - obj_n.stack_height + 1
                    for j in range(n + 1, i + 1):
                        if _dist(_raw_end(obj_n), (objs[j].raw_x, objs[j].raw_y)) < STACK_DISTANCE:
                            objs[j].stack_height -= offset
                    break
                if _dist((obj_n.raw_x, obj_n.raw_y), (obj_i.raw_x, obj_i.raw_y)) < STACK_DISTANCE:
                    obj_n.stack_height = obj_i.stack_height + 1
                    obj_i = obj_n
        elif obj_i.kind == "slider":
            while True:
                n -= 1
                if n < 0:
                    break
                obj_n = objs[n]
                if obj_n.kind == "spinner":
                    continue
                if obj_i.time - obj_n.time > threshold:
                    break
                if _dist(_raw_end(obj_n), (obj_i.raw_x, obj_i.raw_y)) < STACK_DISTANCE:
                    obj_n.stack_height = obj_i.stack_height + 1
                    obj_i = obj_n


def _stacking_old(objs: list[PlayObject], threshold: float) -> None:
    for i, cur in enumerate(objs):
        if cur.stack_height != 0 and cur.kind != "slider":
            continue
        start = cur.end_time
        slider_stack = 0
        # the old algorithm uses the end of the path (Path.PositionAt(1)), even for repeat sliders
        end_pos = cur.path.position_at(1.0) if cur.kind == "slider" else (cur.raw_x, cur.raw_y)
        for j in range(i + 1, len(objs)):
            if objs[j].time - threshold > start:
                break
            if _dist((objs[j].raw_x, objs[j].raw_y), (cur.raw_x, cur.raw_y)) < STACK_DISTANCE:
                cur.stack_height += 1
                start = objs[j].time
            elif _dist((objs[j].raw_x, objs[j].raw_y), end_pos) < STACK_DISTANCE:
                slider_stack += 1
                objs[j].stack_height -= slider_stack
                start = objs[j].time
