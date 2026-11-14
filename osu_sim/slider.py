"""Slider geometry and timing.

Ported from osu! lazer, which reproduces osu!stable:
- curve parsing: osu.Game/Rulesets/Objects/Legacy/ConvertHitObjectParser.cs (convertPoints)
- path shapes:   osu-framework osu.Framework/Utils/PathApproximator.cs, CircularArcProperties.cs
- path length:   osu.Game/Rulesets/Objects/SliderPath.cs (calculatePath, calculateLength, PositionAt)
- timing:        osu.Game.Rulesets.Osu/Objects/Slider.cs (ApplyDefaultsToSelf),
                 osu.Game/Rulesets/Objects/Legacy/LegacyRulesetExtensions.cs (GetPrecisionAdjustedBeatLength),
                 osu.Game/Rulesets/Objects/SliderEventGenerator.cs (ticks, repeats, legacy last tick)

lazer computes positions in single precision (float); we use double. The difference is far below a pixel.
"""
import bisect
import math
from dataclasses import dataclass, field

from osu_sim.beatmap import Beatmap, HitObject, TimingPoint

BEZIER_TOLERANCE = 0.25
CATMULL_DETAIL = 50
ARC_TOLERANCE = 0.1
TAIL_LENIENCY = -36            # ms; stable judges the slider end 36 ms early ("legacy last tick")
BASE_SCORING_DISTANCE = 100

Point = tuple[float, float]


# ---------------------------------------------------------------- path shapes

def _bezier_flat_enough(cp: list[Point]) -> bool:
    for i in range(1, len(cp) - 1):
        dx = cp[i - 1][0] - 2 * cp[i][0] + cp[i + 1][0]
        dy = cp[i - 1][1] - 2 * cp[i][1] + cp[i + 1][1]
        if dx * dx + dy * dy > BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4:
            return False
    return True


def _bezier_subdivide(cp: list[Point]) -> tuple[list[Point], list[Point]]:
    """De Casteljau split at t = 0.5 into left and right halves."""
    n = len(cp)
    mid = list(cp)
    left, right = [None] * n, [None] * n
    for i in range(n):
        left[i] = mid[0]
        right[n - i - 1] = mid[n - i - 1]
        for j in range(n - i - 1):
            mid[j] = ((mid[j][0] + mid[j + 1][0]) / 2, (mid[j][1] + mid[j + 1][1]) / 2)
    return left, right


def _bezier_approximate(cp: list[Point], out: list[Point]) -> None:
    """Append a piecewise-linear approximation of a flat-enough curve (all but its last point)."""
    n = len(cp)
    left, right = _bezier_subdivide(cp)
    joined = left + right[1:]                  # 2n - 1 points
    out.append(cp[0])
    for i in range(1, n - 1):
        k = 2 * i
        out.append((0.25 * (joined[k - 1][0] + 2 * joined[k][0] + joined[k + 1][0]),
                    0.25 * (joined[k - 1][1] + 2 * joined[k][1] + joined[k + 1][1])))


def bezier(cp: list[Point]) -> list[Point]:
    if len(cp) < 2:
        return list(cp)
    out: list[Point] = []
    stack = [list(cp)]                         # depth-first adaptive subdivision, as in osu-framework
    while stack:
        parent = stack.pop()
        if _bezier_flat_enough(parent):
            _bezier_approximate(parent, out)
            continue
        left, right = _bezier_subdivide(parent)
        stack.append(right)
        stack.append(left)
    out.append(cp[-1])
    return out


def _catmull_point(v1, v2, v3, v4, t):
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
                 for a, b, c, d in zip(v1, v2, v3, v4))


def catmull(cp: list[Point]) -> list[Point]:
    out = []
    for i in range(len(cp) - 1):
        v1 = cp[i - 1] if i > 0 else cp[i]
        v2 = cp[i]
        v3 = cp[i + 1] if i < len(cp) - 1 else (2 * v2[0] - v1[0], 2 * v2[1] - v1[1])
        v4 = cp[i + 2] if i < len(cp) - 2 else (2 * v3[0] - v2[0], 2 * v3[1] - v2[1])
        for c in range(CATMULL_DETAIL):
            out.append(_catmull_point(v1, v2, v3, v4, c / CATMULL_DETAIL))
            out.append(_catmull_point(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))
    return out


def _optimise_catmull(sub: list[Point]) -> tuple[list[Point], float]:
    """stable keeps only vertices >6 px apart when drawing; lazer mimics it for catmull. Returns the
    thinned path and the length it removed (which still counts toward the path length)."""
    out, removed_total = [], 0.0
    last_start, removed = None, 0.0
    for i, p in enumerate(sub):
        if last_start is None:
            out.append(p)
            last_start = p
            continue
        dist_from_start = math.dist(last_start, p)
        removed += math.dist(sub[i - 1], p)
        if dist_from_start > 6 or (i + 1) % (CATMULL_DETAIL * 2) == 0 or i == len(sub) - 1:
            out.append(p)
            removed_total += removed - dist_from_start
            last_start, removed = None, 0.0
    return out, removed_total


def _collinear(a: Point, b: Point, c: Point) -> bool:
    return abs((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])) < 1e-3


def circular_arc(cp: list[Point]) -> list[Point] | None:
    """Points on the circle through cp[0], cp[1], cp[2]; None if the arc is degenerate."""
    a, b, c = cp
    if _collinear(a, b, c):
        return None
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    a2, b2, c2 = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    cx = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    cy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    radius = math.hypot(a[0] - cx, a[1] - cy)
    theta_start = math.atan2(a[1] - cy, a[0] - cx)
    theta_end = math.atan2(c[1] - cy, c[0] - cx)
    while theta_end < theta_start:
        theta_end += 2 * math.pi
    direction, theta_range = 1, theta_end - theta_start
    # draw the side of the circle that B lies on
    ortho = (c[1] - a[1], -(c[0] - a[0]))
    if ortho[0] * (b[0] - a[0]) + ortho[1] * (b[1] - a[1]) < 0:
        direction, theta_range = -1, 2 * math.pi - theta_range
    if 2 * radius <= ARC_TOLERANCE:
        n = 2
    else:
        n = max(2, math.ceil(theta_range / (2 * math.acos(1 - ARC_TOLERANCE / radius))))
    if n >= 1000:              # SliderPath gives up on huge arcs and falls back to Bezier
        return None
    return [(cx + radius * math.cos(theta_start + direction * i / (n - 1) * theta_range),
             cy + radius * math.sin(theta_start + direction * i / (n - 1) * theta_range)) for i in range(n)]


# ---------------------------------------------------------------- .osu curve -> segments

def segments(curve_type: str, points: list[Point]) -> list[tuple[str, list[Point]]]:
    """Split a stable curve into (type, control points) segments, following ConvertHitObjectParser.

    `points` includes the slider's start position. Stable coordinates are whole numbers.
    """
    kind = {"B": "bezier", "P": "perfect", "L": "linear", "C": "catmull"}[curve_type]
    if kind == "perfect":
        if len(points) != 3:
            kind = "bezier"
        elif _collinear(*points):
            kind = "linear"            # stable special-cases collinear perfect curves
    out, start = [], 0
    for end in range(1, len(points)):
        if points[end] != points[end - 1]:
            continue
        if kind == "catmull" and end > 1:
            continue                   # stable catmull has no implicit segments
        if end == len(points) - 1:
            continue                   # the last point cannot start a segment
        out.append((kind, points[start:end]))
        start = end                    # the next segment starts on the repeated point (kept once)
    out.append((kind, points[start:]))
    return out


def _sub_path(kind: str, cp: list[Point]) -> tuple[list[Point], float]:
    if kind == "linear":
        return list(cp), 0.0
    if kind == "perfect":
        arc = circular_arc(cp) if len(cp) == 3 else None
        if arc:
            return arc, 0.0
        return bezier(cp), 0.0
    if kind == "catmull":
        return _optimise_catmull(catmull(cp))
    return bezier(cp), 0.0


@dataclass
class SliderPath:
    points: list[Point]        # piecewise-linear path, already trimmed/extended to the expected length
    cumulative: list[float]    # distance along the path at each point
    distance: float            # total length = the .osu pixel length (unless the path is degenerate)

    def position_at(self, progress: float) -> Point:
        """Position at fraction `progress` (0..1) of the path length."""
        d = min(max(progress, 0.0), 1.0) * self.distance
        i = bisect.bisect_left(self.cumulative, d)
        if i <= 0:
            return self.points[0]
        if i >= len(self.points):
            return self.points[-1]
        d0, d1 = self.cumulative[i - 1], self.cumulative[i]
        p0, p1 = self.points[i - 1], self.points[i]
        if abs(d1 - d0) < 1e-3:
            return p0
        w = (d - d0) / (d1 - d0)
        return (p0[0] + (p1[0] - p0[0]) * w, p0[1] + (p1[1] - p0[1]) * w)


def build_path(curve_type: str, control_points: list[Point], expected_distance: float) -> SliderPath:
    path: list[Point] = []
    optimised = 0.0
    for kind, cp in segments(curve_type, control_points):
        if len(cp) == 1:
            path.append(cp[0])
            continue
        sub, removed = _sub_path(kind, cp)
        optimised += removed
        skip_first = bool(path) and bool(sub) and path[-1] == sub[0]
        path.extend(sub[1:] if skip_first else sub)

    # calculateLength: lengths start at the catmull "optimised" length (lazer quirk, kept to match)
    length = optimised
    cumulative = [0.0]
    for p, q in zip(path, path[1:]):
        length += math.dist(p, q)
        cumulative.append(length)

    if expected_distance <= 0:                    # no length in the .osu: keep the curve's own length
        expected_distance = length
    if length != expected_distance:
        if len(path) >= 2 and path[-1] == path[-2] and expected_distance > length:
            cumulative.append(length)  # lazer: a path ending on a duplicate point is not extended
            return SliderPath(path, cumulative, cumulative[-1])
        cumulative.pop()
        end = len(path) - 1
        if length > expected_distance:            # trim points beyond the expected length
            while cumulative and cumulative[-1] >= expected_distance:
                cumulative.pop()
                path.pop(end)
                end -= 1
        if end <= 0:
            cumulative.append(0.0)
            return SliderPath(path, cumulative, 0.0)
        # move the last point along the last segment to land exactly on the expected length
        (x0, y0), (x1, y1) = path[end - 1], path[end]
        seg = math.hypot(x1 - x0, y1 - y0)
        ux, uy = ((x1 - x0) / seg, (y1 - y0) / seg) if seg > 0 else (0.0, 0.0)
        extra = expected_distance - cumulative[-1]
        path[end] = (x0 + ux * extra, y0 + uy * extra)
        cumulative.append(expected_distance)
    return SliderPath(path, cumulative, cumulative[-1])


# ---------------------------------------------------------------- timing

class ControlPoints:
    """Lookups over a beatmap's timing points, built once per beatmap (LegacyBeatmapDecoder semantics)."""

    def __init__(self, tps: list[TimingPoint]):
        bpm = [tp for tp in tps if tp.uninherited]
        self._bpm_times = [tp.time for tp in bpm]
        self._bpm = bpm
        # Every timing point also sets slider velocity: BPM points to 1.0, inherited points to
        # 100 / -beatLength. Points at the same time are merged and the inherited one wins
        # (flushPendingPoints). A NaN beat length gives SV 1.0 and disables ticks.
        self._sv_times: list[float] = []
        self._sv: list[tuple[float, bool]] = []          # (sv, generate_ticks)
        i = 0
        while i < len(tps):
            j = i
            while j < len(tps) and tps[j].time == tps[i].time:
                j += 1
            group = tps[i:j]
            inherited = [tp for tp in group if not tp.uninherited]
            chosen = inherited[-1] if inherited else group[-1]
            bl = chosen.beat_length
            sv = 100.0 / -bl if bl < 0 else 1.0           # NaN < 0 is False -> 1.0
            self._sv_times.append(tps[i].time)
            self._sv.append((min(max(sv, 0.1), 10.0), not math.isnan(bl)))
            i = j

    def timing_point_at(self, time: float) -> TimingPoint:
        """The BPM point in effect at `time`; before the first one, the first one."""
        i = bisect.bisect_right(self._bpm_times, time) - 1
        return self._bpm[max(i, 0)]

    def slider_velocity_at(self, time: float) -> tuple[float, bool]:
        """(SV multiplier, generate ticks) at `time`; (1.0, True) before the first point."""
        i = bisect.bisect_right(self._sv_times, time) - 1
        return self._sv[i] if i >= 0 else (1.0, True)


@dataclass
class SliderTiming:
    start: float
    span_duration: float
    spans: int
    end: float                 # end of the last span
    legacy_last_tick: float    # where stable judges the slider end (end - 36 ms, or half way if short)
    ticks: list[tuple[float, float]] = field(default_factory=list)    # (time, path progress)
    repeats: list[tuple[float, float]] = field(default_factory=list)  # (time, path progress)
    velocity: float = 0.0      # osu! pixels per ms


def slider_timing(obj: HitObject, beatmap: Beatmap, path: SliderPath, cps: ControlPoints) -> SliderTiming:
    tp = cps.timing_point_at(obj.time)
    beat_length = min(max(tp.beat_length, 6), 60000)
    sv, generate_ticks = cps.slider_velocity_at(obj.time)
    # GetPrecisionAdjustedBeatLength: stable clamps the SV-as-beat-length in single precision
    sv_beat_length = -100 / sv
    bpm_multiplier = min(max(float(_f32(-sv_beat_length)), 10), 1000) / 100.0
    velocity = BASE_SCORING_DISTANCE * beatmap.difficulty.slider_multiplier / (beat_length * bpm_multiplier)
    scoring_distance = velocity * beat_length
    # OsuBeatmapConverter: maps before format v8 space ticks by 1 / SV
    tick_multiplier = 1 / sv if beatmap.format_version < 8 else 1.0
    tick_distance = (scoring_distance / beatmap.difficulty.slider_tick_rate * tick_multiplier
                     if generate_ticks else math.inf)

    total = path.distance
    # ConvertHitObjectParser.createSlider: a zero-length slider has no repeats (stable generated
    # them but never judged them)
    spans = obj.slides if abs(total) > 1e-7 else 1
    duration = spans * total / velocity if velocity > 0 else 0.0
    span_duration = duration / spans
    t = SliderTiming(start=obj.time, span_duration=span_duration, spans=spans,
                     end=obj.time + duration, legacy_last_tick=0.0, velocity=velocity)

    # SliderEventGenerator.Generate
    length = min(100000, total)
    tick_distance = min(max(tick_distance, 0), length)
    min_from_end = velocity * 10
    for span in range(spans):
        span_start = obj.time + span * span_duration
        reversed_ = span % 2 == 1
        if tick_distance != 0:
            span_ticks = []
            d = tick_distance
            while d <= length:
                if d >= length - min_from_end:
                    break
                progress = d / length
                time_progress = 1 - progress if reversed_ else progress
                span_ticks.append((span_start + time_progress * span_duration, progress))
                d += tick_distance
            t.ticks += reversed(span_ticks) if reversed_ else span_ticks
        if span < spans - 1:
            t.repeats.append((obj.time + (span + 1) * span_duration, float((span + 1) % 2)))
    final_span_start = obj.time + (spans - 1) * span_duration
    t.legacy_last_tick = max(obj.time + duration / 2, final_span_start + span_duration + TAIL_LENIENCY)
    return t


def _f32(x: float) -> float:
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]
