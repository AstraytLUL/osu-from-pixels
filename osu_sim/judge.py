"""Judge: score a cursor trajectory + key presses against a beatmap, with osu!stable rules.

Works frame by frame, like the game: on each replay frame, new key presses are handled first, then
the game state updates (expired objects are missed, sliders track the cursor, spinners count spins).
It also updates at every checkpoint (tick, slider end, window closing) that falls between frames.

Sources:
- hit windows: osu! wiki Gameplay/Judgement/osu!; lazer osu.Game.Rulesets.Osu/Scoring/OsuHitWindows.cs
  (windows are floor(80 - 6·OD) etc. ms, and a hit needs |error| <= window - 0.5)
- which object a click goes to, and note lock: lazer osu.Game.Rulesets.Osu/UI/LegacyHitPolicy.cs and
  Objects/Drawables/DrawableHitCircle.cs (HitReceptor)
- slider tracking and ticks: lazer Objects/Drawables/SliderInputManager.cs; the slider end is checked
  once at the legacy last tick, as stable did (see _judge_parts)
- slider result from the share of parts hit: osu! wiki Gameplay/Judgement/osu! (Sliders)
- spinner spins required: osu! wiki Gameplay/Judgement/osu! (Spinners); spin counting: lazer
  Skinning/Default/SpinnerRotationTracker.cs

Validation (scripts/validate_judge.py): on 24 real stable replays with 100s, 50s and misses, the
counts match osu!'s exactly on 20. The other 4 are all DT+HR+HD on two maps, and each has one 300
where osu! has a 100; the cause is not found yet (not the hit windows, spinners, slider ends or
key handling, which were each tested). Known approximation: spinner spin counting.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from data.replay import M1, M2, ReplayFrames
from osu_sim.mods import rate as mod_rate
from osu_sim.playable import FOLLOW_AREA, SPINNER_CENTRE, Playable, PlayObject

MISS_WINDOW = 400              # ms; an early click inside this (but outside the 50 window) is a miss
NOTELOCK_LENIENCY = 3          # ms; objects this close in time do not lock each other
SPINNER_MAX_RAD_PER_MS = 0.05  # cap on counted rotation, real time (~477 RPM); approximation
LEFT, RIGHT = M1, M2


@dataclass
class HitWindows:
    w300: int
    w100: int
    w50: int


def hit_windows(od: float) -> HitWindows:
    # stable: ±(80 - 6·OD), ±(140 - 8·OD), ±(200 - 10·OD) ms, truncated to whole ms.
    # The wiki disagrees with itself (Gameplay/Judgement/osu! says truncated, Beatmap/Overall_difficulty
    # says rounded). The code settles it: lazer's OsuHitWindows.cs uses floor(range) - 0.5 with
    # |error| <= window; _result compares exactly that way (for whole-ms replay times it is the
    # same as |error| < floor(range)).
    return HitWindows(int(80 - 6 * od), int(140 - 8 * od), int(200 - 10 * od))


def circle_radius(cs: float) -> float:
    # osu! pixels (wiki: Beatmap/Circle_size); the 1.00041 matches a radius bug in old replays
    return (54.4 - 4.48 * cs) * 1.00041


@dataclass
class Judgement:
    index: int                 # position in beatmap.hit_objects
    kind: str                  # "circle" | "slider" | "spinner"
    time: float                # object start time, ms
    result: int                # 300 / 100 / 50 / 0 (miss)
    offset: float | None = None  # circle / slider head: click time - object time (ms); None if never clicked
    parts_hit: int | None = None       # slider: head + ticks + repeats + tail that were hit
    parts_total: int | None = None
    parts: list[tuple[float, str, bool]] = field(default_factory=list)   # slider: (time, "tick" | "repeat" | "tail", hit)
    spins: float | None = None         # spinner: full spins counted


# ---------------------------------------------------------------- per-object state

@dataclass
class _Part:                   # a slider tick, repeat or tail
    time: float
    kind: str                  # "tick" | "repeat" | "tail"
    hit: bool | None = None    # None = not judged yet


@dataclass
class _State:
    obj: PlayObject
    # circle, or the slider's head
    head_judged: bool = False
    head_hit: bool = False
    head_offset: float | None = None
    head_result: int = 0
    hit_action: int | None = None      # LEFT / RIGHT: the first key that clicked the head
    # slider
    parts: list[_Part] = field(default_factory=list)
    tracking: bool = False
    accept_any_after: float | None = None
    last_pressed: int = 0
    # spinner
    rotation: float = 0.0
    last_angle: float | None = None
    last_t: float | None = None
    done: bool = False                 # fully judged

    @property
    def all_judged(self) -> bool:
        return self.done


def _result(error: float, win: HitWindows) -> int | None:
    """300/100/50, 0 for an early miss, None if outside the MISS window (the click does nothing)."""
    # Strict whole-ms bounds are equivalent to these half-ms bounds only on
    # integer replay times. BC's 60 Hz clock also contains fractional times.
    if error <= win.w300 - .5:
        return 300
    if error <= win.w100 - .5:
        return 100
    if error <= win.w50 - .5:
        return 50
    if error <= MISS_WINDOW:
        return 0
    return None


# ---------------------------------------------------------------- the judge

def judge(pl: Playable, frames: ReplayFrames) -> list[Judgement]:
    win = hit_windows(pl.od)
    rate = mod_rate(pl.mods)
    states = [_State(o) for o in pl.objects]
    for s in states:
        if s.obj.kind == "slider":
            tm = s.obj.timing
            s.parts = sorted([_Part(t, "tick") for t, _ in tm.ticks] + [_Part(t, "repeat") for t, _ in tm.repeats],
                             key=lambda p: p.time) + [_Part(tm.legacy_last_tick, "tail")]

    # The game judges ticks, slider ends and expiring circles at their exact times, holding the last
    # key state and moving the cursor between replay frames. Replay frames are ~16 ms apart, so we
    # also update at every such checkpoint that falls between two frames.
    checkpoints = sorted(_checkpoints(states, win))
    c = 0

    first = [0]                # states[:first[0]] are all fully judged
    ts, xs, ys, ks = frames.t.tolist(), frames.x.tolist(), frames.y.tolist(), frames.keys.tolist()
    prev_held, prev = 0, None
    for t, x, y, keys in zip(ts, xs, ys, ks):
        held = keys & (LEFT | RIGHT)
        new = held & ~prev_held
        # checkpoints strictly between the previous frame and this one
        while c < len(checkpoints) and checkpoints[c] < t:
            ct = checkpoints[c]
            c += 1
            if prev is None:                   # before the first frame: cursor where it first appears, no keys
                _update(_alive(states, first, ct, pl), ct, x, y, 0, pl, win, rate)
                continue
            if ct <= prev[0]:
                continue
            pt, px, py = prev
            w = (ct - pt) / (t - pt)
            _update(_alive(states, first, ct, pl), ct, px + (x - px) * w, py + (y - py) * w, prev_held, pl, win, rate)

        alive = _alive(states, first, t, pl)
        # 1. input: each newly pressed key is a separate click (left first)
        for action in (LEFT, RIGHT):
            if new & action:
                _click(alive, states, action, t, x, y, held, pl, win)
        # 2. update
        _update(alive, t, x, y, held, pl, win, rate)
        prev_held, prev = held, (t, x, y)

    # replay over: anything left is judged as it stands
    for s in states:
        if not s.head_judged and s.obj.kind != "spinner":
            _judge_head(s, 0, None)
        for p in s.parts:
            if p.hit is None:
                p.hit = False
        s.done = True
    return [_judgement(s, pl) for s in states]


def _checkpoints(states: list[_State], win: HitWindows) -> list[float]:
    """Times at which the game state changes on its own: a circle / head window closing, a slider
    tick / repeat / end check, a spinner ending."""
    out = []
    for s in states:
        o = s.obj
        if o.kind == "spinner":
            out.append(o.end_time)
            continue
        out.append(o.time + win.w50)
        out.extend(p.time for p in s.parts)
    return out


def _alive(states: list[_State], first: list[int], t: float, pl: Playable) -> list[_State]:
    """Objects already on screen (time - preempt <= t) and not fully judged, in time order."""
    while first[0] < len(states) and states[first[0]].done:
        first[0] += 1
    alive = []
    for s in states[first[0]:]:
        if s.obj.time - pl.preempt > t:
            break
        if not s.done:
            alive.append(s)
    return alive


def _update(alive: list[_State], t, x, y, held: int, pl: Playable, win: HitWindows, rate: float) -> None:
    for s in alive:
        o = s.obj
        if o.kind in ("circle", "slider") and not s.head_judged and t - o.time >= win.w50:
            _judge_head(s, 0, None)            # expired without a hit
        if o.kind == "circle" and s.head_judged:
            s.done = True
        elif o.kind == "slider":
            _update_tracking(s, t, x, y, held, pl.radius, _in_follow_area(s, t, x, y, pl.radius, s.tracking))
            _judge_parts(s, t)
        elif o.kind == "spinner":
            _spin(s, t, x, y, held, rate)
            if t >= o.end_time:
                s.done = True


def _click(alive, states, action, t, x, y, held, pl: Playable, win: HitWindows) -> None:
    """Give a click to the earliest unjudged circle / slider head under the cursor (it is drawn on top).
    That object consumes the click whatever happens: hit, note lock shake, or nothing."""
    for s in alive:
        o = s.obj
        if o.kind == "spinner" or s.head_judged:
            continue
        if (x - o.x) ** 2 + (y - o.y) ** 2 > pl.radius ** 2:
            continue
        if s.hit_action is None:
            s.hit_action = action              # HitReceptor sets this even when the click shakes
        offset = t - o.time
        result = _result(abs(offset), win)
        if not _hittable(s, alive, states, t, result):
            return
        _judge_head(s, result, offset)
        if o.kind == "slider" and s.head_hit:
            _after_head_hit(s, t, x, y, held, pl.radius)
        return


def _hittable(s: _State, alive: list[_State], states: list[_State], t: float, result: int | None) -> bool:
    """LegacyHitPolicy.CheckHittable: may this click judge object s?"""
    o = s.obj
    if o.index > 0:
        prev = states[o.index - 1]
        if prev.obj.stack_height > 0 and not prev.done:
            return False                       # Ignore: a stacked object before it is still waiting
    if result is None:
        return False                           # Shake: outside the MISS window
    for e in alive:                            # alive is in time order; unjudged earlier objects are all in it
        if e.obj.index >= o.index:
            break
        if not e.done and e.obj.end_time + NOTELOCK_LENIENCY < o.time:
            return False                       # Shake: note lock behind an earlier object
    return abs(o.time - t) < MISS_WINDOW


def _judge_head(s: _State, result: int, offset: float | None) -> None:
    s.head_judged = True
    s.head_offset = offset
    s.head_result = result
    s.head_hit = result > 0
    if s.obj.kind == "circle":
        s.done = True                          # a circle is fully judged at once (matters for a 2nd click this frame)


# ---------------------------------------------------------------- sliders (SliderInputManager)

def _in_follow_area(s: _State, t: float, x: float, y: float, radius: float, expanded: bool) -> bool:
    r = radius * (FOLLOW_AREA if expanded else 1.0)
    bx, by = s.obj.ball_position(t)
    return (x - bx) ** 2 + (y - by) ** 2 <= r * r


def _update_tracking(s: _State, t, x, y, held: int, radius: float, valid_position: bool) -> None:
    head_action = s.hit_action
    if head_action is None:
        s.accept_any_after = None
    if head_action is not None and s.accept_any_after is None:
        # once the other key has been released, either key may track
        other = RIGHT if head_action == LEFT else LEFT
        if not s.last_pressed & other:
            s.accept_any_after = t
    restricted = head_action is not None and (s.accept_any_after is None or t <= s.accept_any_after)
    valid_action = bool(held & head_action) if restricted else bool(held)
    s.last_pressed = held
    parts_left = any(p.hit is None for p in s.parts) or not s.head_judged
    s.tracking = (parts_left or t <= s.obj.end_time) and valid_position and valid_action


def _after_head_hit(s: _State, t, x, y, held: int, radius: float) -> None:
    """PostProcessHeadJudgement: a late head hit settles the ticks the ball already passed."""
    if not _in_follow_area(s, t, x, y, radius, expanded=True):
        return
    r2 = (radius * FOLLOW_AREA) ** 2
    passed = [p for p in s.parts if p.hit is None and p.time <= t]
    all_in_range = True
    for p in passed:
        bx, by = s.obj.ball_position(p.time)
        if (bx - x) ** 2 + (by - y) ** 2 > r2:
            all_in_range = False
            break
    for p in passed:
        p.hit = all_in_range
    _update_tracking(s, t, x, y, held, radius,
                     all_in_range or _in_follow_area(s, t, x, y, radius, expanded=False))


def _judge_parts(s: _State, t: float) -> None:
    """Judge each tick / repeat / tail whose time has come: hit if tracking at that moment.

    The tail is checked once, at the legacy last tick (end - 36 ms, or half way for sliders under
    72 ms), as stable did (see SliderEventGenerator). lazer instead accepts the tail at any moment
    from end - 36 to the end; on 24 validation replays the stable rule matched osu! on one more.
    Parts are only judged once the head has been judged (hit or missed), as in lazer.
    """
    if not s.head_judged:
        return
    for p in s.parts:
        if p.hit is None and t >= p.time:
            p.hit = s.tracking
    if all(p.hit is not None for p in s.parts):
        s.done = True


# ---------------------------------------------------------------- spinners

def _spin(s: _State, t, x, y, held: int, rate: float) -> None:
    """Count rotation of the cursor around the playfield centre while a key is held.

    As in lazer's SpinnerRotationTracker, each step is multiplied by the playback rate (so under DT
    one turn of the cursor counts as 1.5 spins), and spinning counts while start <= t < end.
    Approximation: stable also smooths the spin speed and caps it; we only cap the cursor's real-time
    angular speed at ~477 RPM.
    """
    o = s.obj
    angle = math.atan2(y - SPINNER_CENTRE[1], x - SPINNER_CENTRE[0])
    if s.last_angle is not None and held and o.time <= t < o.end_time:
        d = angle - s.last_angle
        d = (d + math.pi) % (2 * math.pi) - math.pi
        dt = max(t - s.last_t, 0.0)
        cap = SPINNER_MAX_RAD_PER_MS / rate * dt            # map time runs `rate` x faster than real time
        s.rotation += max(-cap, min(cap, d)) * rate
    s.last_angle, s.last_t = angle, t


def spinner_result(spins: float, duration_ms: float, od: float) -> int:
    """osu! wiki (Spinners): GREAT at the required spins, OK one spin short, MEH at 25%, else MISS.
    Counted in half spins, as stable does."""
    half_per_s = 3 + 0.4 * od if od < 5 else 2.5 + 0.5 * od
    needed_half = int(duration_ms / 1000 * half_per_s) + 1  # stable compares count > requirement
    if needed_half <= 1:
        return 300                             # requirement rounds to 0: the spinner clears itself
    half = int(spins * 2)
    if half >= needed_half:
        return 300
    if half >= needed_half - 2:
        return 100
    if half >= needed_half * 0.25:
        return 50
    return 0


# ---------------------------------------------------------------- results

def _judgement(s: _State, pl: Playable) -> Judgement:
    o = s.obj
    if o.kind == "circle":
        return Judgement(o.index, "circle", o.time, s.head_result, s.head_offset)
    if o.kind == "spinner":
        spins = abs(s.rotation) / (2 * math.pi)
        return Judgement(o.index, "spinner", o.time, spinner_result(spins, o.end_time - o.time, pl.od), spins=spins)
    hit = int(s.head_hit) + sum(bool(p.hit) for p in s.parts)
    total = 1 + len(s.parts)
    if hit == total:
        result = 300
    elif hit == 0:
        result = 0
    else:
        result = 100 if hit / total >= 0.5 else 50
    return Judgement(o.index, "slider", o.time, result, s.head_offset if s.head_hit else None,
                     parts_hit=hit, parts_total=total, parts=[(p.time, p.kind, bool(p.hit)) for p in s.parts])


def accuracy(judgements: list[Judgement]) -> float:
    """osu!stable accuracy: sum of results / (300 · number of objects)."""
    return sum(j.result for j in judgements) / (300 * len(judgements))


def unstable_rate(judgements: list[Judgement]) -> float:
    """10 × standard deviation of hit offsets (ms), over circles and slider heads that were hit."""
    offsets = [j.offset for j in judgements if j.offset is not None and (j.kind == "slider" or j.result > 0)]
    return 10 * float(np.std(offsets))


def counts(judgements: list[Judgement]) -> dict[int, int]:
    """Number of 300s, 100s, 50s and misses."""
    return {r: sum(j.result == r for j in judgements) for r in (300, 100, 50, 0)}
