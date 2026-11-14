"""Combo and score (osu!stable, ScoreV1) from the judge's results, for .osr headers and reports.

Combo (osu! wiki, Gameplay/Combo; lazer's legacy judgement types): every circle, slider head,
slider tick, slider repeat, slider end and spinner adds 1. A miss on any of them resets the combo
to 0, except the slider end: missing it only loses its +1. A 100 or 50 keeps the combo.

Score (osu! wiki, Gameplay/Score/ScoreV1/osu!): each circle, slider and spinner result adds
    result + result * (max(combo before it - 1, 0) * difficulty multiplier * mod multiplier / 25)
and slider parts add flat points (head, repeat, end 30; tick 10), spinners 100 per spin up to the
requirement and 1000 per bonus spin. The difficulty multiplier is
    round((HP + CS + OD + clamp(objects / drain seconds * 8, 0, 16)) / 38 * 5)
from the map's unmodded settings, drain time excluding breaks.

"""
from dataclasses import dataclass

from osu_sim.beatmap import Beatmap
from osu_sim.judge import Judgement, hit_windows
from osu_sim.mods import DT, EZ, HD, HR, HT, NC
from osu_sim.playable import Playable

NF, FL, SO = 1, 1024, 4096
MOD_MULTIPLIERS = {NF: 0.5, EZ: 0.5, HT: 0.3, HD: 1.06, HR: 1.06, DT: 1.12, FL: 1.12, SO: 0.9}
TICK_POINTS, PART_POINTS = 10, 30              # slider tick; slider head / repeat / end
SPIN_POINTS, BONUS_SPIN_POINTS = 100, 1000


@dataclass
class _Event:
    time: float
    combo: str                 # "add" (+1), "break" (reset to 0) or "none"
    flat: int = 0              # points without the combo multiplier
    result: int = 0            # a circle / slider / spinner result, scored with the multiplier


def _events(js: list[Judgement], pl: Playable) -> list[_Event]:
    """Every combo / score event of a play, in time order."""
    w50 = hit_windows(pl.od).w50
    ev = []
    for j in js:
        o = pl.objects[j.index]
        if j.kind == "circle":
            t = o.time + (j.offset if j.offset is not None else w50)    # clicked, or missed when the window closed
            ev.append(_Event(t, "add" if j.result else "break", result=j.result))
        elif j.kind == "slider":
            head_hit = j.offset is not None
            ev.append(_Event(o.time + (j.offset if head_hit else w50), "add" if head_hit else "break",
                             flat=PART_POINTS if head_hit else 0))
            for t, kind, hit in j.parts:
                if kind == "tail":
                    ev.append(_Event(t, "add" if hit else "none", flat=PART_POINTS if hit else 0))
                else:
                    ev.append(_Event(t, "add" if hit else "break",
                                     flat=(TICK_POINTS if kind == "tick" else PART_POINTS) if hit else 0))
            ev.append(_Event(o.end_time, "none", result=j.result))     # the slider's own result, at its end
        else:
            ev.append(_Event(o.end_time, "add" if j.result else "break", flat=_spin_points(j, o, pl), result=j.result))
    return sorted(ev, key=lambda e: e.time)                            # stable sort: same-time events keep order


def _spin_points(j: Judgement, o, pl: Playable) -> int:
    """100 per full spin up to the requirement, 1000 per spin beyond it."""
    per_s = 1.5 + 0.2 * pl.od if pl.od < 5 else 1.25 + 0.25 * pl.od            # spins per second required
    required = int((o.end_time - o.time) / 1000 * per_s)
    spins = int(j.spins or 0)
    return SPIN_POINTS * min(spins, required) + BONUS_SPIN_POINTS * max(spins - required, 0)


def max_combo(js: list[Judgement], pl: Playable) -> int:
    combo = best = 0
    for e in _events(js, pl):
        if e.combo == "add":
            combo += 1
        elif e.combo == "break":
            combo = 0
        best = max(best, combo)
    return best


def combo_breaks(js: list[Judgement]) -> dict[str, int]:
    """Combo breaks by cause (the project proposal's diagnostic), one per break event as in _events:
    miss          a circle, a slider head or a spinner missed
    slider_break  a slider tick or repeat dropped (the head was hit or not: each counts)
    end_drop      a slider tail dropped. It does not break combo in osu!stable (the slider just loses
                  its last combo point and drops to a 100 at best), so it is counted next to the breaks,
                  not in them."""
    out = {"miss": 0, "slider_break": 0, "end_drop": 0}
    for j in js:
        if j.kind == "slider":
            out["miss"] += j.offset is None
            for _, kind, hit in j.parts:
                if not hit:
                    out["end_drop" if kind == "tail" else "slider_break"] += 1
        else:
            out["miss"] += j.result == 0
    return out


def difficulty_multiplier(beatmap: Beatmap) -> int:
    """From the unmodded map: HP + CS + OD plus a density term (objects per drain second)."""
    d = beatmap.difficulty
    objs = beatmap.hit_objects
    first = objs[0].time
    last = max(o.end_time if o.end_time is not None else o.time for o in objs)
    drain_s = max((last - first - sum(e - s for s, e in beatmap.breaks)) / 1000, 1.0)
    density = min(max(len(objs) / drain_s * 8, 0), 16)
    return round((d.hp + d.cs + d.od + density) / 38 * 5)


def mod_multiplier(mods: int) -> float:
    m = 1.0
    for bit, factor in MOD_MULTIPLIERS.items():
        if mods & bit:
            m *= factor
    if mods & NC and not mods & DT:            # NC counts as DT (stable replays set both bits)
        m *= MOD_MULTIPLIERS[DT]
    return m


def score_v1(js: list[Judgement], pl: Playable, beatmap: Beatmap) -> int:
    """ScoreV1 of a play; `beatmap` is the unmodded map (for the difficulty multiplier)."""
    factor = difficulty_multiplier(beatmap) * mod_multiplier(pl.mods) / 25
    total, combo = 0, 0
    for e in _events(js, pl):
        if e.result:
            # stable rounds each hit's combo bonus down to a whole number
            total += e.result + int(e.result * max(combo - 1, 0) * factor)
        total += e.flat
        if e.combo == "add":
            combo += 1
        elif e.combo == "break":
            combo = 0
    return total
