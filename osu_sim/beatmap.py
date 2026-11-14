"""Parser for .osu beatmap files (osu!standard). See the project proposal, "Engineering components" 3.

Coordinates are osu! pixels on the 512x384 playfield; times are milliseconds.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Difficulty:
    hp: float                   # HPDrainRate
    cs: float                   # CircleSize: circle radius
    od: float                   # OverallDifficulty: hit windows
    ar: float                   # ApproachRate: how early objects appear
    slider_multiplier: float    # base slider speed, in hundreds of osu! pixels per beat
    slider_tick_rate: float     # slider ticks per beat


@dataclass
class TimingPoint:
    time: float
    beat_length: float          # uninherited: ms per beat; inherited: negative, -100 / speed multiplier
    uninherited: bool           # True = real BPM change, False = slider speed change


@dataclass
class HitObject:
    x: float
    y: float
    time: int
    kind: str                   # "circle" | "slider" | "spinner"
    new_combo: bool
    # sliders only
    curve_type: str | None = None                                  # "B", "P", "L" or "C"
    curve_points: list[tuple[float, float]] = field(default_factory=list)  # includes the start (x, y)
    slides: int = 1             # 1 = no repeat
    length: float = 0.0         # path length in osu! pixels; 0 = the curve's own length
    # spinners only
    end_time: int | None = None


@dataclass
class Beatmap:
    difficulty: Difficulty
    stack_leniency: float
    timing_points: list[TimingPoint]
    hit_objects: list[HitObject]
    format_version: int = 14   # "osu file format vN"; old versions change stacking, tick spacing, offsets
    beatmap_id: str = ""       # [Metadata] BeatmapID, BeatmapSetID and Title ("" in old maps)
    set_id: str = ""
    title: str = ""
    breaks: list[tuple[float, float]] = field(default_factory=list)   # [Events] break periods (start, end), ms


CIRCLE, SLIDER, NEW_COMBO, SPINNER = 1, 2, 4, 8    # bits of a hit object's type field
EARLY_VERSION_TIMING_OFFSET = 24                    # ms added to all times in format < v5 (LegacyBeatmapDecoder)


def parse_beatmap(path: str | Path) -> Beatmap:
    """Read a .osu file into a Beatmap."""
    return parse_beatmap_text(Path(path).read_text(encoding="utf-8-sig"))


def parse_beatmap_text(text: str) -> Beatmap:
    """Parse the same format from text, including IDs used for grouped splits."""
    sections: dict[str, dict[str, str]] = {"General": {}, "Difficulty": {}, "Metadata": {}}   # key: value sections
    general, difficulty, metadata = sections["General"], sections["Difficulty"], sections["Metadata"]
    timing_points: list[TimingPoint] = []
    hit_objects: list[HitObject] = []
    breaks: list[tuple[float, float]] = []
    version = 14

    section = None
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if line.startswith("osu file format v"):
            version = int(line.removeprefix("osu file format v"))
            continue
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section in sections:
            key, value = line.split(":", 1)          # split on the first ":" only
            sections[section][key.strip()] = value.strip()
        elif section == "Events":
            f = line.split(",")
            if f[0] in ("2", "Break"):           # a break: "2,start,end" (other events are storyboard)
                breaks.append((float(f[1]), float(f[2])))
        elif section == "TimingPoints":
            timing_points.append(_parse_timing_point(line))
        elif section == "HitObjects":
            hit_objects.append(_parse_hit_object(line))

    if int(general.get("Mode", 0)) != 0:
        raise ValueError("only osu!standard maps are supported")
    if not hit_objects:
        raise ValueError("Beatmap has no objects")
    if version < 5:
        for tp in timing_points:
            tp.time += EARLY_VERSION_TIMING_OFFSET
        for o in hit_objects:
            o.time += EARLY_VERSION_TIMING_OFFSET
            if o.end_time is not None:
                o.end_time += EARLY_VERSION_TIMING_OFFSET
        breaks = [(s + EARLY_VERSION_TIMING_OFFSET, e + EARLY_VERSION_TIMING_OFFSET) for s, e in breaks]

    return Beatmap(
        format_version=version,
        beatmap_id=metadata.get("BeatmapID", ""),
        set_id=metadata.get("BeatmapSetID", ""),
        title=metadata.get("Title", ""),
        difficulty=Difficulty(
            hp=float(difficulty["HPDrainRate"]),
            cs=float(difficulty["CircleSize"]),
            od=float(difficulty["OverallDifficulty"]),
            ar=float(difficulty.get("ApproachRate", difficulty["OverallDifficulty"])),  # old maps: AR = OD
            # stable clamps these (LegacyBeatmapDecoder.applyDifficultyRestrictions)
            slider_multiplier=min(max(float(difficulty["SliderMultiplier"]), 0.4), 3.6),
            slider_tick_rate=min(max(float(difficulty["SliderTickRate"]), 0.5), 8.0),
        ),
        stack_leniency=float(general.get("StackLeniency", 0.7)),       # 0.7 is osu!'s default
        timing_points=timing_points,
        hit_objects=hit_objects,
        breaks=breaks,
    )


def _parse_timing_point(line: str) -> TimingPoint:
    # time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects
    f = line.split(",")
    return TimingPoint(
        time=float(f[0]),
        beat_length=float(f[1]),
        uninherited=f[6] == "1" if len(f) > 6 else True,  # old maps omit the field: all uninherited
    )


def _parse_hit_object(line: str) -> HitObject:
    # x, y, time, type, hitSound, then type-specific fields
    f = line.split(",")
    x, y, time, type_ = float(f[0]), float(f[1]), int(f[2]), int(f[3])
    obj = HitObject(x=x, y=y, time=time, kind="circle", new_combo=bool(type_ & NEW_COMBO))
    if type_ & SLIDER:
        # f[5] = "P|187:281|229:272": curve type, then the control points after the start (x, y)
        curve_type, *points = f[5].split("|")
        obj.kind = "slider"
        obj.curve_type = curve_type
        obj.curve_points = [(x, y)] + [(float(px), float(py)) for px, py in (p.split(":") for p in points)]
        # ConvertHitObjectParser: at least one span; a missing or non-positive length means
        # "use the curve's own length" (stored as 0, see build_path)
        obj.slides = max(1, int(f[6]))
        obj.length = max(0.0, float(f[7])) if len(f) > 7 else 0.0
    elif type_ & SPINNER:
        obj.kind = "spinner"
        obj.end_time = int(f[5])
    elif not type_ & CIRCLE:
        raise ValueError(f"unknown hit object type {type_}: {line}")  # e.g. osu!mania hold notes
    return obj
