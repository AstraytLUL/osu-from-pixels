import math
from pathlib import Path

import pytest

from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint, parse_beatmap
from osu_sim.playable import make_playable
from osu_sim.slider import ControlPoints, bezier, build_path, circular_arc, segments, slider_timing

# ---------- curve parsing (ConvertHitObjectParser.convertPoints) ----------


def test_repeated_point_splits_a_bezier():
    pts = [(0, 0), (10, 0), (10, 0), (20, 10)]
    assert segments("B", pts) == [("bezier", [(0, 0), (10, 0)]), ("bezier", [(10, 0), (20, 10)])]


def test_last_point_cannot_split():
    pts = [(0, 0), (10, 0), (10, 0)]
    assert segments("B", pts) == [("bezier", pts)]


def test_perfect_curve_rules():
    assert segments("P", [(0, 0), (50, 50), (100, 0)]) == [("perfect", [(0, 0), (50, 50), (100, 0)])]
    assert segments("P", [(0, 0), (50, 0), (100, 0)])[0][0] == "linear"            # collinear
    assert segments("P", [(0, 0), (50, 50), (100, 0), (150, 50)])[0][0] == "bezier"  # not 3 points


def test_catmull_does_not_split():
    pts = [(0, 0), (10, 0), (10, 0), (20, 10)]
    assert segments("C", pts) == [("catmull", pts)]


# ---------- path shapes and length ----------

def test_linear_trimmed_and_extended():
    short = build_path("L", [(0, 0), (100, 0)], 50)
    assert short.distance == 50 and short.position_at(1.0) == pytest.approx((50, 0))
    long = build_path("L", [(0, 0), (100, 0)], 150)
    assert long.distance == 150 and long.position_at(1.0) == pytest.approx((150, 0))
    assert long.position_at(0.5) == pytest.approx((75, 0))


def test_perfect_circle_arc():
    arc = circular_arc([(0, 0), (50, 50), (100, 0)])        # half circle, centre (50, 0), radius 50
    assert all(math.dist(p, (50, 0)) == pytest.approx(50) for p in arc)
    path = build_path("P", [(0, 0), (50, 50), (100, 0)], 50 * math.pi)
    assert path.position_at(0.5) == pytest.approx((50, 50), abs=0.1)
    # the chords are a hair shorter than the true arc, so the path is extended straight past the end
    # to reach the .osu length (as in lazer): it overshoots by ~0.1 px
    assert path.position_at(1.0) == pytest.approx((100, 0), abs=0.2)


def test_bezier_symmetric():
    pts = bezier([(0, 0), (50, 100), (100, 0)])
    assert pts[0] == (0, 0) and pts[-1] == (100, 0)
    length = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
    path = build_path("B", [(0, 0), (50, 100), (100, 0)], length)
    assert path.position_at(1.0) == pytest.approx((100, 0))
    assert path.position_at(0.5) == pytest.approx((50, 50), abs=0.5)  # quadratic Bezier peak, by symmetry


# ---------- timing (Slider.ApplyDefaultsToSelf, SliderEventGenerator) ----------

def make_bm(tick_rate=1.0, slides=1, extra_tps=()):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=tick_rate)
    tps = [TimingPoint(0, 500.0, True), TimingPoint(0, -50.0, False), *extra_tps]   # 120 BPM, SV 2.0
    obj = HitObject(x=0, y=100, time=1000, kind="slider", new_combo=True, curve_type="L",
                    curve_points=[(0, 100), (100, 100)], slides=slides, length=100)
    return Beatmap(difficulty=diff, stack_leniency=0.7, timing_points=tps, hit_objects=[obj])


def timing(bm):
    o = bm.hit_objects[0]
    return slider_timing(o, bm, build_path(o.curve_type, o.curve_points, o.length), ControlPoints(bm.timing_points))


def test_velocity_and_duration():
    # SV 2 at 500 ms/beat, SliderMultiplier 1: 200 px per beat = 0.4 px/ms; 100 px takes 250 ms
    t = timing(make_bm())
    assert t.velocity == pytest.approx(0.4)
    assert t.end == pytest.approx(1250)
    assert t.ticks == []                     # one tick per beat = every 200 px: beyond the 100 px slider


def test_ticks_and_repeats():
    # tick rate 4: a tick every 50 px = every 125 ms; 2 spans of 250 ms
    t = timing(make_bm(tick_rate=4, slides=2))
    assert [round(x) for x, _ in t.ticks] == [1125, 1375]
    assert t.repeats == [(pytest.approx(1250), 1.0)]
    assert t.end == pytest.approx(1500)
    assert t.legacy_last_tick == pytest.approx(1464)    # end - 36


def test_short_slider_last_tick_is_half_way():
    bm = make_bm()
    bm.hit_objects[0].length = 20                        # 50 ms long: under 72 ms
    t = timing(bm)
    assert t.legacy_last_tick == pytest.approx(1025)


def test_control_points_at_same_time_inherited_wins():
    cps = ControlPoints([TimingPoint(0, 500.0, True), TimingPoint(1000, -25.0, False), TimingPoint(1000, 400.0, True)])
    assert cps.slider_velocity_at(1000) == (4.0, True)
    assert cps.timing_point_at(1000).beat_length == 400.0
    assert cps.slider_velocity_at(999) == (1.0, True)


def test_nan_beat_length_disables_ticks():
    cps = ControlPoints([TimingPoint(0, 500.0, True), TimingPoint(100, float("nan"), False)])
    assert cps.slider_velocity_at(200) == (1.0, False)


# ---------- the fixture map: osu! says its max combo is 476 ----------

OSU = Path(__file__).parent / "fixtures" / "1262832_3133639837" / "beatmap.osu"


@pytest.mark.skipif(not OSU.exists(), reason="fixture missing: run scripts/check_download.py")
def test_fixture_max_combo_matches_osu():
    pl = make_playable(parse_beatmap(OSU))
    sliders = [o for o in pl.objects if o.kind == "slider"]
    combo = len(pl.objects) + len(sliders) + sum(len(o.timing.ticks) + len(o.timing.repeats) for o in sliders)
    assert combo == 476
    for o in sliders:                        # every path ends exactly at its .osu length
        assert o.path.distance == pytest.approx(pl.beatmap.hit_objects[o.index].length)


# ---------- unusual .osu fields (ConvertHitObjectParser, applyDifficultyRestrictions) ----------

OSU_TEMPLATE = """osu file format v14

[General]
StackLeniency: 0.7

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:8
ApproachRate:9
SliderMultiplier:{sm}
SliderTickRate:{tick_rate}

[TimingPoints]
0,500,4,2,1,100,1,0

[HitObjects]
{slider}
"""


def parse_slider(tmp_path, slider, sm=1.0, tick_rate=1):
    path = tmp_path / "map.osu"
    path.write_text(OSU_TEMPLATE.format(sm=sm, tick_rate=tick_rate, slider=slider))
    return make_playable(parse_beatmap(path)).objects[0]


def test_zero_slides_is_one_span(tmp_path):
    o = parse_slider(tmp_path, "0,0,1000,2,0,L|100:0,0,100")
    assert o.timing.spans == 1 and o.end_time == pytest.approx(1500)    # 100 px at 0.2 px/ms


def test_zero_length_slider_has_no_repeats(tmp_path):
    o = parse_slider(tmp_path, "100,100,1000,2,0,L|100:100,5,0.0001")
    assert o.path.distance == 0
    assert o.timing.spans == 1 and o.timing.repeats == []


def test_missing_or_zero_length_uses_the_curve_length(tmp_path):
    assert parse_slider(tmp_path, "0,0,1000,2,0,L|100:0,1,0").path.distance == pytest.approx(100)
    assert parse_slider(tmp_path, "0,0,1000,2,0,L|100:0,1,-5").path.distance == pytest.approx(100)
    assert parse_slider(tmp_path, "0,0,1000,2,0,L|100:0,1").path.distance == pytest.approx(100)


def test_slider_multiplier_and_tick_rate_are_clamped(tmp_path):
    slider = "0,0,1000,2,0,L|300:0,1,300"
    assert parse_slider(tmp_path, slider, sm=0.3).end_time == pytest.approx(4750)   # as SM 0.4
    assert parse_slider(tmp_path, slider, sm=4.0).end_time == pytest.approx(parse_slider(tmp_path, slider, sm=3.6).end_time)
    ticks = lambda rate: len(parse_slider(tmp_path, slider, tick_rate=rate).timing.ticks)
    assert ticks(0.25) == ticks(0.5) == 1
    assert ticks(10) == ticks(8)
