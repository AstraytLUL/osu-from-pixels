from pathlib import Path

import pytest

from osu_sim.beatmap import parse_beatmap

# Hitorigoto -TV MIX- [Soliloquy]. The .osu is gitignored: run scripts/check_download.py to fetch it.
OSU = Path(__file__).parent / "fixtures" / "1262832_3133639837" / "beatmap.osu"
pytestmark = pytest.mark.skipif(not OSU.exists(), reason="fixture missing: run scripts/check_download.py")


@pytest.fixture(scope="module")
def bm():
    return parse_beatmap(OSU)


def test_difficulty(bm):
    assert bm.difficulty.cs == 4
    assert bm.difficulty.od == 8.5
    assert bm.difficulty.ar == 9
    assert bm.difficulty.slider_multiplier == 1.7


def test_stack_leniency(bm):
    assert bm.stack_leniency == 0.7


def test_object_counts(bm):
    kinds = [o.kind for o in bm.hit_objects]
    assert kinds.count("circle") == 207
    assert kinds.count("slider") == 132
    assert kinds.count("spinner") == 1
    assert len(kinds) == 340                     # matches the replay's 340 judged objects


def test_first_circle(bm):
    # line 69: 44,136,536,5,4,1:2:0:0:   (type 5 = circle + new combo)
    o = bm.hit_objects[0]
    assert (o.kind, o.x, o.y, o.time, o.new_combo) == ("circle", 44, 136, 536, True)


def test_plain_circle(bm):
    # line 70: 315,196,717,1,0,1:0:0:0:   (type 1 = circle, no new combo)
    o = bm.hit_objects[1]
    assert (o.kind, o.x, o.y, o.time, o.new_combo) == ("circle", 315, 196, 717, False)


def test_first_slider(bm):
    # line 71: 152,304,899,6,0,P|187:281|229:272,1,85   (no trailing sound fields)
    o = bm.hit_objects[2]
    assert (o.kind, o.time, o.new_combo) == ("slider", 899, True)
    assert o.curve_type == "P"
    assert o.curve_points == [(152, 304), (187, 281), (229, 272)]
    assert (o.slides, o.length) == (1, 85)


def test_bezier_slider_keeps_repeated_point(bm):
    # line 90: 108,49,5869,2,0,B|147:51|147:51|187:79,1,85,...   (repeated point = sharp corner)
    o = next(o for o in bm.hit_objects if o.time == 5869)
    assert o.curve_type == "B"
    assert o.curve_points == [(108, 49), (147, 51), (147, 51), (187, 79)]


def test_spinner(bm):
    # line 408: 256,192,86445,12,4,87717,1:2:0:0:   (type 12 = spinner + new combo)
    o = bm.hit_objects[-1]
    assert (o.kind, o.time, o.end_time) == ("spinner", 86445, 87717)


def test_timing_points(bm):
    tps = bm.timing_points
    assert len(tps) == 8
    assert tps[0].time == 536 and tps[0].uninherited
    assert tps[0].beat_length == pytest.approx(363.636363636364)     # 165 BPM
    assert not tps[1].uninherited and tps[1].beat_length == -100     # slider speed 1.0x
