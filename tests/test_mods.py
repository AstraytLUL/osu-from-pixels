import pytest

from osu_sim.beatmap import Beatmap, Difficulty, HitObject
from osu_sim.mods import DT, EZ, HD, HR, HT, NC, apply_mods, rate


def make_map(od=8.5, cs=4, ar=9):
    diff = Difficulty(hp=6, cs=cs, od=od, ar=ar, slider_multiplier=1.7, slider_tick_rate=1)
    objs = [
        HitObject(x=44, y=136, time=536, kind="circle", new_combo=True),
        HitObject(x=152, y=304, time=899, kind="slider", new_combo=True, curve_type="P",
                  curve_points=[(152, 304), (187, 281), (229, 272)], slides=1, length=85),
    ]
    return Beatmap(difficulty=diff, stack_leniency=0.7, timing_points=[], hit_objects=objs)


def test_nomod_is_unchanged_copy():
    m = make_map()
    out = apply_mods(m, 0)
    assert out == m and out is not m


def test_hr_difficulty_caps_at_10():
    d = apply_mods(make_map(od=8.5, cs=4, ar=9), HR).difficulty
    assert d.od == 10 and d.ar == 10                       # 11.9 and 12.6, capped
    assert d.cs == pytest.approx(5.2)


def test_hr_difficulty_below_cap():
    d = apply_mods(make_map(od=5, cs=3, ar=6), HR).difficulty
    assert d.od == pytest.approx(7.0) and d.ar == pytest.approx(8.4) and d.cs == pytest.approx(3.9)


def test_hr_flips_vertically():
    m = apply_mods(make_map(), HR)
    assert (m.hit_objects[0].x, m.hit_objects[0].y) == (44, 384 - 136)
    assert m.hit_objects[1].curve_points == [(152, 80), (187, 103), (229, 112)]


def test_input_not_modified():
    m = make_map()
    apply_mods(m, HR | EZ)
    assert m.hit_objects[0].y == 136 and m.difficulty.od == 8.5


def test_ez_halves():
    d = apply_mods(make_map(od=8, cs=4, ar=9), EZ).difficulty
    assert (d.od, d.cs, d.ar) == (4, 2, 4.5)


def test_dt_and_hd_do_not_change_the_map():
    # replay times are map time, so DT leaves the judge's view of the map alone; HD is visual only
    m = make_map()
    assert apply_mods(m, DT | HD) == m


@pytest.mark.parametrize("mods, expected", [(0, 1.0), (DT, 1.5), (DT | NC, 1.5), (HT, 0.75), (HR | HD, 1.0)])
def test_rate(mods, expected):
    assert rate(mods) == expected
