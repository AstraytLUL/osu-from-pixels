from collections import Counter
from pathlib import Path

import pytest

from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint, parse_beatmap
from osu_sim.judge import circle_radius
from osu_sim.mods import HR
from osu_sim.playable import make_playable, preempt_ms, scale_from_cs


@pytest.mark.parametrize("ar, ms", [(0, 1800), (5, 1200), (9, 600), (10, 450),
                                     (9.3, 554)])   # 1200 - 150 * 4.3 = 554.99997 in floating point, truncated
def test_preempt(ar, ms):
    assert preempt_ms(ar) == ms


def test_scale_matches_radius():
    for cs in (2, 4, 5.2, 7):
        assert 64 * scale_from_cs(cs) == pytest.approx(circle_radius(cs))


def circles(*pts, leniency=0.7):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.4, slider_tick_rate=1)
    objs = [HitObject(x=x, y=y, time=t, kind="circle", new_combo=False) for x, y, t in pts]
    return Beatmap(difficulty=diff, stack_leniency=leniency, timing_points=[TimingPoint(0, 500.0, True)], hit_objects=objs)


def test_stack_of_three():
    pl = make_playable(circles((100, 100, 1000), (100, 100, 1100), (100, 100, 1200)))
    assert [o.stack_height for o in pl.objects] == [2, 1, 0]
    step = pl.scale * -6.4
    assert [(o.x, o.y) for o in pl.objects] == [pytest.approx((100 + 2 * step, 100 + 2 * step)),
                                                 pytest.approx((100 + step, 100 + step)), (100, 100)]


def test_no_stack_when_too_far_apart_in_time():
    # threshold = preempt (600) x leniency (0.7) = 420 ms
    pl = make_playable(circles((100, 100, 1000), (100, 100, 1500)))
    assert [o.stack_height for o in pl.objects] == [0, 0]


def test_no_stack_when_leniency_zero():
    pl = make_playable(circles((100, 100, 1000), (100, 100, 1100), leniency=0))
    assert [o.stack_height for o in pl.objects] == [0, 0]


def test_hr_stacks_the_flipped_positions():
    pl = make_playable(circles((100, 100, 1000), (100, 100, 1100)), HR)
    step = pl.scale * -6.4
    assert (pl.objects[0].x, pl.objects[0].y) == pytest.approx((100 + step, 284 + step))


OSU = Path(__file__).parent / "fixtures" / "1262832_3133639837" / "beatmap.osu"


@pytest.mark.skipif(not OSU.exists(), reason="fixture missing: run scripts/check_download.py")
def test_fixture_stacks():
    pl = make_playable(parse_beatmap(OSU))
    assert Counter(o.stack_height for o in pl.objects) == {0: 316, 1: 14, 2: 10}
    assert (pl.preempt, round(pl.radius, 3)) == (600, 36.495)
