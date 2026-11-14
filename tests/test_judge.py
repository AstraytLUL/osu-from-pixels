from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from data.replay import ReplayFrames, load_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint, parse_beatmap
from osu_sim.judge import accuracy, circle_radius, counts, hit_windows, judge, spinner_result, unstable_rate
from osu_sim.mods import DT, HD, HR
from osu_sim.playable import make_playable

# ---------- rules, on hand-built maps (OD 8.5 -> windows ±29 / ±72 / ±115 ms; CS 4 -> radius 36.49) ----------


def make_map(*objects):
    """objects: (x, y, time) for a circle; (x, y, time, "slider") for a 100 px slider to the right
    (lasts 500 ms: 100 BPM-free beat of 500 ms, SliderMultiplier 1 -> 100 px per beat)."""
    diff = Difficulty(hp=5, cs=4, od=8.5, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = []
    for o in objects:
        if len(o) > 3 and o[3] == "slider":
            objs.append(HitObject(x=o[0], y=o[1], time=o[2], kind="slider", new_combo=False, curve_type="L",
                                  curve_points=[(o[0], o[1]), (o[0] + 100, o[1])], slides=1, length=100))
        else:
            objs.append(HitObject(x=o[0], y=o[1], time=o[2], kind="circle", new_combo=False))
    bm = Beatmap(difficulty=diff, stack_leniency=0.7, timing_points=[TimingPoint(0, 500.0, True)], hit_objects=objs)
    return make_playable(bm)


def frames(rows):
    """rows: (time, x, y, keys)."""
    t, x, y, k = map(np.array, zip(*sorted(rows)))
    return ReplayFrames(t=t.astype(float), x=x.astype(float), y=y.astype(float), keys=k.astype(int), mods=0)


def clicks(*events):
    """events: (time, x, y). Each click presses a key for 10 ms; keys alternate left / right like a
    player's, so clicks closer than 10 ms apart do not merge into one press."""
    pos = {}
    for t, x, y in events:
        pos[t] = pos[t + 10] = (x, y)          # the cursor sits on each click until its release
    rows = []
    for tt in sorted(pos):
        k = 0
        for i, (t, _, _) in enumerate(events):
            if t <= tt < t + 10:
                k |= 1 if i % 2 == 0 else 2
        rows.append((tt, *pos[tt], k))
    return frames(rows)


def results(js):
    return [j.result for j in js]


def test_windows_and_radius():
    w = hit_windows(8.5)
    assert (w.w300, w.w100, w.w50) == (29, 72, 115)
    assert circle_radius(4) == pytest.approx(36.48 * 1.00041)
    assert make_map((100, 100, 1000)).radius == pytest.approx(circle_radius(4), abs=1e-6)


@pytest.mark.parametrize("offset, expected", [
    (0, 300),
    (28, 300), (-28, 300),     # inside: |error| < 29
    (29, 100), (-29, 100),     # the edge itself is outside (wiki: hit error < max hit error)
    (71, 100), (72, 50),
    (114, 50), (-114, 50),
    (-115, 0),                 # early, outside the 50 window but inside the 400 MISS window -> miss
    (-399, 0),
    # fractional times (a policy on the 60 fps grid): lazer's bound is |error| <= 29 - 0.5
    (28.5, 300), (28.75, 100), (-28.75, 100),
])
def test_window_edges(offset, expected):
    js = judge(make_map((100, 100, 1000)), clicks((1000 + offset, 100, 100)))
    assert results(js) == [expected]


def test_click_on_the_expiry_frame_is_a_late_miss():
    # lazer handles input before updating: a click exactly +115 still reaches circle 1 and is
    # judged a (late) miss there, so it does not go on to circle 2
    js = judge(make_map((100, 100, 1000), (100, 100, 1100)), clicks((1115, 100, 100)))
    assert results(js) == [0, 0]
    assert js[0].offset == 115 and js[1].offset is None


def test_fractional_offset_is_kept():
    js = judge(make_map((100, 100, 1000)), clicks((1028.75, 100, 100)))
    assert js[0].offset == 28.75


def test_holding_a_key_is_not_a_click():
    # left held down the whole time: only the press at 0 counts, which is too early for every circle
    rows = [(t, x, 100, 1) for t, x in ((0, 100), (1000, 100), (1500, 200), (2000, 300), (2600, 300))]
    js = judge(make_map((100, 100, 1000), (200, 100, 1500), (300, 100, 2000)), frames(rows))
    assert results(js) == [0, 0, 0]


def test_never_clicked_is_miss():
    js = judge(make_map((100, 100, 1000), (300, 300, 2000)), clicks((2000, 300, 300)))
    assert results(js) == [0, 300]
    assert js[0].offset is None


@pytest.mark.parametrize("offset", [-400, -401, -1000])
def test_click_before_miss_window_does_nothing(offset):
    js = judge(make_map((100, 100, 1000)), clicks((1000 + offset, 100, 100), (1000, 100, 100)))
    assert results(js) == [300]


def test_cursor_must_be_on_circle():
    m = make_map((100, 100, 1000))
    r = m.radius
    assert results(judge(m, clicks((1000, 100 + r - 0.1, 100)))) == [300]   # just inside
    assert results(judge(m, clicks((1000, 100 + r + 0.1, 100)))) == [0]     # just outside -> never hit


def test_note_lock():
    # two circles 50 ms apart; the player clicks the second one first, then the first one
    m = make_map((100, 100, 1000), (300, 300, 1050))
    js = judge(m, clicks((1040, 300, 300), (1045, 100, 100), (1060, 300, 300)))
    # click 1 is on circle 2 while circle 1 is unjudged and earlier -> shake, nothing
    # click 2 hits circle 1 (+45 -> 100); click 3 hits circle 2 (+10 -> 300)
    assert results(js) == [100, 300]


def test_simultaneous_objects_do_not_lock():
    # within 3 ms of each other: either can be hit first (LegacyHitPolicy leniency)
    m = make_map((100, 100, 1000), (300, 300, 1002))
    assert results(judge(m, clicks((1002, 300, 300), (1003, 100, 100)))) == [300, 300]


def test_one_click_hits_one_circle():
    m = make_map((100, 100, 1000), (100, 100, 1010))       # stacked on the same spot
    assert results(judge(m, clicks((1005, 100, 100)))) == [300, 0]


def test_both_keys_on_one_frame_are_two_clicks():
    m = make_map((100, 100, 1000), (100, 100, 1010))
    js = judge(m, frames([(1005, 100, 100, 3), (1020, 100, 100, 0)]))
    assert results(js) == [300, 300]


def test_accuracy_and_ur():
    js = judge(make_map((100, 100, 1000), (100, 100, 2000)), clicks((1010, 100, 100), (2050, 100, 100)))
    assert accuracy(js) == pytest.approx(400 / 600)
    assert unstable_rate(js) == pytest.approx(10 * np.std([10, 50]))


# ---------- sliders: 100 px to the right over 500 ms; parts = head + tail (no ticks: tick rate 1 = 1 per beat
# = 1 per slider, which falls on the end and is dropped) ----------

def follow(start, x0, y0, t0, t1, keys=1, step=10):
    """Frames following the slider ball from t0 to t1 holding `keys`."""
    return [(t, x0 + 100 * (t - start) / 500, y0, keys) for t in range(t0, t1 + 1, step)]


def test_slider_held_to_the_end_is_300_even_with_a_late_head():
    m = make_map((100, 100, 1000, "slider"))
    rows = [(990, 100, 100, 0)] + follow(1000, 100, 100, 1100, 1520) + [(1530, 200, 100, 0)]
    js = judge(m, frames(rows))
    # head at +100 would be a 50 on a circle; for a slider any hit inside the 50 window is a full hit
    assert (js[0].result, js[0].parts_hit, js[0].parts_total, js[0].offset) == (300, 2, 2, 100)


def test_slider_released_early_is_100():
    m = make_map((100, 100, 1000, "slider"))
    rows = [(990, 100, 100, 0)] + follow(1000, 100, 100, 1000, 1200) + [(1210, 140, 100, 0), (1600, 200, 100, 0)]
    js = judge(m, frames(rows))
    assert (js[0].result, js[0].parts_hit) == (100, 1)     # 1 of 2 parts = 50% -> 100


def test_slider_end_needs_tracking_36ms_before_the_end():
    m = make_map((100, 100, 1000, "slider"))
    # release at 1470: the end check is at 1500 - 36 = 1464, so this still counts
    ok = [(990, 100, 100, 0)] + follow(1000, 100, 100, 1000, 1460) + [(1470, 194, 100, 0), (1600, 200, 100, 0)]
    assert judge(m, frames(ok))[0].result == 300
    # release at 1460: before the check
    early = [(990, 100, 100, 0)] + follow(1000, 100, 100, 1000, 1450) + [(1460, 192, 100, 0), (1600, 200, 100, 0)]
    assert judge(m, frames(early))[0].result == 100


def test_slider_head_never_clicked_but_followed():
    # the head is missed, but holding a key on the ball still collects the tail: 1 of 2 parts -> 100
    m = make_map((100, 100, 1000, "slider"))
    rows = follow(1000, 100, 100, 900, 1520, keys=1)
    # the key goes down at 300 ms, before the slider is even on screen (preempt 600): that press does
    # nothing, and the key is then held through the head without a new press
    js = judge(m, frames([(300, 100, 100, 1)] + rows + [(1530, 200, 100, 0)]))
    assert (js[0].result, js[0].parts_hit, js[0].offset) == (100, 1, None)


def test_slider_tracking_lost_outside_follow_circle():
    m = make_map((100, 100, 1000, "slider"))
    r = m.radius * 2.4
    rows = [(990, 100, 100, 0)] + follow(1000, 100, 100, 1000, 1300)
    rows += [(t, 100 + 100 * (t - 1000) / 500, 100 + r + 5, 1) for t in range(1310, 1521, 10)]   # drifted off
    rows += [(1530, 200, 100, 0)]
    assert judge(m, frames(rows))[0].parts_hit == 1


# ---------- spinners ----------

def test_spinner_requirements():
    # OD 8.5: 2.5 + 0.5 * 8.5 = 6.75 half spins per second; a 2 s spinner needs int(13.5) + 1 = 14 half spins = 7 spins
    assert spinner_result(7.0, 2000, 8.5) == 300
    assert spinner_result(6.0, 2000, 8.5) == 100           # one spin short
    assert spinner_result(2.0, 2000, 8.5) == 50            # >= 25%
    assert spinner_result(1.0, 2000, 8.5) == 0
    assert spinner_result(0.0, 100, 8.5) == 300            # too short to require anything


# ---------- real replays ----------

FIX = Path(__file__).parent / "fixtures" / "1262832_3133639837"
FIX_MODDED = Path(__file__).parent / "fixtures" / "1262832_7613193930"
needs = pytest.mark.skipif(not (FIX / "replay.osr").exists() or not (FIX_MODDED / "replay.osr").exists(),
                           reason="fixtures missing: run scripts/check_download.py")


@needs
def test_nomod_ss_fixture():
    # Hitorigoto -TV MIX- [Soliloquy], a 340/340 SS
    replay = load_replay(FIX / "replay.osr")
    js = judge(make_playable(parse_beatmap(FIX / "beatmap.osu"), replay.mods), replay)
    assert Counter((j.kind, j.result) for j in js) == {("circle", 300): 207, ("slider", 300): 132, ("spinner", 300): 1}
    assert accuracy(js) == 1.0
    assert 50 < unstable_rate(js) < 150       # a top player; the measured value is 84.9


@needs
def test_dthrhd_fixture_matches_leaderboard():
    # same map, DT+HR+HD, 99.80%: the leaderboard says 339 x 300 and 1 x 100
    replay = load_replay(FIX_MODDED / "replay.osr")
    assert replay.mods == HD | HR | DT
    js = judge(make_playable(parse_beatmap(FIX_MODDED / "beatmap.osu"), replay.mods), replay)
    assert counts(js) == {300: 339, 100: 1, 50: 0, 0: 0}
    hundred = next(j for j in js if j.result == 100)
    # HR: OD 8.5 x 1.4 -> 10 (capped) -> windows ±20 / ±60 / ±100. This circle was hit 22 ms late.
    assert (hundred.kind, hundred.time, hundred.offset) == ("circle", 58354, 22)
    assert accuracy(js) == pytest.approx(0.998039, abs=1e-6)
