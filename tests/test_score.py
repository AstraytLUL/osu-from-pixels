from pathlib import Path

import pytest
from osrparse import Replay

from data.replay import load_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint, parse_beatmap
from osu_sim.judge import Judgement, judge
from osu_sim.mods import DT, HD, HR, NC
from osu_sim.playable import make_playable
from osu_sim.score import _events, combo_breaks, difficulty_multiplier, max_combo, mod_multiplier, score_v1


def three_circles():
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(100, 100, t, "circle", False) for t in (1000, 2000, 3000)]
    bm = Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], objs)
    return bm, make_playable(bm)


def circles(*results):
    return [Judgement(i, "circle", 1000 * (i + 1), r, 0.0 if r else None) for i, r in enumerate(results)]


def test_combo_breaks_on_a_miss_not_on_a_100():
    _, pl = three_circles()
    assert max_combo(circles(300, 100, 300), pl) == 3
    assert max_combo(circles(300, 0, 300), pl) == 1


def test_missed_slider_end_does_not_break_combo():
    _, pl = three_circles()
    slider = Judgement(1, "slider", 2000, 100, 0.0, parts=[(2200, "tick", True), (2464, "tail", False)])
    js = [circles(300)[0], slider, Judgement(2, "circle", 3000, 300, 0.0)]
    assert max_combo(js, pl) == 4                  # circle, head, tick, circle; the end adds nothing
    slider.parts[0] = (2200, "tick", False)        # a missed tick does break it
    assert max_combo(js, pl) == 2


def test_combo_breaks_by_cause():
    _, pl = three_circles()
    slider = Judgement(1, "slider", 2000, 50, None, parts=[(2200, "tick", False), (2300, "repeat", True),
                                                           (2464, "tail", False)])
    js = [circles(0)[0], slider, Judgement(2, "spinner", 3000, 0)]
    assert combo_breaks(js) == {"miss": 3, "slider_break": 1, "end_drop": 1}   # circle, head, spinner
    assert combo_breaks(circles(300, 100, 50)) == {"miss": 0, "slider_break": 0, "end_drop": 0}


@pytest.mark.skipif(not (Path(__file__).parent / "fixtures" / "1262832_3133639837" / "replay.osr").exists(),
                    reason="fixture missing: run scripts/check_download.py")
def test_combo_breaks_match_the_break_events_on_a_real_play():
    frames = load_replay(FIX / "replay.osr")         # a full combo: knock the cursor off in its middle third
    n = len(frames.t)
    frames.x[n // 3: 2 * n // 3] += 80
    pl = make_playable(parse_beatmap(FIX / "beatmap.osu"), frames.mods)
    js = judge(pl, frames)
    b = combo_breaks(js)
    assert b["miss"] > 0 and b["slider_break"] > 0
    assert b["miss"] + b["slider_break"] == sum(e.combo == "break" for e in _events(js, pl))


def test_score_combo_bonus():
    bm, pl = three_circles()
    dm = difficulty_multiplier(bm)
    # 300 + 300 + 300 * (1 * dm / 25): the third circle is the first with a combo bonus
    assert score_v1(circles(300, 300, 300), pl, bm) == 900 + int(300 * dm / 25)


def test_mod_multipliers():
    assert mod_multiplier(HD | HR) == pytest.approx(1.06 * 1.06)
    assert mod_multiplier(DT | NC) == mod_multiplier(NC) == pytest.approx(1.12)


FIX = Path(__file__).parent / "fixtures" / "1262832_3133639837"


@pytest.mark.skipif(not (FIX / "replay.osr").exists(), reason="fixture missing: run scripts/check_download.py")
def test_fixture_max_combo_matches_osu():
    # a lazer replay: its header score is lazer's standardised score (~1M), not ScoreV1, so only the
    # combo is comparable (ScoreV1 was checked against 422 stable replays: exact without spinners)
    header = Replay.from_path(str(FIX / "replay.osr"))
    frames = load_replay(FIX / "replay.osr")
    pl = make_playable(parse_beatmap(FIX / "beatmap.osu"), frames.mods)
    assert max_combo(judge(pl, frames), pl) == header.max_combo
