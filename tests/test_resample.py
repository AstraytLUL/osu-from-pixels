from pathlib import Path

import numpy as np
import pytest

from data.replay import ReplayFrames, load_replay
from data.resample import flip_vertical, grid_step, map_grid, resample
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint, parse_beatmap
from osu_sim.judge import accuracy, judge
from osu_sim.mods import DT, NC
from osu_sim.playable import make_playable


def replay(rows, mods=0):
    """rows: (time, x, y, keys)."""
    t, x, y, k = map(np.array, zip(*rows))
    return ReplayFrames(t=t.astype(float), x=x.astype(float), y=y.astype(float), keys=k.astype(int), mods=mods)


def test_grid_step_is_60fps_real_time():
    assert grid_step(0) == pytest.approx(1000 / 60)
    assert grid_step(DT) == pytest.approx(25.0)
    assert grid_step(DT | NC) == pytest.approx(25.0)


def test_map_grid_covers_the_map():
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1, slider_tick_rate=1)
    bm = Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)],
                 [HitObject(100, 100, 3000, "circle", True), HitObject(200, 100, 5000, "circle", False)])
    times = map_grid(make_playable(bm))
    assert times[0] == 1000 and times[-1] >= 5500
    assert np.allclose(np.diff(times), 1000 / 60)


def test_cursor_is_interpolated():
    g = resample(replay([(0, 0, 0, 0), (100, 100, 50, 0)]), np.array([0.0, 25.0, 50.0]))
    assert np.allclose(g.x, [0, 25, 50]) and np.allclose(g.y, [0, 12.5, 25])


def test_short_tap_between_grid_points_is_kept():
    # left down 3..8 ms only; grid at 0 and 16.67: frame 0's window is [-8.3, 8.3)
    rp = replay([(0, 0, 0, 0), (3, 0, 0, 1), (8, 0, 0, 0), (40, 0, 0, 0)])
    g = resample(rp, np.array([0.0, 1000 / 60, 2000 / 60]))
    assert list(g.press) == [1, 0, 0]
    assert list(g.held) == [0, 0, 0]                 # already released when the window ends


def test_quick_repress_is_two_clicks():
    # left down 0-20, up for 5 ms, down again 25-60; a held-only grid would see one long hold
    rp = replay([(0, 0, 0, 1), (20, 0, 0, 0), (25, 0, 0, 1), (60, 0, 0, 0), (100, 0, 0, 0)])
    step = 1000 / 60
    g = resample(rp, step * np.arange(6))
    assert list(g.press[:3]) == [1, 0, 1]           # the re-press at 25 falls in frame 2's window [25, 41.7)
    f = g.as_frames()
    down = f.keys & 1
    presses = np.flatnonzero(down & ~np.concatenate([[0], down[:-1]]))
    assert len(presses) == 2                        # a release frame was inserted before frame 2


def test_a_press_on_a_window_boundary_counts_once():
    # 127846 (9 Nov): Auto's click at 131860 fell in two windows of the map grid after rounding, so
    # it clicked twice and every later note of the stream was hit early
    times = 42485.0 + 1000 / 60 * np.arange(6000)                # that map's grid (map_grid)
    rp = replay([(131700, 0, 0, 0), (131860, 0, 0, 2), (131900, 0, 0, 0), (132000, 0, 0, 0)])
    assert resample(rp, times).press.sum() == 2                  # one frame with key 2 (bit value 2), not two


def test_cursor_on_press_frames_is_where_the_player_clicked():
    # cursor moves 1 px per ms; the click is at 5 ms, inside frame 0's window
    rp = replay([(0, 0, 0, 0), (5, 5, 0, 1), (30, 30, 0, 0), (60, 60, 0, 0)])
    g = resample(rp, np.array([0.0, 1000 / 60, 2000 / 60]))
    assert g.x[0] == pytest.approx(5.0)              # not 0: the click position
    assert g.x[1] == pytest.approx(1000 / 60)        # no press: interpolated as usual


def test_flip_vertical():
    g = flip_vertical(resample(replay([(0, 10, 100, 0), (50, 10, 100, 0)]), np.array([0.0])))
    assert g.y[0] == 284


FIX = Path(__file__).parent / "fixtures" / "1262832_3133639837"


@pytest.mark.skipif(not (FIX / "replay.osr").exists(), reason="fixture missing: run scripts/check_download.py")
def test_grid_replay_judges_close_to_the_original():
    rp = load_replay(FIX / "replay.osr")
    pl = make_playable(parse_beatmap(FIX / "beatmap.osu"), rp.mods)
    g = resample(rp, map_grid(pl))
    assert accuracy(judge(pl, g.as_frames())) == pytest.approx(accuracy(judge(pl, rp)), abs=0.01)
