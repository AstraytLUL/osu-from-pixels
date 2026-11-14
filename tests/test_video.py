import pytest
from PIL import Image

from eval.video import render_video, slider_ticks
from osu_sim.auto import auto_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.playable import make_playable
from osu_sim.renderer import Renderer


def small_map():
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(100, 100, 2000, "circle", True),
            HitObject(300, 120, 2400, "slider", False, curve_type="L", curve_points=[(300, 120), (400, 120)],
                      slides=1, length=100)]
    return make_playable(Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], objs))


def test_gif_of_auto(tmp_path):
    pl = small_map()
    out = tmp_path / "auto.gif"
    n = render_video(pl, auto_replay(pl), out, "Auto")
    assert n == {300: 2, 100: 0, 50: 0, 0: 0}
    gif = Image.open(out)
    assert gif.size == (512, 384 + 48) and gif.n_frames > 10


def test_mp4_clip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pl = small_map()
    out = tmp_path / "clip.mp4"
    n = render_video(pl, auto_replay(pl), out, "Auto", start_ms=1500, end_ms=2100)
    assert out.stat().st_size > 0
    assert n == {300: 1, 100: 0, 50: 0, 0: 0}         # only the circle is judged before the clip ends


def test_slider_ticks_of_the_current_span_until_the_ball_passes():
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    # 300 px at 100 px per beat: 1500 ms per span, ticks a beat (500 ms) apart; two spans (one reverse)
    slider = HitObject(100, 200, 2400, "slider", True, curve_type="L", curve_points=[(100, 200), (400, 200)],
                       slides=2, length=300)
    renderer = Renderer(make_playable(Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], [slider])))
    shown = lambda t: sorted(round(x) for x, _, _ in slider_ticks(renderer, t))
    assert shown(2300) == [200, 300]           # before the start: the first span's ticks
    assert shown(3000) == [300]                # the ball has passed the first tick
    assert shown(3800) == []                   # end of the first span
    assert shown(4000) == [200, 300]           # coming back: the second span's ticks
    assert shown(5000) == []                   # the slider is over
