import numpy as np

from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.mods import HD, HR
from osu_sim.playable import make_playable
from osu_sim.renderer import CURSOR_RADIUS, GX, GY, H, PX, W, Renderer, _soft, _to_u8, add_cursor


def one_circle(mods=0):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(x=256, y=192, time=2000, kind="circle", new_combo=True),
            HitObject(x=100, y=100, time=3000, kind="slider", new_combo=True, curve_type="L",
                      curve_points=[(100, 100), (300, 100)], slides=1, length=200)]
    bm = Beatmap(difficulty=diff, stack_leniency=0.7, timing_points=[TimingPoint(0, 500.0, True)], hit_objects=objs)
    return make_playable(bm, mods)


def px(x, y):
    """Rendered pixel (row, col) containing osu! position (x, y)."""
    return int(y // PX), int(x // PX)


def test_shape_and_type():
    f = Renderer(one_circle()).frame(1900)
    assert f.shape == (H, W) == (96, 128) and f.dtype == np.uint8


def test_deterministic():
    r = Renderer(one_circle())
    assert np.array_equal(r.frame(1900, (10, 10)), r.frame(1900, (10, 10)))


def test_blank_before_preempt_and_after_hit():
    r = Renderer(one_circle())                  # AR 9: preempt 600, so the circle appears at 1400
    assert r.frame(1300).max() == 0
    assert r.frame(1950)[px(256 + 30, 192)] > 0   # its ring
    assert r.frame(2050)[px(256, 192)] == 0       # gone after its time (no hit_times given)


def test_hit_times_keep_a_late_circle_on_screen():
    r = Renderer(one_circle(), hit_times={0: 2080})
    assert r.frame(2050)[px(256, 192)] > 0


def test_approach_circle_only_without_hidden():
    t = 2000 - 300                               # approach circle at 1 + 3 * 300/600 = 2.5x the radius
    ring = px(256 + 2.5 * 36.5, 192)
    assert Renderer(one_circle()).frame(t)[ring] > 20        # half faded in at this point
    assert Renderer(one_circle(HD)).frame(t)[ring] == 0


def test_hidden_fades_out_before_the_hit():
    # HD: fade in over 0.4 x 600 = 240 ms from 1400, then out over 180 ms: gone by 1820
    r = Renderer(one_circle(HD))
    assert r.frame(1640)[px(256, 192)] > 0
    assert r.frame(1850)[px(256, 192)] == 0


def test_hr_flips_the_slider():
    f = Renderer(one_circle(HR)).frame(3100)
    assert f[px(200, 384 - 100)] > 0 and f[px(200, 100)] == 0


def test_cursor_visible_on_slider_ball():
    r = Renderer(one_circle())
    ball = (100 + 200 * 0.5, 100)                # half way along the 200 px slider (velocity 0.2 px/ms, 1000 ms)
    f = r.frame(3500, ball)
    row, col = px(*ball)
    assert f[row, col] == 255                    # the cursor dot is brighter than the ball (0.6 = 153)
    assert f[row, col + 4] < 200                 # the ball, further out
    assert f[row - 2:row + 3, col - 2:col + 3].min() < 60   # a dark ring around the dot cuts through the ball


def test_reverse_marker_shows_before_the_slider_starts():
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    kick = HitObject(x=100, y=100, time=3000, kind="slider", new_combo=True, curve_type="L",
                     curve_points=[(100, 100), (160, 100)], slides=3, length=60)    # 300 ms spans
    pl = make_playable(Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], [kick]))
    r = Renderer(pl)
    row, col = px(160, 100)                      # the path's end, where the first reverse happens
    before = r.frame(2900)                       # 100 ms before the slider starts
    assert before[row, col] > 150                # the marker (the body alone is dim)
    after_reverses = r.frame(3000 + 2.5 * 300)   # last span: no reverse left to show
    assert after_reverses[row, col] < 150


def test_cursor_patch_matches_drawing_the_whole_frame():
    rng = np.random.default_rng(0)
    layers = rng.integers(0, 256, (40, H, W), dtype=np.uint8)
    cursors = np.concatenate([rng.uniform(-30, 540, (36, 2)),
                              [[0, 0], [512, 384], [-500, 100], [3.9, 383.2]]])
    d = np.hypot(GX[None] - cursors[:, 0, None, None], GY[None] - cursors[:, 1, None, None])
    img = layers / 255.0 * (1 - _soft(d - (CURSOR_RADIUS + PX)))
    whole = _to_u8(np.maximum(img, _soft(d - CURSOR_RADIUS)))   # the old formula, on whole frames
    assert np.array_equal(add_cursor(layers, cursors), whole)
