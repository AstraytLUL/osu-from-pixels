import numpy as np
import pytest

from data.dataset import MapFrames, ReplayWindows, build_map_frames, mods_from_str, visual_mods
from osu_sim.auto import auto_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.mods import DT, HD, HR, NC
from osu_sim.playable import make_playable
from osu_sim.renderer import Renderer, add_cursor


def small_map(mods=0):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(100, 100, 2000, "circle", True),
            HitObject(300, 120, 2400, "slider", False, curve_type="L", curve_points=[(300, 120), (400, 120)],
                      slides=1, length=100),
            HitObject(200, 300, 3200, "circle", False)]
    bm = Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], objs)
    return make_playable(bm, mods)


def windows(tmp_path, mods=0, canonical_hr=False, window=20):
    pl = small_map(mods)
    return ReplayWindows([(42, pl, auto_replay(pl))], window=window, canonical_hr=canonical_hr, root=tmp_path), pl


def test_mod_helpers():
    assert mods_from_str("NM") == 0 and mods_from_str("HDHR") == HD | HR
    assert visual_mods(DT | NC | 1) == DT              # NF does not change frames; NC counts as DT


def test_cache_round_trip_matches_the_renderer(tmp_path):
    pl = small_map()
    mf = MapFrames(build_map_frames(pl, 42, tmp_path))
    r = Renderer(pl)
    for k in (0, len(mf) // 3, len(mf) // 2, len(mf) - 1):
        assert np.array_equal(mf.frame(k), r.map_layer(mf.times[k]))


def test_sample_is_byte_identical_to_rendering_directly(tmp_path):
    ds, pl = windows(tmp_path)
    s = ds[len(ds) // 2]
    k = int(ds.idx_k[len(ds) // 2])
    g = ds.replays[0][2]
    assert s["frames"].shape == (20, 96, 128) and s["frames"].dtype == np.uint8
    assert np.array_equal(s["frames"][-1], Renderer(pl).frame(g.t[k], (g.x[k], g.y[k])))
    assert np.array_equal(s["frames"][0], Renderer(pl).frame(g.t[k - 19], (g.x[k - 19], g.y[k - 19])))


def test_targets_are_the_next_frame(tmp_path):
    ds, _ = windows(tmp_path)
    g = ds.replays[0][2]
    for i in range(0, len(ds), max(1, len(ds) // 7)):
        s, k = ds[i], int(ds.idx_k[i])
        assert np.allclose(s["action"], [g.x[k + 1] - g.x[k], g.y[k + 1] - g.y[k]])
        assert s["press"].tolist() == [g.press[k + 1] & 1, (g.press[k + 1] >> 1) & 1]
        assert s["held"].tolist() == [g.held[k + 1] & 1, (g.held[k + 1] >> 1) & 1]


def test_windows_cover_only_the_active_part(tmp_path):
    ds, pl = windows(tmp_path)
    times = ds.replays[0][3].times
    t = times[ds.idx_k]
    assert ds.idx_k.min() >= 19 and ds.idx_k.max() < len(times) - 1
    assert t.min() >= pl.objects[0].time - pl.preempt and t.max() <= pl.objects[-1].end_time
    # Auto presses once per object: three presses across all targets
    assert sum(ds[i]["press"].sum() for i in range(len(ds))) == 3


def test_cursor_jitter_targets_lead_back_to_the_player(tmp_path):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path, cursor_jitter=40)
    g = ds.replays[0][2]
    shifted = 0
    for i in range(0, len(ds), 3):
        s, k = ds[i], int(ds.idx_k[i])
        # wherever the cursor was drawn, cursor + move = where the player went next
        assert np.allclose(s["cursor"] + s["action"], [g.x[k + 1], g.y[k + 1]], atol=1e-3)
        on_track = np.allclose(s["cursor"], [g.x[k], g.y[k]], atol=1e-3)
        shifted += not on_track
        assert np.abs(s["cursor"] - [g.x[k], g.y[k]]).max() <= 40 + 1e-3
        # the frames show the shifted cursor: identical to drawing it there directly
        layer = ds.replays[0][3].window(k, 20)[-1:]
        assert np.array_equal(s["frames"][-1:], add_cursor(layer, s["cursor"][None].astype(float)))
    assert 0 < shifted < len(range(0, len(ds), 3))          # about half the samples


def test_cursor_channel_keeps_map_and_cursor_apart(tmp_path):
    pl = small_map()
    drawn = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path)
    split = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path, cursor_channel=True)
    i = len(split) // 2
    k = int(split.idx_k[i])
    s = split[i]["frames"]
    assert s.shape == (40, 96, 128)
    assert np.array_equal(s[:20], split.replays[0][3].window(k, 20))     # the map layers, no cursor
    cursor_only = s[20:]
    assert cursor_only.max() == 255 and (cursor_only > 0).sum(axis=(1, 2)).max() < 30   # one small dot each
    # the same cursors as the drawn-in version: drawing channel 2 onto channel 1 gives it back
    g = split.replays[0][2]
    js = np.arange(k - 19, k + 1)
    assert np.array_equal(add_cursor(s[:20], np.stack([g.x[js], g.y[js]], axis=1)), drawn[i]["frames"])


def test_dt_frames_are_25ms_apart(tmp_path):
    ds, _ = windows(tmp_path, mods=DT)
    assert ds[0]["interval"] == pytest.approx(25.0)


def test_canonical_hr_flips_frames_and_cursor(tmp_path):
    flipped, _ = windows(tmp_path, mods=HR, canonical_hr=False)
    canon, _ = windows(tmp_path, mods=HR, canonical_hr=True)
    i = len(canon) // 2
    a, b = flipped[i], canon[i]
    assert b["cursor"][1] == pytest.approx(384 - a["cursor"][1])
    assert b["action"][1] == pytest.approx(-a["action"][1])
    layer = flipped.replays[0][3].window(int(canon.idx_k[i]), 20)[:, ::-1, :]
    g = canon.replays[0][2]
    js = np.arange(int(canon.idx_k[i]) - 19, int(canon.idx_k[i]) + 1)
    assert np.array_equal(b["frames"], add_cursor(layer, np.stack([g.x[js], g.y[js]], axis=1)))
