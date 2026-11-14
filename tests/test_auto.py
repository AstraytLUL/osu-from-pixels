from pathlib import Path

import numpy as np
import pytest

from osu_sim.auto import auto_replay, key_schedule
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import counts, judge
from osu_sim.mods import DT, HD, HR
from osu_sim.playable import make_playable

OSU = Path(__file__).parent / "fixtures" / "1262832_3133639837" / "beatmap.osu"
needs = pytest.mark.skipif(not OSU.exists(), reason="fixture missing: run scripts/check_download.py")


@needs
@pytest.mark.parametrize("mods", [0, HR, DT, HD | HR | DT])
def test_auto_scores_ss(mods):
    pl = make_playable(parse_beatmap(OSU), mods)
    js = judge(pl, auto_replay(pl))
    assert counts(js) == {300: 340, 100: 0, 50: 0, 0: 0}


@needs
def test_auto_frames_are_time_ordered_and_on_a_60fps_grid():
    pl = make_playable(parse_beatmap(OSU), DT)
    rp = auto_replay(pl)
    assert np.all(np.diff(rp.t) > 0)
    # DT: 60 fps in real time = one frame every 25 map ms (plus extra frames at key events)
    assert np.median(np.diff(rp.t)) <= 25


@needs
def test_keys_alternate_and_never_repress_while_held():
    pl = make_playable(parse_beatmap(OSU))
    sched = key_schedule(pl)
    for (p1, r1, k1), (p2, _, k2) in zip(sched, sched[1:]):
        if k1 == k2:
            assert r1 < p2
