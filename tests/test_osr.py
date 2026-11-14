import numpy as np
from osrparse import Replay

from data.replay import M1, ReplayFrames, load_replay
from eval.osr import MOVE_LEAD_MS, PLAYER_AUTO, save_osr
from osu_sim.auto import auto_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import accuracy, judge
from osu_sim.mods import HR
from osu_sim.playable import make_playable
from osu_sim.score import NF

OSU = """osu file format v14

[General]
StackLeniency: 0.7

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:8
ApproachRate:9
SliderMultiplier:1
SliderTickRate:1

[TimingPoints]
0,500,4,2,1,100,1,0

[HitObjects]
100,100,2000,1,0,0:0:0:0:
300,120,2400,2,0,L|400:120,2,100
200,300,3600,1,0,0:0:0:0:
"""


def test_auto_round_trips_through_an_osr(tmp_path):
    osu = tmp_path / "map.osu"
    osu.write_text(OSU)
    pl = make_playable(parse_beatmap(osu), HR)
    frames = auto_replay(pl)
    path = tmp_path / "auto.osr"
    header = save_osr(frames, path, osu, PLAYER_AUTO)

    raw = Replay.from_path(str(path))
    assert raw.username == PLAYER_AUTO and raw.perfect
    # circle + (head, repeat, end of a 2-span slider; its only tick falls on the end, so none) + circle
    assert raw.max_combo == header["max_combo"] == 5
    assert raw.score == header["score"] > 0
    back = load_replay(path)
    assert back.mods == HR | NF                               # NF added, so osu! never stops the replay
    # the loader keeps stable's leading "no cursor" frame (t = -1, y = -500), as for real replays
    assert (back.t[0], back.y[0]) == (-1, -500)
    # every frame comes back (plus the 1 ms "move first" frames before presses)
    kept = np.isin(back.t, np.round(frames.t))
    t, x, keys = back.t[kept], back.x[kept], back.keys[kept]
    assert np.allclose(t, np.round(frames.t)) and np.allclose(x, frames.x, atol=1e-3)
    assert np.array_equal(keys & 3, frames.keys)               # M1 / M2 (K1 / K2 ride along)
    assert accuracy(judge(pl, back)) == 1.0


def test_a_press_never_moves_the_cursor_in_the_same_frame(tmp_path):
    osu = tmp_path / "map.osu"
    osu.write_text(OSU)
    # the cursor jumps 100 px onto a spot on the very frame the left key goes down
    frames = ReplayFrames(t=np.array([1000.0, 1017.0, 1034.0]), x=np.array([100.0, 200.0, 200.0]),
                          y=np.array([100.0, 100.0, 100.0]), keys=np.array([0, M1, M1]), mods=0)
    save_osr(frames, tmp_path / "jump.osr", osu, PLAYER_AUTO)
    back = load_replay(tmp_path / "jump.osr")
    i = int(np.flatnonzero(back.t == 1017)[0])
    assert back.t[i - 1] == 1017 - MOVE_LEAD_MS                                        # moved ahead of the press,
    assert back.x[i - 1] == 200 and back.keys[i - 1] == 0                              # keys still up
    assert back.x[i] == 200 and back.keys[i] & M1                                      # then pressed, in place


def test_no_fail_changes_only_the_score(tmp_path):
    osu = tmp_path / "map.osu"
    osu.write_text(OSU)
    frames = auto_replay(make_playable(parse_beatmap(osu), HR))
    with_nf = save_osr(frames, tmp_path / "nf.osr", osu, PLAYER_AUTO)
    without = save_osr(frames, tmp_path / "plain.osr", osu, PLAYER_AUTO, no_fail=False)
    assert Replay.from_path(str(tmp_path / "plain.osr")).mods == HR
    assert {k: v for k, v in with_nf.items() if k != "score"} == {k: v for k, v in without.items() if k != "score"}
    assert with_nf["score"] < without["score"]           # NF halves the combo bonus, not the base points
