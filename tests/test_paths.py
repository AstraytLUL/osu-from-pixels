from pathlib import Path

from eval.paths import footage_path, map_name, model_player, mods_name, replay_path
from osu_sim.mods import DT, HD, HR, NC


def test_mods_name():
    assert mods_name(0) == "NM"
    assert mods_name(HR | HD) == "HDHR"                # fixed order, whatever the bits
    assert mods_name(NC | DT) == mods_name(NC) == "DT"


def test_map_name():
    assert map_name(4912305, "Tetoris") == "4912305-tetoris"
    assert map_name(555797, "Everything will freeze") == "555797-everything-will-freeze"
    assert map_name(1, "Exit This Earth's Atomosphere") == "1-exit-this-earth-s-atomosphere"
    assert map_name(7, "") == "7"
    assert len(map_name(1, "x" * 100)) <= 32


def test_paths():
    assert model_player("runs/bc_baseline_seed0/best.pt") == "bc_baseline_seed0-best"
    assert replay_path("auto", 4912305, "Tetoris", HR) == Path("runs/replays/auto/4912305-tetoris_HR.osr")
    assert footage_path("human-1", 4912305, "Tetoris", 0) == Path("runs/footage/human-1/4912305-tetoris_NM.mp4")
    assert footage_path("auto", 4912305, "Tetoris", DT, "gif", 20, 26.5) == \
        Path("runs/footage/auto/4912305-tetoris_DT_20-26.5s.gif")
