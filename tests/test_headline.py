import yaml

from data.dataset import load_maps
from data.splits import band
from eval.headline import cell_maps, checkpoint


def test_cells_are_every_test_map_on_nomod_and_in_band_maps_under_the_mods():
    cfg = yaml.safe_load(open("configs/eval/headline.yaml"))
    cells = cell_maps(cfg)
    test = [e for e in load_maps().values() if e["split"] == "test"]
    in_band = [e for e in test if band(e["stars"]) in ("4_star", "5_star", "6_star")]
    assert len(cells) == len(test) + 3 * len(in_band)
    assert len(set((m, b) for m, b, _ in cells)) == len(cells)
    assert {m for m, _, e in cells if band(e["stars"]) == "9_star"} == {"NM"}


def test_a_local_run_still_training_has_no_checkpoint(tmp_path):
    run = tmp_path / "x_seed0"
    run.mkdir()
    (run / "best.pt").write_bytes(b"")
    (run / "train.log").write_text("x seed 0: 10 windows\n")
    assert checkpoint("x", 0, {"local": str(tmp_path / "x_seed{seed}")}, None) is None
    (run / "train.log").write_text("x seed 0: 10 windows\nbest val accuracy 97.00%\n")
    assert checkpoint("x", 0, {"local": str(tmp_path / "x_seed{seed}")}, None) == run / "best.pt"
