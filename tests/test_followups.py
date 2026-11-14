"""The follow-up analyses' pure parts (eval/diagnose.py, eval/heatmaps.py, eval/test_all.py), and the
split option of eval/compare_runs.py. No network, no checkpoints."""
import json

import numpy as np
import pytest
import yaml

from eval.compare_runs import load_runs
from eval.diagnose import classify, summarise
from eval.heatmaps import CELL_X, CELL_Y, GRID, entropy_bits, mass_near, measures
from eval.test_all import run_prefix, seeds_in


def test_classify_circles():
    assert classify(300, 3.0, 0.2) == "300"
    assert classify(100, 40.0, 0.9) == "100"
    assert classify(0, -150.0, 0.5) == "miss_timing"     # clicked on it, outside the hit window
    assert classify(0, None, 0.8) == "miss_timing"       # the cursor was on it, no click
    assert classify(0, None, 2.5) == "miss_aim"          # the cursor was elsewhere, never clicked it


def test_summarise_rates_and_offsets():
    rows = [{"result": 300, "offset": -4.0, "distance": 0.3, "class": "300"},
            {"result": 100, "offset": 30.0, "distance": 0.7, "class": "100"},
            {"result": 0, "offset": None, "distance": 3.0, "class": "miss_aim"},
            {"result": 0, "offset": None, "distance": 0.5, "class": "miss_timing"}]
    s = summarise(rows, n_clicks=3, n_objects=4)
    assert s["rates"]["miss_aim"] == 0.25 and s["rates"]["miss_timing"] == 0.25
    assert s["mean_offset"] == pytest.approx(13.0) and s["clicks_per_object"] == 0.75
    assert s["median_distance"] == pytest.approx(0.6)


def one_cell(x, y):
    p = np.zeros(GRID)
    p[int(y // 16), int(x // 16)] = 1.0
    return p


def test_heatmap_measures():
    p = one_cell(100, 100)
    assert entropy_bits(p) == 0.0
    assert entropy_bits(np.full(GRID, 1 / 768)) == pytest.approx(np.log2(768))
    m = measures(p, target=(100, 100), upcoming=[(400, 300)], previous=[], radius=36)
    assert m["target"] == 1.0 and m["upcoming"] == 0.0 and m["on_target"] == 1.0
    m = measures(one_cell(400, 300), target=(100, 100), upcoming=[(400, 300)], previous=[], radius=36)
    assert m["target"] == 0.0 and m["upcoming"] == 1.0 and m["on_target"] == 0.0
    half = (one_cell(100, 100) + one_cell(400, 300)) / 2
    assert mass_near(half, [(100, 100), (400, 300)], 52) == pytest.approx(1.0)
    assert len(CELL_X) == 32 and len(CELL_Y) == 24 and CELL_X[0] == 8.0


def test_test_all_finds_both_layouts():
    files = ["README.md", "seed0/best.pt", "seed0/config.yaml", "seed2/best.pt", "seed1/config.yaml"]
    assert seeds_in(files) == [0, 2]                     # seed1 has no best.pt
    assert run_prefix(files, 2) == "seed2/"
    assert run_prefix(["config.yaml", "best.pt"], 0) == ""        # bc-baseline: one run at the root
    assert run_prefix(["README.md"], 0) is None


def test_compare_runs_reads_the_split_asked_for(tmp_path):
    for split, acc in (("val", 0.9), ("test", 0.8)):
        d = tmp_path / "bc-x" / "seed0"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.yaml").write_text(yaml.safe_dump({"name": "x", "seed": 0}))
        (d / "metrics.jsonl").write_text(json.dumps({"epoch": 0, "loss": 1.0, "val_accuracy": 0.9}) + "\n")
        report = {"accuracy": acc, "miss_rate": 0.0, "maps": []}
        (d / f"transfer_{split}.json").write_text(json.dumps({"results": {"NM": report}}))
    assert load_runs(tmp_path, "val")["x"][0]["transfer"]["NM"]["accuracy"] == 0.9
    assert load_runs(tmp_path, "test")["x"][0]["transfer"]["NM"]["accuracy"] == 0.8


def test_heatmap_query_frame():
    from types import SimpleNamespace as NS
    from eval.heatmaps import query_frame
    times = np.arange(0, 2000, 16.0)
    objs = [NS(index=0, kind="circle", time=500.0, end_time=500.0),
            NS(index=1, kind="circle", time=900.0, end_time=900.0),      # 25 frames after the first
            NS(index=2, kind="circle", time=940.0, end_time=940.0)]      # only 2-3 frames after the second
    pl = NS(objects=objs)
    assert query_frame(times, pl, objs[1], "decision", 2) == int(np.searchsorted(times, 500.0)) + 2
    assert query_frame(times, pl, objs[0], "decision", 2) is None       # no previous object
    assert query_frame(times, pl, objs[2], "decision", 2) is None       # too close to decide
    assert query_frame(times, pl, objs[1], "before-hit", 3) == int(np.searchsorted(times, 900.0)) - 3
