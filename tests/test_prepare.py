import pytest
import yaml

from data.dataset import load_maps
from data.prepare import frame_jobs
from data.splits import SPLITS


def test_load_maps_joins_lists_and_refuses_a_map_listed_twice(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"maps": [{"beatmap_id": 1, "split": "train"}]}))
    b.write_text(yaml.safe_dump({"maps": [{"beatmap_id": 2, "split": "val"}]}))
    assert set(load_maps((a, b))) == {1, 2}
    with pytest.raises(ValueError):
        load_maps((a, a))


def test_bc_frame_jobs_match_what_training_reads():
    jobs = frame_jobs(test=False)
    maps = load_maps((SPLITS,))
    train = [b for b, e in maps.items() if e["split"] == "train"]
    val = [b for b, e in maps.items() if e["split"] == "val"]
    test = [b for b, e in maps.items() if e["split"] == "test"]
    assert set(jobs) == {(b, m) for b in train for m in ("NM", "HR")} | {(b, "NM") for b in val}
    assert not any(b in test for b, _ in jobs)
    with_test = frame_jobs(test=True)
    assert (test[0], "DT") in with_test
