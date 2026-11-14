import pytest
import yaml

from data.dataset import load_maps
from data.splits import LEGACY_LISTS, SPLITS, add_maps, band, build, expand, split_files


def test_band_is_whole_stars():
    assert band(4.0) == "4_star"
    assert band(4.99) == "4_star"
    assert band(7.0) == "7_star"
    assert band(8.31) == "8_star"


def test_every_map_is_in_the_file_of_its_split_and_band():
    files = split_files()
    assert files
    for path in files:
        for m in yaml.safe_load(path.read_text(encoding="utf-8"))["maps"]:
            assert m["split"] == path.parent.name
            assert band(m["stars"]) == path.stem
    load_maps((SPLITS,))                               # raises if a map is in two files


def test_splits_keep_every_map_of_the_old_lists_and_only_test_grew():
    old = load_maps(LEGACY_LISTS)
    new = load_maps((SPLITS,))
    for split in ("train", "val", "test"):
        before = {b for b, e in old.items() if e["split"] == split}
        after = {b for b, e in new.items() if e["split"] == split}
        assert before <= after, split                 # append only
        if split != "test":                           # data/find_test_maps.py added test maps on 9 Nov
            assert after == before, split


def test_a_directory_expands_to_its_band_files_in_star_order(tmp_path):
    for split, b in (("test", "10_star"), ("test", "4_star"), ("train", "5_star")):
        (tmp_path / split).mkdir(exist_ok=True)
        (tmp_path / split / f"{b}.yaml").write_text("maps: []\n")
    (tmp_path / "challenge.yaml").write_text("manual: []\n")
    assert [p.relative_to(tmp_path).as_posix() for p in expand([tmp_path])] == \
           ["test/4_star.yaml", "test/10_star.yaml", "train/5_star.yaml"]
    assert expand([tmp_path / "train"]) == (tmp_path / "train" / "5_star.yaml",)


def test_add_maps_appends_by_band_and_build_refuses_to_overwrite(tmp_path):
    add_maps([{"beatmap_id": 1, "stars": 7.2}, {"beatmap_id": 2, "stars": 8.5}], "test", "first", tmp_path)
    add_maps([{"beatmap_id": 3, "stars": 7.9}], "test", "second", tmp_path)
    doc = yaml.safe_load((tmp_path / "test" / "7_star.yaml").read_text(encoding="utf-8"))
    assert [m["beatmap_id"] for m in doc["maps"]] == [1, 3]
    assert doc["meta"]["sources"] == ["first", "second"]
    assert set(load_maps((tmp_path,))) == {1, 2, 3}
    with pytest.raises(ValueError):
        add_maps([{"beatmap_id": 4, "stars": 5.0}], "nonsense", "x", tmp_path)
    with pytest.raises(SystemExit):
        build(root=tmp_path)
