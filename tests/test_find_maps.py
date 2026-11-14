import yaml

from data.find_maps import LISTS_SPEC, excluded_maps, excluded_sets, qualifies


def diff(**kw):
    d = {"mode": 0, "stars": 5.5, "cs": 4.0, "ar": 9.0, "drain_s": 120, "spinners": 3}
    d.update(kw)
    return d


def test_spinner_list_needs_spinners_and_the_v1_filters():
    spec = LISTS_SPEC["spinners"]
    assert qualifies(diff(), spec)
    assert not qualifies(diff(spinners=2), spec)
    assert not qualifies(diff(ar=9.8), spec)              # outside v1's AR range
    assert not qualifies(diff(mode=3), spec)              # mania


def test_existing_lists_are_excluded(tmp_path):
    v1 = tmp_path / "maps_v1.yaml"
    v1.write_text(yaml.safe_dump({"maps": [{"beatmap_id": 1, "beatmapset_id": 10}]}))
    challenge = tmp_path / "maps_challenge.yaml"                      # no set ids, two sections
    challenge.write_text(yaml.safe_dump({"manual": [{"beatmap_id": 2}], "suggested": [{"beatmap_id": 3}]}))
    paths = [v1, challenge, tmp_path / "not_written_yet.yaml"]
    assert excluded_sets(paths) == {10}
    assert excluded_maps(paths) == {1, 2, 3}
