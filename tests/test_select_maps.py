from collections import Counter

import pytest

from data.select_maps import QUOTA, assign_splits, bad_tags, choose, relaxation_needed, style_of


def tags(jumps=0, streams=0, tech=0, **extra):
    t = {"skillset/jumps": jumps, "skillset/streams": streams, "skillset/tech": tech}
    t.update({k.replace("__", "/").replace("_", " "): v for k, v in extra.items()})
    return t


# ---------- style rule ----------

def test_dominant_style():
    assert style_of(tags(jumps=398, streams=10), 20) == "jumps"


def test_ambiguous_style():
    assert style_of(tags(jumps=40, streams=30), 20) is None          # 40 < 1.5 x 30


def test_specific_tags_count_for_their_style():
    # Everything will freeze [Lunatic], real votes: skillset/tech 15 > skillset/streams 5, but the
    # specific stream tags dominate
    t = {"skillset/tech": 15, "skillset/streams": 5, "streams/bursts": 186, "streams/speed": 108,
         "streams/flow aim": 68, "tech/finger control": 77, "expression/chaotic": 40}
    assert style_of(t, 20) == "streams"


def test_mixed_map_is_ambiguous():
    # Night of Knights [Guy's Extra]: jumps 99, streams 186, tech 130 -> 186 < 1.5 x 130
    t = {"skillset/jumps": 99, "streams/speed": 186, "tech/aim control": 130}
    assert style_of(t, 20) is None


def test_other_categories_do_not_count():
    assert style_of({"expression/simple": 500, "skillset/jumps": 25}, 20) == "jumps"


def test_style_below_threshold():
    assert style_of(tags(tech=15), 20) is None
    assert style_of(tags(tech=15), 10) == "tech"


# ---------- filters and relaxation ----------

def diff(**kw):
    d = {"stars": 5.0, "drain": 120, "cs": 4, "ar": 9, "tags": tags(streams=50)}
    d.update(kw)
    return d


def test_strict_pass():
    assert relaxation_needed(diff(), "streams") == 0
    assert relaxation_needed(diff(), "jumps") is None                 # wrong style


@pytest.mark.parametrize("kw, step", [
    ({"nm98": 12, "nm95": 25}, 1),         # needs the 95% floor
    ({"stars": 6.8}, 2),
    ({"drain": 200}, 3),
    ({"tags": tags(streams=12)}, 4),
    ({"stars": 6.8, "drain": 200}, 3),     # the later step wins
])
def test_relaxation_steps(kw, step):
    assert relaxation_needed(diff(**kw), "streams") == step


@pytest.mark.parametrize("kw", [
    {"stars": 4.4}, {"stars": 7.2}, {"drain": 50}, {"drain": 250}, {"cs": 5}, {"ar": 10},
    {"nm98": 5, "nm95": 15}, {"tags": tags(streams=8)},
    {"tags": tags(streams=50, gimmick__2B=6)}, {"tags": tags(streams=50, sliders__complex_sv=9)},
])
def test_rejected(kw):
    assert relaxation_needed(diff(**kw), "streams") is None


def test_few_gimmick_votes_are_ignored():
    assert bad_tags(tags(streams=50, gimmick__2B=4)) == []
    assert relaxation_needed(diff(tags=tags(streams=50, gimmick__2B=4)), "streams") == 0


# ---------- choosing ----------

def cand(i, step=0, mapper=None, set_id=None):
    return {"beatmap_id": i, "step": step, "mapper": mapper if mapper is not None else i, "set_id": set_id or i}


def test_strict_maps_before_relaxed_ones():
    cands = [cand(1, step=2), cand(2), cand(3, step=1), cand(4)]
    chosen = choose(cands, 3, Counter(), set())
    assert [c["beatmap_id"] for c in chosen] == [2, 4, 3]


def test_mapper_cap_and_one_diff_per_set():
    mappers, sets = Counter(), set()
    cands = [cand(1, mapper=7), cand(2, mapper=7), cand(3, mapper=7), cand(4, set_id=1)]
    chosen = choose(cands, 10, mappers, sets)
    assert [c["beatmap_id"] for c in chosen] == [1, 2]                # 3: mapper cap; 4: set 1 taken
    # the cap carries over to the next style
    assert choose([cand(5, mapper=7)], 10, mappers, sets) == []


# ---------- splits ----------

def test_splits_are_stratified_deterministic_and_prefer_test_ready_maps():
    maps = [{"beatmap_id": i} for i in range(QUOTA)]
    ready = {3, 8, 15}
    assign_splits(maps, lambda m: m["beatmap_id"] in ready)
    split = Counter(m["split"] for m in maps)
    assert split == {"train": 14, "val": 3, "test": 3}
    assert {m["beatmap_id"] for m in maps if m["split"] == "test"} == ready
    pilot = Counter(m["split"] for m in maps if m["pilot"])
    assert pilot == {"train": 3, "val": 1}
    again = [{"beatmap_id": i} for i in range(QUOTA)]
    assign_splits(again, lambda m: m["beatmap_id"] in ready)
    assert [m["split"] for m in again] == [m["split"] for m in maps]


def test_splits_fill_test_when_few_maps_are_ready():
    maps = [{"beatmap_id": i} for i in range(QUOTA)]
    assign_splits(maps, lambda m: False)
    assert Counter(m["split"] for m in maps)["test"] == 3
