from collections import Counter

from data.find_test_maps import pick, qualifies, quotas


def diff(**kw):
    d = {"stars": 7.4, "cs": 4.0, "ar": 9.8, "drain": 150, "tags": {"skillset/jumps": 30}}
    return {**d, **kw}


def test_quotas_give_every_style_two_then_fill_the_least_represented():
    assert quotas(Counter({"tech": 1, "streams": 1})) == Counter({"jumps": 2, "streams": 1, "tech": 1})
    assert quotas(Counter()) == Counter({"jumps": 2, "streams": 2, "tech": 2})
    assert quotas(Counter({"tech": 2, "streams": 1, "jumps": 3})) == Counter({"streams": 1})
    assert quotas(Counter({"streams": 1}), target=7) == Counter({"jumps": 3, "streams": 1, "tech": 2})


def test_ar_10_only_above_the_training_bands():
    assert qualifies(diff(), "jumps", 7) == []
    assert qualifies(diff(stars=6.4), "jumps", 6) is None
    assert qualifies(diff(stars=6.4, ar=9.3), "jumps", 6) == []


def test_star_band_style_and_relaxations():
    assert qualifies(diff(), "jumps", 8) is None                    # 7.4* is not 8_star
    assert qualifies(diff(stars=9.5), "jumps", 9) == []
    assert qualifies(diff(stars=10.2), "jumps", 9) is None          # no 10_star band
    assert qualifies(diff(), "streams", 7) is None
    assert qualifies(diff(tags={"skillset/jumps": 12}), "jumps", 7) == ["votes10"]
    assert qualifies(diff(nm98=4, nm95=15), "jumps", 7) == ["acc95"]
    assert qualifies(diff(nm98=4, nm95=6), "jumps", 7) is None


def test_pick_prefers_nomod_replays_then_mod_replays_and_keeps_the_mapper_cap():
    c = lambda i, mapper, relaxed=(), mods_ok=2: {"beatmap_id": i, "beatmapset_id": i, "mapper_id": mapper,
                                                     "relaxed": list(relaxed), "mods_ok": mods_ok}
    cands = [c(1, 9, ["acc95"]), c(2, 9, mods_ok=1), c(3, 9, ["votes10"]), c(4, 8)]
    mappers = Counter({8: 2})
    assert [m["beatmap_id"] for m in pick(cands, 3, mappers, set())] == [3, 2]    # 1: mapper 9 full; 4: mapper 8
    assert mappers[9] == 2
