"""Choose the beatmaps: per-style quotas from osu!'s player-voted tags, filters, a replay availability
check, relaxation steps when a style is short, and stratified splits.

    python -m data.select_maps            # writes configs/legacy/maps/maps_v1.yaml and configs/maps/*.txt

Style labels come from the tags players vote on each difficulty (osu! website): `skillset/jumps`,
`skillset/streams`, `skillset/tech` and each style's specific tags (`streams/bursts`, `jumps/linear`,
...); see style_votes. The most-played maps are mostly jump maps, so every style is
searched and ranked on its own and fills its own quota; one style never fills another's.

Relaxation (only for a style short of its quota, in this order; each map records the step it needed):
  1. NoMod accuracy floor 98% -> 95%   2. stars 4.5-6.5 -> 4.5-7.0
  3. drain up to 240 s                 4. style vote threshold 20 -> 10
API responses are cached in data/raw/cache/select_maps.json, so a rerun makes no repeat calls.
"""
import json
import random
import time
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from data.download import connect, reject_reason, with_backoff

STYLES = ("tech", "streams", "jumps")        # scarcest first: they get first pick of mappers
SKILL_TAG = {s: f"skillset/{s}" for s in STYLES}
QUOTA, SPLIT = 20, {"train": 14, "val": 3, "test": 3}
PILOT = {"train": 3, "val": 1}
SEED = 4001

# filters at relaxation step 0, and the step that widens each one
STARS, STARS_RELAXED, STARS_STEP = (4.5, 6.5), (4.5, 7.0), 2
DRAIN, DRAIN_RELAXED, DRAIN_STEP = (60, 180), (60, 240), 3
VOTES, VOTES_RELAXED, VOTES_STEP = 20, 10, 4
ACC, ACC_RELAXED, ACC_STEP = 0.98, 0.95, 1
CS, AR = (3.5, 4.5), (8.5, 9.5)
DOMINANCE = 1.5                              # top skillset tag needs 1.5x the votes of the next
NM_NEEDED, TEST_NEEDED = 20, 10              # usable replays: NoMod (required), DT/HR/HD on test maps
MAX_PER_MAPPER = 2
BAD_TAG_VOTES = 5                            # gimmick/* or sliders/complex sv with >= 5 votes: excluded
MAX_SETS_PER_STYLE = 150                     # stop scanning a style's search results after this many sets

CACHE = Path("data/raw/cache/select_maps.json")
OUT = Path("configs/legacy/maps/maps_v1.yaml")    # frozen into configs/splits on 9 Nov (data/splits.py)
LISTS = Path("configs/maps")


# ---------------------------------------------------------------- pure rules (unit tested)

def style_votes(tags: dict[str, int]) -> dict[str, int]:
    """Votes per style: the `skillset/<style>` tag plus all of that style's specific tags
    (`streams/bursts`, `jumps/linear`, `tech/finger control`, ...). Players vote the specific tags far
    more than the skillset ones: Everything will freeze [Lunatic] has skillset/tech 15 vs
    skillset/streams 5, but streams/* 393 vs tech/* 92."""
    return {s: sum(n for t, n in tags.items() if t == SKILL_TAG[s] or t.startswith(f"{s}/")) for s in STYLES}


def style_of(tags: dict[str, int], threshold: int) -> str | None:
    """The difficulty's style: the style with the most votes (style_votes), if it has >= threshold
    votes and >= DOMINANCE x the next style. None if none qualifies (ambiguous or too few votes)."""
    votes = sorted(((n, s) for s, n in style_votes(tags).items()), reverse=True)
    (top, style), (second, _) = votes[0], votes[1]
    if top >= threshold and top >= DOMINANCE * second:
        return style
    return None


def bad_tags(tags: dict[str, int]) -> list[str]:
    return [t for t, n in tags.items()
            if n >= BAD_TAG_VOTES and (t.startswith("gimmick/") or t == "sliders/complex sv")]


def relaxation_needed(d: dict, style: str) -> int | None:
    """Smallest relaxation step at which difficulty `d` qualifies for `style`, or None if it never does.

    d: stars, drain, cs, ar, tags (name -> votes), and optionally nm98 / nm95 (usable NoMod replays at
    each accuracy floor; absent = not checked yet, treated as available)."""
    if not (CS[0] <= d["cs"] <= CS[1] and AR[0] <= d["ar"] <= AR[1]) or bad_tags(d["tags"]):
        return None
    steps = [0]
    if style_of(d["tags"], VOTES) != style:
        if style_of(d["tags"], VOTES_RELAXED) != style:
            return None
        steps.append(VOTES_STEP)
    if not STARS[0] <= d["stars"] <= STARS[1]:
        if not STARS_RELAXED[0] <= d["stars"] <= STARS_RELAXED[1]:
            return None
        steps.append(STARS_STEP)
    if not DRAIN[0] <= d["drain"] <= DRAIN[1]:
        if not DRAIN_RELAXED[0] <= d["drain"] <= DRAIN_RELAXED[1]:
            return None
        steps.append(DRAIN_STEP)
    if "nm98" in d and d["nm98"] < NM_NEEDED:
        if d["nm95"] < NM_NEEDED:
            return None
        steps.append(ACC_STEP)
    return max(steps)


def choose(candidates: list[dict], quota: int, mapper_count: Counter, used_sets: set) -> list[dict]:
    """Fill a quota from candidates (in play-count order, each with its `step`): all step-0 maps first,
    then step 1, and so on; one difficulty per mapset, at most MAX_PER_MAPPER maps per mapper overall.
    Updates mapper_count and used_sets."""
    chosen = []
    for step in range(VOTES_STEP + 1):
        for c in candidates:
            if len(chosen) == quota:
                return chosen
            if c["step"] != step or c["set_id"] in used_sets or mapper_count[c["mapper"]] >= MAX_PER_MAPPER:
                continue
            chosen.append(c)
            used_sets.add(c["set_id"])
            mapper_count[c["mapper"]] += 1
    return chosen


def assign_splits(maps: list[dict], test_ok, seed: int = SEED) -> None:
    """Set m["split"] and m["pilot"] for one style's maps. Test maps are preferably those where
    test_ok(m) is True (enough DT/HR/HD replays); val next, the rest train. Deterministic for a seed."""
    order = sorted(maps, key=lambda m: m["beatmap_id"])
    random.Random(seed).shuffle(order)
    test = []
    for m in order:                          # check maps one by one (API calls) until enough pass
        if len(test) == SPLIT["test"]:
            break
        if test_ok(m):
            test.append(m)
    test += [m for m in order if m not in test][:SPLIT["test"] - len(test)]   # short: fill anyway
    rest = [m for m in order if m not in test]
    val, train = rest[:SPLIT["val"]], rest[SPLIT["val"]:]
    for split, group in (("test", test), ("val", val), ("train", train)):
        for i, m in enumerate(group):
            m["split"] = split
            m["pilot"] = i < PILOT.get(split, 0)


# ---------------------------------------------------------------- API (cached)

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"sets": {}, "search": {}, "avail": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


def search_style(api, cache, style: str) -> list[int]:
    """Ranked osu!standard set ids carrying the style's tag, most played first."""
    if style in cache.data["search"]:
        return cache.data["search"][style]
    query = (f'tag="{SKILL_TAG[style]}" stars>={STARS_RELAXED[0]} stars<={STARS_RELAXED[1]} '
             f"length>={DRAIN_RELAXED[0]} length<={DRAIN_RELAXED[1] + 60}")
    ids, cursor = [], None
    while len(ids) < MAX_SETS_PER_STYLE:
        r = with_backoff(lambda: api.search_beatmapsets(query, mode=0, category="ranked",
                                                        sort="plays_desc", cursor=cursor), f"search {style}")
        time.sleep(4)
        ids += [s.id for s in r.beatmapsets]
        cursor = r.cursor
        if not r.beatmapsets or cursor is None:
            break
    cache.data["search"][style] = ids[:MAX_SETS_PER_STYLE]
    cache.save()
    return cache.data["search"][style]


def set_info(api, cache, set_id: int, tag_names: dict[int, str]) -> dict:
    """Per-difficulty facts and tags for one mapset."""
    key = str(set_id)
    if key not in cache.data["sets"]:
        s = with_backoff(lambda: api.beatmapset(set_id), f"set {set_id}")
        time.sleep(4)
        diffs = []
        for b in s.beatmaps or []:
            if b.mode_int != 0:
                continue
            diffs.append({
                "beatmap_id": b.id, "version": b.version, "stars": round(b.difficulty_rating, 2),
                "bpm": b.bpm, "drain": b.hit_length, "cs": b.cs, "ar": b.ar, "od": b.accuracy,
                "playcount": b.playcount, "mapper_id": getattr(b, "user_id", None) or s.user_id,
                "tags": {tag_names.get(t["tag_id"], str(t["tag_id"])): t["count"] for t in (b.top_tag_ids or [])},
            })
        cache.data["sets"][key] = {"set_id": set_id, "artist": s.artist, "title": s.title,
                                   "creator": s.creator, "status": str(s.status), "diffs": diffs}
        cache.save()
    return cache.data["sets"][key]


def usable(api, cache, beatmap_id: int, mods: str) -> dict:
    """Usable replays on one leaderboard (download.py's filters) at each accuracy floor."""
    key = f"{beatmap_id}:{mods}"
    if key not in cache.data["avail"]:
        board = with_backoff(lambda: api.beatmap_scores(beatmap_id, mode="osu", mods=mods, limit=100),
                             f"leaderboard {beatmap_id} {mods}")
        time.sleep(4)
        cache.data["avail"][key] = {
            "98": sum(reject_reason(sc, mods, ACC) is None for sc in board.scores),
            "95": sum(reject_reason(sc, mods, ACC_RELAXED) is None for sc in board.scores),
        }
        cache.save()
    return cache.data["avail"][key]


def candidates_for(api, cache, style: str, tag_names: dict[int, str], have_level0: int) -> list[dict]:
    """Candidate difficulties for a style, in play-count order (one per set: its most played one
    that qualifies). Scans until QUOTA step-0 maps (counting `have_level0`) or the search runs out."""
    out, level0 = [], have_level0
    for set_id in search_style(api, cache, style):
        info = set_info(api, cache, set_id, tag_names)
        if "ranked" not in info["status"].lower():
            continue
        best = None
        for d in sorted(info["diffs"], key=lambda d: -d["playcount"]):
            step = relaxation_needed(d, style)          # static filters only, before any replay check
            if step is None:
                continue
            a = usable(api, cache, d["beatmap_id"], "NM")
            d["nm98"], d["nm95"] = a["98"], a["95"]
            step = relaxation_needed(d, style)
            if step is not None:
                best = {**d, "step": step, "set_id": set_id, "artist": info["artist"], "title": info["title"],
                        "mapper": d["mapper_id"], "creator": info["creator"], "style": style}
                break
        if best:
            out.append(best)
            level0 += best["step"] == 0
            print(f"  {style}: {best['beatmap_id']} {info['artist']} - {info['title']} [{best['version']}] "
                  f"{best['stars']}* step {best['step']} (NM usable {best['nm98']}/{best['nm95']})")
            if level0 >= QUOTA + 5:                     # a few spare for the mapper/set rules
                break
    return out


# ---------------------------------------------------------------- main

def main():
    api = connect()
    cache = Cache(CACHE)
    tl = api.tags()
    tag_names = {t.id: t.name for t in getattr(tl, "tags", tl)}

    mapper_count, used_sets, selected = Counter(), set(), {}
    for style in STYLES:
        print(f"== {style}")
        cands = candidates_for(api, cache, style, tag_names, 0)
        selected[style] = choose(cands, QUOTA, mapper_count, used_sets)
        short = QUOTA - len(selected[style])
        print(f"{style}: {len(selected[style])} maps, steps {dict(Counter(m['step'] for m in selected[style]))}"
              + (f", SHORT by {short}" if short else ""))

    def test_ok(m):
        counts = {mods: usable(api, cache, m["beatmap_id"], mods)["98"] for mods in ("DT", "HR", "HD")}
        m["test_available"] = counts
        return all(n >= TEST_NEEDED for n in counts.values())

    for style in STYLES:
        assign_splits(selected[style], test_ok)

    write_outputs(selected)


def write_outputs(selected: dict[str, list[dict]]) -> None:
    fields = ["beatmap_id", "beatmapset_id", "artist", "title", "version", "mapper_id", "creator", "stars",
              "bpm", "drain_s", "cs", "ar", "od", "style", "style_votes", "playcount", "available",
              "relaxation", "split", "pilot"]
    maps = []
    for style in STYLES:
        for m in sorted(selected[style], key=lambda m: ("train val test".split().index(m["split"]), -m["playcount"])):
            available = {"NM": {"acc98": m["nm98"], "acc95": m["nm95"]}}
            if "test_available" in m:
                available.update({k: {"acc98": v} for k, v in m["test_available"].items()})
            maps.append(dict(zip(fields, [
                m["beatmap_id"], m["set_id"], m["artist"], m["title"], m["version"], m["mapper"], m["creator"],
                m["stars"], m["bpm"], m["drain"], m["cs"], m["ar"], m["od"], style,
                style_votes(m["tags"]), m["playcount"], available,
                m["step"], m["split"], m["pilot"]])))
    header = {
        "generated": date.today().isoformat(), "seed": SEED, "quota_per_style": QUOTA, "splits": SPLIT,
        "pilot": PILOT, "filters": {"stars": STARS, "drain_s": DRAIN, "cs": CS, "ar": AR, "votes": VOTES,
                                    "dominance": DOMINANCE, "nm_usable_needed": NM_NEEDED,
                                    "accuracy_min": ACC, "max_per_mapper": MAX_PER_MAPPER},
        "relaxation_steps": {1: f"NM accuracy floor {ACC} -> {ACC_RELAXED}", 2: f"stars -> {STARS_RELAXED}",
                             3: f"drain -> {DRAIN_RELAXED}", 4: f"votes {VOTES} -> {VOTES_RELAXED}"},
    }
    OUT.write_text("# Generated by data/select_maps.py; review by hand before downloading.\n"
                   + yaml.safe_dump({"meta": header, "maps": maps}, sort_keys=False, allow_unicode=True))
    LISTS.mkdir(parents=True, exist_ok=True)
    groups = {"pilot": [m for m in maps if m["pilot"]]} | {s: [m for m in maps if m["split"] == s] for s in SPLIT}
    for name, group in groups.items():
        lines = [f"{m['beatmap_id']}  # {m['style']}: {m['artist']} - {m['title']} [{m['version']}] {m['stars']}*"
                 for m in group]
        (LISTS / f"{name}.txt").write_text(f"# {name} maps, from {OUT}\n" + "\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(maps)} maps) and {LISTS}/{{pilot,train,val,test}}.txt")


if __name__ == "__main__":
    main()
