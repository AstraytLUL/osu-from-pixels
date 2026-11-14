"""Find more beatmaps beyond the splits (configs/splits), from the osu! beatmap search alone (no player tags):

    python -m data.find_maps spinners    # BC training maps with several spinners, NoMod replays checked

`spinners` adds to the train split (configs/splits/train, data/splits.py) and writes
configs/maps/spinners.txt, to download replays for with data/download.py like the v1 train maps. Spinners are rare in v1 (most maps have 0-2),
so the BC model rarely saw one; these maps each have at least MIN_SPINNERS.

The list was first written on 26-27 Oct as configs/legacy/maps/maps_spinners.yaml.

Rules:
- ranked osu!standard, one difficulty per mapset (its most played one that qualifies), at most
  MAX_PER_MAPPER maps per mapper; the filters below;
- no mapset from any split or configs/splits/challenge.yaml: a new map never shares a song with the
  maps we validate and test on, nor repeats a map we already have.
Search results are ranked by play count, as in data/select_maps.py. Leaderboard checks (spinners only)
are cached in data/raw/cache/find_maps.json, so a rerun makes no repeat calls.
"""
import argparse
import time
from datetime import date
from pathlib import Path

import yaml

from data.download import connect, with_backoff
from data.select_maps import Cache, usable
from data.splits import CHALLENGE, add_maps, split_files

LISTS = Path("configs/maps")
CACHE = Path("data/raw/cache/find_maps.json")
MAX_PER_MAPPER = 2
SEARCH_SLEEP = 4                               # s between search pages (osu! rate limits)
MAX_PAGES = 200                                # 50 mapsets a page

# Per list: search filters, extra rules, how many maps, and the split they join.
# stars/cs/ar/drain_s are inclusive ranges, as in data/select_maps.py (v1: stars 4.5-6.5 relaxed to 7.0,
# drain 60-180 relaxed to 240, CS 3.5-4.5, AR 8.5-9.5).
LISTS_SPEC = {
    "spinners": {"stars": (4.5, 7.0), "cs": (3.5, 4.5), "ar": (8.5, 9.5), "drain_s": (60, 240),
                 "min_spinners": 3, "nm_usable": 10, "count": 20, "split": "train"},
}


# ---------------------------------------------------------------- pure rules (unit tested)

def qualifies(d: dict, spec: dict) -> bool:
    """Whether difficulty `d` (search fields: mode, stars, cs, ar, drain_s, spinners) passes a list's
    static filters. Leaderboard checks come later, and only for the maps that pass."""
    def within(key):
        lo, hi = spec[key]
        return lo <= d[key] <= hi
    return d["mode"] == 0 and all(within(k) for k in ("stars", "cs", "ar", "drain_s")) \
        and d["spinners"] >= spec["min_spinners"]


def excluded_sets(paths: list[Path]) -> set[int]:
    """Mapset ids in existing map lists (the split band files). The challenge list has no set
    ids, so pick() skips any mapset holding one of its beatmap ids (excluded_maps) instead."""
    out = set()
    for p in paths:
        if p.exists():
            for m in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("maps", []):
                if "beatmapset_id" in m:
                    out.add(int(m["beatmapset_id"]))
    return out


def excluded_maps(paths: list[Path]) -> set[int]:
    """Beatmap ids in every section of existing map lists (challenge.yaml has `manual` and
    `suggested` sections, not `maps`)."""
    out = set()
    for p in paths:
        if p.exists():
            for section in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).values():
                if isinstance(section, list):
                    out.update(int(m["beatmap_id"]) for m in section if isinstance(m, dict) and "beatmap_id" in m)
    return out


# ---------------------------------------------------------------- API

def search_pages(api, spec: dict):
    """Ranked osu!standard mapsets roughly within the filters, most played first, one page at a time.
    The search filters a mapset if any difficulty matches, so every difficulty is checked again."""
    q = " ".join(f"{k}>={spec[k][0]} {k}<={spec[k][1]}" for k in ("stars", "cs", "ar"))
    q += f" length>={spec['drain_s'][0]} length<={spec['drain_s'][1] + 60}"   # length is total, not drain
    cursor = None
    for _ in range(MAX_PAGES):
        r = with_backoff(lambda: api.search_beatmapsets(q, mode=0, category="ranked", sort="plays_desc",
                                                        cursor=cursor), "search")
        time.sleep(SEARCH_SLEEP)
        if not r.beatmapsets:
            return
        yield r.beatmapsets
        cursor = r.cursor
        if cursor is None:
            return


def difficulty(s, b) -> dict:
    """One difficulty of a search result, as the fields the lists keep."""
    return {"beatmap_id": b.id, "beatmapset_id": s.id, "artist": s.artist, "title": s.title,
            "version": b.version, "mapper_id": getattr(b, "user_id", None) or s.user_id, "creator": s.creator,
            "mode": b.mode_int, "stars": round(b.difficulty_rating, 2), "bpm": b.bpm, "drain_s": b.hit_length,
            "cs": b.cs, "ar": b.ar, "od": b.accuracy, "circles": b.count_circles, "sliders": b.count_sliders,
            "spinners": b.count_spinners, "playcount": b.playcount}


def pick(api, cache: Cache, spec: dict, skip_sets: set[int], skip_maps: set[int]) -> list[dict]:
    chosen, per_mapper = [], {}
    for page in search_pages(api, spec):
        for s in page:
            beatmaps = s.beatmaps or []
            # a mapset already listed, or holding a listed map (e.g. a challenge map's other difficulties)
            if s.id in skip_sets or any(b.id in skip_maps for b in beatmaps) or len(chosen) >= spec["count"]:
                continue
            diffs = sorted((difficulty(s, b) for b in beatmaps), key=lambda d: -d["playcount"])
            for d in diffs:
                if not qualifies(d, spec):
                    continue
                if per_mapper.get(d["mapper_id"], 0) >= MAX_PER_MAPPER:
                    continue
                if spec["nm_usable"]:                   # enough clean NoMod replays to download
                    d["nm_usable"] = usable(api, cache, d["beatmap_id"], "NM")["98"]
                    if d["nm_usable"] < spec["nm_usable"]:
                        continue
                chosen.append(d)
                per_mapper[d["mapper_id"]] = per_mapper.get(d["mapper_id"], 0) + 1
                print(f"  {len(chosen):>4} {d['beatmap_id']} {d['artist']} - {d['title']} [{d['version']}] "
                      f"{d['stars']}* spinners {d['spinners']}")
                break                                   # one difficulty per mapset
        if len(chosen) >= spec["count"]:
            break
    return chosen


# ---------------------------------------------------------------- main

def write_list(name: str, spec: dict, maps: list[dict]) -> None:
    for m in maps:
        m.pop("mode")
        m["style"] = "spinners" if name == "spinners" else None     # no player tags in search results
    written = add_maps(maps, spec["split"], f"data/find_maps.py {name} ({date.today().isoformat()})")
    LISTS.mkdir(parents=True, exist_ok=True)
    lines = [f"{m['beatmap_id']}  # {m['artist']} - {m['title']} [{m['version']}] {m['stars']}*" for m in maps]
    (LISTS / f"{name}.txt").write_text(f"# {name} maps, added to configs/splits/{spec['split']}\n"
                                       + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"added {len(maps)} maps to {', '.join(map(str, written))}; ids in {LISTS / name}.txt")


def main():
    ap = argparse.ArgumentParser(description="Find spinner maps for BC.")
    ap.add_argument("list", choices=list(LISTS_SPEC))
    ap.add_argument("--count", type=int, help="how many maps (default: the list's own)")
    args = ap.parse_args()
    spec = dict(LISTS_SPEC[args.list])
    if args.count:
        spec["count"] = args.count
    others = split_files() + [CHALLENGE]
    api = connect()
    maps = pick(api, Cache(CACHE), spec, excluded_sets(others), excluded_maps(others))
    if len(maps) < spec["count"]:
        print(f"WARNING: only {len(maps)} of {spec['count']} maps found")
    write_list(args.list, spec, maps)


if __name__ == "__main__":
    main()
