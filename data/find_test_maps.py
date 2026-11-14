"""More test maps per star band, for the headline figure (the project plan M3): at least TARGET maps in
every band from 4_star to 9_star, and at least MIN_PER_STYLE of each style where the search finds them.

    python -m data.find_test_maps            # search and print the picks (cached: a rerun is free)
    python -m data.find_test_maps --write    # also append them to configs/splits/test and write
                                             # configs/maps/test_bands_inband.txt and test_bands_hard.txt

then download their replays like the v1 test maps:

    python -m data.download configs/maps/test_bands_inband.txt --config configs/download.yaml     # NM HR HD DT x10
    python -m data.download configs/maps/test_bands_hard.txt --config configs/download_nm.yaml    # NM x10

Same rules as data/select_maps.py where they apply: per style, the ranked mapsets carrying its
`skillset/<style>` tag, most played first (then, if the style is still short, every ranked mapset in
the band, since many maps only carry the specific tags such as `tech/finger control`); the style from the player-voted tags (style_of), no
gimmick or complex-SV maps, one difficulty per mapset, at most MAX_PER_MAPPER maps per mapper (counting
every BC split); >= NM_NEEDED clean NoMod replays at 98% accuracy. Mapsets in any split (rl
included: RL never trains on a test song) or holding a challenge map are skipped. Changed for test:

- stars within the band ([n, n+1)), not 4.5-6.5; nothing from 10* up (no 10_star band);
- AR up to 10 in the bands above training (7+): nearly every 8* map is AR 9.5-10, and the model saw
  AR 10 through HR (AR x 1.4, capped at 10) in training;
- in-band (4, 6) maps prefer ones with >= MODS_NEEDED replays on HR and on DT (the mod rows of the
  headline table need human references; DT on a 6* map is rare, ~0 replays); 7+ maps are NoMod only;
- relaxations, recorded per map in `relaxed` and only used when a style is short: NoMod floor 98% ->
  95% ("acc95"), style votes 20 -> 10 ("votes10").

Quotas: every style gets up to MIN_PER_STYLE, then a band short of TARGET fills its least
represented style first (ties: jumps, streams, tech, the most common first). A style that comes up
short hands its places to the others, so the band still reaches TARGET; its per-style results then
have to be pooled with the neighbouring bands.
"""
import argparse
import time
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from data.download import connect, with_backoff
from data.select_maps import Cache, bad_tags, set_info, style_of, style_votes, usable
from data.find_maps import excluded_maps, excluded_sets
from data.splits import CHALLENGE, add_maps, band, split_files

BANDS = (4, 5, 6, 7, 8, 9)
IN_BAND = (4, 5, 6)                          # train covers 4.53-6.92*
TARGET = 5
MIN_PER_STYLE = 2
STYLE_ORDER = ("jumps", "streams", "tech")   # tie order for quotas: most common first
CS = (3.5, 4.5)
AR_IN, AR_OUT = (8.5, 9.5), (8.5, 10.0)
DRAIN = (60, 240)
VOTES, VOTES_RELAXED = 20, 10
NM_NEEDED, MODS_NEEDED = 10, 10
MAX_PER_MAPPER = 2
MAX_SETS = 80                                # search results scanned per style and band

CACHE = Path("data/raw/cache/select_maps.json")     # shared with select_maps: its sets and boards are reused
LISTS = Path("configs/maps")
BC_SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- pure rules (unit tested)

def quotas(have: Counter, target: int = TARGET, min_style: int = MIN_PER_STYLE) -> Counter:
    """How many maps each style gets, given the band's current maps per style (`have`)."""
    out = Counter({s: max(min_style - have[s], 0) for s in STYLE_ORDER})
    counts = Counter({s: have[s] + out[s] for s in STYLE_ORDER})
    while sum(counts.values()) < target:
        s = min(STYLE_ORDER, key=lambda s: (counts[s], STYLE_ORDER.index(s)))
        counts[s] += 1
        out[s] += 1
    return +out


def star_range(b: int) -> tuple[float, float]:
    return b, b + 1


def qualifies(d: dict, style: str, b: int) -> list[str] | None:
    """The relaxations difficulty `d` needs to be a `style` test map in band `b` ([] = none), or None
    if it never qualifies. d: select_maps.set_info's fields, plus nm98 / nm95 once checked."""
    lo, hi = star_range(b)
    ar = AR_IN if b in IN_BAND else AR_OUT
    if not (lo <= d["stars"] < hi and CS[0] <= d["cs"] <= CS[1] and ar[0] <= d["ar"] <= ar[1]
            and DRAIN[0] <= d["drain"] <= DRAIN[1]) or bad_tags(d["tags"]):
        return None
    relaxed = []
    if style_of(d["tags"], VOTES) != style:
        if style_of(d["tags"], VOTES_RELAXED) != style:
            return None
        relaxed.append("votes10")
    if "nm98" in d and d["nm98"] < NM_NEEDED:
        if d["nm95"] < NM_NEEDED:
            return None
        relaxed.append("acc95")
    return relaxed


def pick(cands: list[dict], quota: int, mapper_count: Counter, used_sets: set) -> list[dict]:
    """Fill a quota from candidates in play-count order: enough NoMod replays at 98% first, then the
    most mods (HR, DT) with enough replays, then the fewest relaxations. Updates mapper_count and
    used_sets."""
    def rank(c):
        return ("acc95" in c["relaxed"], -c.get("mods_ok", 0), len(c["relaxed"]))
    chosen = []
    for c in sorted(cands, key=rank):                # stable: play-count order within a rank
        if len(chosen) == quota:
            break
        if c["beatmapset_id"] in used_sets or mapper_count[c["mapper_id"]] >= MAX_PER_MAPPER:
            continue
        chosen.append(c)
        used_sets.add(c["beatmapset_id"])
        mapper_count[c["mapper_id"]] += 1
    return chosen


# ---------------------------------------------------------------- API (cached)

def search(api, cache, style: str | None, b: int) -> list[int]:
    """Ranked mapsets in band b, most played first: those tagged `skillset/<style>`, or every one."""
    key = f"test:{style or 'any'}:{b}"
    if key not in cache.data["search"]:
        lo, hi = star_range(b)
        query = ((f'tag="skillset/{style}" ' if style else "") + f"stars>={lo} stars<{hi} "
                 f"length>={DRAIN[0]} length<={DRAIN[1] + 60}")
        ids, cursor = [], None
        while len(ids) < MAX_SETS:
            r = with_backoff(lambda: api.search_beatmapsets(query, mode=0, category="ranked",
                                                            sort="plays_desc", cursor=cursor), f"search {key}")
            time.sleep(4)
            ids += [s.id for s in r.beatmapsets]
            cursor = r.cursor
            if not r.beatmapsets or cursor is None:
                break
        cache.data["search"][key] = ids[:MAX_SETS]
        cache.save()
    return cache.data["search"][key]


def candidates(api, cache, style, b, tag_names, skip_sets, skip_maps, enough) -> list[dict]:
    """A style's candidates in band b, play-count order, one per mapset; stops after `enough`. The
    untagged search is only scanned (and only fetched) if the tagged one runs out."""
    out, seen = [], set()
    def set_ids():
        yield from search(api, cache, style, b)
        yield from search(api, cache, None, b)
    for set_id in set_ids():
        if set_id in skip_sets or set_id in seen:
            continue
        seen.add(set_id)
        info = set_info(api, cache, set_id, tag_names)
        if "ranked" not in info["status"].lower() or any(d["beatmap_id"] in skip_maps for d in info["diffs"]):
            continue
        for d in sorted(info["diffs"], key=lambda d: -d["playcount"]):
            if qualifies(d, style, b) is None:           # static filters before any leaderboard call
                continue
            a = usable(api, cache, d["beatmap_id"], "NM")
            d = {**d, "nm98": a["98"], "nm95": a["95"]}
            relaxed = qualifies(d, style, b)
            if relaxed is None:
                continue
            c = {**d, "relaxed": relaxed, "beatmapset_id": set_id, "artist": info["artist"],
                 "title": info["title"], "creator": info["creator"], "style": style}
            if b in IN_BAND:
                c["mods"] = {m: usable(api, cache, d["beatmap_id"], m)["98"] for m in ("HR", "HD", "DT")}
                c["mods_ok"] = sum(c["mods"][m] >= MODS_NEEDED for m in ("HR", "DT"))
            out.append(c)
            print(f"    {d['beatmap_id']} {info['artist']} - {info['title']} [{d['version']}] {d['stars']}* "
                  f"NM {a['98']}" + (f" mods {c['mods']}" if "mods" in c else "")
                  + (f" relaxed {relaxed}" if relaxed else ""), flush=True)
            break                                        # one difficulty per mapset
        if sum(not c["relaxed"] and c.get("mods_ok", 2) == 2 for c in out) >= enough:
            break
    return out


# ---------------------------------------------------------------- main

def entry(c: dict) -> dict:
    """A split entry, with the v1 fields where they exist."""
    available = {"NM": {"acc98": c["nm98"], "acc95": c["nm95"]}} | {m: {"acc98": n} for m, n in c.get("mods", {}).items()}
    return {"beatmap_id": c["beatmap_id"], "beatmapset_id": c["beatmapset_id"], "artist": c["artist"],
            "title": c["title"], "version": c["version"], "mapper_id": c["mapper_id"], "creator": c["creator"],
            "stars": c["stars"], "bpm": c["bpm"], "drain_s": c["drain"], "cs": c["cs"], "ar": c["ar"],
            "od": c["od"], "style": c["style"], "style_votes": style_votes(c["tags"]),
            "playcount": c["playcount"], "available": available, "relaxed": c["relaxed"], "pilot": False}


def main():
    ap = argparse.ArgumentParser(description="Find test maps until every star band has enough.")
    ap.add_argument("--write", action="store_true", help="append the picks to configs/splits/test")
    args = ap.parse_args()

    files = split_files()
    current = {}
    for p in files:
        for m in yaml.safe_load(p.read_text(encoding="utf-8"))["maps"]:
            current[m["beatmap_id"]] = m
    mapper_count = Counter(m["mapper_id"] for m in current.values() if m["split"] in BC_SPLITS and m.get("mapper_id"))
    used_sets = excluded_sets(files)
    skip_maps = excluded_maps([CHALLENGE])

    api = connect()
    cache = Cache(CACHE)
    tl = api.tags()
    tag_names = {t.id: t.name for t in getattr(tl, "tags", tl)}

    picks = []
    for b in BANDS:
        have = Counter(m["style"] for m in current.values() if m["split"] == "test" and band(m["stars"]) == f"{b}_star")
        want = quotas(have)
        need = sum(want.values())
        if need == 0:
            print(f"== {b}_star: {dict(have)}, enough")
            continue
        print(f"== {b}_star: have {dict(have)}, adding {dict(want)}")
        chosen, pools = [], {}
        for style in STYLE_ORDER:
            print(f"  {style}:")
            pools[style] = candidates(api, cache, style, b, tag_names, used_sets, skip_maps, want[style] + 2)
            got = pick(pools[style], want[style], mapper_count, used_sets)
            chosen += got
        for style in STYLE_ORDER:                        # short styles hand their places to the others
            if len(chosen) >= need:
                break
            chosen += pick(pools[style], need - len(chosen), mapper_count, used_sets)
        if len(chosen) < need:
            print(f"  SHORT: {len(chosen)} of {need}")
        print(f"  {b}_star now: {dict(have + Counter(c['style'] for c in chosen))}")
        picks += chosen

    print(f"\n{len(picks)} picks:")
    for c in picks:
        print(f"  {band(c['stars']):7} {c['style']:8} {c['beatmap_id']:>8} {c['stars']:5.2f}* AR{c['ar']:<4} "
              f"NM {c['nm98']:>3}" + (f" HR {c['mods']['HR']:>3} HD {c['mods']['HD']:>3} DT {c['mods']['DT']:>3}"
                                       if "mods" in c else " " * 21)
              + f"  {c['artist']} - {c['title']} [{c['version']}]" + (f"  relaxed {c['relaxed']}" if c["relaxed"] else ""))
    if not args.write:
        print("\n(dry run: --write appends them to configs/splits/test)")
        return

    written = add_maps([entry(c) for c in picks], "test", f"data/find_test_maps.py ({date.today().isoformat()})")
    for name, group in (("inband", [c for c in picks if c["stars"] < 7]), ("hard", [c for c in picks if c["stars"] >= 7])):
        lines = [f"{c['beatmap_id']}  # {c['style']}: {c['artist']} - {c['title']} [{c['version']}] {c['stars']}*"
                 for c in group]
        (LISTS / f"test_bands_{name}.txt").write_text(
            f"# test maps added by data/find_test_maps.py ({'in-band: NM HR HD DT' if name == 'inband' else '7*+: NM'})\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {', '.join(map(str, written))} and {LISTS}/test_bands_{{inband,hard}}.txt")


if __name__ == "__main__":
    main()
