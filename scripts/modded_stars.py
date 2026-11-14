"""Star ratings of the training-range test maps with a modifier applied (Fig. 7 of the report). Run
from the repo root:

    python scripts/modded_stars.py            # compare the API's ratings with the committed file
    python scripts/modded_stars.py --write    # rewrite report/data/modded_stars.json

For each of the 19 test maps rated 4 to 6 stars, the official star rating under HR, DT and HR+DT, as
returned by the beatmap-attributes endpoint of the osu! API v2 (needs the osu! credentials in .env,
like data/download.py). The committed file was retrieved on 10 Nov 2026. Ratings change when the
game's difficulty algorithm does, so a later run can differ slightly.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from data.download import connect, with_backoff

DATA = Path("report/data")
MODS = ("HR", "DT", "HRDT")
BANDS = ("4_star", "5_star", "6_star")


def maps() -> list[int]:
    """The training-range test maps, in the order of the evaluation results."""
    with open(DATA / "headline_results.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["band"] in BANDS and r["mods"] == "NM"]
    return list(dict.fromkeys(int(r["beatmap_id"]) for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the committed file")
    args = parser.parse_args()
    api = connect()
    stars = {}
    for bid in maps():
        for mods in MODS:
            attrs = with_backoff(lambda: api.beatmap_attributes(bid, mods=mods), f"map {bid} {mods}")
            stars[f"{bid}:{mods}"] = attrs.attributes.star_rating
            time.sleep(1.0)
    path = DATA / "modded_stars.json"
    if args.write:
        path.write_text(json.dumps(stars, indent=1), encoding="utf-8")
        print(f"wrote {len(stars)} ratings to {path}")
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(set(old) ^ set(stars))
    diffs = {k: stars[k] - old[k] for k in stars if k in old}
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    print(f"{len(stars)} ratings from the API, {len(old)} in {path}; keys in only one of them: {missing}")
    print(f"largest difference: {diffs[worst]:+.4f} stars ({worst}); "
          f"{sum(abs(d) > 0.005 for d in diffs.values())} differ by more than 0.005")


if __name__ == "__main__":
    main()
