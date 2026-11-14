"""The map splits: which maps are trained, validated and tested on, by star band.

    python -m data.splits list                 # maps per split and band, with star range and styles
    python -m data.splits build                # once (done 9 Nov): the old map lists -> configs/splits/

configs/splits/<split>/<band>.yaml, where a band is whole stars ("4_star" holds 4.00-4.99*):
  train/    BC training maps: v1 train and the spinner maps (data/select_maps.py, data/find_maps.py)
  val/      NoMod validation, for model selection (v1 val)
  test/     held-out maps, played once at the end (v1 test, and harder bands as they are added)
configs/splits/challenge.yaml is a separate showcase of very hard maps, in no split.

Each band file is {meta, maps}, every map entry as the script that found it wrote it (beatmap_id,
stars, style, ...) and its `split`. data/dataset.load_maps reads any of these files, or a directory of
them (configs/splits for every split). A map is in exactly one file. The splits are frozen: maps can
be added (data/find_maps.py appends), but an entry is never moved or removed once it has been used.

`build` made the splits from configs/legacy/maps/ (maps_v1.yaml, maps_spinners.yaml),
which stay as the record of how they were chosen; the band is the entry's own `stars`.
"""
import argparse
import math
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

SPLITS = Path("configs/splits")
CHALLENGE = SPLITS / "challenge.yaml"
LEGACY = Path("configs/legacy/maps")
LEGACY_LISTS = (LEGACY / "maps_v1.yaml", LEGACY / "maps_spinners.yaml")
NAMES = ("train", "val", "test")


def band(stars: float) -> str:
    """The whole-star band of a star rating: 4.0-4.99 -> "4_star"."""
    return f"{math.floor(stars)}_star"


def _band_key(path: Path) -> tuple[str, int]:
    return path.parent.name, int(path.stem.split("_")[0])


def split_files(root: Path = SPLITS) -> list[Path]:
    """Every band file under `root` (configs/splits, or one split's directory), split by split and
    band by band in star order."""
    return sorted(root.rglob("*_star.yaml"), key=_band_key)


def expand(paths) -> tuple[Path, ...]:
    """Map lists with any directory replaced by the band files in it."""
    out = []
    for p in map(Path, paths):
        out += split_files(p) if p.is_dir() else [p]
    return tuple(out)


def add_maps(maps: list[dict], split: str, source: str, root: Path = SPLITS) -> list[Path]:
    """Append `maps` (entries with `stars`) to `split`'s band files, creating the ones that don't exist
    yet; every map gets split = `split`. Returns the files written."""
    if split not in NAMES:
        raise ValueError(f"unknown split {split!r}; one of {NAMES}")
    by_band: dict[str, list[dict]] = {}
    for m in maps:
        by_band.setdefault(band(m["stars"]), []).append({**m, "split": split})
    written = []
    for b, new in by_band.items():
        path = root / split / f"{b}.yaml"
        if path.exists():
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            doc["meta"]["sources"] = list(dict.fromkeys(doc["meta"]["sources"] + [source]))
        else:
            doc = {"meta": {"split": split, "band": b, "sources": [source]}, "maps": []}
        doc["meta"]["updated"] = date.today().isoformat()
        doc["maps"] += new
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {split}, {b}: see data/splits.py. Add maps, never move or remove one.\n"
                        + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(path)
    return written


def build(lists=LEGACY_LISTS, root: Path = SPLITS) -> None:
    """Write the band files from the old map lists, keeping each list's order within a band."""
    if split_files(root):
        raise SystemExit(f"{root} already has band files: the splits are frozen")
    for path in lists:
        maps = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["maps"]
        for split in NAMES:
            add_maps([m for m in maps if m["split"] == split], split, str(path).replace("\\", "/"), root)


def summary(root: Path = SPLITS) -> list[str]:
    lines = [f"{'split':8}{'band':8}{'maps':>5}  {'stars':11}  styles"]
    for path in split_files(root):
        maps = yaml.safe_load(path.read_text(encoding="utf-8"))["maps"]
        stars = [m["stars"] for m in maps]
        styles = Counter(m.get("style") or "-" for m in maps)
        lines.append(f"{path.parent.name:8}{path.stem:8}{len(maps):>5}  {min(stars):.2f}-{max(stars):.2f}  "
                     + ", ".join(f"{s} {n}" for s, n in styles.most_common()))
    return lines


def main():
    ap = argparse.ArgumentParser(description="The map splits by star band.")
    ap.add_argument("action", choices=["list", "build"])
    args = ap.parse_args()
    if args.action == "build":
        build()
    print("\n".join(summary()))


if __name__ == "__main__":
    main()
