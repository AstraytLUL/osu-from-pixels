"""One command to get the data and pre-render the frames, for teammates (no osu! account needed):

    python -m data.prepare                 # pull data/raw/ from Hugging Face, cache BC frames
    python -m data.prepare --check         # only report what is there and what is missing

Steps:
1. pull: data/raw/ from the private dataset in env HF_DATASET (default below); needs read access and
   `hf auth login` or HF_TOKEN (see data/hub.py). Skipped with --no-pull.
2. check: every listed map has its .osu, and every manifest replay on a listed map has its .osr.
3. cache: render each map's layers once per mod set into $OSU_FRAME_CACHE (data/dataset.py), several
   maps in parallel. Already cached ones are skipped, so it is safe to stop and rerun.
     BC   train maps (configs/splits/train) under NM and HR, val maps under NM: what
          `python -m train.bc --config configs/baseline.yaml` reads. --test adds the test maps
          under NM HR HD DT, for the transfer matrix.
   About 20 MB per map and mod set: the BC set is ~3.5 GB. rl/ renders what it needs on first use.

Using the maps afterwards (e.g. in an RL environment):

    from eval.rollout import split_maps
    from data.dataset import MapFrames, build_map_frames
    maps = split_maps("train", "NM")                         # [(beatmap_id, style, Playable)]
    beatmap_id, _, pl = maps[0]
    layers = MapFrames(build_map_frames(pl, beatmap_id))    # cached map layers on the 60 fps grid
    layers.frame(k)                                          # (96, 128) uint8, no cursor
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from data.dataset import RAW, build_map_frames, cache_dir, load_maps, map_frames_path, mods_from_str
from data.hub import pull
from data.splits import SPLITS
from osu_sim.beatmap import parse_beatmap
from osu_sim.playable import make_playable

BC_LISTS = (SPLITS / "train", SPLITS / "val", SPLITS / "test")
DEFAULT_REPO = "sc4001-project/osu-mod-transfer-raw"


def frame_jobs(test: bool) -> list[tuple[int, str]]:
    """(beatmap_id, mods) pairs to render, in a fixed order."""
    bc = load_maps(BC_LISTS)
    mods_per_split = {"train": ["NM", "HR"], "val": ["NM"]} | ({"test": ["NM", "HR", "HD", "DT"]} if test else {})
    jobs = [(bid, m) for bid, e in bc.items() for m in mods_per_split.get(e["split"], [])]
    return list(dict.fromkeys(jobs))


def render(job: tuple[int, str]) -> tuple[int, str]:
    """Render and cache one (map, mods); runs in a worker process."""
    bid, mods = job
    pl = make_playable(parse_beatmap(RAW / "beatmaps" / f"{bid}.osu"), mods_from_str(mods))
    build_map_frames(pl, bid)
    return job


def check(lists: tuple[Path, ...]) -> list[str]:
    """What is missing from data/raw/ for the listed maps."""
    maps = load_maps(lists)
    missing = [f"beatmaps/{bid}.osu" for bid in maps if not (RAW / "beatmaps" / f"{bid}.osu").exists()]
    manifest = RAW / "manifest.csv"
    if manifest.exists():
        for r in csv.DictReader(manifest.open(newline="")):
            if int(r["beatmap_id"]) in maps and not (RAW / r["osr_path"]).exists():
                missing.append(r["osr_path"])
    else:
        missing.append("manifest.csv")
    return missing


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Pull the data and cache rendered frames.")
    ap.add_argument("--no-pull", action="store_true", help="use data/raw/ as it is")
    ap.add_argument("--check", action="store_true", help="only report what is there and what is missing")
    ap.add_argument("--test", action="store_true", help="also the test maps under NM HR HD DT")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    if not args.no_pull and not args.check:
        pull(os.environ.get("HF_DATASET", DEFAULT_REPO), os.environ.get("HF_TOKEN"))

    lists = BC_LISTS
    missing = check(lists)
    maps = load_maps(lists)
    print(f"{len(maps)} maps in {', '.join(map(str, lists))}; {len(missing)} files missing")
    for path in missing[:20]:
        print(f"  missing data/raw/{path}")
    if missing:
        raise SystemExit("pull again (python -m data.prepare), or ask Astrayt to push the missing files")

    jobs = frame_jobs(args.test)
    todo = [j for j in jobs if not map_frames_path(j[0], mods_from_str(j[1])).with_suffix(".idx.npy").exists()]
    print(f"frame cache {cache_dir()}: {len(jobs) - len(todo)} of {len(jobs)} (map, mods) already cached")
    if args.check or not todo:
        return
    with ProcessPoolExecutor(args.workers) as pool:
        for i, f in enumerate(as_completed(pool.submit(render, j) for j in todo), 1):
            bid, mods = f.result()
            print(f"  {i}/{len(todo)} cached {bid} {mods}", flush=True)
    size = sum(p.stat().st_size for p in cache_dir().glob("*.bin"))
    print(f"done; cache {size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
