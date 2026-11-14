"""The report's results table: every model and seed of an eval config, played on the test split, one
row per (model, seed, map, mods) in one long-format CSV (the project plan M6).

    python -m eval.headline list --config configs/eval/headline.yaml       # runs, and how many rows each has
    python -m eval.headline prepare --config configs/eval/headline.yaml    # once: render every cell's frames
    python -m eval.headline run --config configs/eval/headline.yaml        # resumable
    python -m eval.headline humans --config configs/eval/headline.yaml     # the human reference rows
    CUDA_VISIBLE_DEVICES=0 python -m eval.headline run --config ... --lane 0/2   # parallel lanes: one
                                                                                 # process each, any GPU

The config (configs/eval/headline.yaml) names the split, the motor, the cells (mods x star bands) and
the models: local run folders or Hub repos (train/hub.py pull fetches those). Auto is played once per
map and cell as the model "auto" (it does not depend on the checkpoint), under auto_motor if the
config names one. `humans` judges every
downloaded replay of a cell's map and mods (data/raw/manifest.csv) the same way, as the model "human"
with the score id as its seed: the reference is the top of each leaderboard (>= 98% accuracy, 95% on
the maps data/find_test_maps.py relaxed), not a typical player. Human rows also keep the replay's
leaderboard accuracy and era: our judge follows osu!stable (note lock), and lazer-set replays on dense
maps (9 Nov: 16 of the 20 at 9*) score several points lower under it, while stable replays agree
within 0.05 pp. The reference uses only replays the judge reproduces (within AGREE); see human_ok. Rows hold eval.rollout's
per-map results (accuracy, counts, unstable rate, combo breaks by cause) plus the
map's band, style and stars. A row already in the CSV is never played again, so a rerun picks up
where it stopped; lanes append to the same file (one short write per row).
"""
import argparse
import csv
import os
from pathlib import Path

import yaml

FIELDS = ["model", "seed", "beatmap_id", "mods", "band", "style", "stars", "accuracy", "count_300", "count_100",
          "count_50", "count_miss", "objects", "breaks_miss", "breaks_slider", "end_drops", "unstable_rate",
          "motor", "leaderboard_accuracy", "era"]
AGREE = 0.005                                  # humans: judge within 0.5 pp of the leaderboard (module doc)


def cell_maps(cfg: dict) -> list[tuple[str, int, dict]]:
    """(mods, beatmap_id, split entry) for every cell of the config, in a fixed order."""
    from data.dataset import load_maps
    from data.splits import band
    maps = {b: e for b, e in load_maps().items() if e["split"] == cfg["split"]}
    out = []
    for cell in cfg["cells"]:
        for bid, e in sorted(maps.items(), key=lambda kv: (kv[1]["stars"], kv[0])):
            if cell["bands"] == "all" or band(e["stars"]) in cell["bands"]:
                out.append((cell["mods"], bid, e))
    return out


def runs(cfg: dict, api=None) -> list[tuple[str, int, dict]]:
    """(model, seed, source) for every run of the config. Hub repos with seeds: all are listed."""
    out = []
    for name, m in cfg["models"].items():
        seeds = m["seeds"]
        if seeds == "all":
            from eval.test_all import seeds_in
            org = os.environ.get("HF_ORG", "sc4001-project")
            from huggingface_hub.errors import RepositoryNotFoundError
            try:
                files = api.list_repo_files(f"{org}/{m['hub']}", repo_type="model")
            except RepositoryNotFoundError:          # private: this account cannot see it (yet)
                print(f"{name}: {org}/{m['hub']} not found or no access, skipped", flush=True)
                continue
            seeds = seeds_in(files)
            if "config.yaml" in files and "best.pt" in files:            # bc-baseline: seed 0 at the root
                seeds = sorted(set(seeds) | {0})
        out += [(name, s, m) for s in seeds]
    return out


def checkpoint(name: str, seed: int, m: dict, api) -> Path | None:
    """The run's best.pt, pulled from the Hub first if it is not here yet; None for a local run that
    has not finished training (best.pt exists from the first epoch on, so train.log's last line,
    "best val ...", is what says it is final)."""
    if "local" in m:
        folder = Path(m["local"].format(seed=seed))
        log = folder / "train.log"
        if not log.exists() or "best val" not in log.read_text(encoding="utf-8"):
            return None
        return folder / "best.pt"
    folder = Path(f"runs/{name}_seed{seed}")
    if not (folder / "best.pt").exists():
        from train.hub import pull
        folder = pull(api, f"{os.environ.get('HF_ORG', 'sc4001-project')}/{m['hub']}", seed, folder)
    return folder / "best.pt"


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {(r["model"], int(r["seed"]), int(r["beatmap_id"]), r["mods"]) for r in csv.DictReader(f)}


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def play_cells(policy, model: str, seed: int, cells, cfg: dict, out: Path, done: set) -> int:
    from data.dataset import mods_from_str
    from data.splits import band
    from eval.motor import MOTORS
    from eval.rollout import evaluate
    from osu_sim.beatmap import parse_beatmap
    from osu_sim.playable import make_playable
    motor = cfg.get("auto_motor", cfg["motor"]) if model == "auto" else cfg["motor"]
    n = 0
    for mods, bid, e in cells:
        if (model, seed, bid, mods) in done:
            continue
        pl = make_playable(parse_beatmap(Path("data/raw/beatmaps") / f"{bid}.osu"), mods_from_str(mods))
        r = evaluate(policy, [(bid, e["style"], pl)], auto_gap=False, motor=MOTORS[motor])["maps"][0]
        append(out, {**r, "model": model, "seed": seed, "mods": mods, "band": band(e["stars"]),
                     "stars": e["stars"], "motor": motor})
        n += 1
    return n


def human_ok(row: dict) -> bool:
    """Whether a human row counts in the reference: the judge reproduces its leaderboard accuracy."""
    return abs(float(row["accuracy"]) - float(row["leaderboard_accuracy"])) <= AGREE


def judge_humans(cells, cfg: dict, out: Path, done: set, raw: Path = Path("data/raw")) -> int:
    """A row per downloaded replay of every cell (module doc)."""
    from data.dataset import mods_from_str
    from data.replay import load_replay
    from data.splits import band
    from eval.rollout import result_row
    from osu_sim.beatmap import parse_beatmap
    from osu_sim.judge import judge
    from osu_sim.playable import make_playable
    with (raw / "manifest.csv").open(newline="") as f:
        manifest = list(csv.DictReader(f))
    n = 0
    for mods, bid, e in cells:
        replays = [r for r in manifest if int(r["beatmap_id"]) == bid and r["mods"] == mods]
        pl = make_playable(parse_beatmap(raw / "beatmaps" / f"{bid}.osu"), mods_from_str(mods))
        for r in replays:
            if ("human", int(r["score_id"]), bid, mods) in done:
                continue
            row = result_row(bid, e["style"], pl, judge(pl, load_replay(raw / r["osr_path"])))
            append(out, {**row, "model": "human", "seed": int(r["score_id"]), "mods": mods,
                         "band": band(e["stars"]), "stars": e["stars"], "motor": "human replay",
                         "leaderboard_accuracy": float(r["accuracy"]), "era": r["era"]})
            n += 1
    return n


def main():
    from dotenv import load_dotenv
    load_dotenv()
    ap = argparse.ArgumentParser(description="The report's results table, from an eval config.")
    ap.add_argument("action", choices=["list", "prepare", "run", "humans"])
    ap.add_argument("--config", type=Path, default=Path("configs/eval/headline.yaml"))
    ap.add_argument("--models", nargs="+", help="only these models of the config (auto only if named)")
    ap.add_argument("--lane", default="0/1", help="i/n: take every n-th run starting at i")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.models:
        cfg["models"] = {k: v for k, v in cfg["models"].items() if k in args.models}
    out = Path(cfg["out"])
    cells = cell_maps(cfg)

    if args.action == "prepare":                 # render first, so parallel lanes never write one cache file
        from data.dataset import build_map_frames, mods_from_str
        from osu_sim.beatmap import parse_beatmap
        from osu_sim.playable import make_playable
        for mods, bid, _ in cells:
            build_map_frames(make_playable(parse_beatmap(Path("data/raw/beatmaps") / f"{bid}.osu"),
                                           mods_from_str(mods)), bid)
        print(f"{len(cells)} map x mods cells rendered")
        return

    if args.action == "humans":
        n = judge_humans(cells, cfg, out, done_keys(out))
        print(f"{n} human rows -> {out}")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    # two processes with different --models would otherwise both play auto (9 Nov: every auto row twice)
    todo = ([("auto", 0, None)] if not args.models or "auto" in args.models else []) + runs(cfg, api)
    done = done_keys(out)
    if args.action == "list":
        for name, seed, _ in todo:
            n = sum((name, seed, bid, mods) in done for mods, bid, _ in cells)
            print(f"{name:20} seed {seed}  {n:>4} / {len(cells)} rows")
        return

    from eval.policy import AutoPolicy, load_model_policy
    i, n = (int(v) for v in args.lane.split("/"))
    for name, seed, m in [r for k, r in enumerate(todo) if k % n == i]:
        if all((name, seed, bid, mods) in done for mods, bid, _ in cells):
            print(f"== {name} seed {seed}: done", flush=True)
            continue
        if m is None:
            policy = AutoPolicy()
        else:
            ckpt = checkpoint(name, seed, m, api)
            if ckpt is None:
                print(f"== {name} seed {seed}: still training, skipped (rerun later)", flush=True)
                continue
            policy = load_model_policy(str(ckpt), args.device)
        played = play_cells(policy, name, seed, cells, cfg, out, done)
        print(f"== {name} seed {seed}: {played} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
