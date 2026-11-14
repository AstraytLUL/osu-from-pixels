"""One evaluation for every condition of both experiments (the brief's "Evaluation" rules).

    python -m rl.evaluate --label A_bc --checkpoint runs/baseline_cap_seed0/best.pt \
        --maps heldout --mods HD --out rl/results/exp1.csv

Deterministic policy (the Gaussian mean = the BC-format checkpoint; clicks by the BC click rule),
closed-loop through eval/rollout.py's play() under the speed cap, judged by osu_sim/judge.py: the
same path as every number in the report. PPO and supervised fine-tunes are saved as BC-format
checkpoints (rl/ppo.py export), so this script cannot treat conditions differently.

One row per (label, seed, map, mods), appended to --out; rows already there are skipped (resumable).
Each hit's signed error goes to <out>.offsets.jsonl (one line per row), for the hit-error histograms.
Metrics: accuracy; counts; misses (circles, sliders heads, spinners), split for circles into
"wrong position" (cursor off the circle at its hit time and never clicked on it) and "no click / bad
timing" (cursor on it, or clicked outside the window), as eval/diagnose.py; max combo / full combo;
slider breaks and dropped ends; unstable rate; mean signed hit error (ms, < 0 early) over hit circles
and slider heads.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from data.dataset import load_maps, mods_from_str
from eval.diagnose import circle_rows
from eval.motor import CAP
from eval.policy import AutoPolicy, load_model_policy
from eval.rollout import play
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import accuracy, counts, unstable_rate
from osu_sim.playable import make_playable
from osu_sim.score import combo_breaks, max_combo

MAP_LISTS = Path("rl/maps.yaml")
FIELDS = ["label", "seed", "checkpoint", "beatmap_id", "mods", "stars", "style", "accuracy", "count_300",
          "count_100", "count_50", "count_miss", "miss_circle_position", "miss_circle_click", "objects",
          "max_combo", "full_combo", "max_combo_ratio", "slider_breaks", "end_drops", "unstable_rate",
          "mean_hit_error"]
_FULL_COMBO: dict = {}


def map_list(name: str) -> list[int]:
    """A named map list of rl/maps.yaml (fixed before any RL training), split:<name> (every map of
    that split, e.g. split:val), or comma-separated ids."""
    if name.startswith("split:"):
        return sorted(b for b, e in load_maps().items() if e["split"] == name[6:])
    lists = yaml.safe_load(MAP_LISTS.read_text(encoding="utf-8"))
    if name in lists:
        return [int(b) for b in lists[name]["maps"]]
    return [int(b) for b in name.split(",")]


def osu_path(bid: int) -> Path:
    """The .osu file: data/raw/beatmaps, or data/challenge for the challenge maps (no replays)."""
    p = Path("data/raw/beatmaps") / f"{bid}.osu"
    return p if p.exists() else Path("data/challenge") / f"{bid}.osu"


def playable(bid: int, mods: str):
    return make_playable(parse_beatmap(osu_path(bid)), mods_from_str(mods))


def full_combo(bid: int, mods: str, pl) -> int:
    if (bid, mods) not in _FULL_COMBO:
        _, js = play(AutoPolicy(), bid, pl, motor=CAP, seed=bid)
        _FULL_COMBO[(bid, mods)] = max_combo(js, pl)
    return _FULL_COMBO[(bid, mods)]


def evaluate_map(policy, bid: int, mods: str) -> dict:
    pl = playable(bid, mods)
    g, js = play(policy, bid, pl, motor=CAP, seed=bid)
    n = counts(js)
    circles = circle_rows(pl, g, js)
    b = combo_breaks(js)
    offsets = [j.offset for j in js if j.offset is not None and j.result > 0 and j.kind != "spinner"]
    fc = full_combo(bid, mods, pl)
    mc = max_combo(js, pl)
    return {"beatmap_id": bid, "mods": mods, "accuracy": accuracy(js), "count_300": n[300], "count_100": n[100],
            "count_50": n[50], "count_miss": n[0],
            "miss_circle_position": sum(r["class"] == "miss_aim" for r in circles),
            "miss_circle_click": sum(r["class"] == "miss_timing" for r in circles),
            "objects": len(js), "max_combo": mc, "full_combo": fc, "max_combo_ratio": mc / fc if fc else None,
            "slider_breaks": b["slider_break"], "end_drops": b["end_drop"],
            "unstable_rate": unstable_rate(js) if offsets else None,
            "mean_hit_error": float(np.mean(offsets)) if offsets else None,
            "offsets": [round(o, 1) for o in offsets]}


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {(r["label"], r["seed"], int(r["beatmap_id"]), r["mods"]) for r in csv.DictReader(f)}


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run(label: str, seed, checkpoint: str, maps: list[int], mods_list: list[str], out: Path, device="cuda") -> int:
    info = load_maps()
    done = done_keys(out)
    todo = [(b, m) for m in mods_list for b in maps if (label, str(seed), b, m) not in done]
    if not todo:
        return 0
    policy = load_model_policy(checkpoint, device)
    for bid, mods in todo:
        row = evaluate_map(policy, bid, mods)
        e = info.get(bid, {})
        append(out, {**row, "label": label, "seed": seed, "checkpoint": checkpoint,
                     "stars": e.get("stars"), "style": e.get("style")})
        with out.with_suffix(".offsets.jsonl").open("a") as f:    # per-hit errors, for histograms
            f.write(json.dumps({"label": label, "seed": seed, "beatmap_id": bid, "mods": mods,
                                "offsets": row["offsets"]}) + "\n")
    return len(todo)


def main():
    ap = argparse.ArgumentParser(description="Deterministic closed-loop evaluation for the RL experiments.")
    ap.add_argument("--label", required=True, help="condition, e.g. exp1_A, exp1_B, exp2_B_4952940")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--maps", required=True, help="a list name in rl/maps.yaml, or comma-separated ids")
    ap.add_argument("--mods", nargs="+", default=["NM"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    n = run(args.label, args.seed, args.checkpoint, map_list(args.maps), args.mods, args.out, args.device)
    print(f"{args.label} seed {args.seed}: {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
