"""Experiment 1, condition C: supervised fine-tune of the BC policy on HD re-renders of its own data.

    python -m rl.sft_hd --seed 0 --epochs 20 --name exp1_C20
    python -m rl.sft_hd --seed 0 --epochs 20 --name exp1_Cmix --mix   # addition: C-mix (rl/RESULTS.md)

HD changes only what is drawn (no approach circles, objects fade out early); positions and timing are
the same. So the BC training replays (train split, NoMod + HR) are valid action labels for the same
maps rendered with HD: this re-renders them and fine-tunes from the BC checkpoint with train/bc.py's
own loop (same loss, same closed-loop model selection), selecting the epoch on the val maps played
with HD (the val split, never the held-out maps). Output: rl/runs/exp1_C_seed<seed>/ (best.pt etc.).
"""
import argparse
import copy
import csv
from pathlib import Path

import yaml

from data.dataset import RAW, ReplayWindows, load_maps, mods_from_str
from data.replay import load_replay
from eval.rollout import split_maps
from osu_sim.beatmap import parse_beatmap
from osu_sim.mods import HD
from osu_sim.playable import make_playable
from train.bc import fit, log_line

BC_CONFIG = Path("runs/baseline_cap_seed0/config.yaml")
BC_CHECKPOINT = "runs/baseline_cap_seed0/best.pt"


def hd_windows(d: dict, mix: bool = False) -> ReplayWindows:
    """data/dataset.py from_split for the train split, every map rendered with HD added (mix: each
    replay twice, once as recorded and once with HD added)."""
    maps = load_maps(tuple(Path(p) for p in d["map_lists"]))
    rows = [r for r in csv.DictReader((RAW / "manifest.csv").open(newline=""))
            if int(r["beatmap_id"]) in maps and maps[int(r["beatmap_id"])]["split"] == "train"
            and r["mods"] in d["train_mods"]]
    items, playables = [], {}
    for r in rows:
        bid, recorded = int(r["beatmap_id"]), mods_from_str(r["mods"])
        for m in ([recorded, recorded | HD] if mix else [recorded | HD]):
            if (bid, m) not in playables:
                playables[(bid, m)] = make_playable(parse_beatmap(RAW / r["osu_path"]), m)
            frames = load_replay(RAW / r["osr_path"])
            frames.mods = m
            items.append((bid, playables[(bid, m)], frames))
    return ReplayWindows(items, window=d["window"], canonical_hr=d["canonical_hr"],
                         cursor_jitter=d.get("cursor_jitter", 0.0), cursor_channel=d.get("cursor_channel", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="exp1_C", help="run name: rl/runs/<name>_seed<seed>/")
    ap.add_argument("--mix", action="store_true",
                    help="C-mix: train on the replays as recorded and with HD; select on val with HD and NoMod")
    args = ap.parse_args()
    cfg = copy.deepcopy(yaml.safe_load(BC_CONFIG.read_text()))
    cfg.update(name=args.name, seed=args.seed)
    cfg["train"].update(init=BC_CHECKPOINT, epochs=args.epochs, lr=args.lr)
    out = Path(f"rl/runs/{args.name}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    train_set = hd_windows(cfg["data"], args.mix)
    val_maps = split_maps("val", "HD") + (split_maps("val", "NM") if args.mix else [])
    what = "the replays as recorded and HD re-renders" if args.mix else "HD re-renders"
    log_line(out, f"{args.name} seed {args.seed}: fine-tune {BC_CHECKPOINT} on {what}, {len(train_set):,} windows, "
                  f"{args.epochs} epochs at lr {args.lr}; selection on {len(val_maps)} val plays "
                  f"({'HD and NoMod' if args.mix else 'HD'})")
    best = fit(cfg, train_set, val_maps, out)
    log_line(out, f"best val accuracy {best:.2%}")


if __name__ == "__main__":
    main()
