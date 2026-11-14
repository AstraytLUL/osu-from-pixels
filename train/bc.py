"""Stage 1: behaviour cloning. Learn the next frame's cursor move and keys from players' replays,
pick the checkpoint that *plays* best.

    python -m train.bc --config configs/baseline.yaml                 # writes runs/<name>_seed<seed>/
    python -m train.bc --config configs/baseline.yaml --resume        # continue from its last.pt
    python -m train.bc --config configs/arch/cnn3d.yaml --seed 1      # another seed: runs/cnn3d_seed1/
    python -m train.bc --config configs/arch/cnn3d.yaml --smoke       # a few minutes: does it run at all?

The network comes from the config's `model` section (models/heads.py build_policy): model.backbone
(framestack | cnn3d | cnn_gru), model.width and model.aim_head (linear | softargmax); left out, they
are the baseline's. See docs/architectures.md.

Data: data/dataset.py's training windows (the train split, the config's mods, HR flipped to NoMod
geometry). Each epoch draws `samples_per_epoch` windows at random; the split has ~4.5M, far more
than one pass needs.

Loss, per window (targets are the frame after the window):
  move   SmoothL1 between predicted and true cursor displacement, in units of MOVE_SCALE px; with
         model.aim: absolute, between predicted and true next position instead (same units)
  press  binary cross-entropy per key; positives weighted by press_pos_weight (a key is pressed on
         only ~4% of frames, so unweighted the model learns never to click)
  held   binary cross-entropy per key
  total  move_weight * move + press_weight * press + held_weight * held

Model selection: every `eval.every` epochs the model plays the NoMod val maps closed-loop
(eval/rollout.py) and best.pt keeps the highest mean judged accuracy (the project proposal: selection on
NoMod val accuracy only).
eval.extra_val adds more maps to that play ({split, map_list} entries; no replays needed). eval.max_step / eval.aim_noise set the motor limits for that play
(eval/motor.py; absent = HUMAN's). last.pt is written every epoch, for --resume.

The run directory gets: config.yaml, run.json (git revision, device, parameter count), metrics.jsonl
(one line per epoch), train.log (everything printed), val_epoch_NNN.json (per-map results), last.pt,
best.pt. train/hub.py uploads it to Hugging Face.

Devices: device "auto" picks CUDA, else Apple's GPU (MPS, a Mac), else the CPU. Mixed precision is
on CUDA only.

Fine-tuning: train.init starts from a checkpoint's weights instead of random ones (rl/sft_hd.py).
"""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from data.dataset import MAPS, from_split
from eval.policy import ModelPolicy
from eval.motor import motor_from
from eval.rollout import evaluate, split_maps
from models.backbones import count_parameters
from models.heads import CENTRE, MOVE_SCALE, build_policy


# ---------------------------------------------------------------- setup

def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if mps_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("config asks for CUDA, but torch sees no GPU (a CPU-only torch build?)")
    if name == "mps" and not mps_available():
        raise SystemExit("config asks for MPS (a Mac's GPU), but torch cannot use it here")
    return name


def apply_smoke(cfg: dict) -> dict:
    """A copy of `cfg` shrunk to a few minutes' run (2 epochs of 20k windows, 2 val maps), to check
    that a config runs end to end before the real run. Not for results."""
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    cfg["name"] = f"{cfg['name']}_smoke"
    cfg["train"].update(epochs=2, samples_per_epoch=20_000)
    cfg["eval"].update(every=1, val_maps=2)
    return cfg


def log_line(out: Path, text: str) -> None:
    """Print, and append to the run's train.log."""
    print(text, flush=True)
    with (out / "train.log").open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------- one training step

def bc_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict]:
    """The weighted BC loss (module doc) and its parts, for logging."""
    t = cfg["train"]
    if "position" in out:                       # absolute aim: the next position, from the centre
        centre = torch.tensor(CENTRE, device=out["position"].device)
        move = nn.functional.smooth_l1_loss(out["position"], (batch["target"] - centre) / MOVE_SCALE)
    else:                                       # relative aim: the next move
        move = nn.functional.smooth_l1_loss(out["move"], batch["action"] / MOVE_SCALE)
    pos_weight = torch.full((2,), float(t["press_pos_weight"]), device=out["press"].device)
    press = nn.functional.binary_cross_entropy_with_logits(out["press"], batch["press"], pos_weight=pos_weight)
    held = nn.functional.binary_cross_entropy_with_logits(out["held"], batch["held"])
    total = t["move_weight"] * move + t["press_weight"] * press + t["held_weight"] * held
    return total, {"move": move.item(), "press": press.item(), "held": held.item()}


def train_epoch(model, loader, optimizer, scaler, cfg, device) -> dict:
    """One pass over `loader`; returns the mean loss parts."""
    model.train()
    sums, batches = {"loss": 0.0, "move": 0.0, "press": 0.0, "held": 0.0}, 0
    for batch in loader:
        batch = {k: batch[k].to(device, non_blocking=True) for k in ("frames", "prev", "action", "target", "press", "held")}
        optimizer.zero_grad(set_to_none=True)
        # mixed precision on CUDA only (device_type "cpu" when off: older torch rejects "mps" here)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=device == "cuda"):
            out = model(batch["frames"], batch["prev"])
        loss, parts = bc_loss({k: v.float() for k, v in out.items()}, batch, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError("loss is NaN or infinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        sums["loss"] += loss.item()
        for k, v in parts.items():
            sums[k] += v
        batches += 1
    return {k: v / max(batches, 1) for k, v in sums.items()}


# ---------------------------------------------------------------- the loop

def fit(cfg: dict, train_set, val_maps, out: Path, resume: bool = False, frame_root: Path | None = None) -> float:
    """Train on `train_set` (data/dataset.ReplayWindows), select on closed-loop play of `val_maps`
    (eval/rollout.split_maps). Returns the best validation accuracy."""
    t, e = cfg["train"], cfg["eval"]
    device = pick_device(t["device"])
    seed_everything(cfg["seed"])
    d = cfg["data"]
    cursor_channel = d.get("cursor_channel", False)
    model = build_policy(cfg).to(device)
    if t.get("init"):                           # fine-tune: start from a trained model's weights
        model.load_state_dict(torch.load(t["init"], map_location=device, weights_only=True)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    start_epoch, best = 0, -1.0

    out.mkdir(parents=True, exist_ok=True)
    if resume:
        state = torch.load(out / "last.pt", map_location=device, weights_only=True)
        if state["config"] != cfg:
            raise SystemExit("--resume needs the same config as the run it continues")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best_accuracy"]
    elif (out / "last.pt").exists():
        raise SystemExit(f"{out} already has a run: pass --resume, or use another --out")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "run.json").write_text(json.dumps({"git_revision": git_revision(), "device": device,
                                              "torch": torch.__version__, "train_windows": len(train_set),
                                              "backbone": cfg["model"].get("backbone", "framestack"),
                                              "parameters": count_parameters(model)}, indent=2))

    policy = ModelPolicy(model, d["window"], device, d["canonical_hr"], e["press_threshold"],
                         cursor_channel=cursor_channel)
    for epoch in range(start_epoch, t["epochs"]):
        # a fresh random draw each epoch; seeded by epoch so a resumed run draws the same windows
        sampler = RandomSampler(train_set, replacement=True, num_samples=t["samples_per_epoch"],
                                generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch))
        loader = DataLoader(train_set, batch_size=t["batch_size"], sampler=sampler, num_workers=t["num_workers"],
                            pin_memory=device == "cuda")
        started = time.time()
        losses = train_epoch(model, loader, optimizer, scaler, cfg, device)
        log = {"epoch": epoch, **losses, "train_seconds": round(time.time() - started)}

        if (epoch + 1) % e["every"] == 0 or epoch + 1 == t["epochs"]:
            report = evaluate(policy, val_maps, auto_gap=False, root=frame_root, motor=motor_from(e))
            (out / f"val_epoch_{epoch:03d}.json").write_text(json.dumps(report, indent=2))
            log.update(val_accuracy=report["accuracy"], val_miss_rate=report["miss_rate"])
            if report["accuracy"] > best:
                best = report["accuracy"]
                _save(out / "best.pt", model, optimizer, scaler, cfg, epoch, best)
        _save(out / "last.pt", model, optimizer, scaler, cfg, epoch, best)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        log_line(out, json.dumps(log))
    return best


def _save(path: Path, model, optimizer, scaler, cfg, epoch, best) -> None:
    """Write a checkpoint atomically (a crash mid-write leaves the old one)."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg, "epoch": epoch, "best_accuracy": best}, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Behaviour cloning (Stage 1).")
    ap.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    ap.add_argument("--out", type=Path, help="run directory (default runs/<name>_seed<seed>)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, help="override the config's seed (the run is <name>_seed<seed>)")
    ap.add_argument("--smoke", action="store_true", help="a few minutes' run to check the config works")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.smoke:
        cfg = apply_smoke(cfg)
    d = cfg["data"]
    out = args.out or Path(f"runs/{cfg['name']}_seed{cfg['seed']}")
    map_lists = tuple(Path(p) for p in d.get("map_lists", [MAPS]))     # validation: the val split
    train_set = from_split("train", mods=tuple(d["train_mods"]), window=d["window"], canonical_hr=d["canonical_hr"],
                           map_lists=map_lists,
                           cursor_jitter=d.get("cursor_jitter", 0.0), cursor_channel=d.get("cursor_channel", False))
    val_maps = split_maps("val", "NM")[:cfg["eval"]["val_maps"]]
    for extra in cfg["eval"].get("extra_val", []):      # more maps for selection, e.g. another split
        val_maps += split_maps(extra["split"], "NM", map_lists=(Path(extra["map_list"]),))
    out.mkdir(parents=True, exist_ok=True)
    log_line(out, f"{cfg['name']} seed {cfg['seed']}, backbone {cfg['model'].get('backbone', 'framestack')}: "
                  f"{len(train_set):,} training windows, {len(val_maps)} val maps -> {out}")
    best = fit(cfg, train_set, val_maps, out, args.resume)
    log_line(out, f"best val accuracy {best:.2%}")


if __name__ == "__main__":
    main()
