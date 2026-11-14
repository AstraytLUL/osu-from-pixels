"""How big and how fast is each architecture? Random frames, no data needed.

    python scripts/bench_backbones.py                     # every config in configs/arch/ + the baseline
    python scripts/bench_backbones.py --device mps        # a Mac's GPU; cuda on RunPod
    python scripts/bench_backbones.py --configs configs/arch/cnn3d.yaml --batch 128

Per config: parameters; training windows per second (forward + backward + optimizer step, at the
config's batch size, mixed precision on CUDA as in train/bc.py); and the milliseconds per frame of
closed-loop play (batch 1, no gradient: what validation does every frame). The epoch estimate is
GPU time only: it leaves out data loading (~1,100 windows/s per DataLoader worker) and the
closed-loop validation, which runs at the "play ms" per frame for ~9 maps x ~10k frames.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.backbones import count_parameters                    # noqa: E402
from models.heads import build_policy                            # noqa: E402
from osu_sim.renderer import H, W                                # noqa: E402
from train.bc import pick_device                                 # noqa: E402


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench(cfg: dict, device: str, batch: int, steps: int) -> dict:
    torch.manual_seed(0)
    model = build_policy(cfg).to(device)
    channels = cfg["data"]["window"] * (2 if cfg["data"].get("cursor_channel") else 1)
    frames = torch.randint(0, 256, (batch, channels, H, W), dtype=torch.uint8, device=device)
    target = torch.randn(batch, 2, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    autocast = dict(device_type="cuda" if device == "cuda" else "cpu", enabled=device == "cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**autocast):
            out = model(frames)
        aim = out.get("position", out.get("move")).float()
        loss = (aim - target).abs().mean() + out["press"].float().mean() + out["held"].float().mean()
        loss.backward()
        opt.step()

    model.train()
    for _ in range(2):                       # warm-up (kernel selection, allocation)
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    per_second = batch * steps / (time.perf_counter() - t0)

    model.eval()
    one = frames[:1]
    with torch.no_grad():
        for _ in range(5):
            model(one)
        sync(device)
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            model(one)
            sync(device)
        play_ms = 1000 * (time.perf_counter() - t0) / n
    return {"parameters": count_parameters(model), "train_per_s": per_second, "play_ms": play_ms,
            "epoch_min": cfg["train"]["samples_per_epoch"] / per_second / 60}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", type=Path,
                    default=[Path("configs/baseline.yaml"), *sorted(Path("configs/arch").glob("*.yaml"))])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, help="default: each config's batch size")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device {device}, torch {torch.__version__}\n")
    print(f"{'config':20}{'params':>10}{'train win/s':>13}{'GPU min/epoch':>15}{'play ms/frame':>15}")
    for path in args.configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        batch = args.batch or cfg["train"]["batch_size"]
        try:
            r = bench(cfg, device, batch, args.steps)
        except RuntimeError as e:                # out of memory, or an op the device lacks
            print(f"{cfg['name']:20} failed: {str(e).splitlines()[0]}")
            continue
        print(f"{cfg['name']:20}{r['parameters'] / 1e6:>9.2f}M{r['train_per_s']:>13,.0f}"
              f"{r['epoch_min']:>15.1f}{r['play_ms']:>15.2f}", flush=True)


if __name__ == "__main__":
    main()
