"""Exploration-noise calibration (the brief: the largest std at which stochastic NoMod play stays within
about 2 accuracy points of deterministic play), on training maps only.

    python -m rl.calibrate --maps 6 --aim 0 1 2 4 8 --key 0 0.3 1.0

Plays whole maps through rl/env.py's MapEnv with actions sampled from rl/agent.py's Gaussian, one
setting at a time, and writes rl/runs/calibrate.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import MapFrames, build_map_frames, load_maps
from eval.motor import CAP
from osu_sim.judge import accuracy
from rl import evaluate as rl_eval
from rl.agent import Agent
from rl.env import MapEnv
from rl.ppo import load_bc


@torch.no_grad()
def play(agent, pl, layers, window, threshold, seed, device, stochastic=True) -> float:
    env = MapEnv(pl, layers, window, CAP, threshold)
    obs, _ = env.reset(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    done = False
    while not done:
        d = agent.dist(torch.from_numpy(obs[None]).to(device))
        a = d.mean + d.stddev * torch.randn(d.mean.shape, generator=gen, device=device) if stochastic else d.mean
        (obs, _), _, done, _, _ = env.step(a[0].cpu().numpy())
    return accuracy(env.judge.judgements())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="runs/baseline_cap_seed0/best.pt")
    ap.add_argument("--maps", type=int, default=6)
    ap.add_argument("--aim", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--key", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    bc, cfg = load_bc(args.bc, args.device)
    window, threshold = cfg["data"]["window"], cfg["eval"]["press_threshold"]
    rng = np.random.default_rng(0)
    train = sorted(b for b, e in load_maps().items() if e["split"] == "train" and e.get("style") != "spinners")
    ids = [int(b) for b in rng.choice(train, args.maps, replace=False)]
    maps = [(b, rl_eval.playable(b, "NM")) for b in ids]
    maps = [(b, pl, MapFrames(build_map_frames(pl, b))) for b, pl in maps]
    base = Agent(bc, 1.0, 1.0).to(args.device)
    det = [play(base, pl, l, window, threshold, b, args.device, stochastic=False) for b, pl, l in maps]
    print(f"deterministic: {100 * np.mean(det):.2f}% on {ids}", flush=True)
    rows = [{"aim_std_px": 0, "key_std": 0, "accuracy": float(np.mean(det)), "drop_pp": 0.0}]
    for aim in args.aim:
        for key in args.key:
            if aim == 0 and key == 0:
                continue
            agent = Agent(bc, max(aim, 1e-3), max(key, 1e-3)).to(args.device)
            acc = [play(agent, pl, l, window, threshold, b, args.device) for b, pl, l in maps]
            drop = 100 * (np.mean(det) - np.mean(acc))
            rows.append({"aim_std_px": aim, "key_std": key, "accuracy": float(np.mean(acc)), "drop_pp": float(drop)})
            print(f"aim {aim:>4} px, key {key:>4}: {100 * np.mean(acc):.2f}% (drop {drop:+.2f} pp)", flush=True)
    Path("rl/runs").mkdir(parents=True, exist_ok=True)
    Path("rl/runs/calibrate.json").write_text(json.dumps({"maps": ids, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
