"""The brief's three gates, run once before any training (results go to the pre-run audit).

    python -m rl.gates parity      # G2: the RL env + deterministic BC reproduce eval/rollout.py's numbers
    python -m rl.gates throughput  # G1: agent steps per second, batched over parallel segment envs
    python -m rl.gates hd          # G3: HD frames around a few hit times -> rl/runs/gates/hd_frames.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import MapFrames, build_map_frames
from eval.motor import CAP
from eval.policy import load_model_policy
from eval.rollout import play, split_maps
from osu_sim.judge import accuracy
from rl.env import MapEnv, SegmentEnv

BC = "runs/baseline_cap_seed0/best.pt"
OUT = Path("rl/runs/gates")


def bc_latents(model, obs: np.ndarray, device) -> np.ndarray:
    """The deterministic policy's six action latents (position, press logits, held logits)."""
    with torch.no_grad():
        out = model(torch.from_numpy(obs).to(device))
    return torch.cat([out["position"], out["press"], out["held"]], dim=1).float().cpu().numpy()


def parity(device):
    policy = load_model_policy(BC, device)
    model = policy.model.eval()
    rows = []
    for bid, style, pl in split_maps("val", "NM"):
        layers = MapFrames(build_map_frames(pl, bid))
        _, js = play(policy, bid, pl, motor=CAP, seed=bid)
        env = MapEnv(pl, layers, policy.window, CAP, policy.press_threshold)
        obs, _ = env.reset(bid)
        done = False
        while not done:
            (obs, _), _, done, _, _ = env.step(bc_latents(model, obs[None], device)[0])
        env_acc = accuracy(env.judge.judgements())
        rows.append({"beatmap_id": bid, "eval_accuracy": accuracy(js), "env_accuracy": env_acc})
        print(f"{bid}: eval {accuracy(js):.4%}  env {env_acc:.4%}", flush=True)
    worst = max(abs(r["eval_accuracy"] - r["env_accuracy"]) for r in rows)
    print(f"G2 parity: max |difference| {100 * worst:.4f} pp over {len(rows)} val maps")
    return {"maps": rows, "max_abs_diff_pp": 100 * worst}


def throughput(device, n_envs=16, steps=200):
    policy = load_model_policy(BC, device)
    model = policy.model.eval()
    maps = [(bid, pl, MapFrames(build_map_frames(pl, bid))) for bid, _, pl in split_maps("val", "NM")]
    rng = np.random.default_rng(0)
    envs = [SegmentEnv(maps, np.random.default_rng(rng.integers(2 ** 31))) for _ in range(n_envs)]
    t0 = time.time()
    obs = np.stack([e.reset()[0] for e in envs])
    t_reset = time.time() - t0
    t0 = time.time()
    t_model = 0.0
    for _ in range(steps):
        tm = time.time()
        a = bc_latents(model, obs, device) + rng.normal(0, 0.05, (n_envs, 6))
        t_model += time.time() - tm
        nxt = []
        for e, act in zip(envs, a):
            o, _, _, term, trunc, _ = e.step(act)
            nxt.append(e.reset()[0] if term or trunc else o)
        obs = np.stack(nxt)
    dt = time.time() - t0
    sps = n_envs * steps / dt
    print(f"G1: {n_envs} envs x {steps} steps in {dt:.1f} s = {sps:.0f} agent steps/s "
          f"(policy forward {t_model / dt:.0%} of the time); {n_envs} segment resets took {t_reset:.1f} s")
    return {"n_envs": n_envs, "steps": steps, "steps_per_s": sps, "forward_share": t_model / dt,
            "reset_seconds_for_all": t_reset}


def hd_frames():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data.dataset import mods_from_str
    from osu_sim.beatmap import parse_beatmap
    from osu_sim.playable import make_playable
    bid = split_maps("val", "NM")[0][0]
    pl = make_playable(parse_beatmap(Path("data/raw/beatmaps") / f"{bid}.osu"), mods_from_str("HD"))
    layers = MapFrames(build_map_frames(pl, bid))
    o = [x for x in pl.objects if x.kind == "circle"][10]
    offsets = [-1.0, -0.7, -0.5, -0.3, -0.15, 0.0]       # fractions of preempt before the hit
    fig, ax = plt.subplots(1, len(offsets), figsize=(3 * len(offsets), 2.6))
    for a, f in zip(ax, offsets):
        k = int(np.argmin(np.abs(layers.times - (o.time + f * pl.preempt))))
        a.imshow(layers.frame(k), cmap="gray", vmin=0, vmax=255)
        a.plot(o.x / 4, o.y / 4, "r+", ms=8)
        a.set_title(f"{f:+.2f} preempt", fontsize=9)
        a.axis("off")
    fig.suptitle(f"HD, map {bid}, AR {pl.ar if hasattr(pl, 'ar') else '?'}, preempt {pl.preempt} ms; red + = circle")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "hd_frames.png", dpi=110, bbox_inches="tight")
    print(f"G3: wrote {OUT / 'hd_frames.png'} (preempt {pl.preempt} ms; invisible for the last "
          f"{0.3 * pl.preempt:.0f} ms before the hit)")
    return {"beatmap_id": bid, "preempt_ms": pl.preempt, "invisible_ms": 0.3 * pl.preempt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["parity", "throughput", "hd"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.gate == "parity":
        res = parity(args.device)
    elif args.gate == "throughput":
        res = throughput(args.device, args.envs)
    else:
        res = hd_frames()
    (OUT / f"{args.gate}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
