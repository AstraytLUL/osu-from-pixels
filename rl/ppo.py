"""PPO fine-tuning of the BC policy (the brief's shared setup), one run per config and seed.

    python -m rl.ppo --config rl/configs/exp1_hd.yaml --seed 0
    python -m rl.ppo --config rl/configs/exp2_map.yaml --map 4952940 --seed 0

Order of a run:
1. Load BC (frozen copy as the KL anchor), build the actor-critic (rl/agent.py).
2. Critic warm-up: the policy is frozen; collect rollouts and train only the critic until its
   explained variance stops improving (logged).
3. PPO: collect `n_envs` x `rollout_steps` steps of segments (rl/env.py), GAE with truncation
   bootstrapping, clipped policy loss + decaying KL to BC, early stop on the old-policy KL,
   separate optimisers for actor and critic.
4. Every `eval_every_steps`: export the actor as a BC-format checkpoint (ckpt_<steps>.pt) and play
   the config's curve maps deterministically (rl/evaluate.py) -> curve.csv. Nothing is selected on
   these: the reported model is the final one (final.pt).

Run directory rl/runs/<name>_seed<seed>/: config.yaml, run.json (git commit, dirty flag, device),
metrics.jsonl (one line per update), log.txt, curve.csv, ckpt_*.pt, final.pt.
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

from data.dataset import MapFrames, build_map_frames, load_maps
from eval.motor import CAP
from models.heads import build_policy
from rl import evaluate as rl_eval
from rl.agent import Agent
from rl.env import SegmentEnv


def git_state() -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "rl/"], capture_output=True, text=True).stdout.strip())
    return {"commit": rev, "rl_dirty": dirty}


def training_maps(cfg: dict) -> list[tuple]:
    if cfg["train_maps"] == "train_split":
        ids = sorted(b for b, e in load_maps().items() if e["split"] == "train")
    else:
        ids = [int(b) for b in cfg["train_maps"]]
    mods = cfg["mods"] if isinstance(cfg["mods"], list) else [cfg["mods"]]   # a list: each segment draws
    out = []                                                                  # a (map, mods) pair uniformly
    for bid in ids:
        for m in mods:
            pl = rl_eval.playable(bid, m)
            out.append((bid, pl, MapFrames(build_map_frames(pl, bid))))
    return out


def load_bc(path: str, device: str):
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_policy(state["config"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state["config"]


def export(agent: Agent, bc_cfg: dict, path: Path, extra: dict) -> None:
    """The actor as a BC-format checkpoint: eval.policy.load_model_policy and rl/evaluate.py read it."""
    torch.save({"model": agent.actor.state_dict(), "config": bc_cfg, "ppo": extra}, path)


class Collector:
    """The parallel segment environments and the rollout buffer."""

    def __init__(self, maps, cfg, seed, window, threshold):
        rng = np.random.default_rng(seed)
        self.envs = [SegmentEnv(maps, np.random.default_rng(rng.integers(2 ** 31)), cfg["segment_frames"],
                                window, CAP, threshold) for _ in range(cfg["n_envs"])]
        self.obs = np.stack([e.reset()[0] for e in self.envs])
        self.ep_stats = []                              # per finished segment: reward, accuracy

        self._seg = [{"reward": 0.0, "score": 0, "score_max": 0, "misses": 0, "judged": 0} for _ in self.envs]

    @torch.no_grad()
    def collect(self, agent: Agent, steps: int, device: str, gamma: float, lam: float) -> dict:
        n = len(self.envs)
        obs = np.zeros((steps, n, *self.obs.shape[1:]), np.uint8)
        act = np.zeros((steps, n, 6), np.float32)
        logp = np.zeros((steps, n), np.float32)
        val = np.zeros((steps, n), np.float32)
        rew = np.zeros((steps, n), np.float32)
        term = np.zeros((steps, n), bool)
        trunc_val = np.full((steps, n), np.nan, np.float32)    # V(last obs of a cut segment)
        for t in range(steps):
            x = torch.from_numpy(self.obs).to(device)
            d = agent.dist(x)
            a = d.sample()
            obs[t] = self.obs
            act[t] = a.cpu().numpy()
            logp[t] = d.log_prob(a).sum(-1).cpu().numpy()
            val[t] = agent.value(x).cpu().numpy()
            nxt, cut = [], []
            for i, (e, ai) in enumerate(zip(self.envs, act[t])):
                o, _, r, done, truncated, info = e.step(ai)
                rew[t, i] = r
                s = self._seg[i]
                s["reward"] += r
                for k in ("score", "score_max", "misses", "judged"):
                    s[k] += info[k]
                if done or truncated:
                    self.ep_stats.append(dict(s))
                    self._seg[i] = {"reward": 0.0, "score": 0, "score_max": 0, "misses": 0, "judged": 0}
                    term[t, i] = done
                    if truncated:
                        cut.append((i, o))
                    o = e.reset()[0]
                nxt.append(o)
            if cut:
                v = agent.value(torch.from_numpy(np.stack([o for _, o in cut])).to(device)).cpu().numpy()
                for (i, _), vi in zip(cut, v):
                    trunc_val[t, i] = vi
            self.obs = np.stack(nxt)
        last = agent.value(torch.from_numpy(self.obs).to(device)).cpu().numpy()
        adv = np.zeros((steps, n), np.float32)
        running = np.zeros(n, np.float32)
        for t in range(steps - 1, -1, -1):
            cut = ~np.isnan(trunc_val[t])
            nextv = last if t == steps - 1 else val[t + 1]
            nextv = np.where(term[t], 0.0, np.where(cut, np.nan_to_num(trunc_val[t]), nextv))
            cont = (~term[t]) & (~cut)                # the lambda trace stops at any segment end
            delta = rew[t] + gamma * nextv - val[t]
            running = delta + gamma * lam * cont * running
            adv[t] = running
        flat = lambda a: a.reshape(steps * n, *a.shape[2:])
        return {"obs": flat(obs), "act": flat(act), "logp": flat(logp), "val": flat(val),
                "adv": flat(adv), "ret": flat(adv + val), "rew_mean": float(rew.mean())}


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    v = float(np.var(target))
    return float(1 - np.var(target - pred) / v) if v > 0 else float("nan")


def critic_epochs(agent, opt, batch, cfg, device) -> float:
    n = len(batch["ret"])
    losses = []
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(n).split(cfg["minibatch"]):
            x = torch.from_numpy(batch["obs"][idx.numpy()]).to(device)
            ret = torch.from_numpy(batch["ret"][idx.numpy()]).to(device)
            loss = 0.5 * (agent.value(x) - ret).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_parameters(), cfg["max_grad_norm"])
            opt.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def ppo_epochs(agent, actor_opt, critic_opt, batch, cfg, kl_coef, device) -> dict:
    n = len(batch["ret"])
    adv_all = batch["adv"]
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    logs, stopped = [], False
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(n).split(cfg["minibatch"]):
            i = idx.numpy()
            x = torch.from_numpy(batch["obs"][i]).to(device)
            a = torch.from_numpy(batch["act"][i]).to(device)
            old = torch.from_numpy(batch["logp"][i]).to(device)
            adv = torch.from_numpy(adv_all[i]).to(device)
            ret = torch.from_numpy(batch["ret"][i]).to(device)
            d = agent.dist(x)
            log_ratio = d.log_prob(a).sum(-1) - old
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            if approx_kl > cfg["target_kl"]:
                stopped = True
                break
            pg = -torch.minimum(ratio * adv, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv).mean()
            kl_ref = agent.kl_to_reference(x, d).mean()
            loss = pg + kl_coef * kl_ref
            actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_parameters(), cfg["max_grad_norm"])
            actor_opt.step()
            vloss = 0.5 * (agent.value(x) - ret).square().mean()
            critic_opt.zero_grad(set_to_none=True)
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_parameters(), cfg["max_grad_norm"])
            critic_opt.step()
            logs.append({"policy_loss": pg.item(), "value_loss": vloss.item(), "kl_ref": kl_ref.item(),
                         "approx_kl": approx_kl, "clip_frac": ((ratio - 1).abs() > cfg["clip"]).float().mean().item()})
        if stopped:
            break
    out = {k: float(np.mean([l[k] for l in logs])) for k in logs[0]} if logs else {}
    out.update(minibatches=len(logs), early_stop=stopped)
    return out


def seg_summary(stats: list[dict]) -> dict:
    if not stats:
        return {}
    sm = sum(s["score_max"] for s in stats)
    return {"segments": len(stats), "seg_reward": float(np.mean([s["reward"] for s in stats])),
            "seg_accuracy": sum(s["score"] for s in stats) / sm if sm else None,
            "seg_miss_rate": sum(s["misses"] for s in stats) / max(1, sum(s["judged"] for s in stats))}


def main():
    ap = argparse.ArgumentParser(description="PPO fine-tuning of the BC policy.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", type=int, help="Experiment 2: the one map to train on (overrides train_maps)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--total-steps", type=int, help="override (sanity runs)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="0 = the zero-learning-rate sanity run")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.map:
        cfg["train_maps"] = [args.map]
        cfg["curve_maps"] = str(args.map)
        cfg["name"] = f"{cfg['name']}_{args.map}"
    if args.total_steps:
        cfg["total_steps"] = args.total_steps
    cfg["lr_scale"] = args.lr_scale
    out = args.out or Path(f"rl/runs/{cfg['name']}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": args.seed}, sort_keys=False))
    (out / "run.json").write_text(json.dumps({**git_state(), "device": args.device, "seed": args.seed,
                                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    def log(text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        with (out / "log.txt").open("a") as f:
            f.write(line + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    bc, bc_cfg = load_bc(cfg["bc_checkpoint"], device)
    agent = Agent(bc, cfg["aim_std_px"], cfg["key_std"]).to(device)
    maps = training_maps(cfg)
    log(f"{cfg['name']} seed {args.seed}: {len(maps)} training maps ({cfg['mods']}), "
        f"{cfg['n_envs']} envs x {cfg['rollout_steps']} steps per update, {cfg['total_steps']:,} steps")
    col = Collector(maps, cfg, args.seed, bc_cfg["data"]["window"], bc_cfg["eval"]["press_threshold"])
    s = cfg["lr_scale"]
    actor_opt = torch.optim.Adam(agent.actor_parameters(), lr=cfg["actor_lr"] * s, eps=1e-5)
    critic_opt = torch.optim.Adam(agent.critic_parameters(), lr=cfg["critic_lr"], eps=1e-5)
    gamma, lam = cfg["gamma"], cfg["gae_lambda"]
    metrics = (out / "metrics.jsonl").open("a")

    def curve(steps):
        path = out / f"ckpt_{steps}.pt"
        export(agent, bc_cfg, path, {"steps": steps, "config": cfg})
        if cfg.get("curve_maps"):
            for mods in cfg.get("curve_mods", [cfg["mods"]]):
                rl_eval.run(f"step_{steps}", args.seed, str(path), rl_eval.map_list(cfg["curve_maps"]), [mods],
                            out / "curve.csv", device)
            log(f"curve point at {steps:,} steps written")

    # 2. critic warm-up (policy frozen)
    best_ev, stale = -1e9, 0
    for w in range(cfg["warmup_max_iters"]):
        batch = col.collect(agent, cfg["rollout_steps"], device, gamma, lam)
        ev = explained_variance(batch["val"], batch["ret"])
        vloss = critic_epochs(agent, critic_opt, batch, cfg, device)
        log(f"warm-up {w}: explained variance (before this batch's update) {ev:.3f}, value loss {vloss:.4f}")
        metrics.write(json.dumps({"phase": "warmup", "iter": w, "explained_variance": ev, "value_loss": vloss}) + "\n")
        metrics.flush()
        if ev > best_ev + cfg["warmup_min_gain"]:
            best_ev, stale = ev, 0
        else:
            stale += 1
            if stale >= cfg["warmup_patience"]:
                break
    log(f"critic warm-up done: explained variance {best_ev:.3f}")
    col.ep_stats.clear()

    # 3. PPO
    steps, update = 0, 0
    per_update = cfg["n_envs"] * cfg["rollout_steps"]
    next_curve = 0
    t0 = time.time()
    while steps < cfg["total_steps"]:
        if steps >= next_curve:
            curve(steps)
            next_curve += cfg["eval_every_steps"]
        kl_coef = cfg["kl_coef"] * max(0.0, 1 - steps / (cfg["kl_decay_frac"] * cfg["total_steps"]))
        batch = col.collect(agent, cfg["rollout_steps"], device, gamma, lam)
        ev = explained_variance(batch["val"], batch["ret"])
        rep = ppo_epochs(agent, actor_opt, critic_opt, batch, cfg, kl_coef, device)
        steps += per_update
        update += 1
        seg = seg_summary(col.ep_stats)
        col.ep_stats.clear()
        std = agent.std().detach().cpu().numpy()
        row = {"phase": "ppo", "update": update, "steps": steps, "reward_per_step": batch["rew_mean"],
               "explained_variance": ev, "kl_coef": kl_coef, "aim_std_px": float(std[:2].mean() * 50),
               "key_std": float(std[2:].mean()), **rep, **seg, "elapsed_s": round(time.time() - t0)}
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if update % cfg["log_every"] == 0:
            log(f"update {update} ({steps:,} steps): reward/step {batch['rew_mean']:+.4f}, segment accuracy "
                f"{seg.get('seg_accuracy') or float('nan'):.4f}, EV {ev:.2f}, kl_ref {rep.get('kl_ref', float('nan')):.4f}, "
                f"approx_kl {rep.get('approx_kl', float('nan')):.4f}, minibatches {rep.get('minibatches')}, "
                f"aim std {row['aim_std_px']:.2f} px")
    curve(steps)
    export(agent, bc_cfg, out / "final.pt", {"steps": steps, "config": cfg})
    log(f"done: {steps:,} steps -> {out / 'final.pt'}")


if __name__ == "__main__":
    main()
