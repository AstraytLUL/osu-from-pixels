"""Where does the soft-argmax policy want to go, with and without approach circles (HD)?
docs/architectures.md, "Follow-ups": hypothesis 3, why HD breaks aim as well as timing.

    python -m eval.heatmaps                                   # every softargmax seed, NM vs HD, val maps
    python -m eval.heatmaps --seeds 0 --maps 1                # a quick check
    python -m eval.heatmaps --at before-hit --lead 3 --out runs/diagnostics/heatmaps_late.json --fig-tag _late

Controlled (open-loop): for every hit circle, the network sees a window with Auto's cursor drawn in
(osu_sim/auto.py: always on the right path). NM and HD then differ only in what the map layer shows,
not in where the cursor has been, so a difference in the heatmap is the mod's, not drift from earlier
mistakes. (Closed-loop accuracy is in transfer_val.json.) Which moment (--at):

  decision     (default) `--lead` frames after the previous object ends: the cursor has just left it
               and the next target is still some way off, so the network has to tell from the screen
               where to go. Circles whose decision frame is under 3 frames before their hit are skipped.
  before-hit   `--lead` frames before the circle's hit time. Auto's cursor is then nearly on the
               target already, so "follow the cursor" scores well here without reading the screen:
               a weak test, kept for comparison.

Per circle, from the heatmap p (24 x 32 cells, models/heads.py):

  target     mass within radius + 16 px of the circle the player must hit next
  upcoming   mass near the objects after it that start within one approach time (the competitors:
             under NM their approach circles say they come later; under HD nothing does)
  previous   mass near the previous object (staying put)
  entropy    of p in bits (0 = one cell; 9.6 = flat over all 768 cells)
  on target  whether the heatmap's peak cell lies within the target circle (+ one cell)

Writes runs/diagnostics/heatmaps_val.json and two figures into --fig-out:
  fig8_heatmap_examples   the same moments under NM and HD: the frame, the heatmap over it, the target
                          ring (orange) and the upcoming objects (dashed)
  fig9_heatmap_summary    the five measures, NM vs HD, mean per seed
(--fig-tag is appended to both names, e.g. _late.)
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

GRID = (24, 32)
CELL = 16.0                                  # osu! px per heatmap cell
CELL_X = (np.arange(GRID[1]) + 0.5) * CELL   # cell centres, osu! px
CELL_Y = (np.arange(GRID[0]) + 0.5) * CELL
OFF_SCREEN = (-1000.0, -1000.0)


def mass_near(p: np.ndarray, points: list[tuple[float, float]], r: float) -> float:
    """Heatmap mass (p: (24, 32), sums to 1) in cells whose centre is within r of any point."""
    if not points:
        return 0.0
    gx, gy = np.meshgrid(CELL_X, CELL_Y)
    near = np.zeros(GRID, dtype=bool)
    for x, y in points:
        near |= (gx - x) ** 2 + (gy - y) ** 2 <= r * r
    return float(p[near].sum())


def entropy_bits(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


def peak(p: np.ndarray) -> tuple[float, float]:
    i, j = np.unravel_index(int(np.argmax(p)), GRID)
    return float(CELL_X[j]), float(CELL_Y[i])


def measures(p: np.ndarray, target, upcoming, previous, radius: float) -> dict:
    r = radius + CELL
    px, py = peak(p)
    t_mass = mass_near(p, [target], r)
    # upcoming / previous mass, not counting cells already near the target
    gx, gy = np.meshgrid(CELL_X, CELL_Y)
    not_target = (gx - target[0]) ** 2 + (gy - target[1]) ** 2 > r * r
    q = p * not_target
    return {"target": t_mass, "upcoming": mass_near(q, upcoming, r), "previous": mass_near(q, previous, r),
            "entropy": entropy_bits(p), "on_target": float(np.hypot(px - target[0], py - target[1]) <= r)}


def query_frame(times: np.ndarray, pl, o, at: str, lead: int) -> int | None:
    """The frame whose window the network sees for circle o (module doc), or None to skip o."""
    hit = int(np.searchsorted(times, o.time))
    if at == "before-hit":
        k = hit - lead
    else:                                                     # decision
        if o.index == 0 or pl.objects[o.index - 1].kind == "spinner":
            return None
        k = int(np.searchsorted(times, pl.objects[o.index - 1].end_time)) + lead
        if k > hit - 3:
            return None
    return k if k >= 1 else None


def samples(pl, layers, auto, window: int, lead: int, cursor_channel: bool, at: str = "decision"):
    """(observation, info) for every circle: the window ending at query_frame()."""
    from data.dataset import observation
    times = layers.times
    blank = np.zeros_like(layers.frame(0))
    for o in pl.objects:
        if o.kind != "circle":
            continue
        k = query_frame(times, pl, o, at, lead)
        if k is None:
            continue
        idx = range(k - window + 1, k + 1)
        maps = np.stack([layers.frame(i) if i >= 0 else blank for i in idx])
        curs = np.array([(auto.x[i], auto.y[i]) if i >= 0 else OFF_SCREEN for i in idx])
        upcoming = [(u.x, u.y) for u in pl.objects
                    if u.index > o.index and u.kind != "spinner" and u.time <= o.time + pl.preempt]
        prev = pl.objects[o.index - 1] if o.index > 0 else None
        previous = [prev.end_position] if prev is not None and prev.kind != "spinner" else []
        info = {"index": o.index, "time": o.time, "k": k, "target": (o.x, o.y), "upcoming": upcoming,
                "previous": previous, "radius": pl.radius, "ms_before_hit": float(o.time - times[k])}
        yield observation(maps, curs, cursor_channel), info


def run_map(model, policy_window: int, cursor_channel: bool, bid: int, pl, device: str, lead: int,
            keep_examples: bool, at: str = "decision"):
    import torch
    from data.dataset import MapFrames, build_map_frames
    from data.resample import resample
    from osu_sim.auto import auto_replay
    layers = MapFrames(build_map_frames(pl, bid))
    auto = resample(auto_replay(pl), layers.times)
    obs, infos = [], []
    for x, info in samples(pl, layers, auto, policy_window, lead, cursor_channel, at):
        obs.append(x)
        infos.append(info)
    rows, examples = [], {}
    with torch.no_grad():
        for start in range(0, len(obs), 128):
            batch = torch.from_numpy(np.stack(obs[start:start + 128])).to(device)
            heat = model(batch)["heatmap"].float().cpu().numpy()
            for b, p in enumerate(heat):
                info = infos[start + b]
                m = measures(p, info["target"], info["upcoming"], info["previous"], info["radius"])
                rows.append({"index": info["index"], "ms_before_hit": info["ms_before_hit"], **m})
                if keep_examples:
                    examples[info["index"]] = {"heatmap": p, "frame": obs[start + b][policy_window - 1],
                                               "measures": m, **info}
    return rows, examples


def main():
    ap = argparse.ArgumentParser(description="Soft-argmax heatmaps under NM vs HD (open-loop, Auto's cursor).")
    ap.add_argument("--config", default="softargmax")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--mods", nargs="+", default=["NM", "HD"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--maps", type=int, help="only the first n maps")
    ap.add_argument("--at", choices=["decision", "before-hit"], default="decision", help="which moment (module doc)")
    ap.add_argument("--lead", type=int, default=2,
                    help="frames after the previous object (decision) or before the hit (before-hit)")
    ap.add_argument("--fig-tag", default="", help="suffix for the figure names, e.g. _late")
    ap.add_argument("--example-map", type=int, help="beatmap id for fig8 (default: the first jumps map)")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/diagnostics/heatmaps_val.json"))
    ap.add_argument("--fig-out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    from eval.diagnose import find_checkpoints
    from eval.policy import load_model_policy
    from eval.rollout import split_maps

    ckpts = find_checkpoints(args.root, [args.config], args.seeds)
    if not ckpts:
        raise SystemExit(f"no best.pt for {args.config} under {args.root}/*_seed*/")
    maps_by_mod = {m: split_maps(args.split, m)[:args.maps] for m in args.mods}
    first = maps_by_mod[args.mods[0]]
    example_map = args.example_map or next((bid for bid, style, _ in first if style == "jumps"), first[0][0])

    results, examples = {}, {}
    for name, seed, best in ckpts:
        policy = load_model_policy(str(best), args.device)
        model = policy.model.eval()
        if "heatmap" not in model(_probe(policy, args.device)):
            raise SystemExit(f"{best} has no heatmap output: use a soft-argmax checkpoint")
        for mods, maps in maps_by_mod.items():
            rows = []
            for bid, style, pl in maps:
                keep = seed == ckpts[0][1] and bid == example_map
                r, ex = run_map(model, policy.window, policy.cursor_channel, bid, pl, args.device, args.lead, keep,
                                args.at)
                rows += [{**x, "beatmap_id": bid, "style": style} for x in r]
                if keep:
                    examples[mods] = ex
            key = f"{name}/seed{seed}/{mods}"
            results[key] = {"seed": seed, "mods": mods, "n": len(rows), "at": args.at, "lead": args.lead,
                            "ms_before_hit": statistics.median(x["ms_before_hit"] for x in rows),
                            **{m: statistics.mean(x[m] for x in rows) for m in
                               ("target", "upcoming", "previous", "entropy", "on_target")}}
            r = results[key]
            print(f"{key}: {r['n']} circles ({r['ms_before_hit']:.0f} ms before the hit, median), "
                  f"target {r['target']:.2f}, upcoming {r['upcoming']:.2f}, "
                  f"previous {r['previous']:.2f}, entropy {r['entropy']:.2f} bits, "
                  f"peak on target {r['on_target']:.0%}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")

    from eval.report_figures import setup_style
    setup_style()
    args.fig_out.mkdir(parents=True, exist_ok=True)
    if args.at == "decision":
        moment = f"{args.lead} frames after the previous object"
    else:
        moment = f"{args.lead} frames before each circle's hit"
    fig_examples(examples, args.mods, example_map, args.fig_out, moment, args.fig_tag)
    fig_summary(results, args.mods, args.fig_out, moment, args.fig_tag)


def _probe(policy, device):
    import torch
    channels = policy.window * (2 if policy.cursor_channel else 1)
    return torch.zeros(1, channels, 96, 128, dtype=torch.uint8, device=device)


# ---------------------------------------------------------------- figures

def pick_examples(examples_nm: dict, n: int = 4) -> list[int]:
    """Circles where the NM heatmap puts at least half its mass on the target while other objects are
    on screen (so there is a choice to make), spread evenly over the map."""
    good = sorted(i for i, e in examples_nm.items() if e["upcoming"] and e["measures"]["target"] >= 0.5)
    if not good:
        good = sorted(i for i, e in examples_nm.items() if e["upcoming"])
    if len(good) <= n:
        return good
    return [good[round(k * (len(good) - 1) / (n - 1))] for k in range(n)]


def fig_examples(examples: dict, mods: list[str], bid: int, out: Path,
                 moment: str = "2 frames after the previous object", tag: str = "") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from eval.report_figures import INK2, save
    if not examples or mods[0] not in examples:
        return
    chosen = pick_examples(examples[mods[0]])
    blue = LinearSegmentedColormap.from_list("heat", [(0.16, 0.47, 0.84, 0.0), (0.16, 0.47, 0.84, 0.9)])
    fig, axes = plt.subplots(len(mods), len(chosen), figsize=(3.0 * len(chosen), 2.45 * len(mods)), squeeze=False)
    for r, mods_name in enumerate(mods):
        for c, idx in enumerate(chosen):
            ax = axes[r][c]
            e = examples.get(mods_name, {}).get(idx)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if e is None:
                ax.set_visible(False)
                continue
            ax.imshow(e["frame"], cmap="gray", vmin=0, vmax=255, extent=(0, 512, 384, 0))
            p = e["heatmap"] / e["heatmap"].max()
            ax.imshow(p, cmap=blue, vmin=0, vmax=1, extent=(0, 512, 384, 0), interpolation="nearest")
            tx, ty = e["target"]
            ax.add_patch(Circle((tx, ty), e["radius"] + 6, fill=False, ec="#eb6834", lw=2))
            for ux, uy in e["upcoming"]:
                ax.add_patch(Circle((ux, uy), e["radius"] + 6, fill=False, ec="#eb6834", lw=1, ls=(0, (2, 2))))
            ax.set_xlim(0, 512)
            ax.set_ylim(384, 0)
            if c == 0:
                ax.set_ylabel(mods_name, fontsize=10, fontweight="bold", color=INK2)
            if r == 0:
                ax.set_title(f"circle {idx}", fontsize=9, loc="left", fontweight="normal")
    fig.suptitle(f"Soft-argmax heatmap (blue), {moment}, map {bid}, "
                 "same cursor history in both rows. Orange ring: the circle to hit; dashed: objects after it.",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig8_heatmap_examples" + tag)


def fig_summary(results: dict, mods: list[str], out: Path,
                moment: str = "2 frames after the previous object", tag: str = "") -> None:
    import matplotlib.pyplot as plt
    from eval.report_figures import INK2, SURFACE, save
    colors = {"NM": "#52514e", "HD": "#1baf7a", "DT": "#2a78d6", "HR": "#898781", "HRDT": "#eb6834"}
    shares = [("target", "mass on the target"), ("upcoming", "mass on later objects"),
              ("previous", "mass on the previous object"), ("on_target", "peak on the target")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [4, 1.2]})
    x = np.arange(len(shares))
    w = 0.8 / len(mods)
    for i, m in enumerate(mods):
        per_seed = [r for r in results.values() if r["mods"] == m]
        off = (i - (len(mods) - 1) / 2) * w
        for j, (key, _) in enumerate(shares):
            vals = [100 * r[key] for r in per_seed]
            ax.bar(x[j] + off, statistics.mean(vals), w * 0.92, color=colors.get(m, INK2),
                   label=m if j == 0 else None, zorder=2)
            ax.scatter([x[j] + off] * len(vals), vals, s=10, color=SURFACE, edgecolors=INK2, linewidths=0.6, zorder=3)
        ent = [r["entropy"] for r in per_seed]
        ax2.bar(i, statistics.mean(ent), 0.7, color=colors.get(m, INK2), zorder=2)
        ax2.scatter([i] * len(ent), ent, s=10, color=SURFACE, edgecolors=INK2, linewidths=0.6, zorder=3)
    ax.set_xticks(x, [label for _, label in shares])
    ax.set_ylabel("% (mean over circles; dots: seeds)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title(f"Where the heatmap puts its weight, {moment}", loc="left")
    ax2.set_xticks(range(len(mods)), mods)
    ax2.set_ylabel("bits")
    ax2.set_title("Entropy", loc="left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig9_heatmap_summary" + tag)


if __name__ == "__main__":
    main()
