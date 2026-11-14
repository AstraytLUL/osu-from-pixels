"""Why does a model miss: aim or timing? Plays the val maps with checkpoints and classifies every hit
circle (docs/architectures.md, "Follow-ups"; the project proposal's aim and timing diagnostics).

    python -m eval.diagnose --configs baseline cnn3d cnn_gru --mods NM DT HD HRDT        # all their seeds
    python -m eval.diagnose --configs cnn_gru --seeds 0 --maps 1 --mods HRDT              # a quick check
    CUDA_VISIBLE_DEVICES=1 python -m eval.diagnose --configs cnn_gru --out runs/diagnostics/diagnose_gru.json
    (parallel runs: give each its own --out named diagnose*.json; eval.report_figures reads them all)

Checkpoints: runs/<config>_seed<N>/best.pt (training writes them; eval.test_all and train.hub pull
fetch them). Each map is played closed-loop exactly as eval.rollout.evaluate does (CAP motor unless
--motor says otherwise, the same seed per map), then every circle is put in one class (sliders and
spinners are left out: their judgement mixes head, ticks and tail, and spinners need no aim):

  300 / 100 / 50   hit; 100 and 50 are hits with a timing error
  miss_timing      a miss with the cursor on the circle at its hit time (a click came at the wrong
                   time, or no click came), or a click on it outside the hit window
  miss_aim         a miss with the cursor off the circle at its hit time and no click ever on it

and for every circle: the cursor's distance from its centre at its hit time, in units of the circle
radius (1 = the rim; HR's circles are smaller, so radii compare across mods), and for clicked ones the
click offset in map ms (negative = early; the hit windows are in map ms, so DT compares with NM). Also
the clicks per object (1 is ideal; under 1 means objects never clicked).

Writes one JSON (per config, seed, mod: the counts, and the distances and offsets for figures) and
prints a summary; eval.report_figures turns it into fig6 / fig7 when it exists.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

MODS = ("NM", "HR", "DT", "HD", "HRDT")
CLASSES = ("300", "100", "50", "miss_timing", "miss_aim")


def classify(result: int, offset: float | None, distance_over_radius: float) -> str:
    """One circle's class (module doc)."""
    if result > 0:
        return str(result)
    if offset is not None or distance_over_radius <= 1.0:
        return "miss_timing"
    return "miss_aim"


def circle_rows(pl, g, judgements) -> list[dict]:
    """Per circle: result, offset, cursor distance / radius at its hit time (cursor interpolated on
    the frame grid)."""
    rows = []
    for j in judgements:
        if j.kind != "circle":
            continue
        o = pl.objects[j.index]
        cx = float(np.interp(o.time, g.t, g.x))
        cy = float(np.interp(o.time, g.t, g.y))
        d = float(np.hypot(cx - o.x, cy - o.y)) / pl.radius
        rows.append({"result": j.result, "offset": j.offset, "distance": d,
                     "class": classify(j.result, j.offset, d)})
    return rows


def clicks(g) -> int:
    """Key presses in a grid replay (a frame whose press bits are set: one click, maybe two keys)."""
    return int(np.count_nonzero(g.press))


def summarise(rows: list[dict], n_clicks: int, n_objects: int) -> dict:
    n = len(rows)
    counts = {c: sum(r["class"] == c for r in rows) for c in CLASSES}
    offsets = [r["offset"] for r in rows if r["offset"] is not None and r["result"] > 0]
    return {"circles": n, "counts": counts,
            "rates": {c: counts[c] / n if n else 0.0 for c in CLASSES},
            "median_distance": statistics.median(r["distance"] for r in rows) if rows else None,
            "distances": [round(r["distance"], 3) for r in rows],
            "offsets": [round(o, 1) for o in offsets],
            "mean_offset": statistics.mean(offsets) if offsets else None,
            "offset_sd": statistics.pstdev(offsets) if len(offsets) > 1 else None,
            "clicks_per_object": n_clicks / n_objects if n_objects else None}


def find_checkpoints(root: Path, configs: list[str], seeds: list[int] | None) -> list[tuple[str, int, Path]]:
    import yaml
    out = []
    for best in sorted(root.glob("*_seed*/best.pt")):
        cfg = yaml.safe_load((best.parent / "config.yaml").read_text())
        if cfg["name"] in configs and (seeds is None or cfg["seed"] in seeds):
            out.append((cfg["name"], cfg["seed"], best))
    return out


def main():
    ap = argparse.ArgumentParser(description="Aim vs timing: classify every circle of the val maps.")
    ap.add_argument("--configs", nargs="+", default=["baseline", "cnn3d", "cnn_gru"])
    ap.add_argument("--seeds", nargs="+", type=int, help="default: every seed found")
    ap.add_argument("--mods", nargs="+", default=list(MODS))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--maps", type=int, help="only the first n maps (a quick check)")
    ap.add_argument("--root", type=Path, default=Path("runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--motor", choices=["human", "cap", "free"], default="cap", help="eval/motor.py's motor")
    ap.add_argument("--out", type=Path, help="default runs/diagnostics/diagnose_<split>[_<motor>].json (no suffix for human, as before)")
    args = ap.parse_args()

    from eval.motor import MOTORS
    from eval.policy import load_model_policy
    from eval.rollout import play, split_maps

    motor = MOTORS[args.motor]
    if args.out is None:                        # one file per motor: report_figures merges every diagnose*.json
        args.out = Path(f"runs/diagnostics/diagnose_{args.split}{'' if args.motor == 'human' else '_' + args.motor}.json")
    ckpts = find_checkpoints(args.root, args.configs, args.seeds)
    if not ckpts:
        raise SystemExit(f"no best.pt for {args.configs} under {args.root}/*_seed*/ "
                         "(python -m eval.test_all run, or python -m train.hub pull <repo> --seed N)")
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name, seed, best in ckpts:
        policy = load_model_policy(str(best), args.device)
        for mods in args.mods:
            key = f"{name}/seed{seed}/{mods}"
            if key in results:
                print(f"{key}: already done", flush=True)
                continue
            rows, n_clicks, n_objects = [], 0, 0
            for bid, style, pl in split_maps(args.split, mods)[:args.maps]:
                g, js = play(policy, bid, pl, motor=motor, seed=bid)        # as eval.rollout.evaluate
                rows += circle_rows(pl, g, js)
                n_clicks += clicks(g)
                n_objects += sum(o.kind != "spinner" for o in pl.objects)
            results[key] = {"config": name, "seed": seed, "mods": mods, "motor": args.motor,
                            **summarise(rows, n_clicks, n_objects)}
            r = results[key]
            print(f"{key}: circles {r['circles']}, miss aim {r['rates']['miss_aim']:.1%}, "
                  f"miss timing {r['rates']['miss_timing']:.1%}, median distance {r['median_distance']:.2f} r, "
                  f"mean offset {r['mean_offset'] or 0:+.1f} ms, clicks/object {r['clicks_per_object']:.2f}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results))                    # after every cell: resumable
    print(f"wrote {args.out}")
    print_table(results)


def print_table(results: dict) -> None:
    """Mean over seeds per config and mod."""
    groups = {}
    for r in results.values():
        groups.setdefault((r["config"], r["mods"]), []).append(r)
    print(f"\n{'config':14}{'mod':6}{'seeds':>6}{'miss aim':>10}{'miss time':>11}{'100+50':>8}"
          f"{'med dist':>10}{'offset':>9}{'off sd':>8}{'clk/obj':>9}")
    for (name, mods), rs in sorted(groups.items()):
        m = lambda f: statistics.mean(f(r) for r in rs)                          # noqa: E731
        print(f"{name:14}{mods:6}{len(rs):>6}{m(lambda r: r['rates']['miss_aim']):>10.1%}"
              f"{m(lambda r: r['rates']['miss_timing']):>11.1%}"
              f"{m(lambda r: r['rates']['100'] + r['rates']['50']):>8.1%}"
              f"{m(lambda r: r['median_distance']):>9.2f}r{m(lambda r: r['mean_offset'] or 0):>+8.1f}"
              f"{m(lambda r: r['offset_sd'] or 0):>8.1f}{m(lambda r: r['clicks_per_object']):>9.2f}")


if __name__ == "__main__":
    main()
