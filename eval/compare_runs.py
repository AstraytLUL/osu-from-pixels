"""Compare training runs across seeds: the architecture experiments' results table
(docs/architectures.md). Reads each run's transfer_val.json, metrics.jsonl and run.json from run
folders laid out like the Hub repos (train/hub.py); standard library only.

    # fetch every run's small files (not the checkpoints) into runs/hub/<repo>/[seedN/]
    python -c "from huggingface_hub import snapshot_download as s; [s('sc4001-project/' + r, local_dir='runs/hub/' + r, allow_patterns=['*.json', '*.jsonl', '*.yaml', '*.log']) for r in ['bc-baseline', 'bc-cnn3d', 'bc-cnn-gru', 'bc-softargmax', 'bc-framestack-half', 'bc-framestack-double']]"
    python -m eval.compare_runs                      # tables to the terminal
    python -m eval.compare_runs --markdown out.md    # and as Markdown
    python -m eval.compare_runs --split test         # the final results (eval/test_all.py adds them)

A run folder is any folder under --root holding transfer_<split>.json; its config name and seed come
from its config.yaml (bc-baseline keeps seed 0 at the repo root). Every number is the mean over
seeds of the per-seed value, +- the sample standard deviation over seeds; a per-seed value is the
mean over the 9 val maps (each map weighs the same, as in eval/rollout.evaluate).
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

MODS = ("NM", "HR", "DT", "HD", "HRDT")
STYLES = ("jumps", "streams", "tech")
ORDER = ("baseline", "framestack_half", "framestack_double", "cnn3d", "cnn_gru", "cnn_gru_e40", "softargmax")


def load_runs(root: Path, split: str = "val") -> dict[str, dict[int, dict]]:
    """config name -> seed -> {"transfer": ..., "metrics": [...], "run": {...}}, from the runs under
    `root` that have transfer_<split>.json (val: the diagnostic; test: the final results)."""
    runs = defaultdict(dict)
    for tv in sorted(root.rglob(f"transfer_{split}.json")):
        if ".cache" in tv.parts:
            continue
        d = tv.parent
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        metrics = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line.strip()]
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        runs[cfg["name"]][cfg["seed"]] = {"transfer": json.loads(tv.read_text())["results"],
                                          "metrics": metrics, "run": run}
    return dict(runs)


def ms(values: list[float], scale: float = 100.0, digits: int = 1) -> str:
    v = [x * scale for x in values]
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{statistics.mean(v):.{digits}f} ± {statistics.stdev(v):.{digits}f}"


def per_seed(runs, name, fn) -> list[float]:
    return [fn(r) for _, r in sorted(runs[name].items())]


def mean_maps(report: dict, style: str | None = None, key: str = "accuracy") -> float:
    rows = [m for m in report["maps"] if style is None or m["style"] == style]
    return statistics.mean(m[key] for m in rows)


def miss_rate(report: dict) -> float:
    return statistics.mean(m["count_miss"] / (m["count_300"] + m["count_100"] + m["count_50"] + m["count_miss"])
                           for m in report["maps"])


def tables(runs, split: str = "val") -> list[tuple[str, list[str], list[list[str]]]]:
    names = [n for n in ORDER if n in runs] + sorted(n for n in runs if n not in ORDER)
    out = []

    head = ["config", "seeds", "params"] + list(MODS)
    rows = []
    for n in names:
        params = next((r["run"].get("parameters") for r in runs[n].values() if r["run"].get("parameters")), None)
        rows.append([n, str(len(runs[n])), f"{params / 1e6:.2f}M" if params else "?"]
                    + [ms(per_seed(runs, n, lambda r, m=m: r["transfer"][m]["accuracy"])) for m in MODS])
    out.append((f"Accuracy (%), {split} maps, mean ± std over seeds", head, rows))

    head = ["config"] + [f"gap {m}" for m in MODS[1:]]
    rows = [[n] + [ms(per_seed(runs, n, lambda r, m=m: r["transfer"]["NM"]["accuracy"] - r["transfer"][m]["accuracy"]))
                   for m in MODS[1:]] for n in names]
    out.append(("Transfer gap (points): NM accuracy minus mod accuracy, per seed", head, rows))

    for m in ("NM", "DT", "HD", "HRDT"):
        head = ["config"] + list(STYLES)
        rows = [[n] + [ms(per_seed(runs, n, lambda r, s=s: mean_maps(r["transfer"][m], s))) for s in STYLES] for n in names]
        out.append((f"{m} accuracy (%) per style", head, rows))

    head = ["config"] + [f"miss {m}" for m in MODS]
    rows = [[n] + [ms(per_seed(runs, n, lambda r, m=m: miss_rate(r["transfer"][m]))) for m in MODS] for n in names]
    out.append(("Miss rate (% of objects)", head, rows))

    head = ["config", "best epoch", "best NM val", "last-3 mean NM val", "last-epoch loss"]
    rows = []
    for n in names:
        best_ep = per_seed(runs, n, lambda r: max(r["metrics"], key=lambda x: x.get("val_accuracy", -1))["epoch"])
        rows.append([n, ", ".join(str(e) for e in best_ep),
                     ms(per_seed(runs, n, lambda r: max(x.get("val_accuracy", 0) for x in r["metrics"]))),
                     ms(per_seed(runs, n, lambda r: statistics.mean(x["val_accuracy"] for x in r["metrics"][-3:]))),
                     ms(per_seed(runs, n, lambda r: r["metrics"][-1]["loss"]), scale=1, digits=3)])
    out.append(("Training: selected epoch per seed, and whether it was still improving", head, rows))
    return out


def render(title, head, rows, markdown=False) -> str:
    if markdown:
        lines = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines) + "\n"
    widths = [max(len(x) for x in col) for col in zip(head, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([title, fmt.format(*head)] + [fmt.format(*r) for r in rows]) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Results table over configs and seeds.")
    ap.add_argument("--root", type=Path, default=Path("runs/hub"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--markdown", type=Path, help="also write the tables as Markdown")
    args = ap.parse_args()
    runs = load_runs(args.root, args.split)
    if not runs:
        raise SystemExit(f"no transfer_{args.split}.json under {args.root}")
    ts = tables(runs, args.split)
    print("\n".join(render(*t) for t in ts))
    if args.markdown:
        args.markdown.write_text("\n".join(render(*t, markdown=True) for t in ts))
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
