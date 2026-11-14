"""Figures for the report: the architecture experiments (docs/architectures.md), from the run folders
eval/compare_runs.py reads (runs/hub/, fetched from the Hub; see its docstring).

    python -m eval.report_figures                       # -> docs/figures/*.png and *.pdf
    python -m eval.report_figures --root runs/hub --out docs/figures
    python -m eval.report_figures --split test          # the final results: figN_..._test.png

  fig1_accuracy_by_mod     every config's val accuracy under NM / HR / DT / HD / HRDT: seeds as dots,
                           mean +- std; the dashed line is the baseline's mean
  fig2_style_breakdown     accuracy per map style (jumps / streams / tech) under NM, DT and HRDT for the
                           baseline, the 3D CNN and the CNN + GRU: where each gain or loss comes from
  fig3_per_map             per val map, a new architecture against the baseline (mean over seeds):
                           DT for the 3D CNN, HRDT for the CNN + GRU; above the diagonal = better
  fig4_size_control        accuracy against parameter count: the baseline at three widths (the line)
                           against the new architectures (points); a gain the line does not reach is
                           not explained by size
  fig5_training_curves     NoMod val accuracy per epoch, one panel per config (thin: seeds; thick:
                           mean; dashed: the baseline's mean); the selected epochs marked
  fig6_miss_breakdown      (when eval/diagnose.py's JSON exists) the share of circles missed because
                           the cursor was elsewhere (aim) or was there at the wrong moment (timing)
  fig7_click_offsets       (likewise) when the hit circles were clicked, relative to their time

Numbers are from the val maps by default (9 maps, 3 per style: diagnostics); --split test draws the
same figures from the test maps (the final results; fig5 is about training and stays val).
Colours: one fixed colour per config across all figures (the baseline and the size controls in
neutral greys, the three new architectures in blue / orange / aqua); every mark is also labelled, so
colour never carries identity alone.
"""
import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402

from eval.compare_runs import load_runs                # noqa: E402

MODS = ("NM", "HR", "DT", "HD", "HRDT")
MOD_TITLES = {"NM": "NoMod", "HR": "HR (flipped, smaller)", "DT": "DT (1.5x speed)",
              "HD": "HD (no approach circle)", "HRDT": "HR + DT"}
STYLES = ("jumps", "streams", "tech")
ORDER = ("baseline", "baseline_e40", "framestack_half", "framestack_double", "cnn3d", "cnn_gru", "cnn_gru_e40",
         "softargmax")
LABEL = {"baseline": "Baseline (frame stack)", "framestack_half": "Frame stack x0.5",
         "framestack_double": "Frame stack x2", "cnn3d": "3D CNN", "cnn_gru": "CNN + GRU",
         "cnn_gru_e40": "CNN + GRU, 40 epochs", "softargmax": "Soft-argmax aim",
         "baseline_e40": "Baseline, 40 epochs"}
COLOR = {"baseline": "#0b0b0b", "framestack_half": "#898781", "framestack_double": "#898781",
         "cnn3d": "#2a78d6", "cnn_gru": "#eb6834", "cnn_gru_e40": "#eb6834", "softargmax": "#1baf7a",
         "baseline_e40": "#0b0b0b"}
MARKER = {"baseline": "o", "framestack_half": "v", "framestack_double": "^", "cnn3d": "o", "cnn_gru": "o",
          "cnn_gru_e40": "s", "softargmax": "o", "baseline_e40": "s"}
STYLE_MARKER = {"jumps": "o", "streams": "s", "tech": "^"}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
        "text.color": INK,
    })


# ---------------------------------------------------------------- data

def seeds(runs, name):
    return [runs[name][s] for s in sorted(runs[name])]


def acc(run, mod, style=None) -> float:
    rows = [m for m in run["transfer"][mod]["maps"] if style is None or m["style"] == style]
    return 100 * statistics.mean(m["accuracy"] for m in rows)


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def params(runs, name) -> float | None:
    return next((r["run"]["parameters"] for r in seeds(runs, name) if r["run"].get("parameters")), None)


SUFFIX = ""                    # "_test" for --split test


def save(fig, out: Path, stem: str) -> None:
    stem += SUFFIX
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / stem}.png / .pdf")


# ---------------------------------------------------------------- figures

def fig1_accuracy_by_mod(runs, names, out):
    fig, axes = plt.subplots(1, len(MODS), figsize=(13, 3.0), sharey=True)
    ys = np.arange(len(names))[::-1]
    for ax, mod in zip(axes, MODS):
        base = mean_std([acc(r, mod) for r in seeds(runs, "baseline")])[0] if "baseline" in runs else None
        lo, hi = 100, 0
        for y, n in zip(ys, names):
            vals = [acc(r, mod) for r in seeds(runs, n)]
            m, s = mean_std(vals)
            lo, hi = min(lo, *vals, m - s), max(hi, *vals, m + s)
            ax.scatter(vals, [y] * len(vals), s=14, color=COLOR[n], alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(m, y, xerr=s, fmt=MARKER[n], color=COLOR[n], ms=7, elinewidth=2, capsize=0,
                        mec=SURFACE, mew=1.2, zorder=3)
        if base is not None:
            ax.axvline(base, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        pad = max(0.6, 0.08 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(MOD_TITLES[mod], loc="left")
        ax.set_xlabel("accuracy (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [LABEL[n] for n in names])
    axes[0].set_ylim(-0.6, len(names) - 0.4)
    split = "test" if SUFFIX else "val"
    fig.suptitle(f"Closed-loop accuracy on the {split} maps under each mod (dots: seeds; marker and bar: mean ± std; "
                 "dashed: baseline mean). Note each panel's own x range.", x=0.01, ha="left", fontsize=9,
                 color=INK2, fontweight="normal", y=1.04)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig1_accuracy_by_mod")


def fig2_style_breakdown(runs, out, chosen=("baseline", "cnn3d", "cnn_gru"), mods=("NM", "DT", "HRDT")):
    chosen = [n for n in chosen if n in runs]
    fig, axes = plt.subplots(1, len(mods), figsize=(11, 3.3), sharey=False)
    x = np.arange(len(STYLES))
    width = 0.24
    for ax, mod in zip(axes, mods):
        for i, n in enumerate(chosen):
            off = (i - (len(chosen) - 1) / 2) * width
            for j, st in enumerate(STYLES):
                vals = [acc(r, mod, st) for r in seeds(runs, n)]
                m, s = mean_std(vals)
                ax.scatter([x[j] + off] * len(vals), vals, s=12, color=COLOR[n], alpha=0.35, linewidths=0, zorder=2)
                ax.errorbar(x[j] + off, m, yerr=s, fmt=MARKER[n], color=COLOR[n], ms=7, elinewidth=2, capsize=0,
                            mec=SURFACE, mew=1.2, zorder=3, label=LABEL[n] if j == 0 else None)
        ax.set_xticks(x, STYLES)
        ax.set_xlim(-0.55, len(STYLES) - 0.45)
        ax.set_title(MOD_TITLES[mod], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(chosen), bbox_to_anchor=(0.01, 1.08))
    fig.tight_layout(w_pad=2)
    save(fig, out, "fig2_style_breakdown")


def fig3_per_map(runs, out, pairs=(("DT", "cnn3d"), ("HRDT", "cnn_gru"))):
    pairs = [(m, n) for m, n in pairs if n in runs and "baseline" in runs]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (mod, n) in zip(axes, pairs):
        maps = runs["baseline"][min(runs["baseline"])]["transfer"][mod]["maps"]
        xs, ys, styles = [], [], []
        for i, row in enumerate(maps):
            xs.append(statistics.mean(100 * r["transfer"][mod]["maps"][i]["accuracy"] for r in seeds(runs, "baseline")))
            ys.append(statistics.mean(100 * r["transfer"][mod]["maps"][i]["accuracy"] for r in seeds(runs, n)))
            styles.append(row["style"])
        lo = min(xs + ys) - 4
        hi = min(100.5, max(xs + ys) + 3)
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=COLOR[n], alpha=0.06, lw=0, zorder=0)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        for st in STYLES:
            idx = [k for k, s in enumerate(styles) if s == st]
            ax.scatter([xs[k] for k in idx], [ys[k] for k in idx], marker=STYLE_MARKER[st], s=55, color=COLOR[n],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3, label=st)
        better = sum(y > x for x, y in zip(xs, ys))
        ax.text(0.04, 0.96, f"{LABEL[n]} better on {better} of {len(xs)} maps", transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5, color=INK)
        ax.text(0.96, 0.34, "below the line:\nbaseline better", transform=ax.transAxes, va="bottom",
                ha="right", fontsize=8, color=MUTED)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Baseline, {mod} accuracy (%)")
        ax.set_ylabel(f"{LABEL[n]}, {mod} accuracy (%)")
        ax.set_title(f"{MOD_TITLES[mod]}: per {'test' if SUFFIX else 'val'} map", loc="left")
        ax.legend(title="map style", loc="lower right", title_fontsize=8,
                  handletextpad=0.2, borderaxespad=0.6, markerscale=0.9)
        for h in ax.get_legend().legend_handles:
            h.set_color(INK2)
    fig.tight_layout(w_pad=3)
    save(fig, out, "fig3_per_map")


def fig4_size_control(runs, out, mods=("NM", "DT", "HD", "HRDT")):
    family = [n for n in ("framestack_half", "baseline", "framestack_double") if n in runs and params(runs, n)]
    others = [n for n in ("cnn3d", "cnn_gru", "softargmax") if n in runs and params(runs, n)]
    fig, axes = plt.subplots(1, len(mods), figsize=(13, 3.2))
    for ax, mod in zip(axes, mods):
        px = [params(runs, n) / 1e6 for n in family]
        ms_ = [mean_std([acc(r, mod) for r in seeds(runs, n)]) for n in family]
        ax.plot(px, [m for m, _ in ms_], color=MUTED, lw=2, zorder=1)
        for n, x, (m, s) in zip(family, px, ms_):
            ax.errorbar(x, m, yerr=s, fmt=MARKER[n], color=COLOR[n], ms=7, elinewidth=2, capsize=0,
                        mec=SURFACE, mew=1.2, zorder=3)
        for n in others:
            m, s = mean_std([acc(r, mod) for r in seeds(runs, n)])
            ax.errorbar(params(runs, n) / 1e6, m, yerr=s, fmt="o", color=COLOR[n], ms=7, elinewidth=2,
                        capsize=0, mec=SURFACE, mew=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2.5, 3.5, 5, 7.5], ["1.5", "2.5", "3.5", "5", "7.5"])
        ax.minorticks_off()
        ax.set_xlim(1.3, 9)
        ax.set_xlabel("parameters (millions, log)")
        ax.set_title(MOD_TITLES[mod], loc="left")
    axes[0].set_ylabel("accuracy (%)")
    # one legend: the size line and each architecture
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MUTED, lw=2, marker="o", mfc=MUTED, mec=SURFACE,
                      label="Frame stack at x0.5 / x1 (baseline, black) / x2 width")]
    handles += [Line2D([], [], ls="", marker="o", color=COLOR[n], mec=SURFACE, ms=7, label=LABEL[n]) for n in others]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.01, 1.08))
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig4_size_control")


def fig5_training_curves(runs, names, out):
    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    curve = lambda r: [100 * e["val_accuracy"] for e in r["metrics"]]           # noqa: E731
    base_mean = None
    if "baseline" in runs:
        base_mean = np.mean([curve(r) for r in seeds(runs, "baseline")], axis=0)
        base_epochs = np.arange(1, len(base_mean) + 1)
    all_vals = [v for n in names for r in seeds(runs, n) for v in curve(r)]
    ymin = max(0, np.percentile(all_vals, 3) - 1)
    for ax, n in zip(axes, names):
        cs = [curve(r) for r in seeds(runs, n)]
        epochs = np.arange(1, len(cs[0]) + 1)
        if base_mean is not None and n != "baseline":
            ax.plot(base_epochs, base_mean, color=INK2, lw=1.2, ls=(0, (3, 2)), zorder=1, label="baseline mean")
        for c in cs:
            ax.plot(epochs, c, color=COLOR[n], lw=0.8, alpha=0.45, zorder=2)
            best = int(np.argmax(c))
            ax.scatter(epochs[best], c[best], s=30, color=COLOR[n], edgecolors=SURFACE, linewidths=1, zorder=4)
        ax.plot(epochs, np.mean(cs, axis=0), color=COLOR[n], lw=2, zorder=3)
        ax.set_title(LABEL[n], loc="left")
        ax.set_ylim(ymin, 100)
        ax.set_xticks([1] + list(range(5, len(epochs) + 1, 5 if len(epochs) <= 20 else 10)))
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[:len(names)]:
        ax.set_xlabel("epoch")
    for ax in axes[::cols]:
        ax.set_ylabel("NoMod val accuracy (%)")
    fig.suptitle("NoMod val accuracy per epoch (thin: seeds, dot: the epoch kept as best.pt; thick: mean; "
                 "dashed: baseline mean). Selection uses NoMod val only.", x=0.01, ha="left", fontsize=9,
                 color=INK2, y=1.01)
    fig.tight_layout()
    save(fig, out, "fig5_training_curves")


def load_diagnostics(path: Path) -> dict:
    """eval/diagnose.py's output (one JSON, or a folder of diagnose*.json from parallel runs),
    grouped: (config, mod) -> [one entry per seed]."""
    import json
    files = sorted(path.glob("diagnose*.json")) if path.is_dir() else [path]
    merged = {}
    for f in files:
        merged.update(json.loads(f.read_text()))
    groups = {}
    for r in merged.values():
        groups.setdefault((r["config"], r["mods"]), []).append(r)
    return groups


def fig6_miss_breakdown(diag: dict, out: Path, mods=("NM", "DT", "HD", "HRDT")):
    names = [n for n in ORDER if any((n, m) in diag for m in mods)]
    mods = [m for m in mods if any((n, m) in diag for n in names)]
    fig, axes = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 0.5 * len(names) + 1.3), sharey=True)
    axes = np.atleast_1d(axes)
    ys = np.arange(len(names))[::-1]
    aim_c, time_c = INK2, AXIS
    for ax, mod in zip(axes, mods):
        hi = 0
        for y, n in zip(ys, names):
            rs = diag.get((n, mod), [])
            if not rs:
                continue
            aim = 100 * statistics.mean(r["rates"]["miss_aim"] for r in rs)
            tim = 100 * statistics.mean(r["rates"]["miss_timing"] for r in rs)
            ax.barh(y, aim, 0.62, color=aim_c, zorder=2, label="aim: cursor elsewhere" if y == ys[0] else None)
            ax.barh(y, tim, 0.62, left=aim, color=time_c, edgecolor=SURFACE, linewidth=2, zorder=2,
                    label="timing: cursor there, wrong or no click" if y == ys[0] else None)
            ax.text(aim + tim, y, f"  {aim + tim:.1f}%", va="center", ha="left", fontsize=7.5, color=INK2)
            hi = max(hi, aim + tim)
        ax.set_xlim(0, hi * 1.25 + 0.5)
        ax.set_title(MOD_TITLES[mod], loc="left")
        ax.set_xlabel("circles missed (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [LABEL.get(n, n) for n in names])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 1.06))
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig6_miss_breakdown")


def fig7_click_offsets(diag: dict, out: Path, mods=("NM", "DT", "HRDT")):
    names = [n for n in ORDER if any((n, m) in diag for m in mods)]
    mods = [m for m in mods if any((n, m) in diag for n in names)]
    fig, axes = plt.subplots(1, len(mods), figsize=(3.6 * len(mods), 0.5 * len(names) + 1.3), sharey=True)
    axes = np.atleast_1d(axes)
    ys = np.arange(len(names))[::-1]
    for ax, mod in zip(axes, mods):
        for y, n in zip(ys, names):
            offs = [o for r in diag.get((n, mod), []) for o in r["offsets"]]
            if not offs:
                continue
            q1, med, q3 = np.percentile(offs, [25, 50, 75])
            lo, hi = np.percentile(offs, [5, 95])
            ax.plot([lo, hi], [y, y], color=COLOR.get(n, INK2), lw=1, zorder=2)
            ax.plot([q1, q3], [y, y], color=COLOR.get(n, INK2), lw=6, solid_capstyle="butt", zorder=3)
            ax.scatter(med, y, s=30, color=SURFACE, edgecolors=COLOR.get(n, INK2), linewidths=1.5, zorder=4)
            ax.text(hi, y + 0.28, f"mean {np.mean(offs):+.1f} ms", fontsize=7, color=INK2, ha="right")
        ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.set_title(MOD_TITLES[mod], loc="left")
        ax.set_xlabel("click time - circle time (map ms; < 0 early)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys, [LABEL.get(n, n) for n in names])
    fig.suptitle("When hit circles were clicked (line: 5-95%, bar: middle 50%, dot: median; all seeds pooled). "
                 "The hit windows are in map ms, so DT compares with NM.", x=0.01, ha="left", fontsize=9,
                 color=INK2, y=1.03)
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig7_click_offsets")


def main():
    global SUFFIX
    ap = argparse.ArgumentParser(description="Report figures for the architecture experiments.")
    ap.add_argument("--root", type=Path, default=Path("runs/hub"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--diagnostics", type=Path, default=Path("runs/diagnostics"),
                    help="eval/diagnose.py's output (a JSON, or a folder of diagnose*.json), for fig6 / fig7")
    args = ap.parse_args()
    setup_style()
    SUFFIX = "_test" if args.split == "test" else ""
    runs = load_runs(args.root, args.split)
    if not runs:
        raise SystemExit(f"no transfer_{args.split}.json under {args.root}")
    names = [n for n in ORDER if n in runs]
    args.out.mkdir(parents=True, exist_ok=True)
    fig1_accuracy_by_mod(runs, names, args.out)
    fig2_style_breakdown(runs, args.out)
    fig3_per_map(runs, args.out)
    fig4_size_control(runs, args.out)
    if args.split == "val":
        fig5_training_curves(runs, names, args.out)
        if args.diagnostics.exists() and (args.diagnostics.is_file() or any(args.diagnostics.glob("diagnose*.json"))):
            diag = load_diagnostics(args.diagnostics)
            fig6_miss_breakdown(diag, args.out)
            fig7_click_offsets(diag, args.out)


if __name__ == "__main__":
    main()
