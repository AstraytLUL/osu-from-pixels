"""The report's figures, and every number its text quotes, from saved evaluation files only.

    python -m report.figures --snapshot   # once, on the machine that holds the runs: copy the raw
                                          # evaluation files into report/data/ (runs/ is not committed)
    python -m report.figures              # report/data/ -> report/figures/fig*.png, report/numbers.md
    python -m report.figures --frames     # also Figures 1 and 4, which draw game frames (needs data/raw)

Inputs (report/data/, written by --snapshot):
    headline_results.csv   runs/headline/results.csv: one row per (model, seed, map, mods), written by
                           eval/headline.py; human replays are the model "human"
    modded_stars.json      star rating of each in-band test map with HR / DT / HRDT applied (osu! API)
    curves/<run>.csv       rl/runs/<run>/curve.csv: the PPO learning-curve evaluations (rl/ppo.py)
and the committed RL evaluation files rl/results/exp1.csv, exp2.csv, exp2_transfer.csv.

Conventions, the same as eval/headline.py and rl/report.py: a metric is averaged over maps per seed
(every map weighs the same), then reported as mean ± sample sd over seeds. Human rows count only
when our judge reproduces the replay's leaderboard accuracy within 0.5 pp (eval/headline.py
human_ok). Intervals are 95% percentile bootstraps over maps (10,000 resamples, seed 0).
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA, FIGS = HERE / "data", HERE / "figures"
RL = Path("rl/results")
BANDS = ["4_star", "5_star", "6_star", "7_star", "8_star", "9_star"]
IN_BAND = BANDS[:3]
STYLES = ["jumps", "streams", "tech"]
BASE = "baseline_cap"                          # the headline model (configs/baseline_cap.yaml)
VARIANTS = [("baseline_cap", "baseline"), ("framestack_half_cap", "half width"),
            ("framestack_double_cap", "double width"), ("cnn3d_cap", "3D CNN"), ("cnn_gru_cap", "CNN + GRU"),
            ("softargmax_cap", "soft-argmax"), ("prev_action", "prev. actions"), ("prev_moves", "prev. moves")]
EXP2 = [(4952940, "jump"), (555797, "stream"), (4377326, "tech")]      # rl/maps.yaml exp2
TRAIN_RANGE = (4.53, 6.92)                     # star range of the training maps (configs/splits/train)

# colours: three categorical hues that stay apart under colour-vision deficiency, plus neutrals;
# every series also has its own marker shape, so nothing is told apart by colour alone
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GREY, GRID = "#0b0b0b", "#52514e", "#8f8e89", "#e4e3df"
STYLE_LOOK = {"jumps": (BLUE, "o"), "streams": (ORANGE, "s"), "tech": (AQUA, "^")}


# ---------------------------------------------------------------- data

def snapshot() -> None:
    """Copy the raw evaluation files the figures need out of runs/ (gitignored) into report/data/."""
    (DATA / "curves").mkdir(parents=True, exist_ok=True)
    with open("runs/headline/results.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    keep = [i for i, name in enumerate(rows[0]) if name not in ("score_v2", "passed")]   # columns of removed code
    with open(DATA / "headline_results.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([r[i] for i in keep] for r in rows)
    shutil.copy("runs/headline/modded_stars.json", DATA / "modded_stars.json")
    for p in sorted(Path("rl/runs").glob("exp*/curve.csv")):
        shutil.copy(p, DATA / "curves" / f"{p.parent.name}.csv")
    print(f"snapshot -> {DATA}")


def read(path: Path) -> list[dict]:
    """A CSV as dicts, numbers parsed where a cell is one."""
    out = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
            r["beatmap_id"] = int(r["beatmap_id"])
            out.append(r)
    return out


def human_ok(r: dict) -> bool:
    return abs(r["accuracy"] - r["leaderboard_accuracy"]) <= 0.005


def acc(r: dict) -> float:
    return 100 * r["accuracy"]


def breaks(r: dict) -> float:
    """Combo breaks per 100 objects: missed circles / slider heads plus dropped slider ticks / repeats."""
    return 100 * (r["breaks_miss"] + r["breaks_slider"]) / r["objects"]


class Table:
    """The headline results with the selections the report needs."""

    def __init__(self, rows: list[dict], modded: dict[str, float]):
        self.rows = rows
        for r in rows:                          # star rating as played: with the mod applied
            r["played"] = r["stars"] if r["mods"] == "NM" else modded[f"{r['beatmap_id']}:{r['mods']}"]

    def sel(self, model: str, mods: str, bands=None, style=None) -> list[dict]:
        return [r for r in self.rows if r["model"] == model and r["mods"] == mods
                and (bands is None or r["band"] in bands) and (style is None or r["style"] == style)
                and (model != "human" or human_ok(r))]

    def seeds(self, rs: list[dict], f=acc) -> np.ndarray:
        """The map-averaged value of f, one per seed."""
        by = defaultdict(list)
        for r in rs:
            by[r["seed"]].append(f(r))
        return np.array([np.mean(v) for _, v in sorted(by.items())])

    def per_map(self, rs: list[dict], f=acc) -> dict[int, float]:
        """The seed-averaged value of f, one per map."""
        by = defaultdict(list)
        for r in rs:
            by[r["beatmap_id"]].append(f(r))
        return {b: float(np.mean(v)) for b, v in by.items()}


def pm(v: np.ndarray, d: int = 1) -> str:
    v = np.asarray(v, dtype=float)
    if not len(v):
        return "–"
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def boot(x: np.ndarray, y: np.ndarray | None = None, n: int = 10000) -> tuple[float, float]:
    """95% bootstrap interval over maps of mean(x), or of mean(x) - mean(y) for two map sets."""
    rng = np.random.default_rng(0)
    d = [rng.choice(x, len(x)).mean() - (rng.choice(y, len(y)).mean() if y is not None else 0) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def md_table(head: list[str], lines: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(l) + " |" for l in lines] + [""]


# ---------------------------------------------------------------- numbers

def numbers(t: Table) -> str:
    """report/numbers.md: every table and quoted number of Sections 3-6 that comes from the headline
    table (the RL tables are rl/results/tables.md)."""
    star = [b[0] + "★" for b in BANDS]
    md = ["# Numbers quoted in the report (generated by `python -m report.figures`; do not edit by hand)", "",
          "Models: mean ± sd over 3 seeds of the map-averaged value. Humans: mean over kept replays.", ""]

    md += ["## Test maps and usable human replays (Table 1)", ""]
    lines = []
    for b in BANDS:
        maps = {r["beatmap_id"]: r["style"] for r in t.sel("auto", "NM", [b])}
        allh = [r for r in t.rows if r["model"] == "human" and r["mods"] == "NM" and r["band"] == b]
        lines.append([b[0] + "★"] + [str(sum(s == st for s in maps.values())) for st in STYLES]
                     + [str(len(maps)), f"{sum(map(human_ok, allh))} of {len(allh)}"])
    md += md_table(["Band", "Jumps", "Streams", "Tech", "Total", "Human NoMod replays kept"], lines)
    hum = [r for r in t.rows if r["model"] == "human"]
    md += [f"Human replays judged: {len(hum)}; kept: {sum(map(human_ok, hum))}. By mod (kept / judged): "
           + ", ".join(f"{m} {sum(human_ok(r) for r in hum if r['mods'] == m)} / {sum(r['mods'] == m for r in hum)}"
                       for m in ("NM", "HR", "DT")) + ".", ""]

    md += ["## Natural difficulty: NoMod by star band (Table 2)", ""]
    rows_ = [["Auto"] + [pm(t.seeds(t.sel("auto", "NM", [b])), 2) for b in BANDS],
             ["Human"] + [f"{np.mean([acc(r) for r in t.sel('human', 'NM', [b])]):.1f}" for b in BANDS],
             ["BC"] + [pm(t.seeds(t.sel(BASE, "NM", [b]))) for b in BANDS],
             ["BC combo breaks / 100"] + [pm(t.seeds(t.sel(BASE, "NM", [b]), breaks), 2) for b in BANDS],
             ["  of which misses"] + [f"{t.seeds(t.sel(BASE, 'NM', [b]), lambda r: 100 * r['breaks_miss'] / r['objects']).mean():.2f}" for b in BANDS],
             ["  of which slider breaks"] + [f"{t.seeds(t.sel(BASE, 'NM', [b]), lambda r: 100 * r['breaks_slider'] / r['objects']).mean():.2f}" for b in BANDS],
             ["BC slider-end drops / 100"] + [f"{t.seeds(t.sel(BASE, 'NM', [b]), lambda r: 100 * r['end_drops'] / r['objects']).mean():.2f}" for b in BANDS],
             ["Human combo breaks / 100"] + [f"{np.mean([breaks(r) for r in t.sel('human', 'NM', [b])]):.2f}" for b in BANDS],
             ["BC objects missed, %"] + [pm(t.seeds(t.sel(BASE, "NM", [b]), lambda r: 100 * r["count_miss"] / r["objects"]), 2) for b in BANDS],
             ["Human objects missed, %"] + [f"{np.mean([100 * r['count_miss'] / r['objects'] for r in t.sel('human', 'NM', [b])]):.2f}" for b in BANDS],
             ["BC unstable rate"] + [pm(t.seeds(t.sel(BASE, "NM", [b]), lambda r: r["unstable_rate"]), 0) for b in BANDS],
             ["Human unstable rate"] + [f"{np.mean([r['unstable_rate'] for r in t.sel('human', 'NM', [b])]):.0f}" for b in BANDS]]
    md += md_table([""] + star, rows_)
    br = [t.seeds(t.sel(BASE, "NM", [b]), breaks).mean() for b in BANDS]
    md += [f"Combo breaks grow by a factor of {(br[-1] / br[0]) ** (1 / 5):.2f} per star band (geometric mean, 4★ to 9★).", ""]
    mr = [t.seeds(t.sel(BASE, "NM", [b]), lambda r: 100 * r["count_miss"] / r["objects"]).mean() for b in BANDS]
    md += [f"Objects missed grow by a factor of {(mr[-1] / mr[0]) ** (1 / 5):.2f} per star band (geometric mean, 4★ to 9★).", ""]
    md += ["Objects missed on the 19 in-band maps, %: " + "; ".join(
        f"{m} {pm(t.seeds(t.sel(BASE, m, IN_BAND), lambda r: 100 * r['count_miss'] / r['objects']), 1)}" for m in ("NM", "HR", "DT", "HRDT")) + ".", ""]

    md += ["BC by style (NoMod):", ""]
    md += md_table(["Style"] + star, [[st] + [pm(t.seeds(t.sel(BASE, "NM", [b], st))) for b in BANDS] for st in STYLES])

    md += ["Gap to the in-band mean (4–6★), with a 95% bootstrap interval over maps:", ""]
    inb = np.array(list(t.per_map(t.sel(BASE, "NM", IN_BAND)).values()))
    lines = []
    for b in BANDS[3:]:
        x = np.array(list(t.per_map(t.sel(BASE, "NM", [b])).values()))
        lo, hi = boot(inb, x)
        lines.append([b[0] + "★", f"{inb.mean() - x.mean():.1f}", f"[{lo:.1f}, {hi:.1f}]", f"{x.min():.1f} to {x.max():.1f}"])
    md += md_table(["Band", "Gap, pp", "95% interval", "Per-map accuracy range"], lines)
    md += [f"In-band mean over the 19 maps: {inb.mean():.2f}%.", ""]

    below = total = 0
    for b in BANDS[3:]:
        bc = t.per_map(t.sel(BASE, "NM", [b]))
        hmin = defaultdict(lambda: 101.0)
        for r in t.sel("human", "NM", [b]):
            hmin[r["beatmap_id"]] = min(hmin[r["beatmap_id"]], acc(r))
        below += sum(bc[m] < hmin[m] for m in bc)
        total += len(bc)
    md += [f"On {below} of the {total} maps at 7–9★ the policy (seed mean) is below the lowest kept human replay.", ""]

    md += ["## Synthetic difficulty: mods on the in-band maps (Table 3)", ""]
    lines = [[m] + [pm(t.seeds(t.sel(BASE, m, [b]))) for b in IN_BAND] + [pm(t.seeds(t.sel(BASE, m, IN_BAND)))]
             for m in ("NM", "HR", "DT", "HRDT")]
    lines += [[f"Human, {m}"] + [f"{np.mean([acc(r) for r in t.sel('human', m, [b])]):.1f} (n={len(t.sel('human', m, [b]))})"
                                 for b in IN_BAND] + [""] for m in ("HR", "DT")]
    md += md_table(["Mods", "4★", "5★", "6★", "all 19 maps"], lines)
    md += ["BC by style on the 19 in-band maps:", ""]
    md += md_table(["Mods"] + STYLES, [[m] + [pm(t.seeds(t.sel(BASE, m, IN_BAND, st))) for st in STYLES]
                                       for m in ("NM", "HR", "DT", "HRDT")])
    md += ["BC unstable rate on the 19 in-band maps: " + "; ".join(
        f"{m} {pm(t.seeds(t.sel(BASE, m, IN_BAND), lambda r: r['unstable_rate']), 0)}" for m in ("NM", "HR", "DT", "HRDT")) + ".", ""]
    md += ["Per seed (all 19 in-band maps): " + "; ".join(
        f"{m} " + " / ".join(f"{v:.1f}" for v in t.seeds(t.sel(BASE, m, IN_BAND))) for m in ("NM", "HR", "DT", "HRDT")), ""]
    nm = t.per_map(t.sel(BASE, "NM", IN_BAND))
    lines = []
    for m in ("HR", "DT", "HRDT"):
        mod = t.per_map(t.sel(BASE, m, IN_BAND))
        d = np.array([nm[b] - mod[b] for b in nm])
        lo, hi = boot(d)
        lines.append([m, f"{d.mean():.1f}", f"[{lo:.1f}, {hi:.1f}]", f"{(d > 0).sum()} of {len(d)}"])
    md += ["Cost of each mod (NoMod minus mod, paired by map), 95% bootstrap interval over the 19 maps:", ""]
    md += md_table(["Mods", "Cost, pp", "95% interval", "Maps that get worse"], lines)

    md += ["Accuracy by star rating as played (whole-star bins; n = maps):", ""]
    lines = []
    for m in ("NM", "HR", "DT", "HRDT"):
        cells = []
        for s in range(4, 13):
            rs = [r for r in t.sel(BASE, m) if int(r["played"]) == s]
            cells.append(f"{pm(t.seeds(rs))} (n={len({r['beatmap_id'] for r in rs})})" if rs else "")
        lines.append([m] + cells)
    for m in ("NM", "DT"):
        cells = []
        for s in range(4, 13):
            rs = [r for r in t.sel("human", m) if int(r["played"]) == s]
            cells.append(f"{np.mean([acc(r) for r in rs]):.1f} (n={len(rs)} replays)" if rs else "")
        lines.append([f"Human, {m}"] + cells)
    md += md_table(["Mods"] + [f"{s}★" for s in range(4, 13)], lines)

    md += ["## Model variation (Table 5)", ""]
    cells = [("NM", [b]) for b in BANDS] + [("NM", IN_BAND), ("HR", IN_BAND), ("DT", IN_BAND), ("HRDT", IN_BAND)]
    head = ["Variant"] + [f"NoMod {s}" for s in star] + ["NoMod 4–6★", "HR 4–6★", "DT 4–6★", "HR+DT 4–6★"]
    md += md_table(head, [[name] + [pm(t.seeds(t.sel(mdl, m, bs))) for m, bs in cells] for mdl, name in VARIANTS])
    md += ["Combo breaks per 100 objects (NoMod):", ""]
    md += md_table(["Variant"] + star, [[name] + [f"{t.seeds(t.sel(mdl, 'NM', [b]), breaks).mean():.1f}" for b in BANDS]
                                        for mdl, name in VARIANTS[:6]])
    md += ["Per seed, in-band HR+DT and DT:", ""]
    md += md_table(["Variant", "HR+DT per seed", "DT per seed"],
                   [[name, " / ".join(f"{v:.1f}" for v in t.seeds(t.sel(mdl, "HRDT", IN_BAND))),
                     " / ".join(f"{v:.1f}" for v in t.seeds(t.sel(mdl, "DT", IN_BAND)))] for mdl, name in VARIANTS[:6]])
    md += ["Difference to the baseline, paired by map (seed means), 95% bootstrap interval over maps:", ""]
    lines = []
    for mdl, name in VARIANTS[1:6]:
        for m, bs, what in (("NM", ["8_star", "9_star"], "NoMod 8–9★"), ("DT", IN_BAND, "DT 4–6★"), ("HRDT", IN_BAND, "HR+DT 4–6★")):
            a, b = t.per_map(t.sel(mdl, m, bs)), t.per_map(t.sel(BASE, m, bs))
            d = np.array([a[k] - b[k] for k in a])
            lo, hi = boot(d)
            lines.append([name, what, f"{d.mean():+.1f}", f"[{lo:+.1f}, {hi:+.1f}]", f"{(d > 0).sum()} of {len(d)}"])
    md += md_table(["Variant", "Cell", "Difference, pp", "95% interval", "Maps better"], lines)
    md += ["The first architecture study's runs (epoch selected under the earlier noisy motor), same evaluation:", ""]
    orig = [("baseline", "baseline"), ("framestack_half", "half width"), ("framestack_double", "double width"),
            ("cnn3d", "3D CNN"), ("cnn_gru", "CNN + GRU"), ("softargmax", "soft-argmax")]
    md += md_table(head, [[name] + [pm(t.seeds(t.sel(mdl, m, bs))) for m, bs in cells] for mdl, name in orig])

    md += ["## Circle misses by cause at 8–9★ (Table 4): BC seed 0, from the RL evaluation files", ""]
    a = [r for r in read(RL / "exp2_transfer.csv") + read(RL / "exp2.csv")
         if r["label"] == "exp2_A" and r["beatmap_id"] != 555797]
    lines = []
    for st in STYLES:
        x = [r for r in a if r["style"] == st]
        pos, click, miss = (sum(r[k] for r in x) for k in ("miss_circle_position", "miss_circle_click", "count_miss"))
        lines.append([st, str(len(x)), f"{pos:.0f}", f"{click:.0f}", f"{miss - pos - click:.0f}", f"{100 * pos / (pos + click):.0f}%"])
    md += md_table(["Style", "Maps", "Aim", "Click / timing", "Other objects", "Aim share of circle misses"], lines)
    return "\n".join(md)


# ---------------------------------------------------------------- figures
# One name per thing in every figure: "baseline policy", "top human players", "no modifier",
# "1.5x speed" (Double Time), "smaller circles" (Hard Rock), "Hidden". Colours: MAP_LOOK is the map
# type (jump / stream / tech) wherever it appears; MOD_LOOK is the modifier; blue / red bars are
# better / worse than the baseline. Every series also has its own marker or a direct label.

MAP_LOOK = {"jumps": (BLUE, "o", "jump"), "streams": (ORANGE, "s", "stream"), "tech": (AQUA, "^", "tech")}
VIOLET, RED, LIGHT = "#6250d6", "#e34948", "#c9c8c2"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):          # the report's font, where it is installed
        for root in ("/mnt/c/Windows/Fonts", "C:/Windows/Fonts"):
            if (Path(root) / f).exists():
                font_manager.fontManager.addfont(str(Path(root) / f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


def dots(ax, x, y, colour, marker, size=22, alpha=1.0, **kw):
    """Markers with a thin white ring, so overlapping points stay separate."""
    return ax.scatter(x, y, s=size, c=colour, marker=marker, edgecolors="white", linewidths=0.6, alpha=alpha, **kw)


def missed(r: dict) -> float:
    """Objects missed, as a percentage of the map's objects."""
    return 100 * r["count_miss"] / r["objects"]


def paired(t: Table, model: str, mods: str, bands: list[str]) -> tuple[float, float, float]:
    """Accuracy of `model` minus the baseline's, paired by map (seed means): mean and 95% interval."""
    a, b = t.per_map(t.sel(model, mods, bands)), t.per_map(t.sel(BASE, mods, bands))
    d = np.array([a[k] - b[k] for k in a])
    lo, hi = boot(d)
    return float(d.mean()), lo, hi


def fig_frames(plt) -> None:
    """Fig. 1: what the policy sees. One moment of a test map without a modifier, with Hard Rock and
    with Hidden (128x96 frames, brightened for print). The upcoming circles carry their hit order
    inside them, as the game itself draws it."""
    from matplotlib import patheffects
    from data.dataset import RAW, mods_from_str
    from osu_sim.beatmap import parse_beatmap
    from osu_sim.playable import make_playable
    from osu_sim.renderer import PX, Renderer
    bid, t_ms = 2014470, 79_955.0               # a slider under way and three circles 0.2-0.5 s ahead
    bm = parse_beatmap(RAW / "beatmaps" / f"{bid}.osu")
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.95))
    names = (("NM", "No modifier"), ("HR", "Hard Rock: flipped, smaller circles"),
             ("HD", "Hidden: no approach ring, objects fade"))
    outline = [patheffects.withStroke(linewidth=1.4, foreground="black")]
    for ax, (mods, name) in zip(axes, names):
        pl = make_playable(bm, mods_from_str(mods))
        first = next(i for i, o in enumerate(pl.objects) if o.kind == "slider" and o.time <= t_ms <= o.end_time)
        ball = pl.objects[first].ball_position(t_ms)
        frame = Renderer(pl).frame(t_ms, tuple(ball)).astype(float) / 255
        ax.imshow(frame ** 0.55, cmap="gray", vmin=0, vmax=1, interpolation="nearest")     # gamma: visible in print
        for n, o in enumerate(pl.objects[first + 1:first + 4], start=1):                 # the next three circles
            if t_ms >= o.time - pl.preempt:       # on screen (Hard Rock shows objects later)
                ax.text(o.x / PX, o.y / PX + 0.3, str(n), color="white", fontsize=7, fontweight="bold", ha="center",
                        va="center", path_effects=outline)
        if mods == "NM":                          # name the parts once, on the first panel
            o1 = pl.objects[first + 1]
            notes = [("cursor", (ball[0] / PX, ball[1] / PX), (124, 90), "right"),
                     ("slider", ((ball[0] + 14) / PX, (ball[1] - 26) / PX), (124, 38), "right"),
                     ("circle", ((o1.x - 24) / PX, (o1.y + 16) / PX), (3, 90), "left"),
                     ("approach ring", ((o1.x - 60) / PX, (o1.y - 45) / PX), (3, 12), "left")]
            for text, xy, at, ha in notes:
                ax.annotate(text, xy, xytext=at, color="white", fontsize=6.8, ha=ha, va="center",
                            arrowprops={"arrowstyle": "-", "color": "white", "lw": 0.6, "shrinkA": 1, "shrinkB": 1})
        ax.set_title(name, fontsize=8, fontweight="normal", loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIGS / "fig1_frames.png")
    plt.close(fig)


def fig_overview(plt) -> None:
    """Fig. 2: the system in a few words. The policy network in a closed loop with the simulator, and
    the two training stages. The caption and Section 4 carry the detail."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.5, 1.85))
    ax.set(xlim=(0, 100), ylim=(0, 29))
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.27, body, ha="center", va="center", fontsize=7.4, color=MUTED)

    def arrow(a, b, colour=INK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=colour, shrinkA=0, shrinkB=0))

    # the loop, left to right: frames -> policy network (encoder, outputs) -> simulator -> next frame
    ax.add_patch(FancyBboxPatch((23.5, 9.3), 43, 13.6, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f1f0ec", ec="none"))
    ax.text(45, 21.2, "Policy network", fontsize=7.6, ha="center", va="center", color=MUTED)
    box(2, 10.5, 15, 8.4, "Input", "last 20 frames")
    box(25.5, 10.5, 17, 8.4, "Encoder", "4 conv layers")
    box(47.5, 10.5, 17, 8.4, "Outputs", "cursor, 2 keys")
    box(73, 10.5, 25, 8.4, "Simulator", "next frame, score")
    for x0, x1 in ((17.3, 25.2), (42.8, 47.2), (64.8, 72.7)):
        arrow((x0, 14.7), (x1, 14.7))
    ax.plot([85.5, 85.5, 9.5], [19.2, 26.6, 26.6], color=INK, lw=0.9, solid_capstyle="round")
    arrow((9.5, 26.6), (9.5, 19.3))
    ax.text(47.5, 27.4, "next frame, 60 times a second", fontsize=7.2, ha="center", va="bottom", color=MUTED)
    # the two training stages
    ax.text(24, 3.2, "Stage 1: imitate human replays", fontsize=7.8, ha="center", va="center", color=INK,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": BLUE, "lw": 0.9})
    ax.text(76, 3.2, "Stage 2: PPO, reward from the score", fontsize=7.8, ha="center", va="center", color=INK,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": ORANGE, "lw": 0.9})
    arrow((34, 5.4), (34, 9.0), colour=BLUE)
    arrow((85.5, 10.2), (85.5, 5.5), colour=ORANGE)
    arrow((62, 5.4), (58, 9.0), colour=ORANGE)
    fig.savefig(FIGS / "fig2_overview.png")
    plt.close(fig)


def fig_difficulty(t: Table, plt) -> None:
    """Fig. 3: accuracy and missed objects against map difficulty, without modifiers: the cloned
    policy against top human players."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    centre = np.arange(4.5, 10)
    a.axvspan(*TRAIN_RANGE, color="#f1f0ec", lw=0)
    a.text(np.mean(TRAIN_RANGE), 57, "difficulty range of\nthe training maps", ha="center", va="bottom", color=MUTED, fontsize=7.5)
    stars = {r["beatmap_id"]: r["stars"] for r in t.sel(BASE, "NM")}
    pm_ = t.per_map(t.sel(BASE, "NM"))
    dots(a, [stars[m] for m in pm_], list(pm_.values()), LIGHT, "o", size=16)
    hum = [np.mean([acc(r) for r in t.sel("human", "NM", [bd])]) for bd in BANDS]
    a.plot(centre, hum, color=GREY, marker="D", ms=3.5, lw=1.6)
    s = [t.seeds(t.sel(BASE, "NM", [bd])) for bd in BANDS]
    mean = [v.mean() for v in s]
    a.errorbar(centre, mean, yerr=[v.std(ddof=1) for v in s], color=INK, marker="o", ms=3.5, lw=1.8, capsize=2)
    a.text(8.05, 101.3, "top human players", color=MUTED, fontsize=7.8, ha="center", va="bottom")
    a.text(8.3, 86.5, "baseline\npolicy", fontsize=7.8, ha="right", va="center", fontweight="bold")
    a.annotate("", (9.72, mean[-1] + 0.6), (9.72, hum[-1] - 0.6), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.7})
    a.text(9.82, (mean[-1] + hum[-1]) / 2, f"{hum[-1] - mean[-1]:.0f}%\nbelow", fontsize=7.5, va="center")
    a.text(5.72, 96.4, f"about {np.mean(hum[:3]) - np.mean(mean[:3]):.0f}% below", fontsize=7.5, ha="center", va="top")
    a.text(7.15, 60.5, "light dots: single maps", fontsize=7, color=MUTED, va="center")
    a.set(xlabel="map difficulty (star rating)", ylabel="accuracy, %", xlim=(3.9, 10.95), ylim=(55, 104),
          title="(a) Accuracy falls beyond the training range")
    a.set_xticks(range(4, 11))
    a.set_yticks(range(60, 101, 10))

    ms = [t.seeds(t.sel(BASE, "NM", [bd]), missed) for bd in BANDS]
    hm = [max(np.mean([missed(r) for r in t.sel("human", "NM", [bd])]), 0.01) for bd in BANDS]
    b.errorbar(centre, [v.mean() for v in ms], yerr=[v.std(ddof=1) for v in ms], color=INK, marker="o", ms=3.5, lw=1.8, capsize=2)
    b.plot(centre, hm, color=GREY, marker="D", ms=3.5, lw=1.6)
    b.set_yscale("log")
    b.set_yticks([0.01, 0.1, 1, 10])
    b.set_yticklabels(["≤ 0.01", "0.1", "1", "10"])
    b.set(xlabel="map difficulty (whole-star range)", ylabel="objects missed, % (log scale)", xlim=(4, 10), ylim=(0.007, 40),
          title="(b) Misses grow steadily")
    b.set_xticks(centre)
    b.set_xticklabels([bd[0] for bd in BANDS])
    b.text(6.9, 5.0, "baseline policy", ha="right", fontsize=7.8, fontweight="bold")
    b.text(9.6, 0.017, "top human players", ha="right", color=MUTED, fontsize=7.8, va="center")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGS / "fig3_difficulty.png")
    plt.close(fig)


def fig_misses(plt) -> None:
    """Fig. 4: (a, b) a moment of a 9-star jump map and of a 9-star stream map, as the policy sees
    them; (c) the cause of every missed circle on the 8- and 9-star test maps, by map type (baseline
    policy, first training run; from the PPO evaluation files)."""
    from matplotlib.patches import Patch
    from data.dataset import RAW
    from osu_sim.beatmap import parse_beatmap
    from osu_sim.playable import make_playable
    from osu_sim.renderer import PX, Renderer
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.75), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    shots = ((4669413, "jump", "(a) Jump map: far apart"), (4798513, "stream", "(b) Stream map: a dense run"))
    for ax, (bid, kind, title) in zip(axes, shots):
        pl = make_playable(parse_beatmap(RAW / "beatmaps" / f"{bid}.osu"), 0)
        best = (-1.0, 0.0, [])
        for o in pl.objects:                      # the moment that shows the pattern most clearly
            t = o.time - 60.0
            vis = [q for q in pl.objects if q.kind == "circle" and q.time - pl.preempt <= t < q.time]
            if len(vis) < 3 or any(not (70 < q.x < 442 and 60 < q.y < 324) for q in vis):
                continue                          # every object well inside the picture
            gaps = [np.hypot(b.x - a.x, b.y - a.y) for a, b in zip(vis, vis[1:])]
            if kind == "jump":                    # three circles, each far from the others
                apart = min(np.hypot(b.x - a.x, b.y - a.y) for i, a in enumerate(vis) for b in vis[i + 1:])
                score = apart if len(vis) == 3 else -1.0
            else:                                 # many circles in a row, evenly and closely spaced
                score = len(vis) if 14 < min(gaps) and max(gaps) < 60 else -1.0
            if score > best[0]:
                best = (score, t, vis)
        _, t, vis = best
        prev = max((q for q in pl.objects if q.time <= t), key=lambda q: q.time, default=vis[0])
        frame = Renderer(pl).frame(t, tuple(prev.ball_position(t))).astype(float) / 255
        ax.imshow(frame ** 0.55, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if kind == "jump":
            for n, q in enumerate(vis, start=1):
                ax.text(q.x / PX, q.y / PX, str(n), color="white", fontsize=7.5, fontweight="bold", ha="center", va="center")
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)

    c = axes[2]
    rows_ = [r for r in read(RL / "exp2_transfer.csv") + read(RL / "exp2.csv")
             if r["label"] == "exp2_A" and r["beatmap_id"] != 555797]
    names = [("jumps", "Jump maps (6)"), ("streams", "Stream maps (4)"), ("tech", "Tech maps (2)")]
    for yi, (st, _) in zip((2, 1, 0), names):
        x = [r for r in rows_ if r["style"] == st]
        aim, tim = sum(r["miss_circle_position"] for r in x), sum(r["miss_circle_click"] for r in x)
        share = 100 * aim / (aim + tim)
        c.barh(yi, share, height=0.5, color=INK, zorder=2)
        c.barh(yi, 100 - share, left=share, height=0.5, color=LIGHT, zorder=2)
        if share > 25:
            c.text(share / 2, yi, f"{aim:.0f}", color="white", fontsize=7.5, ha="center", va="center", zorder=3)
        if share < 75:
            c.text(share + (100 - share) / 2, yi, f"{tim:.0f}", color=INK, fontsize=7.5, ha="center", va="center", zorder=3)
    c.set_yticks((2, 1, 0))
    c.set_yticklabels([n for _, n in names], color=INK)
    c.yaxis.tick_right()
    c.set(xlim=(0, 100), ylim=(-0.5, 2.5), xlabel="share of missed circles, % (numbers: circles)")
    c.set_title("(c) Why circles are missed", fontsize=8, fontweight="bold", loc="left")
    c.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=INK, label="aim miss: the cursor is not on the circle"),
                        Patch(color=LIGHT, label="timing miss: the cursor is on it, the click is absent or mistimed")],
               loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.2, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIGS / "fig4_misses.png")
    plt.close(fig)


def fig_modifiers(t: Table, plt) -> None:
    """Fig. 7: accuracy against difficulty with the modifier applied: is a sped-up easy map as hard as
    an unmodified map of the same star rating?"""
    fig, ax = plt.subplots(figsize=(6.5, 2.25))
    look = {"NM": (GREY, "o", "no modifier"), "DT": (VIOLET, "^", "1.5× speed (Double Time)"),
            "HRDT": (RED, "D", "1.5× speed and smaller circles\n(Double Time + Hard Rock)")}
    ends = {}
    for m, (c, mk, name) in look.items():
        rs = t.sel(BASE, m)
        a_ = t.per_map(rs)
        played = {r["beatmap_id"]: r["played"] for r in rs}
        dots(ax, [played[k] for k in a_], list(a_.values()), c, mk, size=20, alpha=0.6 if m == "NM" else 0.8, label=name)
        xs, ys = [], []
        for s in range(4, 13):                 # the mean of each whole-star range with at least 2 maps
            v = [a_[k] for k in a_ if int(played[k]) == s]
            if len(v) >= 2:
                xs.append(s + 0.5)
                ys.append(np.mean(v))
        ax.plot(xs, ys, color=c, lw=1.8, zorder=3)
        ends[m] = (xs, ys)
    ax.annotate("", (7.5, 74.3), (7.5, 95.3), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.7})
    ax.text(7.1, 82.5, "about 20% below\nunmodified maps of the\nsame rating (7 and 8 stars)", fontsize=7.5,
            ha="right", va="center")
    ax.legend(loc="lower left", handletextpad=0.3, labelspacing=0.5, borderaxespad=0.3)
    ax.set(xlabel="difficulty with the modifier applied (star rating)", ylabel="accuracy, %", xlim=(3.9, 12.5), ylim=(45, 102))
    ax.set_xticks(range(4, 13))
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_modifiers.png")
    plt.close(fig)


def fig_architectures(t: Table, plt) -> None:
    """Fig. 5: each architecture's accuracy change against the baseline, on harder maps and under the
    speed modifiers (paired by map, with 95% bootstrap intervals over maps)."""
    rows = [("softargmax_cap", "Soft-argmax aim head"), ("cnn3d_cap", "3D convolutions"), ("cnn_gru_cap", "CNN per frame + GRU"),
            ("framestack_double_cap", "Double width"), ("framestack_half_cap", "Half width")]
    panels = [("NM", ["8_star", "9_star"], "(a) Harder maps\n(8 and 9 stars, no modifier)"),
              ("DT", IN_BAND, "(b) 1.5× speed\n(training-range maps)"),
              ("HRDT", IN_BAND, "(c) 1.5× speed, smaller circles\n(training-range maps)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (mods, bands, title) in zip(axes, panels):
        for yi, (mdl, _) in zip(y, rows):
            d, lo, hi = paired(t, mdl, mods, bands)
            ax.barh(yi, d, height=0.56, color=BLUE if d >= 0 else RED, zorder=2)
            ax.plot([lo, hi], [yi, yi], color=INK, lw=0.9, zorder=3)
            ax.text((hi if d >= 0 else lo) + (0.7 if d >= 0 else -0.7), yi, f"{d:+.1f}".replace("-", "−"), va="center",
                    ha="left" if d >= 0 else "right", fontsize=7.5)
        ax.axvline(0, color=INK, lw=0.8, zorder=4)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set(xlim=(-21, 22), ylim=(-0.6, len(rows) - 0.4))
        ax.set_xticks([-10, 0, 10, 20])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([name for _, name in rows], color=INK)
    axes[1].set_xlabel("accuracy change against the baseline, % (blue: better, red: worse)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGS / "fig5_architectures.png")
    plt.close(fig)


def curve(run: str, mods: str = "NM") -> tuple[np.ndarray, np.ndarray]:
    """(environment steps, map-averaged accuracy %) of one PPO run's learning-curve evaluations."""
    pts = defaultdict(list)
    for r in read(DATA / "curves" / f"{run}.csv"):
        if r["mods"] == mods:
            pts[int(r["label"].split("_")[1])].append(acc(r))
    steps = sorted(pts)
    return np.array(steps), np.array([np.mean(pts[s]) for s in steps])


def fig_ppo(plt) -> None:
    """Fig. 6: PPO on one hard map: the gain on that map during training, and the gain of the final
    policies on ten other hard maps."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.25), sharey=True)
    look = {"jump": MAP_LOOK["jumps"], "stream": MAP_LOOK["streams"], "tech": MAP_LOOK["tech"]}
    stars = {4952940: "8.0", 555797: "8.3", 4377326: "8.0"}
    for bid, cat in EXP2:
        col, mk, _ = look[cat]
        runs = [curve(f"exp2_B_{bid}_seed{s}") for s in range(3)]
        for steps, v in runs:
            a.plot(steps / 1e6, v - v[0], color=col, lw=0.7, alpha=0.45)
        n = min(len(v) for _, v in runs)
        mean = np.mean([v[:n] - v[0] for _, v in runs], axis=0)
        a.plot(runs[0][0][:n] / 1e6, mean, color=col, lw=1.8, marker=mk, ms=3.5, label=f"the {cat} map ({stars[bid]} stars)")
    a.set(xlabel="PPO training, millions of steps", ylabel="accuracy gain over the baseline\npolicy, %",
          title="(a) On the map it is fine-tuned on", xlim=(0, 1.02), ylim=(-1.5, 10.5))
    a.legend(loc="upper left", title="policy fine-tuned on", title_fontsize=8, handlelength=1.6, labelspacing=0.25, alignment="left")

    tr = read(RL / "exp2_transfer.csv")
    base = {r["beatmap_id"]: acc(r) for r in tr if r["label"] == "exp2_A"}
    b.axhline(0, color=GREY, lw=0.8)
    for bid, cat in EXP2:
        by = defaultdict(list)
        for r in tr:
            if r["label"] == f"exp2_B_{bid}":
                by[r["beatmap_id"]].append(acc(r) - base[r["beatmap_id"]])
        dots(b, [base[m] for m in by], [np.mean(v) for v in by.values()], look[cat][0], look[cat][1], size=24)
    b.text(86, 8.6, "larger gains where the\nbaseline policy was weak", fontsize=7.6, ha="center", color=MUTED)
    b.set(xlabel="baseline policy's accuracy on that map, %", title="(b) On ten other hard maps it never trained on", xlim=(62, 98))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGS / "fig6_ppo.png")
    plt.close(fig)


def fig_hidden(plt) -> None:
    """Fig. 8: adapting to the Hidden modifier: accuracy with Hidden and without it, for the cloned
    policy and four ways of adapting it."""
    e1 = read(RL / "exp1.csv")

    def cond(label: str, mods: str) -> np.ndarray:
        by = defaultdict(list)
        for r in e1:
            if r["label"] == label and r["mods"] == mods:
                by[r["seed"]].append(acc(r))
        return np.array([np.mean(v) for v in by.values()])
    rows = [("exp1_A", "Baseline policy, not adapted"), ("exp1_B", "PPO with Hidden"),
            ("exp1_C", "Supervised fine-tuning,\nHidden frames only"), ("exp1_D", "… followed by PPO,\nwith and without Hidden"),
            ("exp1_Cmix", "Supervised fine-tuning,\nHidden and normal frames mixed")]
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    y = np.arange(len(rows))[::-1].astype(float)
    for yi, (label, _) in zip(y, rows):
        for off, mods, colour in ((0.17, "HD", INK), (-0.17, "NM", LIGHT)):
            v = cond(label, mods)
            ax.barh(yi + off, v.mean(), height=0.3, color=colour, zorder=2)
            if len(v) > 1:
                ax.plot([v.min(), v.max()], [yi + off] * 2, color=GREY if mods == "HD" else INK, lw=0.9, zorder=3)
            ax.text(max(v.max(), v.mean()) + 1.2, yi + off, f"{v.mean():.1f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in rows], color=INK)
    ax.set(xlabel="accuracy on the 19 training-range test maps, %", xlim=(0, 108), ylim=(-0.6, len(rows) - 0.4))
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK, label="played with Hidden (the new condition)"),
                       Patch(color=LIGHT, label="played without a modifier (what it could do before)")],
              loc="upper center", bbox_to_anchor=(0.42, 1.2), ncol=2, handlelength=1.2, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_hidden.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="The report's figures and quoted numbers.")
    ap.add_argument("--snapshot", action="store_true", help="copy the raw evaluation files from runs/ into report/data/ first")
    ap.add_argument("--frames", action="store_true", help="also render Figures 1 and 4, which draw game frames (needs data/raw)")
    args = ap.parse_args()
    if args.snapshot:
        snapshot()
    t = Table(read(DATA / "headline_results.csv"), json.loads((DATA / "modded_stars.json").read_text()))
    FIGS.mkdir(exist_ok=True)
    (HERE / "numbers.md").write_text(numbers(t), encoding="utf-8")
    plt = style()
    fig_overview(plt)
    fig_difficulty(t, plt)
    fig_modifiers(t, plt)
    fig_architectures(t, plt)
    fig_ppo(plt)
    fig_hidden(plt)
    if args.frames:
        fig_frames(plt)
        fig_misses(plt)
    print(f"wrote {HERE / 'numbers.md'} and {FIGS}/fig*.png")


if __name__ == "__main__":
    main()
