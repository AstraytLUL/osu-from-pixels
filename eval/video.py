"""Gameplay footage from the simulator: the 128x96 frames the policy sees, scaled up 4x, with the
cursor drawn in, plus a strip underneath with the keys, judgement counts and accuracy so far.
For watching a player's replay, Auto, or (later) a model's rollout.

    python -m eval.video --score 890068308 --start 30 --duration 15             # a player's replay, 15 s clip
    python -m eval.video --beatmap 553131 --mods HR --start 20 --duration 6 --format gif   # Auto

Files are named by eval/paths.py, e.g. runs/footage/human-890068308/553131-<title>_NM_30-45s.mp4
and runs/footage/auto/553131-<title>_HR_20-26s.gif (--out overrides).

.mp4 needs the imageio-ffmpeg package (it bundles ffmpeg); .gif needs only Pillow but is large,
so keep GIFs to short clips. Video runs at 60 fps in real time, one frame per grid step
(data/resample.py), so DT footage plays 1.5x faster in map time, as in the game.

Unlike the training frames, circles here disappear when the player hit (or missed) them, as in
the game, using the judge's results (Renderer's `hit_times`), and slider ticks are drawn (on the
scaled-up picture, so the frames the policy sees are unchanged). The cursor and keys shown are the
replay resampled onto the grid, i.e. what the policy would be trained on; the judgement counts
come from judging the replay itself.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.dataset import RAW, mods_from_str
from eval.paths import footage_path
from data.replay import ReplayFrames, load_replay
from data.resample import map_grid, resample
from osu_sim.auto import auto_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import Judgement, hit_windows, judge
from osu_sim.playable import Playable, make_playable
from osu_sim.renderer import H, PX, W, Renderer, add_cursor

SCALE = 4                      # 128x96 -> 512x384 (nearest neighbour: each policy pixel stays visible)
HUD = 48                       # height of the strip under the playfield, px
POPUP_MS = 400                 # how long a judgement stays on screen, map ms
GIF_STRIDE = 3                 # GIFs keep every 3rd frame: 20 fps, 50 ms per frame (GIF timing is in 10 ms)
COLOURS = {300: (102, 204, 255), 100: (120, 230, 90), 50: (255, 210, 60), 0: (255, 70, 70)}
LABELS = {300: "300", 100: "100", 50: "50", 0: "X"}


def judged_at(j: Judgement, pl: Playable, w50: float) -> float:
    """Map time a judgement appears: a circle when clicked (or when its 50 window closes, if never
    clicked); a slider or spinner at its end."""
    o = pl.objects[j.index]
    if o.kind == "circle":
        return o.time + (j.offset if j.offset is not None else w50)
    return o.end_time


def head_gone_at(j: Judgement, pl: Playable, w50: float) -> float:
    """Map time a circle or slider head stops being drawn: when clicked, or when its 50 window closes."""
    o = pl.objects[j.index]
    return o.time + (j.offset if j.offset is not None else w50)


def render_video(pl: Playable, frames: ReplayFrames, out: Path, title: str,
                 start_ms: float | None = None, end_ms: float | None = None) -> dict[int, int]:
    """Write the footage of `frames` played on `pl` to `out` (.mp4 or .gif); returns the counts shown."""
    w50 = hit_windows(pl.od).w50
    js = judge(pl, frames)
    shown_at = np.array([judged_at(j, pl, w50) for j in js])
    hit_times = {j.index: head_gone_at(j, pl, w50) for j in js if pl.objects[j.index].kind != "spinner"}
    renderer = Renderer(pl, hit_times)

    times = map_grid(pl)
    keep = np.ones(len(times), dtype=bool)
    if start_ms is not None:
        keep &= times >= start_ms
    if end_ms is not None:
        keep &= times <= end_ms
    g = resample(frames, times)
    ks = np.flatnonzero(keep)
    if out.suffix == ".gif":
        ks = ks[::GIF_STRIDE]

    font = ImageFont.load_default(size=16)
    images = []
    for k in ks:
        t = times[k]
        layer = renderer.map_layer(t)[None]
        frame = add_cursor(layer, np.array([[g.x[k], g.y[k]]]))[0]
        images.append(_compose(frame, t, (g.press[k] | g.held[k]), js, shown_at, pl, title, font,
                               slider_ticks(renderer, t)))

    _write(images, out)
    done = [j.result for j, s in zip(js, shown_at) if s <= times[ks[-1]]]
    return {r: done.count(r) for r in (300, 100, 50, 0)}


def slider_ticks(renderer: Renderer, t: float) -> list[tuple[float, float, float]]:
    """(x, y, opacity) of the slider ticks on screen at map time t: those of the slider's current
    span (the first span before the slider starts) that the ball hasn't reached yet, faded with the
    slider body, as osu! shows them."""
    pl, out = renderer.pl, []
    lo = np.searchsorted(renderer.starts, t - 60000)               # as Renderer.map_layer: nothing lasts a minute
    hi = np.searchsorted(renderer.starts, t + pl.preempt, side="right")
    for o in pl.objects[lo:hi]:
        if o.kind != "slider" or not o.timing.ticks or t > o.end_time:
            continue
        a = renderer.body_alpha(o, t)
        span = int(max(t - o.time, 0) / o.timing.span_duration)
        for tick_time, progress in o.timing.ticks:
            if tick_time > t and int((tick_time - o.time) / o.timing.span_duration) == span:
                x, y = o.path_point(progress)
                out.append((x, y, a))
    return out


def _compose(frame, t, keys, js, shown_at, pl, title, font, ticks=()) -> np.ndarray:
    """One output image: the scaled-up frame, slider ticks and judgement popups on the playfield,
    the HUD strip."""
    big = Image.fromarray(frame).resize((W * SCALE, H * SCALE), Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (W * SCALE, H * SCALE + HUD), (20, 20, 20))
    canvas.paste(big, (0, 0))
    draw = ImageDraw.Draw(canvas)
    to_px = SCALE / PX         # osu! pixels -> output pixels (1.0 at the defaults)

    # slider ticks: small dots, a tenth of the circle radius
    dot = max(2.0, 0.1 * pl.radius * to_px)
    for x, y, a in ticks:
        grey = int(230 * a)
        draw.ellipse((x * to_px - dot, y * to_px - dot, x * to_px + dot, y * to_px + dot), fill=(grey, grey, grey))

    # popups: judgements made in the last POPUP_MS, drawn at the object
    for j, s in zip(js, shown_at):
        if s <= t < s + POPUP_MS:
            o = pl.objects[j.index]
            x, y = (o.x, o.y) if o.kind != "spinner" else (256, 192)
            draw.text((x * to_px, y * to_px), LABELS[j.result], fill=COLOURS[j.result], font=font, anchor="mm")

    # HUD: judgement counts and accuracy so far, time, keys
    done = [j.result for j, s in zip(js, shown_at) if s <= t]
    n = {r: done.count(r) for r in (300, 100, 50, 0)}
    acc = sum(done) / (300 * len(done)) if done else 1.0
    y0 = H * SCALE
    draw.text((8, y0 + 4), f"{title}   t = {t / 1000:6.2f} s", fill="white", font=font)
    draw.text((8, y0 + 24), f"300 {n[300]}   100 {n[100]}   50 {n[50]}   miss {n[0]}   acc {acc:.2%}",
              fill="white", font=font)
    for i, name in enumerate(("L", "R")):
        down = bool(keys & (1 << i))
        x = W * SCALE - 96 + i * 46
        draw.rectangle((x, y0 + 10, x + 38, y0 + 38), fill=(60, 220, 90) if down else (70, 70, 70))
        draw.text((x + 19, y0 + 24), name, fill="black" if down else "white", font=font, anchor="mm")
    return np.asarray(canvas)


def _write(images: list[np.ndarray], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        pics = [Image.fromarray(im) for im in images]
        pics[0].save(out, save_all=True, append_images=pics[1:], duration=int(1000 / 60 * GIF_STRIDE), loop=0)
    elif out.suffix == ".mp4":
        import imageio_ffmpeg                    # only needed for .mp4
        h, w = images[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(str(out), (w, h), fps=60, macro_block_size=1)
        writer.send(None)                        # start the generator
        for im in images:
            writer.send(np.ascontiguousarray(im))
        writer.close()
    else:
        raise SystemExit(f"unknown video format {out.suffix}: use .mp4 or .gif")


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description="Render simulator footage of a replay or of Auto.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--score", type=int, help="a replay in data/raw (its map and mods come from the manifest)")
    src.add_argument("--beatmap", type=int, help="play this map with Auto (data/raw/beatmaps/<id>.osu)")
    ap.add_argument("--auto", action="store_true", help="with --beatmap: Auto plays (the default)")
    ap.add_argument("--mods", default="NM", help="with --beatmap: NM, HR, HD, DT, HDHR, ...")
    ap.add_argument("--start", type=float, help="map time to start at, seconds")
    ap.add_argument("--duration", type=float, help="seconds of map time")
    ap.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--out", type=Path, help="default: eval/paths.py's naming rules")
    args = ap.parse_args()

    if args.score is not None:
        rows = {r["score_id"]: r for r in csv.DictReader((RAW / "manifest.csv").open(newline=""))}
        row = rows.get(str(args.score))
        if row is None:
            raise SystemExit(f"score {args.score} is not in {RAW / 'manifest.csv'}")
        beatmap_id, mods, player = int(row["beatmap_id"]), mods_from_str(row["mods"]), f"human-{args.score}"
        beatmap = parse_beatmap(RAW / row["osu_path"])
        pl = make_playable(beatmap, mods)
        frames = load_replay(RAW / row["osr_path"])
        frames.mods = mods
        title = f"score {args.score} ({row['mods']}, map {beatmap_id})"
    else:
        beatmap_id, mods, player = args.beatmap, mods_from_str(args.mods), "auto"
        beatmap = parse_beatmap(RAW / "beatmaps" / f"{args.beatmap}.osu")
        pl = make_playable(beatmap, mods)
        frames = auto_replay(pl)
        title = f"Auto ({args.mods}, map {beatmap_id})"

    start = args.start * 1000 if args.start is not None else None
    end = start + args.duration * 1000 if start is not None and args.duration is not None else None
    out = args.out or footage_path(player, beatmap_id, beatmap.title, mods, args.format,
                                   args.start, end / 1000 if end is not None else None)
    n = render_video(pl, frames, out, title, start, end)
    print(f"wrote {out}: 300 {n[300]}, 100 {n[100]}, 50 {n[50]}, miss {n[0]} (up to the clip's end)")


if __name__ == "__main__":
    main()
