"""Training windows for behaviour cloning: the last W frames the policy would see, and what the player
did next. No torch needed: a torch DataLoader takes any object with __len__ / __getitem__.

    python -m data.dataset build --split train --mods NM HR    # render + cache the map layers once
    python -m data.dataset build --split test --mods NM DT --map-lists configs/splits/test
    python -m data.dataset stats --split train --mods NM HR    # windows per style / mod, sanity checks

One sample (see data/resample.py for the frame grid it lives on), for frame k of a replay:
  frames     (W, 96, 128) uint8   frames k-W+1 .. k: the map layer with the player's cursor drawn in;
             (2W, 96, 128)        with cursor_channel: the W map layers without the cursor, then the
                                  W cursors alone on black (the cursor in "its own colour")
  cursor     (2,) float32         cursor at frame k, osu! pixels
  action     (2,) float32         cursor at frame k+1 minus cursor at frame k (osu! pixels)
  target     (2,) float32         the player's cursor at frame k+1 (osu! pixels), for absolute aim
  press      (2,) float32         left / right pressed in frame k+1's window (0 or 1)
  held       (2,) float32         left / right down at the end of frame k+1's window
  prev       (4W,) float32        the player's own last W actions (prev_actions): per frame k-W+1 .. k,
                                  the cursor move into it and whether a key was pressed / held; read
                                  only by models with prev_action (models/heads.py)
  interval   ()   float32         map ms per frame (16.67 NoMod, 25 DT): for the conditioning experiments
  window_300 ()   float32         the 300 hit window (ms), likewise
  beatmap_id, mods (int)          for per-map / per-mod breakdowns

Frames come from a cache: the map layer (everything but the cursor) is the same for every replay of a
map under the same visual mods, so it is rendered once per (map, mods) and stored compressed (most of
each frame is black: ~10x smaller, and a frame decompresses in microseconds). The player's cursor is
drawn on per sample (osu_sim.renderer.add_cursor), so a cached frame plus cursor is byte-identical to
Renderer.frame. The cache lives in $OSU_FRAME_CACHE (default data/cache/frames, gitignored); keep it
outside OneDrive.

Circles disappear at their own time in the cached layer (the renderer's default), not when this
player hit them; closed-loop play uses the same convention, so training and play agree.

HR replays can be canonicalised (canonical_hr=True): frames flipped top-to-bottom and the cursor
mirrored (y -> 384 - y), so they look like NoMod geometry with HR's smaller circles and faster
approach. Use it for training only: the HR test condition is the flipped map.
"""
import argparse
import csv
import os
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from data.replay import load_replay
from data.resample import PLAYFIELD_HEIGHT, GridReplay, flip_vertical, map_grid, resample
from data.splits import SPLITS, expand
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import hit_windows
from osu_sim.mods import DT, EZ, HD, HR, HT, NC
from osu_sim.playable import Playable, make_playable
from osu_sim.renderer import RENDER_VERSION, H, W, Renderer, add_cursor

RAW = Path("data/raw")
MAPS = SPLITS                                  # the default map lists: every split's band files (data/splits.py)
WINDOW = 20
PLAYFIELD_WIDTH = 512
VISUAL_MODS = EZ | HD | HR | DT | HT          # mods that change the frames (DT/HT through the grid spacing)
MOD_BITS = {"HD": HD, "HR": HR, "DT": DT}
PREV_SCALE = 50.0                              # osu! px per unit of a previous move (models/heads.py MOVE_SCALE)


def cache_dir() -> Path:
    load_dotenv()                                # OSU_FRAME_CACHE may be set in .env
    return Path(os.path.expanduser(os.environ.get("OSU_FRAME_CACHE", "data/cache/frames")))


def visual_mods(mods: int) -> int:
    if mods & NC:
        mods |= DT
    return mods & VISUAL_MODS


def mods_from_str(s: str) -> int:
    """"NM" -> 0, "HDHR" -> HD | HR (the manifest's `mods` column)."""
    return 0 if s == "NM" else sum(MOD_BITS[s[i:i + 2]] for i in range(0, len(s), 2))


# ---------------------------------------------------------------- map-layer cache

class MapFrames:
    """The cached map layer of one (map, visual mods) on its frame grid."""

    def __init__(self, path: Path):
        self.times = np.load(path.with_suffix(".times.npy"))
        self.offsets = np.load(path.with_suffix(".idx.npy"))
        self.blob = np.memmap(path.with_suffix(".bin"), dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.times)

    def frame(self, k: int) -> np.ndarray:
        raw = zlib.decompress(self.blob[self.offsets[k]:self.offsets[k + 1]].tobytes())
        return np.frombuffer(raw, dtype=np.uint8).reshape(H, W)

    def window(self, k: int, size: int) -> np.ndarray:
        """Frames k-size+1 .. k as (size, H, W); frames before the grid start are blank."""
        out = np.zeros((size, H, W), dtype=np.uint8)
        for i, j in enumerate(range(k - size + 1, k + 1)):
            if j >= 0:
                out[i] = self.frame(j)
        return out


def map_frames_path(beatmap_id: int, mods: int, root: Path | None = None) -> Path:
    # keyed on the renderer version too, so frames drawn by an older renderer are never reused
    return (root or cache_dir()) / f"{beatmap_id}_{visual_mods(mods)}_r{RENDER_VERSION}"


def build_map_frames(pl: Playable, beatmap_id: int, root: Path | None = None) -> Path:
    """Render and cache the map layer of a playable map on its grid (skips if already cached)."""
    path = map_frames_path(beatmap_id, pl.mods, root)
    if path.with_suffix(".idx.npy").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    times = map_grid(pl)
    renderer = Renderer(pl)
    offsets = [0]
    tmp = path.with_suffix(".bin.part")
    with tmp.open("wb") as f:
        for t in times:
            blob = zlib.compress(renderer.map_layer(t).tobytes(), 1)
            f.write(blob)
            offsets.append(offsets[-1] + len(blob))
    tmp.replace(path.with_suffix(".bin"))
    np.save(path.with_suffix(".times.npy"), times)
    np.save(path.with_suffix(".idx.npy"), np.array(offsets, dtype=np.int64))   # written last: marks "done"
    return path


# ---------------------------------------------------------------- windows

class ReplayWindows:
    """Every usable frame of a set of replays, as training samples (see the module doc).

    items: (beatmap_id, playable, replay frames) per replay; the playable must carry the replay's mods.
    A frame k is usable if it has a full window behind it, a next frame, and lies between the first
    object appearing (time - preempt) and the last object ending.

    cursor_jitter (osu! pixels, 0 = off): in half the samples, the whole cursor history is shifted by
    a random offset of up to this much in x and y, and the target move becomes "back to where the
    player went next" (the absolute target, where the player went, stays the same). Plain BC only ever sees the player's own cursor, so a policy that drifts off
    it has never learned to come back; this shows it how (a cheap stand-in for DAgger).

    cursor_channel: give the cursor its own channels instead of drawing it into the map (see
    `observation`). In grayscale the cursor is one more bright blob among circles and slider balls;
    on its own channel the network cannot mistake it for an object.
    """

    def __init__(self, items, window: int = WINDOW, canonical_hr: bool = False, root: Path | None = None,
                 cursor_jitter: float = 0.0, cursor_channel: bool = False):
        self.window, self.canonical_hr, self.root = window, canonical_hr, root
        self.cursor_jitter, self.cursor_channel = cursor_jitter, cursor_channel
        self.replays: list[tuple[int, int, GridReplay, MapFrames, float]] = []
        idx_r, idx_k = [], []
        for beatmap_id, pl, frames in items:
            mf = MapFrames(build_map_frames(pl, beatmap_id, root))
            g = resample(frames, mf.times)
            if canonical_hr and pl.mods & HR:
                g = flip_vertical(g)
            w300 = float(hit_windows(pl.od).w300)
            first = pl.objects[0].time - pl.preempt
            last = max(o.end_time for o in pl.objects)
            ks = np.flatnonzero((mf.times >= first) & (mf.times <= last))
            ks = ks[(ks >= window - 1) & (ks < len(mf.times) - 1)]
            idx_r.append(np.full(len(ks), len(self.replays), dtype=np.int32))
            idx_k.append(ks.astype(np.int32))
            self.replays.append((beatmap_id, pl.mods, g, mf, w300))
        self.idx_r = np.concatenate(idx_r) if idx_r else np.zeros(0, np.int32)
        self.idx_k = np.concatenate(idx_k) if idx_k else np.zeros(0, np.int32)

    def __len__(self):
        return len(self.idx_r)

    def __getitem__(self, i: int) -> dict:
        beatmap_id, mods, g, mf, w300 = self.replays[self.idx_r[i]]
        k = int(self.idx_k[i])
        js = np.arange(k - self.window + 1, k + 1)
        layers = mf.window(k, self.window)
        if self.canonical_hr and mods & HR:
            layers = layers[:, ::-1, :]          # the cursor in g is already mirrored
        cursors = np.stack([g.x[js], g.y[js]], axis=1)
        if self.cursor_jitter:
            # a fresh generator seeded from the OS: DataLoader worker processes would otherwise share one
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                offset = rng.uniform(-self.cursor_jitter, self.cursor_jitter, 2)
                cursors = np.clip(cursors + offset, 0, [PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT])
        frames = observation(layers, cursors, self.cursor_channel)
        cursor = cursors[-1]
        return {
            "frames": frames,
            "prev": prev_actions(cursors, g.press[js], g.held[js]),
            "cursor": cursor.astype(np.float32),
            "action": np.array([g.x[k + 1] - cursor[0], g.y[k + 1] - cursor[1]], dtype=np.float32),
            "target": np.array([g.x[k + 1], g.y[k + 1]], dtype=np.float32),
            "press": np.array([g.press[k + 1] & 1, (g.press[k + 1] >> 1) & 1], dtype=np.float32),
            "held": np.array([g.held[k + 1] & 1, (g.held[k + 1] >> 1) & 1], dtype=np.float32),
            "interval": np.float32(g.t[1] - g.t[0]),
            "window_300": np.float32(w300),
            "beatmap_id": beatmap_id,
            "mods": mods,
        }


def observation(layers: np.ndarray, cursors: np.ndarray, cursor_channel: bool = False) -> np.ndarray:
    """What the policy sees, from W map layers (W, 96, 128) and W cursor positions (W, 2). Used by
    both training (ReplayWindows) and play (eval/policy.py), so the two cannot differ.

    cursor_channel=False: (W, 96, 128), the cursor drawn into each map layer (byte-identical to
    Renderer.frame). True: (2W, 96, 128), the map layers untouched, then the cursors on black."""
    if cursor_channel:
        return np.concatenate([layers, add_cursor(np.zeros_like(layers), cursors)])
    return add_cursor(layers, cursors)


def prev_actions(cursors: np.ndarray, press: np.ndarray, held: np.ndarray) -> np.ndarray:
    """What the player (or the policy, eval/policy.py) did in the last W frames, flattened to (4W,):
    per frame, the cursor move into it (in PREV_SCALE px; 0 for the first frame and next to an
    off-screen cursor, as before a map starts in closed-loop play) and "any key pressed" / "any key
    held" (0 or 1). The frames show the cursor only to within a few pixels and never show the keys."""
    move = np.zeros_like(cursors, dtype=np.float32)
    step = np.diff(cursors, axis=0)
    visible = (cursors[1:, 0] >= 0) & (cursors[:-1, 0] >= 0)
    move[1:] = np.where(visible[:, None], step / PREV_SCALE, 0.0)
    keys = np.stack([np.asarray(press) > 0, np.asarray(held) > 0], axis=1).astype(np.float32)
    return np.concatenate([move, keys], axis=1).reshape(-1)


def load_maps(map_lists: tuple[Path, ...] = (MAPS,)) -> dict[int, dict]:
    """beatmap_id -> its entry (style, split, stars, ...) over one or more map lists: band files of
    configs/splits (data/splits.py), directories of them, or older lists such as configs/legacy/maps/."""
    maps = {}
    for path in expand(map_lists):
        for m in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["maps"]:
            if m["beatmap_id"] in maps:
                raise ValueError(f"map {m['beatmap_id']} is in more than one map list")
            maps[m["beatmap_id"]] = m
    return maps


def from_split(split: str, mods: tuple[str, ...] = ("NM",), window: int = WINDOW, canonical_hr: bool = True,
               raw: Path = RAW, map_lists: tuple[Path, ...] = (MAPS,), root: Path | None = None,
               limit: int | None = None, cursor_jitter: float = 0.0, cursor_channel: bool = False) -> ReplayWindows:
    """Windows for every manifest replay whose map is in `split` of the map lists and whose mods are in `mods`."""
    maps = load_maps(map_lists)
    rows = [r for r in csv.DictReader((raw / "manifest.csv").open(newline=""))
            if int(r["beatmap_id"]) in maps and maps[int(r["beatmap_id"])]["split"] == split and r["mods"] in mods]
    items, playables = [], {}
    for r in rows[:limit]:
        bid, m = int(r["beatmap_id"]), mods_from_str(r["mods"])
        if (bid, m) not in playables:
            playables[(bid, m)] = make_playable(parse_beatmap(raw / r["osu_path"]), m)
        frames = load_replay(raw / r["osr_path"])
        frames.mods = m                          # the normalised mods (NC counts as DT, SD/PF ignored)
        items.append((bid, playables[(bid, m)], frames))
    return ReplayWindows(items, window=window, canonical_hr=canonical_hr, root=root,
                         cursor_jitter=cursor_jitter, cursor_channel=cursor_channel)


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description="Build the frame cache / summarise the BC windows.")
    ap.add_argument("action", choices=["build", "stats"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--mods", nargs="+", default=["NM"])
    ap.add_argument("--map-lists", nargs="+", type=Path, default=[MAPS], help="band files or directories (configs/splits)")
    args = ap.parse_args()
    print(f"frame cache: {cache_dir()}")
    ds = from_split(args.split, tuple(args.mods), map_lists=tuple(args.map_lists))   # builds missing map layers
    maps = load_maps(tuple(args.map_lists))
    per = Counter()
    for (bid, mods, g, mf, w300), n in zip(ds.replays, np.bincount(ds.idx_r, minlength=len(ds.replays))):
        per[(maps[bid]["style"], mods)] += int(n)
    size = sum(p.stat().st_size for p in cache_dir().glob("*.bin"))
    print(f"{len(ds.replays)} replays, {len(ds)} windows; cache {size / 1e6:.0f} MB")
    for (style, mods), n in sorted(per.items()):
        print(f"  {style:<8} mods {mods:>3}: {n} windows")
    if args.action == "stats" and len(ds):
        s = [ds[int(i)] for i in np.random.default_rng(0).integers(0, len(ds), 2000)]
        act = np.stack([x["action"] for x in s])
        print(f"  cursor step |dx,dy| (px): median {np.median(np.hypot(*act.T)):.1f}, 99th {np.percentile(np.hypot(*act.T), 99):.1f}")
        print(f"  press rate L/R: {np.mean([x['press'] for x in s], axis=0)}; held rate L/R: {np.mean([x['held'] for x in s], axis=0)}")


if __name__ == "__main__":
    main()
