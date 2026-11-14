"""Closed-loop play: a policy (eval/policy.py) plays a whole map from its own frames, and the judge
scores the result.

Each step the policy's own cursor is drawn into the next frame, so its mistakes stay on screen and it
has to recover from them; that is what makes this an evaluation of playing, not of predicting a
player's next move. Frames come from the same map-layer cache as training (data/dataset.py), so the
policy sees exactly what it was trained on (circles disappear at their own time).

A Motor (eval/motor.py) carries out each chosen cursor position: a top speed and landing noise, the
same for every policy. The default is CAP (the speed cap, no landing noise); FREE puts the cursor exactly where the policy says.

    from eval.rollout import split_maps, evaluate
    report = evaluate(AutoPolicy(), split_maps("val"))       # Auto: ~100% on every map
"""
from collections import deque
from pathlib import Path

import numpy as np

from data.dataset import MAPS, RAW, MapFrames, build_map_frames, load_maps, mods_from_str
from data.resample import GridReplay
from eval.motor import CAP, Motor
from eval.policy import AutoPolicy, Policy
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import Judgement, accuracy, counts, judge, unstable_rate
from osu_sim.playable import Playable, make_playable
from osu_sim.score import combo_breaks
from osu_sim.renderer import H, W

START = (256.0, 192.0)         # cursor position at the start of a map: the playfield centre
OFF_SCREEN = (-1000.0, -1000.0)


def rollout(policy: Policy, pl: Playable, layers: MapFrames, motor: Motor = CAP, seed: int = 0) -> GridReplay:
    """Play `pl` with `policy` over the map's whole frame grid; returns what it did (the cursor where
    `motor` put it). `seed` drives the motor's noise."""
    times = layers.times
    n = len(times)
    x, y = np.full(n, START[0]), np.full(n, START[1])
    press, held = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
    policy.reset(pl, times)
    rng = np.random.default_rng(seed)
    # the last `window` map layers and cursor positions; frames before the grid starts are fully
    # blank: no map, and the cursor far off screen, where nothing of it is drawn
    maps = deque([np.zeros((H, W), dtype=np.uint8)] * policy.window, maxlen=policy.window)
    cursors = deque([OFF_SCREEN] * policy.window, maxlen=policy.window)
    for k in range(n - 1):
        maps.append(layers.frame(k))
        cursors.append((x[k], y[k]))
        tx, ty, press[k + 1], held[k + 1] = policy.act(k, np.stack(maps), np.array(cursors))
        x[k + 1], y[k + 1] = (tx, ty) if motor.free else motor.execute(x[k], y[k], tx, ty, rng)
    return GridReplay(t=times, x=x, y=y, press=press, held=held, mods=pl.mods)


def play(policy: Policy, beatmap_id: int, pl: Playable, root: Path | None = None, motor: Motor = CAP,
         seed: int = 0) -> tuple[GridReplay, list[Judgement]]:
    """Roll out one map and judge it."""
    layers = MapFrames(build_map_frames(pl, beatmap_id, root))
    g = rollout(policy, pl, layers, motor, seed)
    return g, judge(pl, g.as_frames())


def split_maps(split: str, mods: str = "NM", raw: Path = RAW,
               map_lists: tuple[Path, ...] = (MAPS,)) -> list[tuple[int, str, Playable]]:
    """(beatmap_id, style, playable) for every map of a split in the map lists (by default every
    band file of configs/splits, data/splits.py), under `mods`."""
    m = mods_from_str(mods)
    return [(bid, e["style"], make_playable(parse_beatmap(raw / "beatmaps" / f"{bid}.osu"), m))
            for bid, e in load_maps(map_lists).items() if e["split"] == split]


def result_row(beatmap_id: int, style: str, pl: Playable, js: list[Judgement]) -> dict:
    """One play's results (a policy's, or a human replay judged the same way, eval/headline.py)."""
    n = counts(js)
    b = combo_breaks(js)
    return {"beatmap_id": beatmap_id, "style": style, "accuracy": accuracy(js),
            "count_300": n[300], "count_100": n[100], "count_50": n[50], "count_miss": n[0],
            "unstable_rate": unstable_rate(js) if any(j.offset is not None for j in js) else None,
            "objects": len(js),
            "breaks_miss": b["miss"], "breaks_slider": b["slider_break"], "end_drops": b["end_drop"]}


def evaluate(policy: Policy, maps: list[tuple[int, str, Playable]], auto_gap: bool = True,
             root: Path | None = None, motor: Motor = CAP) -> dict:
    """Closed-loop results per map, and their mean (every map weighs the same): accuracy, counts,
    unstable rate and combo breaks by cause (osu_sim/score.py combo_breaks; per 100 objects in the
    mean). The map is always played to the end, as with No Fail.

    auto_gap: also play each map with Auto through the same rollout, and report how far below it the
    policy is (percentage points). Auto is ~100%, so this mostly checks the rollout itself; under a
    motor it also shows what the limits alone cost. The motor's noise is seeded by beatmap id."""
    rows = []
    for beatmap_id, style, pl in maps:
        _, js = play(policy, beatmap_id, pl, root, motor, beatmap_id)
        row = result_row(beatmap_id, style, pl, js)
        if auto_gap:
            _, auto_js = play(AutoPolicy(), beatmap_id, pl, root, motor, beatmap_id)
            row["auto_accuracy"] = accuracy(auto_js)
            row["gap_pp"] = 100 * (row["auto_accuracy"] - row["accuracy"])
        rows.append(row)
    return {"accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "miss_rate": float(np.mean([r["count_miss"] / (r["count_300"] + r["count_100"] + r["count_50"] + r["count_miss"])
                                        for r in rows])),
            "breaks_per_100": float(np.mean([100 * (r["breaks_miss"] + r["breaks_slider"]) / r["objects"] for r in rows])),
            "end_drops_per_100": float(np.mean([100 * r["end_drops"] / r["objects"] for r in rows])),
            "maps": rows}
