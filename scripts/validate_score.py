"""Our max combo and ScoreV1 vs the .osr header (what osu!stable computed), on stable replays whose
300/100/50/miss counts our judge reproduces exactly."""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from osrparse import Replay

from data.dataset import mods_from_str
from data.replay import load_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import counts, judge
from osu_sim.playable import make_playable
from osu_sim.score import max_combo, score_v1

RAW = Path("data/raw")
rows = [r for r in csv.DictReader((RAW / "manifest.csv").open()) if r["era"] == "stable"]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
stats = Counter()
bad = []
for r in rows[:limit]:
    header = Replay.from_path(str(RAW / r["osr_path"]))
    mods = mods_from_str(r["mods"])
    bm = parse_beatmap(RAW / r["osu_path"])
    pl = make_playable(bm, mods)
    frames = load_replay(RAW / r["osr_path"])
    frames.mods = mods
    js = judge(pl, frames)
    n = counts(js)
    if (n[300], n[100], n[50], n[0]) != (header.count_300, header.count_100, header.count_50, header.count_miss):
        stats["judge differs (skipped)"] += 1
        continue
    stats["compared"] += 1
    c, s = max_combo(js, pl), score_v1(js, pl, bm)
    stats["combo exact"] += c == header.max_combo
    stats["score exact"] += s == header.score
    stats["score within 0.1%"] += abs(s - header.score) <= 0.001 * header.score
    if c != header.max_combo or s != header.score:
        spinners = sum(j.kind == "spinner" for j in js)
        stats[f"inexact, spinners={min(spinners, 1)}, mods={r['mods']}"] += 1
        if spinners == 0:
            bad.append((r["score_id"], r["beatmap_id"], r["mods"], c, header.max_combo, s, header.score))
print(dict(stats))
for b in bad[:15]:
    print("score_id %s map %s %s: combo ours %d osu %d, score ours %d osu %d (diff %d)" % (*b, b[5] - b[6]))
