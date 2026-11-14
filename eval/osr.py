"""Write a trajectory (our Auto, or the BC model) as an .osr replay to watch in osu!.

    python -m eval.osr --beatmap 553131 --mods HR       # -> runs/replays/auto/553131-<title>_HR.osr (eval/paths.py)

osu! finds the map by the MD5 of its .osu file, so the map must be installed in your client in the
same version as data/raw/beatmaps/<id>.osu (it is the current ranked version, so normally it is).
Open the .osr with osu! (double-click, or drag it onto the client) to watch it.

These replays are for watching locally only: never submit them or play them against online servers.
The player name says who played: "SC4001 Auto" (osu_sim/auto.py, not osu!'s own Autoplay) or
"SC4001 BC" (the model).

Format (osrparse writes the container): frames store time deltas in whole ms, in map time (as real
replays do, also under DT); keys use stable's bits, left = M1|K1 and right = M2|K2. Like stable, two
"no cursor" frames come first. The header's counts, max combo and score come from our judge and
osu_sim/score.py (score exact on maps without spinners, within 0.1% with them).

A press never moves the cursor in the same frame: when a frame presses a key and also moves the
cursor, the move is written as its own frame MOVE_LEAD_MS earlier (with the keys as they were): far
enough ahead that osu! (which updates every ~1 ms at 960 fps) applies the move in an earlier update. osu!stable's
playback judges a click from where the cursor was before that frame's move: the model often
arrives in the circle on the very frame it clicks (20-35 px per frame in jumps), and stable
counted those clicks as misses (Highscore's jumps, 2:12-2:24), while our judge, which uses the
press frame's position as stable does for players' own replays, gave 300s. Humans' replays rarely
have a click and a big move in one frame, so this costs nothing there.

Every replay gets No Fail (NF) added to its mods: osu! replays its health bar during playback and
stops a replay that runs out of health, which a model's play often would (our judge has no health
bar). NF changes nothing else but the score: its 0.5x multiplier halves each hit's combo bonus
(osu_sim/score.py), and the header includes that.
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

from data.dataset import RAW, mods_from_str
from data.replay import M1, M2, ReplayFrames
from eval.paths import replay_path
from osu_sim.auto import auto_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import counts, judge
from osu_sim.playable import make_playable
from osu_sim.score import NF, max_combo, score_v1

PLAYER_AUTO = "SC4001 Auto"
PLAYER_MODEL = "SC4001 BC"
GAME_VERSION = 20240101        # any recent stable build number
K1, K2 = 4, 8
MOVE_LEAD_MS = 8               # a press's cursor move is written this much earlier (half a 60 fps frame)


def save_osr(frames: ReplayFrames, path: Path, osu_path: Path, player: str, no_fail: bool = True) -> dict:
    """Write `frames` (map time, cursor, M1/M2 key bits, mods) played on the map at `osu_path` as an
    osu!standard replay, with NF added unless no_fail=False. Returns the header's numbers (counts,
    max combo, score)."""
    beatmap = parse_beatmap(osu_path)
    mods = frames.mods | NF if no_fail else frames.mods
    pl = make_playable(beatmap, mods)                            # NF doesn't change the map, only the score
    js = judge(pl, frames)
    n = counts(js)
    combo = max_combo(js, pl)
    full_combo = sum(1 + len(j.parts) for j in js)               # every circle, slider part and spinner
    header = {"count_300": n[300], "count_100": n[100], "count_50": n[50], "count_miss": n[0],
              "max_combo": combo, "score": score_v1(js, pl, beatmap)}

    events = [ReplayEventOsu(0, 256.0, -500.0, Key(0)),          # stable's leading "no cursor" frames
              ReplayEventOsu(-1, 256.0, -500.0, Key(0))]
    clock = -1                                                   # the time those two frames add up to
    last_x, last_y, last_bits = 256.0, -500.0, 0
    for t, x, y, keys in zip(frames.t, frames.x, frames.y, frames.keys):
        ms = int(round(t))
        bits = (M1 | K1 if keys & M1 else 0) | (M2 | K2 if keys & M2 else 0)
        pressed = bits & ~last_bits & (M1 | M2)
        if pressed and (x, y) != (last_x, last_y):
            # move first, then press (see the module doc): the cursor reaches its spot a little early,
            # but never before the previous frame
            move_ms = max(ms - MOVE_LEAD_MS, clock + 1)
            if move_ms < ms:
                events.append(ReplayEventOsu(move_ms - clock, float(x), float(y), Key(last_bits)))
                clock = move_ms
        events.append(ReplayEventOsu(ms - clock, float(x), float(y), Key(bits)))
        clock, last_x, last_y, last_bits = ms, x, y, bits
    replay = Replay(
        mode=GameMode.STD, game_version=GAME_VERSION, beatmap_hash=hashlib.md5(Path(osu_path).read_bytes()).hexdigest(),
        username=player, replay_hash="", count_geki=0, count_katu=0, **header, perfect=combo == full_combo,
        mods=Mod(int(mods)), life_bar_graph=None, timestamp=datetime.now(timezone.utc),
        replay_data=events, replay_id=0, rng_seed=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    replay.write_path(str(path))
    return header


def main():
    ap = argparse.ArgumentParser(description="Write our Auto's play of a map as an .osr replay.")
    ap.add_argument("--beatmap", type=int, required=True, help="data/raw/beatmaps/<id>.osu")
    ap.add_argument("--mods", default="NM", help="NM, HR, HD, DT, HDHR, ...")
    ap.add_argument("--osu", type=Path, help="the map's .osu file, if not in data/raw (e.g. data/challenge/<id>.osu)")
    ap.add_argument("--out", type=Path, help="default: eval/paths.py's naming rules")
    args = ap.parse_args()

    osu = args.osu or RAW / "beatmaps" / f"{args.beatmap}.osu"
    beatmap = parse_beatmap(osu)
    mods = mods_from_str(args.mods)
    frames = auto_replay(make_playable(beatmap, mods))
    out = args.out or replay_path("auto", args.beatmap, beatmap.title, mods)
    h = save_osr(frames, out, osu, PLAYER_AUTO)
    print(f"wrote {out}: 300 {h['count_300']}, 100 {h['count_100']}, 50 {h['count_50']}, miss {h['count_miss']}, "
          f"max combo {h['max_combo']}, score {h['score']:,}")


if __name__ == "__main__":
    main()
