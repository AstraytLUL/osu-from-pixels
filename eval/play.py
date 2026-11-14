"""Let a trained model (or Auto) play one map, closed-loop, and keep what it did: the judged result,
an .osr to watch in osu! (eval/osr.py) and, if asked, footage (eval/video.py).

    python -m eval.play --checkpoint runs/baseline_seed0/best.pt --beatmap 553131 --mods HR
    python -m eval.play --auto --beatmap 553131 --mods DT --footage mp4
    python -m eval.play --checkpoint ... --beatmap 555797 --osu data/challenge/555797.osu
    python -m eval.play --checkpoint ... --beatmap 553131 --motor free     # no motor limits (eval/motor.py)
    python -m eval.play --checkpoint ... --beatmap 553131 --motor cap      # the speed cap, no landing noise

Files are named by eval/paths.py, e.g. runs/replays/baseline_seed0-best/553131-<title>_HR.osr
and runs/footage/baseline_seed0-best/553131-<title>_HR.mp4. With --motor free the folder gets
a -free suffix (runs/replays/baseline_seed0-best-free/...), with --motor cap a -cap one. Replays written before the motor
limits existed (2026-10-28) are free ones in the unsuffixed folders.
"""
import argparse
from pathlib import Path

from data.dataset import RAW, mods_from_str
from eval.motor import MOTORS
from eval.osr import PLAYER_AUTO, PLAYER_MODEL, save_osr
from eval.paths import footage_path, model_player, replay_path
from eval.policy import AutoPolicy, load_model_policy
from eval.rollout import play
from eval.video import render_video
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import accuracy, counts
from osu_sim.playable import make_playable


def main():
    ap = argparse.ArgumentParser(description="Play one map with a model or Auto; write .osr and footage.")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--checkpoint", help="a .pt written by train/bc.py")
    who.add_argument("--auto", action="store_true", help="Auto, through the same rollout")
    ap.add_argument("--beatmap", type=int, required=True, help="data/raw/beatmaps/<id>.osu, unless --osu is given")
    ap.add_argument("--osu", type=Path, help="the map's .osu file, if not in data/raw (e.g. data/challenge/<id>.osu)")
    ap.add_argument("--mods", default="NM", help="NM, HR, HD, DT, HDHR, ...")
    ap.add_argument("--footage", choices=["mp4", "gif"], help="also render footage of the whole map")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--motor", choices=["free", "cap", "human"], default="cap",
                    help="eval/motor.py's HUMAN, CAP (its speed cap, no landing noise) or FREE")
    ap.add_argument("--seed", type=int, default=0, help="the motor noise's seed")
    args = ap.parse_args()

    osu = args.osu or RAW / "beatmaps" / f"{args.beatmap}.osu"
    beatmap = parse_beatmap(osu)
    mods = mods_from_str(args.mods)
    pl = make_playable(beatmap, mods)
    if args.auto:
        policy, name_in_game, player = AutoPolicy(), PLAYER_AUTO, "auto"
    else:
        policy, name_in_game, player = load_model_policy(args.checkpoint, args.device), PLAYER_MODEL, model_player(args.checkpoint)

    if args.motor != "human":
        player += "-" + args.motor
    g, js = play(policy, args.beatmap, pl, motor=MOTORS[args.motor], seed=args.seed)
    frames = g.as_frames()
    osr = replay_path(player, args.beatmap, beatmap.title, mods)
    h = save_osr(frames, osr, osu, name_in_game)
    written = [osr]
    if args.footage:
        video = footage_path(player, args.beatmap, beatmap.title, mods, args.footage)
        render_video(pl, frames, video, f"{name_in_game} ({args.mods}, map {args.beatmap})")
        written.append(video)
    n = counts(js)
    print(f"{name_in_game} on {args.beatmap} {args.mods}: accuracy {accuracy(js):.2%} "
          f"(300 {n[300]}, 100 {n[100]}, 50 {n[50]}, miss {n[0]}), max combo {h['max_combo']}, "
          f"score {h['score']:,}; wrote " + ", ".join(map(str, written)))


if __name__ == "__main__":
    main()
