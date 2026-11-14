"""Where replays and footage are saved, and what they are called. Every command that writes them
(eval/osr.py, eval/play.py, eval/video.py) takes its default path from here.

    runs/
      <run name>/                                   a training run (train/bc.py): checkpoints, metrics
      replays/<player>/<map>_<MODS>.osr             replays to watch in osu!
      footage/<player>/<map>_<MODS>.mp4             simulator footage of a whole map
      footage/<player>/<map>_<MODS>_<a>-<b>s.gif    ... or of a clip, from a s to b s of map time

<player>  who played:
            auto                    our scripted Auto (osu_sim/auto.py)
            <run>-<checkpoint>      a trained model, e.g. bc_baseline_seed0-best (runs/bc_baseline_seed0/best.pt)
            human-<score_id>        a player's replay from data/raw
<map>     <beatmap id>-<title slug>, e.g. 4912305-tetoris: the id keeps names unique, the slug
          makes them readable
<MODS>    NM, or the mods in a fixed order (EZ HD HR DT HT), e.g. HDHR, HDDT

runs/ is gitignored: nothing here is committed.
"""
import re
from pathlib import Path

RUNS = Path("runs")
MOD_ORDER = (("EZ", 2), ("HD", 8), ("HR", 16), ("DT", 64), ("HT", 256))
NC = 512


def mods_name(mods: int) -> str:
    """64 -> "DT", 8 | 16 -> "HDHR", 0 -> "NM" (NC counts as DT)."""
    if mods & NC:
        mods |= 64
    name = "".join(n for n, bit in MOD_ORDER if mods & bit)
    return name or "NM"


def map_name(beatmap_id: int, title: str = "") -> str:
    """4912305, "Tetoris" -> "4912305-tetoris" (just the id if there is no title)."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:30].rstrip("-")
    return f"{beatmap_id}-{slug}" if slug else str(beatmap_id)


def model_player(checkpoint: str | Path) -> str:
    """runs/bc_baseline_seed0/best.pt -> "bc_baseline_seed0-best"."""
    p = Path(checkpoint)
    return f"{p.parent.name}-{p.stem}"


def replay_path(player: str, beatmap_id: int, title: str, mods: int) -> Path:
    return RUNS / "replays" / player / f"{map_name(beatmap_id, title)}_{mods_name(mods)}.osr"


def footage_path(player: str, beatmap_id: int, title: str, mods: int, ext: str = "mp4",
                 start_s: float | None = None, end_s: float | None = None) -> Path:
    clip = ""
    if start_s is not None or end_s is not None:
        clip = f"_{_seconds(start_s or 0)}-{_seconds(end_s) if end_s is not None else 'end'}s"
    return RUNS / "footage" / player / f"{map_name(beatmap_id, title)}_{mods_name(mods)}{clip}.{ext}"


def _seconds(s: float) -> str:
    return f"{s:g}"                            # 20.0 -> "20", 12.5 -> "12.5"
