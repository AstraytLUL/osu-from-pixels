"""Validate the judge against osu!'s own counts on real replays with mistakes. Run from the repo root:

    python scripts/validate_judge.py [max replays per map]

Downloads (and caches, under data/raw/judge_validation, gitignored) .osu + .osr for stable-era scores with
100s/50s/misses, judges them, prints ours vs osu!'s counts.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import requests
from dotenv import load_dotenv
from ossapi import Ossapi

sys.path.insert(0, os.getcwd())
from data.replay import load_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.judge import counts, judge
from osu_sim.playable import make_playable

load_dotenv()
work = Path("data/raw/judge_validation")
work.mkdir(parents=True, exist_ok=True)
api = Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"], os.environ["OSU_REDIRECT_URI"])
OK_MODS = {"NM", "HD", "HR", "DT", "NC", "EZ", "HT", "NF", "SD", "PF", "FL", "CL", "TD"}
MAX_PER_MAP = int(sys.argv[1]) if len(sys.argv) > 1 else 3

manifest_f = work / "manifest.json"
manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {}


def retry(fn, *a, **k):
    for i in range(5):
        try:
            return fn(*a, **k)
        except requests.exceptions.JSONDecodeError:
            time.sleep(60 * (i + 1))
    raise RuntimeError("rate limited")


for bid in [814293, 816600, 2935781, 1860433, 658127, 2097898, 129891, 22423]:
    osu = work / f"{bid}.osu"
    if not osu.exists():
        with urllib.request.urlopen(f"https://osu.ppy.sh/osu/{bid}") as r:
            osu.write_bytes(r.read())
    if sum(1 for v in manifest.values() if v["beatmap_id"] == bid) < MAX_PER_MAP:
        r = retry(api.beatmap_scores, bid)
        time.sleep(4)
        picked = 0
        for sc in r.scores:
            acr = [m.acronym for m in sc.mods]
            st = sc.statistics
            imperfect = (st.ok or 0) + (st.meh or 0) + (st.miss or 0) > 0
            if not (sc.has_replay or sc.replay) or "CL" not in acr or not set(acr) <= OK_MODS or not imperfect:
                continue
            if str(sc.id) in manifest:
                picked += 1
                continue
            if picked >= MAX_PER_MAP:
                break
            raw = retry(api.download_score, sc.id, raw=True)
            (work / f"{sc.id}.osr").write_bytes(raw)
            manifest[str(sc.id)] = {"beatmap_id": bid, "mods": acr,
                                    "api": [st.great or 0, st.ok or 0, st.meh or 0, st.miss or 0]}
            manifest_f.write_text(json.dumps(manifest, indent=1))
            picked += 1
            time.sleep(7)

ok = 0
for sid, m in manifest.items():
    rp = load_replay(work / f"{sid}.osr")
    pl = make_playable(parse_beatmap(work / f"{m['beatmap_id']}.osu"), rp.mods)
    c = counts(judge(pl, rp))
    ours = [c[300], c[100], c[50], c[0]]
    same = ours == m["api"]
    ok += same
    print(f"{m['beatmap_id']:>8} {sid:>11} {'+'.join(m['mods']):>12}  osu! {m['api']}  ours {ours}  {'OK' if same else 'DIFF'}")
print(f"{ok}/{len(manifest)} match")
