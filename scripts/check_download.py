"""Fetch a judge test fixture (one replay of beatmap 1262832, filtered by MODS, plus the .osu) and print the era check. Run from the repo root: python scripts/check_download.py"""
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from ossapi import Ossapi
from osrparse import Replay

load_dotenv()
api = Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"],
             os.environ["OSU_REDIRECT_URI"])          # browser opens once; token is cached after

BEATMAP_ID = 1262832                                     # any ranked osu!std map; swap for a 5-star one
MODS = "NM"                                              # leaderboard mod filter (supporter only); None = any mods


# The API can return a score in either of two shapes: lazer (Score: list of mods,
# great/ok/meh/miss, has_replay) or legacy (_LegacyScore: one Mod bitfield,
# count_300/100/50/miss, a `mode` attribute, downloaded through a per-mode endpoint).
def is_legacy(sc):
    return hasattr(sc, "mode")


def mod_str(mods):
    if isinstance(mods, list):                           # CL only marks a stable-era score; not a real mod
        return "".join(m.acronym for m in mods if m.acronym != "CL") or "NM"
    return str(mods) or "NM"


def has_replay(sc):
    return bool(getattr(sc, "has_replay", None) or sc.replay)


def counts(sc):
    st = sc.statistics
    if is_legacy(sc):
        return st.count_300, st.count_100, st.count_50, st.count_miss
    return st.great, st.ok, st.meh, st.miss


# 3a. one authenticated download
bs = api.beatmap_scores(beatmap_id=BEATMAP_ID, mods=MODS)
s = next(sc for sc in bs.scores if has_replay(sc))
n300, n100, n50, nmiss = (c or 0 for c in counts(s))
print("score", s.id, "legacy" if is_legacy(s) else "lazer", "mods", mod_str(s.mods),
      "acc", s.accuracy, "300/100/50/miss", n300, n100, n50, nmiss)

if is_legacy(s):
    raw = api.download_score_mode(s.mode, s.id, raw=True)
else:
    raw = api.download_score(s.id, raw=True)             # raw=True gives the .osr bytes
Path("tests/fixtures/check.osr").write_bytes(raw)

r = Replay.from_path("tests/fixtures/check.osr")
print("parsed mods", r.mods, "misses", r.count_miss, "frames", len(r.replay_data))
print(r.replay_data[:3])

# 3b. judge test fixture: replay + beatmap + expected numbers
fx = Path(f"tests/fixtures/{BEATMAP_ID}_{s.id}")
fx.mkdir(parents=True, exist_ok=True)
(fx / "replay.osr").write_bytes(raw)
with urllib.request.urlopen(f"https://osu.ppy.sh/osu/{BEATMAP_ID}") as resp:
    (fx / "beatmap.osu").write_bytes(resp.read())
expected = {
    "beatmap_id": BEATMAP_ID, "score_id": s.id, "mods": mod_str(s.mods),
    "accuracy": s.accuracy,
    "count_300": n300, "count_100": n100, "count_50": n50, "count_miss": nmiss,
    "api_format": "legacy" if is_legacy(s) else "lazer",
}
(fx / "expected.json").write_text(json.dumps(expected, indent=2))
print("fixture written to", fx)

# 3c. era check: a legacy-shaped score, or a lazer one with a legacy_score_id, is stable-era
legacy_id = s.id if is_legacy(s) else s.legacy_score_id
print("legacy_score_id", legacy_id, "->", "stable" if legacy_id else "lazer")
print("mods agree?", mod_str(s.mods), "(API) vs", r.mods, "(file)")
