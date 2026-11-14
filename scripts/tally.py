"""Replay availability from player top plays (the "player crawl"; see docs/data_collection.md). Run from the repo root."""
import collections
import os
import time

from dotenv import load_dotenv
from ossapi import Cursor, Ossapi, RankingType

load_dotenv()
api = Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"],
             os.environ["OSU_REDIRECT_URI"])


def mod_str(mods):                                       # lazer: list of mods; legacy: one Mod
    if isinstance(mods, list):                           # CL only marks a stable-era score; not a real mod
        return "".join(m.acronym for m in mods if m.acronym != "CL") or "NM"
    return str(mods) or "NM"


def has_replay(sc):                                      # legacy scores have no has_replay
    return bool(getattr(sc, "has_replay", None) or sc.replay)


# players ranked roughly 5k-50k: pages 100..1000 of the performance ranking (50 per page)
users = []
for page in (100, 300, 600, 1000):
    r = api.ranking("osu", RankingType.PERFORMANCE, cursor=Cursor(page=page))
    users += [u.user.id for u in r.ranking[:5]]
    time.sleep(1)

tally = collections.Counter()
clean = collections.Counter()                            # accuracy >= 0.98, survives the filter
for uid in users:
    for sc in api.user_scores(uid, "best", mode="osu", limit=100):
        if not has_replay(sc):
            continue
        if not (4.0 <= sc.beatmap.difficulty_rating <= 6.5):
            continue
        tally[mod_str(sc.mods)] += 1
        if sc.accuracy >= 0.98:
            clean[mod_str(sc.mods)] += 1
    time.sleep(1)

print(f"{'mods':>12}  {'all':>5}  {'>=98%':>5}")
for mods, n in tally.most_common():
    print(f"{mods:>12}  {n:>5}  {clean[mods]:>5}")
print("players:", len(users))
print("total with replay in 4-6.5 star band:", sum(tally.values()), "| >=98%:", sum(clean.values()))
calls = 4 + len(users)
print(f"clean NoMod replays: {clean['NM']} from {calls} calls ({clean['NM'] / calls:.1f} per call)")
