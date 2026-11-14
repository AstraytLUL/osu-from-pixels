"""Replay availability from mod-filtered map leaderboards (supporter token; see docs/data_collection.md). Run from the repo root."""
import collections
import os
import time

import requests
from dotenv import load_dotenv
from ossapi import Ossapi

load_dotenv()
api = Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"],
             os.environ["OSU_REDIRECT_URI"])

def has_replay(sc):                                      # legacy scores have no has_replay
    return bool(getattr(sc, "has_replay", None) or sc.replay)


def beatmap_scores(bid, mods, lb_type, tries=5):
    # Filtered leaderboards hit Cloudflare's rate limit after a handful of quick calls: it answers
    # 429 with an HTML page, which ossapi fails to parse as JSON. Back off and retry.
    for attempt in range(1, tries + 1):
        try:
            # global is the default board; ossapi only accepts RankingType values for type
            return api.beatmap_scores(beatmap_id=bid, mods=mods,
                                      type=None if lb_type == "global" else lb_type)
        except requests.exceptions.JSONDecodeError:
            print(f"  rate-limited on {bid} {mods} {lb_type}; waiting {60 * attempt}s")
            time.sleep(60 * attempt)
    raise RuntimeError("still rate-limited; wait a few minutes and run again")


MAPS = [814293, 816600, 2935781]          # replace with three 5-star maps you know
tally = collections.Counter()
calls = 0
for bid in MAPS:
    for mods in ("NM", "HD", "HR", "HDHR"):
        for lb_type in ("global", "country"):
            r = beatmap_scores(bid, mods, lb_type)
            calls += 1
            n = sum(1 for sc in r.scores if has_replay(sc) and sc.accuracy >= 0.98)
            tally[(bid, mods, lb_type)] += n
            print(bid, mods, lb_type, n)
            time.sleep(4)
for k, n in sorted(tally.items()):
    print(k, n)
nm = sum(n for (_, mods, _), n in tally.items() if mods == "NM")
print(f"clean NoMod replays: {nm} from {calls} calls ({nm / calls:.1f} per call)")
