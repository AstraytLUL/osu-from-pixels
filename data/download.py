"""Collect replays (.osr) and beatmaps (.osu) from per-map global leaderboards into data/raw/.

    python -m data.download configs/maps.txt             # safe to Ctrl+C and run again
    python -m data.download configs/maps.txt --dry-run   # leaderboard calls only, no downloads
    python -m data.download --from-manifest other/manifest.csv   # exactly those scores, no leaderboards

For each map and each mod set in configs/download.yaml: read the global leaderboard, keep clean
scores (reject_reason) up to the per-map and per-player caps (select), download each replay, check
that it parses, and append a row to data/raw/manifest.csv. Replays that fail go to
data/raw/failures.csv and are skipped on later runs; delete a row there to retry it.
"""
import argparse
import csv
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from ossapi import Ossapi

from data.replay import load_replay
from osu_sim.beatmap import parse_beatmap

RAW = Path("data/raw")
MANIFEST = RAW / "manifest.csv"
FAILURES = RAW / "failures.csv"
MANIFEST_COLUMNS = [
    "score_id", "beatmap_id", "user_id", "mods", "mods_raw", "legacy_mods", "accuracy",
    "count_300", "count_100", "count_50", "count_miss", "era", "legacy_score_id",
    "style", "split",                  # filled by later steps (data/select_maps.py, the map split)
    "osr_path", "osu_path", "downloaded_at",
]
FAILURE_COLUMNS = ["score_id", "beatmap_id", "mods", "reason", "failed_at"]


# The API returns scores in two shapes (see scripts/check_download.py). Lazer: `mods` is a list of
# NonLegacyMod, statistics are great/ok/meh/miss, replay flag `has_replay`. Legacy: `mods` is one
# Mod bitfield, statistics are count_300/100/50/miss, replay flag `replay`, and a `mode` attribute.
def is_legacy(sc):
    return hasattr(sc, "mode")


# Mod normalisation. An allow-list: a score is kept only if every mod is studied, ignored or an alias.
STUDIED = ("HD", "HR", "DT")                 # canonical order (legacy bit order) for the `mods` column
IGNORED = {"SD", "PF", "NF", "CL"}           # don't change the replay; CL only marks a stable-era score
ALIASES = {"NC": "DT"}                       # nightcore is DT with a pitch change


def raw_mods(sc):
    """Acronyms as the API gives them (CL included). A legacy NC decomposes to [DT, NC], PF to [SD, PF]."""
    if is_legacy(sc):
        return [str(m) for m in sc.mods.decompose()]
    return [m.acronym for m in sc.mods]


def rejected_mods(acronyms):
    """Mods we don't accept: TD, FL, EZ, HT, SO, RX, AP, and anything lazer-only."""
    return [a for a in acronyms if a not in IGNORED and ALIASES.get(a, a) not in STUDIED]


def normalise(acronyms):
    """Accepted acronyms -> canonical string of studied mods: [HR, CL, HD] -> "HDHR", [NC, PF] -> "DT"."""
    kept = {ALIASES.get(a, a) for a in acronyms}
    return "".join(m for m in STUDIED if m in kept) or "NM"


def mod_str(sc):
    return normalise(raw_mods(sc))


def mod_key(mods):
    """"HDHR" -> {"HD", "HR"}, "NM" -> {}: compares mod strings regardless of acronym order."""
    return frozenset() if mods == "NM" else frozenset(mods[i:i + 2] for i in range(0, len(mods), 2))


def has_replay(sc):
    return bool(getattr(sc, "has_replay", None) or sc.replay)


def counts(sc):
    """(300, 100, 50, miss) in either shape."""
    st = sc.statistics
    if is_legacy(sc):
        c = (st.count_300, st.count_100, st.count_50, st.count_miss)
    else:
        c = (st.great, st.ok, st.meh, st.miss)
    return tuple(n or 0 for n in c)            # absent judgements come back as None


def legacy_score_id(sc):
    return sc.id if is_legacy(sc) else sc.legacy_score_id


def era(sc):
    return "stable" if legacy_score_id(sc) else "lazer"


def reject_reason(sc, mod_set, accuracy_min):
    """Why a leaderboard score is not kept, or None if it is. Reasons are short so they tally."""
    if (sc.mode_int if is_legacy(sc) else sc.ruleset_id) != 0:
        return "not osu!standard"
    if not has_replay(sc):
        return "no replay"
    bad = rejected_mods(raw_mods(sc))
    if bad:
        return f"mod {'+'.join(bad)}"
    if mod_key(mod_str(sc)) != mod_key(mod_set):
        return f"mods {mod_str(sc)}"
    # lazer lets players customise a mod (DT at 1.3x, HD without fade-out); that isn't the mod we
    # study. Settings on ignored mods (e.g. SD's restart-on-fail) don't change the replay.
    if not is_legacy(sc) and any(m.settings for m in sc.mods if ALIASES.get(m.acronym, m.acronym) in STUDIED):
        return "custom mod settings"
    if sc.accuracy < accuracy_min:
        return "low accuracy"
    return None


def cap_key(mods, cap_groups):
    """The per-player cap a mod set counts against: its group in cap_groups (e.g. NM + HR), or its own."""
    key = mod_key(mods)
    for group in cap_groups:
        if key in {mod_key(m) for m in group}:
            return frozenset(mod_key(m) for m in group)
    return frozenset([key])


def select(scores, mod_set, cfg, seen, per_player, have):
    """Split one leaderboard into kept scores and (score, reason) skips, in leaderboard order.

    seen: score ids already in the manifest or failures file. per_player: Counter of
    (user_id, cap_key) over the manifest and all maps so far, updated here as scores are kept.
    have: replays this map and mod set already has; keeping stops at cfg["per_map"].
    """
    kept, skipped = [], []
    for sc in scores:
        reason = "already seen" if sc.id in seen else reject_reason(sc, mod_set, cfg["accuracy_min"])
        player = (sc.user_id, cap_key(mod_set, cfg["cap_groups"]))
        if reason is None and per_player[player] >= cfg["per_player"]:
            reason = "player cap"
        if reason is None and have + len(kept) >= cfg["per_map"]:
            reason = "map full"
        if reason:
            skipped.append((sc, reason))
        else:
            per_player[player] += 1
            kept.append(sc)
    return kept, skipped


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest_row(sc, beatmap_id, legacy_mods, osr_path, osu_path):
    n300, n100, n50, nmiss = counts(sc)
    return {
        "score_id": sc.id, "beatmap_id": beatmap_id, "user_id": sc.user_id,
        "mods": mod_str(sc), "mods_raw": "".join(raw_mods(sc)) or "NM",
        "legacy_mods": legacy_mods, "accuracy": sc.accuracy,
        "count_300": n300, "count_100": n100, "count_50": n50, "count_miss": nmiss,
        "era": era(sc), "legacy_score_id": legacy_score_id(sc) or "",
        "style": "", "split": "",
        "osr_path": osr_path, "osu_path": osu_path,          # relative to data/raw/
        "downloaded_at": now(),
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row, columns):
    # One open/close per row, so every finished replay is on disk if the run is interrupted
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_map_ids(path):
    """Beatmap ids, one per line or as a YAML list ("- 123"); "#" starts a comment."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip().removeprefix("-").strip()
        if line:
            ids.append(int(line))
    return list(dict.fromkeys(ids))                          # drop repeats, keep order


class RateLimited(Exception):
    pass


def with_backoff(call, what, tries=5):
    # Cloudflare answers a burst with HTTP 429 and an HTML page (docs/data_collection.md). ossapi raises
    # JSONDecodeError for it on JSON endpoints, but download_score(raw=True) swallows that error
    # and returns the HTML page as the "replay" bytes, so check for it here.
    for attempt in range(1, tries + 1):
        try:
            result = call()
            if isinstance(result, bytes) and result.lstrip()[:1] == b"<":
                raise RateLimited
            return result
        except (requests.exceptions.JSONDecodeError, RateLimited):
            print(f"  rate-limited on {what}; waiting {60 * attempt}s")
            time.sleep(60 * attempt)
    raise RuntimeError(f"still rate-limited on {what}; wait a few minutes and run again")


def fetch_beatmap(beatmap_id, path):
    """Save the .osu (once); return why the map can't be used, or None."""
    if path.exists():
        return None
    # requests, not urllib: it ships its own CA certificates (python.org builds on macOS don't)
    resp = requests.get(f"https://osu.ppy.sh/osu/{beatmap_id}", timeout=30)
    resp.raise_for_status()
    data = resp.content                                      # empty for an unknown id
    text = data.decode("utf-8-sig", errors="replace")
    if not text.startswith("osu file format"):
        return "no .osu file for this id"
    if re.search(r"^Mode:\s*[123]", text, re.M):             # 1-3 = taiko, catch, mania
        return "not an osu!standard map"
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    try:
        parse_beatmap(tmp)                                   # the simulator must be able to read it
    except Exception as e:
        tmp.unlink()
        return f"unparseable .osu ({type(e).__name__}: {e})"
    tmp.replace(path)
    return None


def fetch_replay(api, score_id, old_endpoint, download_sleep):
    """Download replays/{score_id}.osr and check it parses. Returns (legacy mods, None) or (None, reason).

    old_endpoint: the score id is an old per-mode id (legacy-shape score), served by download_score_mode.
    """
    osr = RAW / "replays" / f"{score_id}.osr"
    if osr.exists():
        path = osr               # saved by an earlier run that stopped before writing its row
    else:
        path = osr.with_suffix(".part")
        if old_endpoint:
            call = lambda: api.download_score_mode("osu", score_id, raw=True)
        else:
            call = lambda: api.download_score(score_id, raw=True)
        try:
            path.write_bytes(with_backoff(call, f"replay {score_id}"))
        except ValueError as e:  # ossapi raises ValueError when the API answers with an error
            return None, f"download failed: {e}"
        finally:
            time.sleep(download_sleep)
    try:
        frames = load_replay(path)
        if len(frames.t) == 0:
            raise ValueError("no frames")
    except Exception as e:       # osrparse raises assorted errors on a corrupt file
        path.unlink()
        return None, f"bad .osr: {type(e).__name__}: {e}"
    path.replace(osr)
    return frames.mods, None


def record_failure(score_id, beatmap_id, mods, reason):
    append_row(FAILURES, {"score_id": score_id, "beatmap_id": beatmap_id, "mods": mods,
                          "reason": reason, "failed_at": now()}, FAILURE_COLUMNS)
    print(f"  failed {score_id}: {reason}")


def collect(api, maps, cfg, rows, seen, dry_run):
    """Leaderboard mode: per map and mod set, keep clean scores up to the caps and download them."""
    per_player = Counter((int(r["user_id"]), cap_key(r["mods"], cfg["cap_groups"])) for r in rows)
    per_map = Counter((int(r["beatmap_id"]), mod_key(r["mods"])) for r in rows)
    done = Counter()                                         # replays kept (or, dry run, planned)
    short = []                                               # (map, mod set, replays) below per_map
    for bid in maps:
        problem = None if dry_run else fetch_beatmap(bid, RAW / "beatmaps" / f"{bid}.osu")
        if problem:
            print(f"{bid}: skipped, {problem}")
            continue
        for mod_set in cfg["mod_sets"]:
            have = per_map[(bid, mod_key(mod_set))]
            if have >= cfg["per_map"]:                       # no API call for a full slot
                print(f"{bid} {mod_set}: full ({have})")
                continue
            board = with_backoff(lambda: api.beatmap_scores(
                bid, mode="osu", mods=mod_set, limit=cfg["leaderboard_limit"]),  # type=None: global
                f"leaderboard {bid} {mod_set}")
            time.sleep(cfg["leaderboard_sleep"])
            kept, skipped = select(board.scores, mod_set, cfg, seen, per_player, have)
            why = ", ".join(f"{r} {n}" for r, n in Counter(r for _, r in skipped).most_common())
            print(f"{bid} {mod_set}: have {have}, keep {len(kept)}, skip {len(skipped)} ({why})")
            if have + len(kept) < cfg["per_map"]:            # a warning only: maps are never dropped
                short.append((bid, mod_set, have + len(kept)))
            for sc in kept:
                if dry_run:
                    done[mod_set] += 1
                    print(f"  would download {sc.id} user {sc.user_id} {mod_str(sc)} "
                          f"acc {sc.accuracy:.4f} {era(sc)}")
                    continue
                legacy_mods, reason = fetch_replay(api, sc.id, is_legacy(sc), cfg["download_sleep"])
                if reason:
                    record_failure(sc.id, bid, mod_str(sc), reason)
                    continue
                row = manifest_row(sc, bid, legacy_mods, f"replays/{sc.id}.osr", f"beatmaps/{bid}.osu")
                append_row(MANIFEST, row, MANIFEST_COLUMNS)
                done[mod_set] += 1
                print(f"  kept {sc.id} user {sc.user_id} {row['mods']} acc {sc.accuracy:.4f} {row['era']}")
    verb = "would download" if dry_run else "downloaded"
    print(f"{verb} {sum(done.values())} replays: " +
          ", ".join(f"{m} {done[m]}" for m in cfg["mod_sets"]))
    for bid, mod_set, n in short:
        note = " (DT is test-only: fine on a training map)" if mod_set == "DT" else ""
        print(f"WARNING: {bid} {mod_set} short, {n} of {cfg['per_map']}{note}")


def rebuild(api, listed, seen, download_sleep, dry_run):
    """Manifest mode: download exactly the scores in `listed` (rows of another manifest) and
    report the ones that are gone. No leaderboard calls, so caps and filters don't apply."""
    gone, done = [], 0
    bad_maps = set()
    for bid in dict.fromkeys(int(r["beatmap_id"]) for r in listed):
        problem = None if dry_run else fetch_beatmap(bid, RAW / "beatmaps" / f"{bid}.osu")
        if problem:
            print(f"{bid}: skipped, {problem}")
            bad_maps.add(bid)
    for r in listed:
        sid, bid = int(r["score_id"]), int(r["beatmap_id"])
        if sid in seen:
            continue
        if bid in bad_maps:
            gone.append(sid)
            continue
        if dry_run:
            done += 1
            print(f"  would download {sid} map {bid} {r['mods']}")
            continue
        # a legacy-shape score is listed under its old per-mode id (score_id == legacy_score_id)
        legacy_mods, reason = fetch_replay(api, sid, r["legacy_score_id"] == r["score_id"], download_sleep)
        if reason:
            record_failure(sid, bid, r["mods"], reason)
            gone.append(sid)
            continue
        row = {c: r.get(c, "") for c in MANIFEST_COLUMNS} | {"legacy_mods": legacy_mods,
                                                             "downloaded_at": now()}
        append_row(MANIFEST, row, MANIFEST_COLUMNS)
        done += 1
        print(f"  kept {sid} map {bid} {r['mods']}")
    print(f"{'would download' if dry_run else 'downloaded'} {done} of {len(listed)} listed replays; "
          f"{len(listed) - done - len(gone)} already here; {len(gone)} gone: {gone}")


def connect():
    load_dotenv()
    # Browser login once, then the token is cached. In WSL the browser can't open: copy the URL.
    return Ossapi(int(os.environ["OSU_CLIENT_ID"]), os.environ["OSU_CLIENT_SECRET"],
                  os.environ["OSU_REDIRECT_URI"])


def main():
    ap = argparse.ArgumentParser(description="Download replays and beatmaps into data/raw/.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("maps", type=Path, nargs="?", help="text or YAML file of beatmap ids")
    source.add_argument("--from-manifest", type=Path, help="download exactly the scores in this manifest")
    ap.add_argument("--config", type=Path, default=Path("configs/download.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # read the input before connect(), so a bad path fails before the browser login
    listed = read_rows(args.from_manifest) if args.from_manifest else None
    maps = None if args.from_manifest else read_map_ids(args.maps)

    (RAW / "replays").mkdir(parents=True, exist_ok=True)
    (RAW / "beatmaps").mkdir(parents=True, exist_ok=True)
    rows = read_rows(MANIFEST)
    seen = {int(r["score_id"]) for r in rows + read_rows(FAILURES)}

    api = connect()
    if args.from_manifest:
        rebuild(api, listed, seen, cfg["download_sleep"], args.dry_run)
    else:
        collect(api, maps, cfg, rows, seen, args.dry_run)


if __name__ == "__main__":
    main()
