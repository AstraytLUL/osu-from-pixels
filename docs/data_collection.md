# Data collection: measurements and decisions

What we measured before building the downloader, and what it decided. Moved here from the
pre-repo checklist (`TODO_before_repo.md`, removed). Useful for the report's data section.

## Which source of replays (Oct 24)

The osu! API offers two routes to replays. We measured clean NoMod replays per API call on each.

| Route | Setup | Result |
| --- | --- | --- |
| Player crawl: each player's top 100 scores | 20 players ranked ~5k-50k, 24 calls | 703 replays in the 4-6.5* band, 414 at >= 98%, but ~90% DT/NC. Clean NoMod 9, HD 32, HR 9, HDHR 6: **0.4 clean NoMod per call** |
| Map leaderboards, filtered by mod (needs a supporter token) | 3 maps (814293, 816600, 2935781) x NM/HD/HR/HDHR x global/country, 24 calls | Global NoMod board: 50/50 clean replays on all three maps, **50 per call**. HD 17-50, HR 5-50, HDHR 7-50 per map. Country boards add 0-22, mostly overlapping the global top 50 |

**Decision:** collect from per-map global leaderboards (NM for training, HR as a training top-up,
HD held back for Stage 3, DT test-only). The player crawl is dropped. Data is not the bottleneck,
but the same top players appear on every board, so replays are capped per player
(`configs/download.yaml`). Top players' own best scores are mostly DT, which is why the crawl fails
for NoMod; on the maps' NoMod boards they are plentiful.

## Rate limits (Oct 24)

The API sits behind Cloudflare. About 7 leaderboard calls in quick succession get HTTP 429 with an
HTML page (ossapi raises `requests.exceptions.JSONDecodeError`; `download_score(raw=True)` instead
returns the HTML as the "replay"). 4 s between leaderboard calls and 7 s between replay downloads
ran clean: the full v1 download (1,100 replays) took 2.5 h with no 429s. `data/download.py` backs
off 60 s, 120 s, ... on a 429.

## Replay eras (Oct 24, the judge fixtures)

- The API can return a score in two shapes. Lazer shape: a list of mods (`CL` marks a stable-era
  score; it is not a real mod), great/ok/meh/miss statistics. Legacy shape: one mod bitfield,
  count_300/100/50/miss, downloaded through a per-mode endpoint. `data/download.py` handles both.
- The `.osr` header's mod bitfield agrees with the API's mods once `CL` is dropped.
- Stable replays start with setup frames (time 0 and -1 at (256, -500)) and a backwards time jump;
  lazer replays start without them. `data/replay.py` trims both.

## Open team decisions (from the checklist; not code)

- [ ] Agree the Oct 31 go/no-go sentence. Proposed: "The BC policy tracks the cursor on validation
      maps and its judge accuracy beats the brightest-approach-circle heuristic." If it fails, RL is
      cut; if there is no usable baseline at all, switch to the fallback topic.
- [ ] Agree a branch per component, merging into `main` only through PRs that keep `pytest` green.
- [ ] Invite both teammates to the GitHub repo and the `sc4001-project` Hugging Face organisation.
