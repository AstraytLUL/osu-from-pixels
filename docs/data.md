# Data: replays, maps and splits

<!-- Moved from the README on 14 Nov 2026; links are relative to docs/. -->

Replays and beatmaps are never committed; they belong to their players, mappers and ppy. The team
shares them through the private Hugging Face dataset `sc4001-project/osu-mod-transfer-raw` (owned
by the `sc4001-project` organisation; ask Astrayt to add you; read access is enough to pull).

**For teammates: one command** ([data/prepare.py](../data/prepare.py)) pulls everything and pre-renders
the frames the BC training reads; `--check` only reports what is missing:

```
hf auth login                  # once (a read token is enough)
python -m data.prepare         # pull data/raw/, cache BC frames (~3.5 GB in $OSU_FRAME_CACHE)
```

Its docstring shows how to load a split's maps and their frames in code (`split_maps("train", ...)`).

**More maps** ([data/find_maps.py](../data/find_maps.py), needs osu! credentials): from the beatmap
search alone, excluding every mapset already in a split (so nothing overlaps val, test or challenge);
new maps are appended to a split's band files. `spinners`: 20 BC training maps with >= 3 spinners
each (spinners are rare in v1), replays downloaded like the v1 train maps. (An `rl` list of 499
maps without replays, for the self-imitation plan, was collected and later removed with it.)

```
python -m data.find_maps spinners
python -m data.download configs/maps/spinners.txt --config configs/download_nm.yaml
python -m data.download configs/maps/spinners.txt --config configs/download_train_mods.yaml
python -m data.hub push
```

**Get the data, step by step** (what `data.prepare` does first):

```
hf auth login                  # once; or set HF_TOKEN (a read token is enough)
python -m data.hub pull        # fills data/raw/ (69 MB)
```

`data/raw/` then holds `manifest.csv` (one row per replay: score, map, player, mods, accuracy,
300/100/50/miss counts, era stable|lazer, file paths), `replays/{score_id}.osr` and
`beatmaps/{beatmap_id}.osu`.

**Splits** ([data/splits.py](../data/splits.py)). Every map list is in
[configs/splits/](../configs/splits/), one file per split and whole-star band:
`configs/splits/<split>/<N>_star.yaml` (`4_star` holds 4.00-4.99*), for the splits `train`, `val`
and `test`; [configs/splits/challenge.yaml](../configs/splits/challenge.yaml) is a
separate showcase. A config's `map_lists: [configs/splits]` reads them all. They were built on 9 Nov
from the original lists, which stay in [configs/legacy/maps/](../configs/legacy/maps/) as the record of
how the maps were chosen. Maps can be added to a split, never moved or removed.

```
python -m data.splits list       # maps per split and band, star range, styles
```

**What's in it (v1, collected Oct 26).** 60 maps, 20 per style, 4.5-6.5*, now in the train, val and
test splits (originally [configs/legacy/maps/maps_v1.yaml](../configs/legacy/maps/maps_v1.yaml); plain
id lists for downloading in [configs/maps/](../configs/maps/)). Splits are by map, stratified by style:
14 train / 3 val / 3 test per style.

| Split | Maps | NM | HR | HD | DT |
| --- | --- | --- | --- | --- | --- |
| train | 42 | 420 | 170 | 208 | - |
| val | 9 | 90 | 3 | 3 | 3 |
| test | 9 | 90 | 79 | 90 | 76 |

All replays are >= 98% accuracy, from 1,110 players, at most 3 per player across NM + HR. 507 are
stable-era and 814 lazer-era (`era` column). 89 more replays are on maps outside the v1 list (an
earlier pilot list and a test run); they are in the manifest but in no split. The 9 HR/HD/DT
replays on a val map also come from that test run. Use `configs/splits/`, not the manifest, to
decide what belongs to which split.

**Why per-map leaderboards, and the rate limits:** see
[docs/data_collection.md](data_collection.md) (the measurements behind the data method, plus
open team decisions).

**How the maps were chosen** ([data/select_maps.py](../data/select_maps.py)). Styles come from the
tags players vote on each difficulty on the osu! website (`skillset/<style>` plus that style's
specific tags such as `streams/bursts`): a style must have >= 20 votes and 1.5x the next style.
Each style is searched and filled separately, ranked by play count, because the most-played maps
are mostly jump maps. Filters: ranked, one difficulty per mapset, <= 2 maps per mapper, 4.5-6.5*,
60-180 s drain, CS 3.5-4.5, AR 8.5-9.5, no gimmick tags, >= 20 usable NoMod replays. A short style
relaxes in a fixed order (recorded per map); only tech needed it (2 maps, stars up to 7.0).

[configs/splits/challenge.yaml](../configs/splits/challenge.yaml) is a separate challenge list of hard maps
(Everything will freeze [Time Freeze], 8.31*, plus unconfirmed suggestions). It is never used for
training, validation or model selection. The PPO experiments (rl/) train on 555797: put its
`.osu` in `data/challenge/555797.osu` (download it from the osu! website; it is not on the Hub).

**Collecting more data** (osu! credentials on a supporter account, for mod-filtered leaderboards):

```
python -m data.select_maps                                   # rewrites configs/legacy/maps/maps_v1.yaml + configs/maps/*.txt
python -m data.download configs/maps/test.txt --dry-run      # leaderboard calls only
python -m data.download configs/maps/test.txt --config configs/download.yaml           # test: NM HR HD DT x10
python -m data.download configs/maps/val.txt --config configs/download_nm.yaml         # val: NM x10
python -m data.download configs/maps/train.txt --config configs/download_nm.yaml       # train: NM x10
python -m data.download configs/maps/train.txt --config configs/download_train_mods.yaml  # train: HR HD x5
python -m data.hub push                                      # merges the Hub's manifest first, then uploads
```

About 7 s per replay (osu! rate limits; the full v1 download took 2.5 h). A rerun skips everything
already in the manifest; failed replays go to `data/raw/failures.csv`. `select_maps` caches its API
calls in `data/raw/cache/` (not uploaded). `--from-manifest path/to/manifest.csv` re-downloads
exactly the listed scores instead of reading leaderboards.

