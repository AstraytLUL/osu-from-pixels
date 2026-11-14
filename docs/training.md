# Training and evaluation

<!-- Moved from the README on 14 Nov 2026; links are relative to docs/. -->

## Training windows (behaviour cloning)

[data/resample.py](../data/resample.py) puts each replay on the frame grid the policy sees: 60 frames per
real second, so 16.7 map ms apart (25 under DT). Per frame: the cursor, and per key whether it was
pressed in the frame's window and whether it is held at its end (so neither short taps nor quick
re-presses are lost); on a frame with a press the cursor is where the player actually clicked.
Judging the grid version of a replay instead of the original costs 0.37% accuracy on average (the
±8 ms timing shift); see the module docstring.

[data/dataset.py](../data/dataset.py) turns the replays of a split into samples: the last 20 frames
(the map with the player's cursor drawn in) and the next action (cursor displacement, press and
held per key), plus the frame interval and 300 window for the conditioning experiments. HR
replays can be flipped back to normal geometry for training (`canonical_hr`). The map layer is
rendered once per map and mod set and cached compressed in `$OSU_FRAME_CACHE`:

```
python -m data.dataset build --split train --mods NM HR   # render + cache the map layers
python -m data.dataset stats --split train --mods NM HR   # windows per style / mod, label rates
```

```python
from data.dataset import from_split
ds = from_split("train", mods=("NM", "HR"))     # a torch DataLoader accepts it as is
sample = ds[0]                                  # dict of numpy arrays: frames (20, 96, 128) uint8, action, press, held, ...
```

Train, NM + HR: 590 replays, 4.56M windows, an 800 MB cache that takes 8 min to build (once). Loading
runs at about 1,100 samples/s per DataLoader worker.

## Training and evaluation (behaviour cloning)

```
python -m train.bc --config configs/baseline.yaml          # -> runs/baseline_seed0/ (best.pt, last.pt, metrics.jsonl)
python -m eval.play --checkpoint runs/baseline_seed0/best.pt --beatmap 553131 --mods HR --footage mp4
python -m eval.transfer_matrix --checkpoint runs/baseline_seed0/best.pt --split val --mods NM HR DT HD
```

Every run so far, what changed between them and what we learned: [docs/bc_stage1.md](bc_stage1.md).
The runs before the baseline (trained on v1 maps only) are kept in `configs/legacy/` and `runs/legacy/`.
The baseline ([configs/baseline.yaml](../configs/baseline.yaml)) predicts the cursor's next *position*
(absolute aim), is trained with cursor jitter, and adds 20 spinner-heavy maps to the v1 train maps.

- [models/](../models/): a frame-stack CNN (stride-2 convolutions, pixel x/y channels) with three heads:
  cursor position (or move, for relative aim), press and held per key.
- [train/bc.py](../train/bc.py): the loss and the loop. Each epoch the model plays the NoMod val maps
  closed-loop, and best.pt keeps the most accurate one.
- [eval/policy.py](../eval/policy.py): the model and Auto behind one interface;
  [eval/rollout.py](../eval/rollout.py) plays a whole map from the policy's own frames (its cursor drawn
  back in) and judges it.
- [eval/play.py](../eval/play.py): one map, with an .osr to watch in osu! and optional footage.
  `python -m eval.osr --beatmap <id> --mods HR` writes Auto's replay. Local viewing only.

A closed-loop map takes ~12 s on CPU, so validating on the 9 val maps adds ~2 min per epoch.

### Architecture experiments and evaluating any model the same way

The network is chosen in the config: `model.backbone` (`framestack` | `cnn3d` | `cnn_gru`),
`model.width`, `model.aim_head` (`linear` | `softargmax`); left out, they are the baseline's. Configs:
`configs/arch/` (the five experiments), `configs/followup/` (40-epoch runs). Plan, hypotheses,
results and figures: [docs/architectures.md](architectures.md).

```
bash scripts/run_experiment.sh configs/arch/cnn3d.yaml 0      # train + val transfer matrix + upload; rerun to resume
python -m train.hub push runs/cnn3d_seed0                     # upload a run: sc4001-project/bc-cnn3d, folder seed0/
python -m train.hub pull sc4001-project/bc-cnn3d --seed 0     # download one: runs/cnn3d_seed0/
python -m eval.test_all run                                   # every run on the Hub on the TEST maps (final only)
python -m eval.compare_runs --split test                      # mean ± std tables over seeds
python -m eval.report_figures --split test                    # docs/figures/
python -m eval.diagnose --configs baseline cnn_gru            # aim vs timing misses, click offsets
python -m eval.heatmaps                                       # soft-argmax: where the network wants to go
```

To compare a new model (an RL policy, say) with these: name its config, run it through
`scripts/run_experiment.sh` (or train it, then `train.hub push` its run folder with best.pt,
config.yaml, metrics.jsonl and transfer_val.json), and add its repo to `eval/test_all.py`'s `REPOS`
and its name to `ORDER` in `eval/compare_runs.py` / `eval/report_figures.py`. Same maps, same motor,
same selection rule (NoMod val only), so the numbers compare.

## Footage

[eval/video.py](../eval/video.py) renders simulator footage: the 128x96 frames the policy sees, scaled
up 4x, with the cursor, judgement popups, keys and running accuracy. A 15 s clip takes ~4 s.

```
python -m eval.video --score 890068308 --start 30 --duration 15                       # a replay in data/raw
python -m eval.video --beatmap 553131 --mods HR --start 20 --duration 6 --format gif   # Auto
```

Use .mp4 for whole maps (a GIF of a full map is hundreds of MB).

## Where outputs go

Everything is under `runs/` (gitignored), named by [eval/paths.py](../eval/paths.py):

```
runs/<run name>/                              a training run: config, metrics, last.pt, best.pt
runs/replays/<player>/<map>_<MODS>.osr        replays to watch in osu!
runs/footage/<player>/<map>_<MODS>.mp4        footage of a whole map
runs/footage/<player>/<map>_<MODS>_<a>-<b>s.gif   ... or of a clip from a s to b s
runs/hub/<repo>/[seed<N>/]                    runs' small files fetched from the Hub (eval/compare_runs.py)
runs/diagnostics/                             eval/diagnose.py and eval/heatmaps.py output (copies in docs/diagnostics/)
```

- `<player>`: `auto` (our Auto), `<run>-<checkpoint>` for a model (e.g. `bc_baseline_seed0-best`),
  `human-<score_id>` for a player's replay. In osu! the player name is "SC4001 Auto" or "SC4001 BC".
- `<map>`: `<beatmap id>-<title>`, e.g. `4912305-tetoris`.
- `<MODS>`: `NM`, or mods in the order EZ HD HR DT HT, e.g. `HDHR`.

