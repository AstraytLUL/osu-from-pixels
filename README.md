# Learning to Play osu! from Pixels

**Does a policy trained on easy maps extrapolate to hard ones?**
SC4001 (Neural Networks and Deep Learning, NTU) group project.

We train a policy that plays the rhythm game osu! from raw pixels (the last 20 frames at 128x96,
60 fps), by behaviour cloning on top players' replays of 4–6.9★ maps. We then ask how it does on
harder maps it has never seen (natural difficulty, up to 10★) and under game modifiers that change
one factor at a time (synthetic difficulty: Hard Rock, Double Time, Hidden). Everything is played
closed-loop in our own deterministic osu! simulator. The policy, the scripted Auto player and human
replays are all scored by the same judge.

The report draft is [report/draft.md](report/draft.md).

## Results at a glance

Behaviour cloning (3 seeds, test maps, NoMod, 75 px/frame speed cap):

| Star band | 4★ | 5★ | 6★ | 7★ | 8★ | 9★ |
|---|---|---|---|---|---|---|
| Top human replays | 99.9 | 99.9 | 99.8 | 99.5 | 99.3 | 97.9 |
| BC policy | 99.0 ± 0.4 | 99.4 ± 0.0 | 98.6 ± 0.2 | 96.1 ± 0.9 | 92.7 ± 0.2 | 78.2 ± 1.8 |

- **Inside the training range** (4–6★) the policy plays within ~1 pp of top players.
- **Past it**, it falls 3, 7 and 20 pp below top players at 7, 8 and 9★ (its own accuracy drops by
  3, 6 and 21 pp from its 4–6★ level).
- **Hard Rock** costs under 1 pp and **Double Time** 13 pp. At equal star rating, Double Time maps are
  only a few points below unmodded ones, but Hard Rock + Double Time maps are more than 20 pp below.
- **No architecture helps both kinds of difficulty.** A spatial (soft-argmax) aim head extrapolates
  best to harder maps. Temporal models (3D CNN, CNN + GRU) are best under HR + DT.
- **Feeding the policy its own past actions** makes it collapse (copycat).

PPO fine-tuning from the BC policy ([rl/RESULTS.md](rl/RESULTS.md)):

- **On a single hard map:** accuracy rises 4–7 pp, with 3 seeds per map and no forgetting.
- **On the other 8–9★ test maps it never trained on:** accuracy also rises, by +1.9 to +4.1 pp on
  average and up to +9.6 pp.
- **Under Hidden** (no approach circles; the policy misses ~80% of objects), PPO gains only 5 pp.
  Supervised fine-tuning on Hidden re-renders reaches 84% but forgets NoMod, unless the NoMod data
  is mixed back in.

## Repository layout

```
osu_sim/     the simulator: .osu parser, sliders, mods, judge (osu!stable rules), Auto, renderer, RL env
data/        data pipeline: map selection, replay download, splits, 60 fps resampling, training windows
models/      policy networks: frame-stack CNN, 3D CNN, CNN + GRU backbones; linear / soft-argmax aim heads
train/       behaviour cloning (train/bc.py) and the Hugging Face run store (train/hub.py)
eval/        closed-loop rollouts, the headline evaluation, diagnostics, replays (.osr) and footage
rl/          PPO fine-tuning experiments: code, configs, raw results, RESULTS / AUDIT / LOG
configs/     one YAML per experiment; configs/splits/ holds the frozen map splits
scripts/     one-off scripts: judge / score validation, test fixtures, data-collection tallies
tests/       pytest suite (simulator, data, training, evaluation, RL environment)
docs/        reference docs, the architecture study, project proposal and milestones
report/      the report draft
```

## Setup

Python 3.11 or newer.

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # install the CUDA build of torch from pytorch.org first for GPU training
```

Settings go in `.env` at the repo root (gitignored):

```
HF_DATASET=sc4001-project/osu-mod-transfer-raw   # the team's private dataset of replays and maps
HF_TOKEN=<token>                                 # or `hf auth login`
OSU_FRAME_CACHE=~/osu_frame_cache                # rendered map layers (GBs); keep off synced folders
OSU_CLIENT_ID=<id>                               # only to collect new data from the osu! API
OSU_CLIENT_SECRET=<secret>
OSU_REDIRECT_URI=http://localhost:7270/
```

## Data

Replays and beatmaps belong to their players, mappers and ppy, so they are never committed. The team
shares them through a private Hugging Face dataset. With read access:

```
python -m data.prepare               # pull data/raw/ and pre-render the training frames (~3.5 GB)
python -m data.splits list           # maps per split and star band
```

The frozen splits are in [configs/splits/](configs/splits/): train 62 maps (4.53–6.92★), val 9,
test 37 (6–7 per band, 4★ to 9★). Collection, selection rules and the dataset's contents are
described in [docs/data.md](docs/data.md).

## Reproducing the results

Behaviour cloning and the headline table (the report's Sections 6.1–6.5):

```
python -m train.bc --config configs/baseline_cap.yaml --seed 0      # also seeds 1, 2 -> runs/baseline_cap_seed{0,1,2}/
python -m train.bc --config configs/arch_cap/softargmax.yaml --seed 0   # each of configs/arch_cap/*, configs/prev_*.yaml
python -m eval.headline prepare --config configs/eval/headline.yaml # render every test cell once
python -m eval.headline run --config configs/eval/headline.yaml     # every model x seed x map x mods -> runs/headline/results.csv
python -m eval.headline humans --config configs/eval/headline.yaml  # the human reference rows
```

PPO experiments (the report's Section 6.6): the one-line commands are in
[rl/RESULTS.md](rl/RESULTS.md#reproduce). They end with `python -m rl.report`, which regenerates
every RL table and figure from the raw CSVs in [rl/results/](rl/results/).

The report's figures and every number it quotes:

```
python -m report.figures --snapshot --frames   # runs/ + rl/ -> report/data/, report/figures/, report/numbers.md
python -m report.figures                       # the same from the committed report/data/ alone (no runs needed)
```

Watch any policy in osu!:

```
python -m eval.play --checkpoint runs/baseline_cap_seed0/best.pt --beatmap 4912305 --mods HR
# -> runs/replays/<run>-best-cap/<map>_HR.osr; open it with osu! (local viewing only)
```

## Tests

```
pytest -q
```

336 tests. Tests that need real replay fixtures skip unless `python scripts/check_download.py` has
fetched them (osu! credentials needed).

## Documentation

- [docs/simulator.md](docs/simulator.md): what the simulator reproduces and how it was validated
- [docs/data.md](docs/data.md): data collection, map selection, splits
- [docs/training.md](docs/training.md): training windows, BC training, evaluation, footage, output paths
- [docs/architectures.md](docs/architectures.md): the architecture study (plan, results, figures)
- [docs/bc_stage1.md](docs/bc_stage1.md), [docs/data_collection.md](docs/data_collection.md): lab notes
- [rl/RESULTS.md](rl/RESULTS.md): the PPO experiments

## Acknowledgements

The simulator ports osu! gameplay rules from [osu! lazer](https://github.com/ppy/osu) and the
[osu! wiki](https://osu.ppy.sh/wiki); each file names its sources. Replays and maps come from the
osu! API v2. This project is not affiliated with ppy. Never submit generated replays to osu! servers.

## License

The code is released under the [MIT License](LICENSE). The licence does not cover replays, beatmaps
or their audio and images, which belong to their players, mappers and ppy and are not in this
repository.
