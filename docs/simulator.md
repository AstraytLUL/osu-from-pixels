# The osu! simulator

<!-- Moved from the README on 14 Nov 2026; links are relative to docs/. -->

`osu_sim/` reproduces osu!stable gameplay, ported from osu! lazer and the osu! wiki (each file's
docstring names its sources):

| File | What it does |
| --- | --- |
| `beatmap.py` | parses `.osu` files |
| `slider.py` | slider curves (linear, perfect circle, Bezier, Catmull), length, velocity, ticks, repeats |
| `mods.py` | HR (flip, CS/AR/OD), EZ, playback rate for DT/NC/HT |
| `playable.py` | a map ready to play: mods applied, slider geometry and timing, stacking |
| `judge.py` | scores a cursor trajectory + key presses: hit windows, note lock, sliders, spinners, accuracy, UR |
| `auto.py` | Auto, a scripted perfect player (oracle upper bound, DAgger expert) |
| `renderer.py` | 128x96 grayscale frames at any map time, with HD and the cursor drawn on top |

```python
from data.replay import load_replay
from osu_sim.beatmap import parse_beatmap
from osu_sim.playable import make_playable
from osu_sim.judge import judge, accuracy
from osu_sim.auto import auto_replay
from osu_sim.renderer import Renderer

replay = load_replay("data/raw/replays/<score_id>.osr")
pl = make_playable(parse_beatmap("data/raw/beatmaps/<beatmap_id>.osu"), replay.mods)
print(accuracy(judge(pl, replay)))           # the human's accuracy under our judge
print(accuracy(judge(pl, auto_replay(pl))))  # Auto: 1.0
frame = Renderer(pl).frame(30_000, (256, 192))   # (96, 128) uint8 at map time 30 s
```

Times are map time in ms (under DT a 60 fps frame is 25 map ms apart); positions are osu! pixels
(512x384).

**Validation.** Max combo matches osu! on 10/10 maps. Auto scores SS on 10 maps x 4 mod sets. On real
replays the judge reproduces osu!'s 300/100/50/miss counts exactly on 20/24 stable replays with
mistakes (`scripts/validate_judge.py`) and on 34/39 stable NoMod replays of the v1 pilot; the rest
differ by one object (0.1-0.3% accuracy). Lazer-era scores use lazer's own scoring, so compare the
judge against stable-era scores only.

Combo and score ([osu_sim/score.py](../osu_sim/score.py), `scripts/validate_score.py`): on the 422
stable replays whose counts the judge reproduces, max combo matches the replay header on all 422, and
ScoreV1 on all maps without spinners exactly; with spinners it is within 0.1% (421/422), since spin
counting is approximate.

