# BC stage 1: what we tried, what changed, what we learned

Behaviour cloning (the project proposal, Stage 1): a CNN looks at the last 20 frames (128x96, 60 fps) and
predicts the next frame's cursor and keys, learned from players' replays. This page records every
training run so far, the code changes between them, and the open problems. Runs are on branch
`behavior-cloning/astrayt`; each run's config, metrics and per-map validation results are in
`runs/<name>_seed0/` (not committed: ask Astrayt for a run folder).

**Legacy runs.** Every run before `baseline` trained on the v1 maps only, and most were selected
with an older click rule, so they don't compare directly with `baseline`. Their configs are in
`configs/legacy/` and their run folders, replays and footage in `runs/legacy/` (checkpoints still
load: `python -m eval.play --checkpoint runs/legacy/bc_abs_seed0/best.pt ...`). They stay as the
history of how the baseline was reached; any comparison for the report should be re-run against
`baseline` on the same data and click rule.

**Where it ended up:** `configs/baseline.yaml` plays the NoMod val maps at **98.5%** (circles 98.6%,
sliders 98.3%, 11 of 12 spinners), Tetoris [Insane] (test) at 98.9% with 3 misses and Everything
will freeze [Time Freeze] (challenge, 8.31*) at 92.5%. The transfer gaps are large and clean: DT on
streams and tech, and HD everywhere (see "Transfer").

## Setup shared by all runs

- **Data:** maps_v1 train split (42 maps), NoMod + HR replays (HR flipped to NoMod geometry),
  ~4.5M training windows; `baseline` adds 20 spinner maps (7.3M windows). 500k windows drawn per
  epoch, 20 epochs, batch 256, AdamW 3e-4.
- **Network:** FrameStackCNN (4 stride-2 convs + x/y coordinate channels, flatten, 512 features)
  and three heads: aim (2), press (2 logits), held (2 logits).
- **Loss:** SmoothL1(aim) + BCE(press, positives x10) + BCE(held), weights 1/1/1.
- **Selection:** after every epoch the model *plays* the 9 NoMod val maps closed-loop in our
  simulator (its own cursor is drawn into its next frames) and the judge scores it; `best.pt` is
  the epoch with the highest mean accuracy. 9 maps is few: expect +-5-10 points between epochs.

## Runs

Each run changes one thing from the run it builds on. "Best val" is what training reported, with
the click rule of the time; the last column replays each best.pt with the final click rule (see
"The click rule" below), so those numbers compare like for like.

| Run | Change | Best val during training (epoch) | Val misses then | Best.pt, final click rule |
| --- | --- | --- | --- | --- |
| `bc_baseline` | first run: relative aim (predict the move) | 17.9% (17) | 3,693 | |
| `bc_jitter` | + cursor jitter (48 px) | 75.0% (10) | 549 | |
| `bc_jitter_r2` | same, on re-rendered frames (reverse-arrow fix) | 46.1% (8) | 1,859 | |
| `bc_cursor` | + cursor on its own input channels | 76.2% (17) | 615 | |
| `bc_abs` | + **absolute aim** (predict the position) | 88.1% (7) | 119 | |
| `bc_abs_nochan` | `bc_abs` without the cursor channels | 90.8% (9) | 45 | 98.6% |
| **`baseline`** | `bc_abs_nochan` + 20 spinner maps in training | 86.3% (16) | | **98.5%** |

`baseline`'s epochs were chosen with the old click rule, and its old-rule score (86.3%) is lower
than `bc_abs_nochan`'s because the old rule dropped its sliders more; with the final rule the two
are level on NoMod, and `baseline` is ahead under DT (86.4% vs 78.9%) and HD and on spinners.

Held-out maps (NoMod, accuracy / misses / max combo; replays in `runs/replays/<run>-best/`):

| Map | `bc_cursor` | `bc_abs` | `bc_abs_nochan` | `baseline`, final click rule |
| --- | --- | --- | --- | --- |
| Tetoris [Insane] (test) | 80.7% / 33 / 68 | 88.2% / 6 / 265 | 87.3% / 2 / 160 | **98.9% / 3 / 574** |
| Everything will freeze [Time Freeze] (challenge) | 53.8% / 551 / 71 | 72.5% / 284 / 75 | 80.9% / 217 / 146 | **92.5% / 78 / 336** |

### What each step showed

1. **`bc_baseline`: good predictions, bad play.** Open-loop (on players' cursors) it predicted
   moves well (5.2 px error vs 8.3 px for "don't move") and 82% of presses. Closed-loop the
   cursor drifted away and stayed there: a third of the time at the playfield edge, median 170 px
   from the next object. Plain BC only ever sees the player's own cursor, so once its cursor is
   somewhere a player's never is, it has no idea how to come back.
2. **`bc_jitter`: teach recovery.** In half the training samples the whole cursor history is
   shifted by up to 48 px, and the target becomes "move back to where the player went next" (a
   cheap stand-in for DAgger). 18% -> 75%.
3. **`bc_jitter_r2`: the 75% was partly luck.** The only difference is a renderer fix that changes
   a few frames per kick slider, yet the run peaked 29 points lower and then collapsed. Training
   with relative aim plus jitter is unstable from run to run.
4. **`bc_cursor`: the cursor on its own channels.** In grayscale the cursor is one more bright dot
   among circles and slider balls. Given the 20 cursor frames as separate input channels (map
   frames unchanged), the network can't confuse the two: 37% after one epoch (vs 2%), 76% best,
   and no collapse.
5. **`bc_abs`: predict where, not how far.** A relative move adds its error to every later frame;
   an absolute position doesn't carry an error into the next frame. 88%, with streams (77.5 ->
   88.6%) and tech (63.5 -> 85.7%) gaining most, and every epoch from the 3rd on above
   `bc_cursor`'s best. Its aim loss is *higher* (0.078 vs 0.046): naming an exact spot is harder
   than guessing a small step, but its errors don't accumulate, and that is what matters in play.
6. **`bc_abs_nochan`: the cursor channels are no longer needed.** With absolute aim, drawing the
   cursor into the map frames (half the input channels) does as well: 90.8% vs 88.1%, within the
   noise of 9 val maps, with fewer misses (45 vs 119).
7. **The click rule, not the model, was losing sliders and spinners** (see below): with it fixed,
   circles were already at 99%, and sliders went from 79% to 98%.
8. **`baseline`: spinner maps.** The spinner maps are 38% of its training windows (they are long
   and have 15 replays each), so this is a bigger data change than "+20 maps" suggests. With the
   final click rule, spinners score 79% (vs 53% without the spinner maps), DT 86.4% (vs 78.9%).

### The click rule

The network outputs probabilities; a rule turns them into key presses (`eval/policy.py`
`ModelPolicy`). Three versions:

1. **Raw:** a key is down on every frame its probability is above 0.5. The press head is unsure for
   a frame or two around each note, so this clicked twice per object (1.98 clicks per object, both
   keys down 1,600 frames per map).
2. **Rising edge** (used during all training runs above): one click when the press probability
   crosses 0.5, keys alternating; the key stays down while "held" > 0.5. Clicks per object 0.91.
   But it lost sliders and every spinner, and the network was not to blame: "held" was above 0.5
   on 97% of slider-body frames and 94% of spinner frames, and on 0% of idle frames.
   - In slider bodies "held" hovers just above 0.5 (median 0.57); one frame's dip released the key
     for good, dropping the slider's end (29-43% of slider tails were missed).
   - A key could only go down on a click, and a spinner rarely starts with one (the press
     probability peaked at 0.26-0.65), so no key was ever down during a spinner. The model spun
     (up to 28 turns in 5 s) for nothing.
3. **Final:** once down, a key is released only when "held" falls to 0.3 (hysteresis), and "held"
   above 0.5 with no key down presses one. Chosen on NoMod val (a decoding choice; nothing else was
   tried on mods or test):

| Rule (`bc_abs_nochan` / `baseline`, NoMod val) | Mean | Circles | Sliders | Spinners hit |
| --- | --- | --- | --- | --- |
| Rising edge | 90.8 / 86.3 | 99.2 / 98.6 | 79.4 / 69.9 | 0 / 0 of 12 |
| + release at 0.3 | 98.7 / 98.4 | 99.2 / 98.6 | 98.2 / 98.0 | 9 / 9 |
| + a held key can start without a click | 91.2 / 90.9 | 87.6 / 91.4 | 93.6 / 88.6 | 11 / 10 |
| **both (final)** | **98.6 / 98.5** | 98.9 / 98.4 | 98.4 / 98.3 | 11 / 11 |

Model selection during training uses the same rule, so runs from now on pick their best epoch
with it.

**Human-like idle movement.** In calm sections and breaks (e.g. on Everything will freeze), the
model's cursor "dances": small loops and wiggles while clicking or waiting, as players' cursors do.
Nothing rewards this; it is imitation of how players move when the map doesn't demand much. It
suggests the model learned player movement, not only "go to the next circle".

## Transfer (val maps, not test)

Each best.pt under each mod, final click rule, on the 9 val maps (a diagnostic: the test maps stay
untouched until final results, and nothing is selected on mod results). Closed-loop accuracy:

| Mods | `bc_abs_nochan` | `baseline` | `baseline`: jumps / streams / tech | `baseline` miss rate |
| --- | --- | --- | --- | --- |
| NM | 98.6% | 98.5% | 99.7 / 98.5 / 97.2 | 0.6% |
| HR | 98.5% | 97.7% | 98.3 / 97.7 / 97.0 | 0.4% |
| DT | 78.9% | 86.4% | 98.1 / 92.0 / 69.2 | 7.0% |
| HD | 14.7% | 18.8% | 21.5 / 17.7 / 17.2 | 66.9% |
| HRDT | 63.4% | 67.7% | 72.5 / 70.7 / 59.8 | 15.3% |

- **HR: no gap.** Expected, since training includes HR replays flipped to NoMod geometry (HR's
  smaller circles and stricter timing are seen in training; only the flip is new). Whether the HR
  column counts as zero-shot is an open team decision (the project proposal says both).
- **DT: the gap is in tech (and streams), not jumps.** Jumps hold at 98%; streams lose 6 points and
  tech 28. Dense, irregular patterns are where 1.5x speed breaks the learned timing (hypotheses 2
  and 7).
- **HD: collapse, in aim as well as timing.** Auto still plays the HD renders at 96-100%, so the
  simulator is fine; the model is not. With the rising-edge rule, on one map it clicked 47 times for
  570 objects (566 under NoMod) and its cursor was a median 124 px from each object at its hit time
  (7 px under NoMod; the circle radius is 39). Hypothesis 3 expected HD to hurt timing more than
  aim; instead the model seems to use the approach circle to find *which* object is next, not only
  *when* to click.

## Other changes made along the way

- **Renderer (`osu_sim/renderer.py`, `RENDER_VERSION` 2):** a slider's reverse arrow now shows from
  when the slider appears, not from when it starts; a kick slider's reverse was visible for only
  ~5 frames. The frame cache is keyed on the renderer version, so stale frames are never reused.
- **Rollout:** frames before the map starts are blank with the cursor off screen, as in training.
- **Tools:** `.osr` export with real ScoreV1 score and max combo (validated exactly against 422
  stable replays' combo) and No Fail added so osu! plays the whole replay; footage as MP4/GIF
  (with slider ticks); naming rules for outputs (`eval/paths.py`).
- **Replays that osu! plays back as we judge them.** In Highscore's jumps (2:12-2:24) osu!stable's
  playback missed ~7 of 8 clicks per burst that our judge scored 300. The model often arrives in
  the circle on the very frame it clicks (the frame before is outside, 20-35 px per frame), a
  pattern players' replays rarely have, and stable's playback judged those clicks from where the
  cursor was before that frame's move. Our judge's rule (the position on the press frame) is the
  right one for players' replays: it reproduces osu!'s counts on 86% of 657 stable replays, vs 31%
  with the previous frame's position. So the writer changed, not the judge: the cursor move of a
  press frame is written 8 ms ahead as its own frame. Checked in osu! (960 fps): the replay now
  matches the simulator footage. Replays written before this change still show the problem.

- **Motor limits (`eval/motor.py`).** The policy names where the cursor goes; the rollout used to
  put it there exactly, however far. A Motor carries the move out like a hand: at most `max_step`
  px per frame toward the target, then Gaussian landing noise of `noise` x the step length per axis
  (still = exact, fast = imprecise). `HUMAN` (75 px, 0.3) comes from 600 train replays: players'
  99.9th-percentile step is 73-76 px per frame, and their click error grows with arrival speed
  (rms 11.6 px arriving at 0-2 px per frame, 16.7 px at 20-35 px). Every policy, Auto included,
  plays under the same motor. **`HUMAN` is the default** in the rollout, `eval.transfer_matrix`,
  `eval.play` and BC validation; `--motor free` (or `max_step: null`, `aim_noise: 0` in a
  config's `eval`) turns it off. Every result above this point was measured free, before the
  limits existed, and so were the replays and footage made until then.
  Cost to `baseline` on NoMod val: none from the cap (98.48%, identical per map), 98.37% with the
  noise as well. BC already moves like its players (steps 8.8 px median vs 8.0; 1.3% of clicks
  arrive at over 20 px per frame, for both). The limits matter for RL, which otherwise learns to
  snap onto objects: RL should train and be judged under `HUMAN`.

## Known problems

- **Spinner direction:** players spin both ways, so the move target during a spinner is two-sided
  and the model sometimes hesitates or reverses (79% spinner accuracy). RL should fix the rest: the
  judge rewards a held, fast spin in either direction.
- **Variance:** one seed per config and 9 val maps. The proposal promises 3 seeds per
  configuration for final numbers.
- **Selection used the old click rule:** `baseline`'s best epoch was chosen under the rising-edge
  rule. Retraining (or re-scoring saved epochs) under the final rule might pick a different epoch.
- **Replays need No Fail:** the bot isn't guaranteed to pass a map. On Everything will freeze
  [Time Freeze] (challenge, 8.31*, 92.5% in the simulator) it misses too much in some sections and
  osu!'s health bar drains to zero, so without NF the replay would end at the fail. `save_osr`
  adds NF to every replay so the whole play can be watched. Our simulator doesn't model health,
  so its results are for a full play: whether the bot would pass without NF isn't measured.

## For the report: analyses to run

- **Is demonstration quality worse on tech maps?** Tech maps are niche: fewer people play them,
  so their leaderboards are shallower and the replays we keep may come from weaker or fewer
  players than on popular stream and jump maps. That would confound "tech is hardest" (tech is the
  lowest style in every condition: 97.2% NoMod val, 69.2% under DT) with "tech has the worst
  teachers". Signs so far: tech was the only style that needed relaxed selection (2 maps up to
  7.0*; `relaxation` in configs/maps_v1.yaml), and tech misses most under transfer. To measure,
  per style, from data/raw/manifest.csv, configs/maps_v1.yaml and our judge:
  - usable replays per leaderboard (`available` in maps_v1.yaml) and play counts;
  - replay accuracy and 100 / 50 / miss counts (the manifest), and unstable rate (judge);
  - distinct players, how many hit the per-player cap, and the stable / lazer era mix;
  - whether a model trained on tech replays with matched quality (e.g. only the top 5 per map,
    or an accuracy floor matched across styles) closes part of the tech gap.
  Report it alongside the per-style results, either way.

## Pinned ideas (not run)

- **RX two-stage training:** stage 1 trains aim only and selects on play where a scripted Relax
  clicker presses the keys (so accuracy measures aim alone); stage 2 fine-tunes aim + keys.
  The clicker would also split any model's misses into aim vs timing.
- **Auto as a teacher, not as plain extra data.** Mixing Auto's replays in with players' would
  give the aim head two conflicting styles to average (Auto moves in straight lines and snaps
  exactly; players curve and cut corners). Better uses: DAgger (the model plays, Auto labels where
  to go from the states it actually reaches: the principled version of cursor jitter), spinner
  hold labels, or pre-training on the 499 RL maps before fine-tuning on players.
- Spinner windows drawn more often; mirroring every training spin to one direction.
- Spatial soft-argmax aim head; a larger aim loss weight.

## Reproduce

```
python -m data.prepare                                   # data + frames (see README, "For teammates")
python -m train.bc --config configs/baseline.yaml        # ~70 min on an RTX 4060 Ti
python -m eval.play --checkpoint runs/baseline_seed0/best.pt --beatmap 4912305 --footage mp4
python -m eval.transfer_matrix --checkpoint runs/baseline_seed0/best.pt --split val --mods NM HR DT HD HRDT
```
