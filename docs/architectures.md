# Architecture experiments: does how the network looks at the frames change how it transfers?

the project proposal lists "backbone choice" among the interventions tested for transfer. This page is the
plan, written before any results, and the place the results go. Everything except the network is
the BC baseline (configs/baseline.yaml, docs/bc_stage1.md): same data, cursor jitter, loss,
optimizer, epochs, click rule, motor and model selection. A difference in results is the
architecture's.

## Why look at architecture

The baseline (98.5% NoMod val) is near its ceiling on NoMod. Its transfer gaps are where the room
is: DT 86.4% (tech 69%), HD 18.8% (aim collapses too: a median 124 px from each object, vs 7 px
under NoMod), HRDT 67.7%. The baseline stacks the 20 frames as input channels of one 2D CNN, so
its first layer mixes all of time at once, at the speed it was trained on. Each experiment below
changes how the network handles time or space, and asks whether that moves a gap.

## The experiments

| Config | What changes | Question | Params |
| --- | --- | --- | --- |
| `configs/baseline.yaml` | nothing (reference); seeds 1, 2 added to Astrayt's seed 0 | | 3.4M |
| `configs/arch/cnn3d.yaml` | 3D CNN: convolutions over (time, y, x); time kept as an axis and halved step by step | does modelling local motion help under DT? | ~3.7M |
| `configs/arch/cnn_gru.yaml` | each frame encoded alone by one shared 2D CNN, then a GRU reads them in order | does memory help under HD, where objects fade before their hit time? | ~2.6M |
| `configs/arch/softargmax.yaml` | aim head: a heatmap over the playfield (24 x 32 cells), the position is its centre of mass | does pointing in image space aim better; what does the heatmap show under HD? | ~3.5M |
| `configs/arch/framestack_half.yaml` | baseline, half the channels | size control | ~1.6M |
| `configs/arch/framestack_double.yaml` | baseline, twice the channels | size control | ~7.3M |

Parameter counts are estimates; each run records the exact number in run.json, and
`python scripts/bench_backbones.py` prints them (and speed) without data.

Details that matter for a fair comparison:

- All backbones keep the baseline's stride-2 layers and x / y coordinate channels, and end in a
  flatten (not a global average), so none of them loses *where* things are by construction.
- cnn3d keeps every frame in its first layer: the approach circle shrinks by under a pixel per
  frame, and shrinking time early would blur that.
- cnn_gru's CNN never mixes frames: anything about time has to come from the GRU.
- softargmax uses the baseline's backbone. Its heatmap reads the 24 x 32 map (local detail) and the
  6 x 8 map upsampled (enough context to tell *which* object is next). Its output dict carries the
  heatmap (`out["heatmap"]`) for analysis; the loss is the same SmoothL1 on the position.
- The size controls separate "a better architecture" from "a bigger network": if
  framestack_double gains as much as cnn3d, cnn3d's gain is not evidence for the architecture.

## Hypotheses (written before results)

1. **cnn3d narrows the DT gap a little, not most of it.** It models change over a few frames rather
   than a fixed stacked picture, but every motion it has seen is still NoMod speed. If it closes
   most of the gap, temporal structure is enough; if not, DT needs data (time-warp augmentation),
   not architecture.
2. **cnn_gru helps HD at most a little.** Under HD an object is visible from ~600 ms to ~180 ms
   before its hit time, so it is inside the 333 ms window when it must be clicked: memory *could*
   carry it. But NoMod training never needs memory (objects stay visible), so the network has no
   reason to learn to use it. A gain would mean structure alone helps; no gain would mean HD needs
   HD data (Stage 3 in the project proposal).
3. **softargmax changes accuracy little, but explains the HD failure.** Under HD the heatmap
   should spread over several objects (it does not know which is next), not stay on one.
4. **Size does not explain the gaps:** framestack_half and framestack_double land within noise of
   the baseline under every mod.
5. NoMod stays within noise of 98.5% for every architecture (it is near the ceiling already).

## Protocol

- Train on NoMod + HR (flipped to NoMod geometry), exactly as the baseline.
- Model selection on NoMod val only (best.pt). Mod results are never used to choose anything.
- 3 seeds per config (0, 1, 2); report mean +- std. 9 val maps are noisy: one model's epochs differ
  by 5-10 points.
- Diagnostic: the transfer matrix on val maps for NM HR DT HD HRDT (`transfer_val.json`).
  Final numbers: the test split, run once per checkpoint after every experiment is done.
- Report per mod and per style the accuracy and the transfer gap, NoMod minus mod, and split the
  misses into aim (cursor error at hit time) and timing (hit offset, UR).

## Running

```
python -m pytest tests/test_backbones.py -q                   # contract tests, no data needed
python scripts/bench_backbones.py --device cuda               # size and speed, no data needed
python -m train.bc --config configs/arch/cnn3d.yaml --smoke   # a few minutes, with data: does it run?
bash scripts/run_experiment.sh configs/arch/cnn3d.yaml 0      # train + transfer_val.json + upload
```

`scripts/run_experiment.sh` resumes after an interruption (rerun the same command) and uploads the
run to the private model repo `sc4001-project/bc-<name>`, folder `seed<N>/` (train/hub.py). Get a
run back with `python -m train.hub pull sc4001-project/bc-cnn3d --seed 0`; the baseline's seed 0
with `python -m train.hub pull sc4001-project/bc-baseline --seed 0`.

Order, by priority: baseline seeds 1 and 2; cnn3d; cnn_gru; softargmax; size controls.

## Follow-ups (after the first 3-seed results; done)

The val results (docs/architectures_results.md, docs/figures/) left four open points. No new
architectures: the five above were pre-registered, and trying more on 9 val maps invites picking a
lucky one. Instead:

| Follow-up | Why | How |
| --- | --- | --- |
| Final results on the test maps | everything so far is val (diagnostic); the project proposal: test once, at the end | `eval/test_all.py` |
| CNN + GRU for 40 epochs (3 seeds), optionally the baseline for 40 (control) | its NoMod val accuracy was still rising at epoch 20, so its results may understate it | `configs/followup/` |
| 3D CNN seeds 3 and 4 | its DT gain (8.9 vs 11.9 points of gap) overlaps the baseline within one std | `scripts/run_experiment.sh configs/arch/cnn3d.yaml 3` |
| Aim vs timing | why CNN + GRU leads under HRDT but trails under DT; whether HD breaks aim or timing | `eval/diagnose.py` -> fig6, fig7 |
| Soft-argmax heatmaps, NM vs HD | hypothesis 3: does the network still know *which* object is next without approach circles? | `eval/heatmaps.py` -> fig8, fig9 |

```
python -m eval.test_all prepare                                  # render test (and val) frames, all mods
bash scripts/run_experiment.sh configs/followup/cnn_gru_e40.yaml 0   # and seeds 1, 2
bash scripts/run_experiment.sh configs/arch/cnn3d.yaml 3             # and 4
CUDA_VISIBLE_DEVICES=0 python -m eval.test_all run --lane 0/2    # every run on the Hub, test maps
python -m eval.diagnose --configs baseline cnn3d cnn_gru cnn_gru_e40 --mods NM DT HD HRDT
python -m eval.heatmaps
python -m eval.compare_runs --split test --markdown docs/architectures_results_test.md
python -m eval.report_figures && python -m eval.report_figures --split test
```

## Results

**Status (Nov 1): done.** Every config has 3 seeds (cnn3d 4: seeds 0, 1, 2, 4; seed 3 is missing;
baseline_e40 1, as a control only). Final numbers are on the **test** maps, run once after everything
else was fixed. Full tables: [architectures_results_test.md](architectures_results_test.md) (test) and
[architectures_results.md](architectures_results.md) (val, the diagnostic). Raw diagnostics:
[diagnostics/](diagnostics/). Figures: [figures/](figures/).

### In short

1. **No architecture fixes transfer.** NoMod is 98.4-99.4% for all; HD collapses for all (9-18%).
2. **The 3D CNN's DT gain on val did not hold on test** (val 87.9 vs 86.6; test 87.9 vs 89.0). It
   trades timing for aim: under DT it halves timing misses (8.4% -> 4.0% of circles) and doubles aim
   misses (1.3% -> 2.6%).
3. **CNN + GRU is the best under HR + DT, on both splits and every seed**, and the gain is timing:
   test 80.6 ± 0.6 (40 epochs) / 77.6 ± 4.8 (20) vs 74.5 ± 2.6. Under HRDT its circles missed with the
   cursor on them drop from 12.6% to 3.9%, and 100s + 50s from 39.5% to 23%. Under plain DT it is
   *worse* (85.1 vs 89.0), mostly from aim.
4. **Fitting NoMod harder hurts HR + DT, except for the GRU.** The baseline at twice the width or 40
   epochs gains on NoMod (99.4, 99.2) and loses on HRDT (65.6, 64.5); the GRU at 40 epochs gains on both.
5. **HD breaks aim, not mainly timing (hypothesis 3 rejected).** Under HD 92-94% of circles are missed
   with the cursor off them at their hit time (median 2.8 radii away), clicks halve (0.5 per object),
   and nearly every circle is missed (98-99%; the remaining accuracy is slider and spinner credit).
   Open-loop with Auto's cursor, the soft-argmax heatmap moves from the next circle to later ones
   (target mass 0.56 -> 0.36, later objects 0.06 -> 0.20) while getting *more* peaked (entropy 4.47
   -> 3.93 bits): without approach circles the network confidently picks the wrong object. The
   approach circle tells it *which* object is next, not only *when*; closed-loop, a wrong pick puts
   the cursor somewhere unfamiliar and the error compounds.

### Test maps (final), mean ± std over seeds

| Config | Seeds | Params | NM | HR | DT | HD | HRDT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3 | 3.41M | 99.0 ± 0.3 | 98.6 ± 0.1 | 89.0 ± 1.2 | 14.0 ± 2.3 | 74.5 ± 2.6 |
| framestack_half | 3 | 1.65M | 98.6 ± 0.3 | 98.7 ± 0.3 | 88.0 ± 1.2 | 18.7 ± 0.3 | 69.6 ± 2.5 |
| framestack_double | 3 | 7.29M | 99.4 ± 0.1 | 98.3 ± 0.2 | 90.7 ± 2.5 | 18.0 ± 2.1 | 65.6 ± 5.1 |
| cnn3d | 4 | 3.66M | 98.5 ± 0.3 | 97.7 ± 0.3 | 87.9 ± 2.7 | 17.1 ± 4.9 | 72.4 ± 4.3 |
| cnn_gru | 3 | 2.62M | 98.4 ± 0.3 | 98.0 ± 0.5 | 83.5 ± 3.6 | 11.0 ± 3.4 | 77.6 ± 4.8 |
| cnn_gru_e40 | 3 | 2.62M | 98.6 ± 0.4 | 97.9 ± 0.3 | 85.1 ± 1.4 | 9.2 ± 3.5 | **80.6 ± 0.6** |
| softargmax | 3 | 3.52M | 99.2 ± 0.3 | 98.9 ± 0.2 | 89.6 ± 2.5 | 12.0 ± 2.4 | 71.6 ± 3.0 |
| baseline_e40 | 1 | 3.41M | 99.2 | 98.7 | 86.9 | 18.2 | 64.5 |

Val (the same runs) for the claims that changed between splits: DT, baseline 86.6 ± 2.6 vs cnn3d
87.9 ± 2.6; HRDT, baseline 71.2 ± 3.4 vs cnn_gru 78.8 ± 3.8 vs cnn_gru_e40 81.1 ± 0.5.

### Aim or timing (val maps, circles only, mean over seeds; eval/diagnose.py, fig6, fig7)

"Aim": missed with the cursor off the circle at its hit time and never clicked on it. "Timing":
missed with the cursor on it (wrong moment or no click), or clicked outside the hit window. Offset:
mean click time minus circle time for hit circles, map ms (< 0 early).

| Config | Mod | Aim miss | Timing miss | 100 + 50 | Offset | Clicks / object |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | NM | 0.1% | 0.3% | 1.6% | -14.9 | 1.01 |
| baseline | DT | 1.3% | 8.4% | 9.4% | +13.3 | 0.95 |
| baseline | HD | 91.8% | 6.5% | 1.3% | +37.0 | 0.53 |
| baseline | HRDT | 3.0% | 12.6% | 39.5% | +20.3 | 0.91 |
| cnn3d | DT | 2.6% | 4.0% | 8.7% | +14.0 | 0.99 |
| cnn3d | HRDT | 4.8% | 5.4% | 42.7% | +20.5 | 0.99 |
| cnn_gru | DT | 6.0% | 7.0% | 10.0% | +13.2 | 0.96 |
| cnn_gru | HRDT | 5.2% | 3.9% | 29.3% | +15.2 | 0.99 |
| cnn_gru_e40 | DT | 5.2% | 6.0% | 9.5% | +11.0 | 0.97 |
| cnn_gru_e40 | HRDT | 5.5% | 3.9% | 22.6% | +13.5 | 0.99 |

Every model clicks ~15 ms early under NoMod and ~11-14 ms late under DT: a timing learned in
frames, stretched by DT's longer frames (hypothesis 2's mechanism). The NoMod bias is the same for
all architectures (possibly from the press labels' ±8 ms frame window); it does not affect the
comparison.

### Soft-argmax heatmaps (val maps, 3 seeds, 2,597 circles each; eval/heatmaps.py, fig8, fig9)

Open-loop: the window ends 2 frames after the previous object (median 117 ms before the hit), with
Auto's cursor drawn in, so NM and HD differ only in what is on screen.

| | Mass on the target | on later objects | on the previous object | Entropy (bits) | Peak on target |
| --- | --- | --- | --- | --- | --- |
| NM | 0.56 | 0.06 | 0.32 | 4.47 | 47% |
| HD | 0.36 | 0.20 | 0.33 | 3.93 | 41% |

### The hypotheses

| # | Hypothesis | Outcome |
| --- | --- | --- |
| 1 | cnn3d narrows the DT gap a little | **Not supported.** Suggestive on val, gone on test; timing gains cancelled by aim losses |
| 2 | cnn_gru helps HD at most a little | **Supported** (it is lower than the baseline) |
| 3 | HD hurts timing more than aim; the heatmap spreads | **Rejected.** Aim misses dominate; the heatmap does not spread, it moves to later objects |
| 4 | Size does not explain the gaps | **Supported.** More width (and more epochs) lowers HRDT |
| 5 | NoMod within noise for all | **Supported** (98.4-99.4% on test) |
| - | not predicted | CNN + GRU leads under HRDT through timing, on both splits and all seeds |

### Figures

| File | Shows |
| --- | --- |
| fig1_accuracy_by_mod(_test) | every config under every mod, seeds and mean ± std |
| fig2_style_breakdown(_test) | NM / DT / HRDT per map style for baseline, cnn3d, cnn_gru |
| fig3_per_map(_test) | per map: cnn3d vs baseline under DT, cnn_gru vs baseline under HRDT |
| fig4_size_control(_test) | accuracy against parameters: width does not explain the gains |
| fig5_training_curves | NoMod val per epoch; cnn_gru still rising at 20, settled by ~25 |
| fig6_miss_breakdown | aim vs timing misses per mod |
| fig7_click_offsets | click time relative to the circle, NM / DT / HRDT |
| fig8_heatmap_examples | the same moments under NM and HD, heatmap over the frame |
| fig9_heatmap_summary | where the heatmap's mass goes, NM vs HD |

### Caveats

- 9 maps per split (3 per style): the 3D CNN result shows how much one set of 9 maps can mislead.
- cnn3d has 4 seeds (seed 3 did not finish); baseline_e40 has 1, so it is a control, not a result.
- The aim / timing and heatmap analyses are on the val maps; the HRDT timing explanation for the
  GRU is an interpretation of those numbers, not a tested mechanism.
- All results are in our simulator (validated against real replays; README "Simulator"), not the game.

### Reproduce

```
python -c "from huggingface_hub import snapshot_download as s; [s('sc4001-project/' + r, local_dir='runs/hub/' + r, allow_patterns=['*.json', '*.jsonl', '*.yaml', '*.log']) for r in ['bc-baseline', 'bc-framestack-half', 'bc-framestack-double', 'bc-cnn3d', 'bc-cnn-gru', 'bc-cnn-gru-e40', 'bc-baseline-e40', 'bc-softargmax']]"
python -m eval.compare_runs --split test                     # the tables, from runs/hub/
python -m eval.report_figures --diagnostics docs/diagnostics && python -m eval.report_figures --split test
```

Retraining and re-evaluating everything: "Running" and "Follow-ups" above (about 25 GPU hours on
two RTX PRO 4500s).
