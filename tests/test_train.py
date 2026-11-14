import numpy as np
import pytest
import torch

from data.dataset import PREV_SCALE, ReplayWindows, prev_actions
from eval.policy import AutoPolicy, ModelPolicy, load_model_policy
from eval.rollout import evaluate, play
from models.heads import BCPolicy
from osu_sim.auto import auto_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.judge import accuracy
from osu_sim.mods import DT, HR
from osu_sim.playable import make_playable
from train.bc import bc_loss, fit


def small_map(mods=0):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(100, 100, 2000, "circle", True),
            HitObject(300, 120, 2400, "slider", False, curve_type="L", curve_points=[(300, 120), (400, 120)],
                      slides=2, length=100),
            HitObject(200, 300, 3600, "circle", False)]
    return make_playable(Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], objs), mods)


def config(cursor_channel=False, **train):
    cfg = {"name": "test", "seed": 0,
           "data": {"train_mods": ["NM"], "canonical_hr": True, "window": 20, "cursor_channel": cursor_channel},
           "model": {"features": 64},
           "train": {"batch_size": 16, "epochs": 1, "samples_per_epoch": 32, "lr": 1e-3, "weight_decay": 0.0,
                     "move_weight": 1.0, "press_weight": 1.0, "held_weight": 1.0, "press_pos_weight": 10.0,
                     "device": "cpu", "num_workers": 0},
           "eval": {"every": 1, "val_maps": None, "press_threshold": 0.5}}
    cfg["train"].update(train)
    return cfg


@pytest.mark.parametrize("mods", [0, HR, DT])
def test_auto_plays_ss_through_the_rollout(tmp_path, mods):
    pl = small_map(mods)
    g, js = play(AutoPolicy(), 42, pl, root=tmp_path)
    assert accuracy(js) == 1.0
    assert g.press.sum() >= 3                      # it clicked; the cursor came from Auto, not START


def test_untrained_model_waits():
    model = BCPolicy(window=20, features=64)
    out = model(torch.zeros(2, 20, 96, 128, dtype=torch.uint8))
    assert out["move"].abs().max() == 0
    assert torch.sigmoid(out["press"]).max() < 0.1 and torch.sigmoid(out["held"]).max() < 0.5


@pytest.mark.parametrize("cursor_channel", [False, True])
def test_model_policy_plays_a_map(tmp_path, cursor_channel):
    model = BCPolicy(window=20, features=64, cursor_channel=cursor_channel)
    policy = ModelPolicy(model, window=20, cursor_channel=cursor_channel)
    report = evaluate(policy, [(42, "jumps", small_map(HR))], root=tmp_path)
    row = report["maps"][0]
    assert row["auto_accuracy"] == 1.0 and row["count_miss"] == 3      # untrained: never clicks
    assert row["gap_pp"] == pytest.approx(100.0)


def test_clicks_are_rising_edges_with_alternating_keys():
    policy = ModelPolicy(BCPolicy(window=20, features=64), window=20)
    policy.reset(small_map(), None)
    # press probability over 6 frames: above the threshold for 3 frames, then again later
    presses, helds = zip(*(policy._click(p, h) for p, h in
                           [(0.2, 0.1), (0.7, 0.9), (0.8, 0.9), (0.6, 0.2), (0.1, 0.1), (0.9, 0.1)]))
    assert presses == (0, 1, 0, 0, 0, 2)          # one click per crossing; left, then right
    assert helds == (0, 1, 1, 0, 0, 2)            # left stays down while held, released after


def test_held_dips_above_the_release_threshold_keep_the_key_down():
    policy = ModelPolicy(BCPolicy(window=20, features=64), window=20, release_below=0.3)
    policy.reset(small_map(), None)
    helds = [policy._click(p, h)[1] for p, h in [(0.7, 0.9), (0.1, 0.45), (0.1, 0.6), (0.1, 0.2)]]
    assert helds == [1, 1, 1, 0]                  # the dip to 0.45 doesn't drop the slider; 0.2 does


def test_a_held_probability_can_put_a_key_down_without_a_click():
    policy = ModelPolicy(BCPolicy(window=20, features=64), window=20, hold_starts_key=True)
    policy.reset(small_map(), None)
    # a spinner: the press head never crosses its threshold, but "held" is high throughout
    out = [policy._click(p, h) for p, h in [(0.1, 0.2), (0.3, 0.8), (0.3, 0.8), (0.3, 0.1)]]
    assert out == [(0, 0), (1, 1), (0, 1), (0, 0)]   # one press, held, then released


def test_untrained_absolute_model_rests_at_the_centre(tmp_path):
    policy = ModelPolicy(BCPolicy(window=20, features=64, aim="absolute"), window=20)
    g, _ = play(policy, 42, small_map(), root=tmp_path)
    assert np.allclose(g.x[1:], 256) and np.allclose(g.y[1:], 192)


def test_absolute_position_is_flipped_back_on_hr():
    model = BCPolicy(window=20, features=64, aim="absolute")
    torch.nn.init.constant_(model.move.bias, 1.0)          # always (256 + 50, 192 + 50) in NoMod geometry
    policy = ModelPolicy(model, window=20)
    layers, cursors = np.zeros((20, 96, 128), np.uint8), np.full((20, 2), 100.0)
    policy.reset(small_map(), None)
    assert policy.act(0, layers, cursors)[:2] == pytest.approx((306, 242))
    policy.reset(small_map(HR), None)
    assert policy.act(0, layers, cursors)[:2] == pytest.approx((306, 384 - 242))


@pytest.mark.parametrize("aim", ["relative", "absolute"])
def test_model_can_memorise_one_batch(tmp_path, aim):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path)
    keys = ("frames", "action", "target", "press", "held")
    samples = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 16))]
    batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys}
    torch.manual_seed(0)
    model = BCPolicy(window=20, features=64, aim=aim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = config()
    first = bc_loss(model(batch["frames"]), batch, cfg)[0].item()
    for _ in range(60):
        opt.zero_grad()
        loss = bc_loss(model(batch["frames"]), batch, cfg)[0]
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


def test_fit_writes_checkpoints_and_resumes(tmp_path):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path / "cache")
    val = [(42, "jumps", pl)]
    out = tmp_path / "run"
    best = fit(config(), ds, val, out, frame_root=tmp_path / "cache")
    assert 0.0 <= best <= 1.0
    assert (out / "last.pt").exists() and (out / "best.pt").exists() and (out / "metrics.jsonl").exists()
    with pytest.raises(SystemExit):
        fit(config(), ds, val, out, frame_root=tmp_path / "cache")                 # would overwrite the run
    with pytest.raises(SystemExit):
        fit(config(lr=1e-2), ds, val, out, resume=True, frame_root=tmp_path / "cache")   # not the same run
    assert fit(config(), ds, val, out, resume=True, frame_root=tmp_path / "cache") == best   # nothing left to do


def test_fit_with_the_cursor_channel(tmp_path):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path / "cache", cursor_channel=True)
    fit(config(cursor_channel=True), ds, [(42, "jumps", pl)], tmp_path / "run", frame_root=tmp_path / "cache")
    policy = load_model_policy(tmp_path / "run" / "last.pt")
    assert policy.cursor_channel and policy.model.backbone.convs[0].in_channels == 42


def test_fit_with_absolute_aim(tmp_path):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path / "cache")
    cfg = config()
    cfg["model"]["aim"] = "absolute"
    fit(cfg, ds, [(42, "jumps", pl)], tmp_path / "run", frame_root=tmp_path / "cache")
    assert load_model_policy(tmp_path / "run" / "last.pt").model.aim == "absolute"


def test_prev_actions_are_moves_and_any_key_bits():
    cursors = np.array([[-1000, -1000], [100, 100], [110, 90], [110, 90]], dtype=float)
    prev = prev_actions(cursors, np.array([0, 0, 1, 0]), np.array([0, 0, 1, 2])).reshape(4, 4)
    assert prev[:2].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]          # nothing before; off-screen step is 0
    assert prev[2].tolist() == pytest.approx([10 / PREV_SCALE, -10 / PREV_SCALE, 1, 1])
    assert prev[3].tolist() == [0, 0, 0, 1]                            # right key held counts as held


def test_fit_and_play_with_the_previous_action_input(tmp_path):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path / "cache")
    assert ds[0]["prev"].shape == (80,)
    cfg = config()
    cfg["model"].update(aim="absolute", prev_action=True)
    fit(cfg, ds, [(42, "jumps", pl)], tmp_path / "run", frame_root=tmp_path / "cache")
    policy = load_model_policy(tmp_path / "run" / "last.pt")
    assert policy.model.prev_action and policy.model.press.in_features == 64 + 64
    _, js = play(policy, 42, small_map(HR), root=tmp_path / "cache")   # its own actions fed back
    assert len(js) == 3 and len(policy.key_history) == 20


def test_prev_action_input_changes_the_output():
    torch.manual_seed(0)
    model = BCPolicy(window=20, features=64, aim="absolute", prev_action=True)
    torch.nn.init.normal_(model.move.weight)                 # the head starts at zero
    frames = torch.zeros(1, 20, 96, 128, dtype=torch.uint8)
    a = model(frames, torch.zeros(1, 80))["position"]
    b = model(frames, torch.ones(1, 80))["position"]
    assert not torch.allclose(a, b)


def test_moves_only_previous_action_ignores_the_key_bits():
    torch.manual_seed(0)
    model = BCPolicy(window=20, features=64, aim="absolute", prev_action=True, prev_keys=False)
    torch.nn.init.normal_(model.move.weight)
    frames = torch.zeros(1, 20, 96, 128, dtype=torch.uint8)
    keys = torch.zeros(1, 20, 4)
    keys[..., 2:] = 1                                        # every key bit on, no moves
    moves = torch.zeros(1, 20, 4)
    moves[..., :2] = 1
    out = lambda prev: model(frames, prev.view(1, 80))["position"]
    assert torch.allclose(out(torch.zeros(1, 20, 4)), out(keys))
    assert not torch.allclose(out(torch.zeros(1, 20, 4)), out(moves))
