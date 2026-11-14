"""The architecture experiments (models/backbones.py, models/heads.py, docs/architectures.md): every
backbone and aim head must fit the same contract as the baseline, train, save, load and play."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data.dataset import ReplayWindows
from eval.policy import ModelPolicy, load_model_policy
from eval.rollout import play
from models.backbones import BACKBONES, FrameStackCNN, build_backbone, count_parameters
from models.heads import GRID, BCPolicy, build_policy
from osu_sim.auto import auto_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.mods import HR
from osu_sim.playable import make_playable
from train.bc import apply_smoke, bc_loss, fit

ARCH_CONFIGS = sorted(Path("configs/arch").glob("*.yaml"))
VARIANTS = [{"backbone": b} for b in BACKBONES] + [{"backbone": "framestack", "aim_head": "softargmax"},
                                                   {"backbone": "framestack", "width": 0.5}]


def small_map(mods=0):
    diff = Difficulty(hp=5, cs=4, od=8, ar=9, slider_multiplier=1.0, slider_tick_rate=1)
    objs = [HitObject(100, 100, 2000, "circle", True),
            HitObject(300, 120, 2400, "slider", False, curve_type="L", curve_points=[(300, 120), (400, 120)],
                      slides=2, length=100),
            HitObject(200, 300, 3600, "circle", False)]
    return make_playable(Beatmap(diff, 0.7, [TimingPoint(0, 500.0, True)], objs), mods)


def config(cursor_channel=False, **model):
    return {"name": "test", "seed": 0,
            "data": {"train_mods": ["NM"], "canonical_hr": True, "window": 20, "cursor_channel": cursor_channel},
            "model": {"features": 64, "aim": "absolute", **model},
            "train": {"batch_size": 16, "epochs": 1, "samples_per_epoch": 32, "lr": 1e-3, "weight_decay": 0.0,
                      "move_weight": 1.0, "press_weight": 1.0, "held_weight": 1.0, "press_pos_weight": 10.0,
                      "device": "cpu", "num_workers": 0},
            "eval": {"every": 1, "val_maps": None, "press_threshold": 0.5}}


@pytest.mark.parametrize("name", BACKBONES)
@pytest.mark.parametrize("cursor_channel", [False, True])
def test_backbone_maps_a_window_to_one_feature_vector(name, cursor_channel):
    net = build_backbone(name, window=20, frame_channels=2 if cursor_channel else 1, features=64)
    x = torch.randint(0, 256, (3, 40 if cursor_channel else 20, 96, 128), dtype=torch.uint8)
    assert net(x).shape == (3, 64)


def test_default_backbone_is_the_baseline_and_its_state_dict_is_unchanged():
    old_cfg = config()
    del old_cfg["model"]["aim"]                         # a config from before these options existed
    model = build_policy(old_cfg)
    assert isinstance(model.backbone, FrameStackCNN) and model.aim == "relative"
    # the baseline's parameter names, so checkpoints trained before this change still load
    assert set(model.state_dict()) == set(BCPolicy(20, 64).state_dict())


def test_temporal_backbones_see_the_order_of_frames():
    """cnn3d and cnn_gru treat the window as a sequence: reversing time changes their features."""
    x = torch.randint(0, 256, (2, 20, 96, 128), dtype=torch.uint8)
    for name in ("cnn3d", "cnn_gru"):
        torch.manual_seed(0)
        net = build_backbone(name, 20, 1, 64)
        assert not torch.allclose(net(x), net(x.flip(1)))


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(map(str, v.values())))
def test_untrained_policy_rests_at_the_centre_with_keys_up(variant):
    model = build_policy(config(**variant))
    out = model(torch.randint(0, 256, (2, 20, 96, 128), dtype=torch.uint8))
    assert torch.allclose(out["position"], torch.zeros(2, 2), atol=1e-5)
    assert torch.sigmoid(out["press"]).max() < 0.1 and torch.sigmoid(out["held"]).max() < 0.5


def test_softargmax_heatmap_is_a_distribution_and_its_centre_is_the_position():
    model = build_policy(config(aim_head="softargmax"))
    torch.nn.init.normal_(model.heatmap[-1].weight)                 # a non-flat heatmap
    out = model(torch.randint(0, 256, (2, 20, 96, 128), dtype=torch.uint8))
    p = out["heatmap"]
    assert p.shape == (2, *GRID) and torch.allclose(p.sum(dim=(1, 2)), torch.ones(2))
    # centre of mass by hand, from cell centres 16 px apart starting at 8 px
    ys, xs = (torch.arange(GRID[0]) * 16 + 8.0), (torch.arange(GRID[1]) * 16 + 8.0)
    x = (p.sum(dim=1) * xs).sum(dim=1)
    y = (p.sum(dim=2) * ys).sum(dim=1)
    assert torch.allclose(out["position"], torch.stack([(x - 256) / 50, (y - 192) / 50], dim=1), atol=1e-4)


def test_softargmax_needs_absolute_aim_and_the_framestack_backbone():
    with pytest.raises(ValueError):
        build_policy(config(aim="relative", aim_head="softargmax"))
    with pytest.raises(ValueError):
        build_policy(config(backbone="cnn3d", aim_head="softargmax"))
    with pytest.raises(ValueError):
        build_policy(config(backbone="resnet"))


def test_width_scales_the_network():
    sizes = [count_parameters(build_policy(config(width=w))) for w in (0.5, 1.0, 2.0)]
    assert sizes[0] < sizes[1] < sizes[2]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(map(str, v.values())))
def test_policy_can_memorise_one_batch(tmp_path, variant):
    """Learning at all: a broken architecture (a wrong reshape, a dead head) cannot fit 16 windows.
    It asks whether, not how fast: up to 300 steps, stopping once the loss is below 30% of where it
    started (the baseline needs well under 80; cnn_gru, carrying frames through its memory, more)."""
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path)
    keys = ("frames", "action", "target", "press", "held")
    samples = [ds[i] for i in range(0, len(ds), max(1, len(ds) // 16))]
    batch = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys}
    torch.manual_seed(0)
    cfg = config(**variant)
    model = build_policy(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = bc_loss(model(batch["frames"]), batch, cfg)[0].item()
    for _ in range(300):
        opt.zero_grad()
        loss = bc_loss(model(batch["frames"]), batch, cfg)[0]
        if loss.item() < 0.3 * first:
            break
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(map(str, v.values())))
def test_fit_saves_a_checkpoint_that_loads_as_the_same_architecture_and_plays(tmp_path, variant):
    pl = small_map()
    ds = ReplayWindows([(42, pl, auto_replay(pl))], window=20, root=tmp_path / "cache")
    fit(config(**variant), ds, [(42, "jumps", pl)], tmp_path / "run", frame_root=tmp_path / "cache")
    run_json = (tmp_path / "run" / "run.json").read_text()
    assert '"parameters"' in run_json and (tmp_path / "run" / "train.log").exists()
    policy = load_model_policy(tmp_path / "run" / "best.pt")
    assert type(policy.model.backbone) is type(build_policy(config(**variant)).backbone)
    _, judgements = play(policy, 42, small_map(HR), root=tmp_path / "cache")     # HR: the flip path too
    assert len(judgements) == 3


def test_model_policy_flips_a_softargmax_position_back_on_hr():
    torch.manual_seed(0)
    model = build_policy(config(aim_head="softargmax"))
    torch.nn.init.normal_(model.heatmap[-1].weight)          # a heatmap away from the centre
    policy = ModelPolicy(model, window=20)
    # an empty map and the cursor on the horizontal centre line: the flipped view is the same view,
    # so the network names the same spot both times, and only the policy's flip back differs
    layers, cursors = np.zeros((20, 96, 128), np.uint8), np.tile([100.0, 192.0], (20, 1))
    policy.reset(small_map(), None)
    x0, y0 = policy.act(0, layers, cursors)[:2]
    policy.reset(small_map(HR), None)
    x1, y1 = policy.act(0, layers, cursors)[:2]
    assert x1 == pytest.approx(x0) and y1 == pytest.approx(384 - y0)


def test_smoke_shrinks_a_copy_of_the_config():
    cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    small = apply_smoke(cfg)
    assert small["name"] == "baseline_smoke" and small["train"]["epochs"] == 2 and small["eval"]["val_maps"] == 2
    assert cfg["train"]["epochs"] == 20 and cfg["name"] == "baseline"          # the original is untouched


@pytest.mark.parametrize("path", ARCH_CONFIGS, ids=lambda p: p.stem)
def test_arch_configs_differ_from_the_baseline_only_in_name_and_model(path):
    base = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    cfg = yaml.safe_load(path.read_text())
    assert cfg["name"] == path.stem
    for key in base:
        if key not in ("name", "model"):
            assert cfg[key] == base[key], key
    assert {k: v for k, v in cfg["model"].items() if k not in ("backbone", "width", "aim_head")} == base["model"]
    build_policy(cfg)                                                            # and it builds


@pytest.mark.parametrize("path", sorted(Path("configs/arch_cap").glob("*.yaml")), ids=lambda p: p.stem)
def test_cap_variants_are_the_arch_configs_selected_like_baseline_cap(path):
    arch = yaml.safe_load((Path("configs/arch") / path.name).read_text())
    cfg = yaml.safe_load(path.read_text())
    base, base_cap = (yaml.safe_load(Path(f"configs/{n}.yaml").read_text()) for n in ("baseline", "baseline_cap"))
    assert cfg["name"] == f"{path.stem}_cap"
    assert cfg["eval"] == base_cap["eval"] and arch["eval"] == base["eval"]
    assert {k: v for k, v in cfg.items() if k not in ("name", "eval")} == \
           {k: v for k, v in arch.items() if k not in ("name", "eval")}
