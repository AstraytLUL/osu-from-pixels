"""The BC policy network: an encoder from models/backbones.py plus three small output heads.

Outputs, for the frame after the last one in the window (data/dataset.py's targets):
  move      (B, 2)  aim="relative": cursor displacement, in units of MOVE_SCALE osu! pixels (the
                    median step is ~7 px and the 99th percentile ~49 px, so targets are mostly
                    within [-1, 1])
  position  (B, 2)  aim="absolute" instead: where the cursor is next, as an offset from the playfield
                    centre in units of MOVE_SCALE (so within about [-5, 5] x [-4, 4])
  press     (B, 2)  logits: left / right pressed in that frame's window
  held      (B, 2)  logits: left / right down at the window's end
  heatmap   (B, 24, 32)  aim_head="softargmax" only: where the network wants the cursor, as
                    probabilities over a 24 x 32 grid of the playfield (16 osu! px per cell); for
                    analysis, the loss does not use it

Relative aim adds up its errors: each move is a little off, and the cursor drifts away from where
the objects are. Absolute aim names the spot itself, so an error in one frame is not carried into
the next.

How the position is named (aim_head, absolute aim only):
  linear      a linear layer on the feature vector (the baseline): the network squeezes the frames
              into `features` numbers and reads two coordinates back out of them
  softargmax  the network colours in a heatmap over the playfield and the position is the heatmap's
              centre of mass ("spatial soft-argmax"), so where to go never leaves image space. The
              heatmap comes from the backbone's 24 x 32 map plus its 6 x 8 map upsampled (local detail
              plus the context to tell *which* object is next). framestack backbone only.

prev_action (S2 in the project plan): the policy also gets its own last W actions (data/dataset.py
prev_actions: cursor moves and whether a key was pressed / held), through a small MLP whose output is
concatenated with the frame features before the heads. The frames show the cursor only to within a
few pixels and never the keys, so this is the information the pixels lack. Linear aim head only.
With prev_keys false the key bits are zeroed and only the cursor moves remain: 9 Nov, the full input
fell to ~73% NoMod val (98.5% without it) with a much lower press loss, the copycat failure (de Haan
et al. 2019), and moves-only tests whether the key history is what it copies.

build_policy(cfg) makes the network a config describes (training and checkpoint loading both use it).
"""
import torch
from torch import nn

from models.backbones import FrameStackCNN, build_backbone

MOVE_SCALE = 50.0              # osu! pixels per unit of the move / position output
CENTRE = (256.0, 192.0)        # playfield centre, osu! pixels: position output 0
GRID = (24, 32)                # soft-argmax heatmap cells (rows, columns): 16 osu! px each
AIM_HEADS = ("linear", "softargmax")
PREV_FEATURES = 64             # prev_action: size of the previous-action features


class BCPolicy(nn.Module):
    def __init__(self, window: int = 20, features: int = 512, cursor_channel: bool = False,
                 aim: str = "relative", backbone: str = "framestack", width: float = 1.0,
                 aim_head: str = "linear", prev_action: bool = False, prev_keys: bool = True):
        super().__init__()
        if aim not in ("relative", "absolute"):
            raise ValueError(f"aim must be 'relative' or 'absolute', not {aim!r}")
        if aim_head not in AIM_HEADS:
            raise ValueError(f"aim_head must be one of {AIM_HEADS}, not {aim_head!r}")
        self.aim, self.aim_head, self.prev_action, self.prev_keys = aim, aim_head, prev_action, prev_keys
        self.backbone = build_backbone(backbone, window, 2 if cursor_channel else 1, features, width)
        heads_in = features
        if prev_action:
            if aim_head != "linear":
                raise ValueError("prev_action needs the linear aim head")
            self.prev = nn.Sequential(nn.Linear(4 * window, 128), nn.ReLU(), nn.Linear(128, PREV_FEATURES), nn.ReLU())
            heads_in += PREV_FEATURES
        if aim_head == "softargmax":
            if aim != "absolute":
                raise ValueError("aim_head softargmax names a position: it needs aim: absolute")
            if not isinstance(self.backbone, FrameStackCNN):
                raise ValueError("aim_head softargmax needs the framestack backbone (its 24 x 32 map)")
            fine, coarse = self.backbone.map_channels
            self.heatmap = nn.Sequential(nn.Conv2d(fine + coarse, 64, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(64, 1, 1))
            # cell centres in osu! pixels, for the centre of mass
            ys = (torch.arange(GRID[0]) + 0.5) * (384.0 / GRID[0])
            xs = (torch.arange(GRID[1]) + 0.5) * (512.0 / GRID[1])
            self.register_buffer("cell_x", xs.repeat(GRID[0]))          # row-major, like flatten
            self.register_buffer("cell_y", ys.repeat_interleave(GRID[1]))
            # a flat heatmap at first: its centre of mass is the playfield centre (rest there)
            nn.init.zeros_(self.heatmap[-1].weight)
            nn.init.zeros_(self.heatmap[-1].bias)
        else:
            self.move = nn.Linear(heads_in, 2)    # the move or the position, depending on `aim`
            # Start as a player who waits: no movement (relative) or resting at the centre (absolute).
            # Otherwise an untrained policy drifts into a corner.
            nn.init.zeros_(self.move.weight)
            nn.init.zeros_(self.move.bias)
        self.press = nn.Linear(heads_in, 2)
        self.held = nn.Linear(heads_in, 2)
        # keys up at first (a press probability of ~5%, near the data's 3.8%), not a random key held
        nn.init.constant_(self.press.bias, -3.0)
        nn.init.constant_(self.held.bias, -1.0)

    def forward(self, frames: torch.Tensor, prev: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """prev: (B, 4W), data/dataset.py prev_actions; read only with prev_action."""
        if self.aim_head == "softargmax":
            fine, coarse, f = self.backbone.forward_maps(frames)
            up = nn.functional.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
            logits = self.heatmap(torch.cat([fine, up], dim=1)).flatten(1).float()   # (B, 768)
            p = logits.softmax(dim=1)
            # elementwise, not a matmul: autocast leaves it in float32 (positions need the precision)
            x, y = (p * self.cell_x).sum(dim=1), (p * self.cell_y).sum(dim=1)
            position = torch.stack([(x - CENTRE[0]) / MOVE_SCALE, (y - CENTRE[1]) / MOVE_SCALE], dim=1)
            return {"position": position, "press": self.press(f), "held": self.held(f),
                    "heatmap": p.view(-1, *GRID)}
        f = self.backbone(frames)
        if self.prev_action:
            if not self.prev_keys:                  # per frame (dx, dy, press, held): keep the moves
                prev = prev.view(len(prev), -1, 4) * prev.new_tensor([1.0, 1.0, 0.0, 0.0])
                prev = prev.reshape(len(prev), -1)
            f = torch.cat([f, self.prev(prev)], dim=1)
        aim = "move" if self.aim == "relative" else "position"
        return {aim: self.move(f), "press": self.press(f), "held": self.held(f)}


def build_policy(cfg: dict) -> BCPolicy:
    """The network a run config describes (train/bc.py's config.yaml, also stored in checkpoints).
    Missing keys mean the baseline's choices, so configs and checkpoints from before these options
    existed load as they were trained."""
    d, m = cfg["data"], cfg["model"]
    return BCPolicy(d["window"], m["features"], d.get("cursor_channel", False), m.get("aim", "relative"),
                    backbone=m.get("backbone", "framestack"), width=float(m.get("width", 1.0)),
                    aim_head=m.get("aim_head", "linear"), prev_action=m.get("prev_action", False),
                    prev_keys=m.get("prev_keys", True))
