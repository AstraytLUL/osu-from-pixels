"""Image encoders: a window of frames -> one feature vector. The architecture experiments
(docs/architectures.md) swap these and keep everything else fixed.

Every backbone takes the observation of data/dataset.py, (B, frame_channels * window, 96, 128) uint8:
the window's frames oldest first, then (with cursor_channel) the window's cursor images. It returns
(B, features) float. build_backbone() makes one by name, from the config's model.backbone.

    framestack  FrameStackCNN, the baseline: every frame is an input channel of one 2D CNN, so time is
                mixed in the first layer ("early fusion"). `width` scales its channels (size control).
    cnn3d       Conv3DNet: 3D convolutions over (time, y, x); time is kept as an axis and shrunk step
                by step, so the network sees how a patch changes over a few frames.
    cnn_gru     FrameGRU: one small 2D CNN looks at each frame on its own (shared weights), and a GRU
                reads the per-frame summaries in order ("late fusion" with memory).

Two choices of the baseline are kept in all of them, since they matter at our resolution:
- stride 2 everywhere, not the Atari 8x8 stride-4 first layer: the timing signal is the approach
  circle shrinking by under a pixel per frame (AR9 at 128x96), and the cursor dot is ~3 px wide;
- two extra input channels holding each pixel's x and y ("CoordConv"), and a flatten instead of a
  global average at the end, so the features keep *where* things are, which aiming needs.
"""
import torch
from torch import nn

from osu_sim.renderer import H, W


def _coords() -> torch.Tensor:
    """(2, H, W): x and y of every pixel in [-1, 1]."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    return torch.stack([xs, ys])


def _split_frames(frames: torch.Tensor, window: int, frame_channels: int) -> torch.Tensor:
    """(B, frame_channels * window, H, W) uint8 -> (B, window, frame_channels, H, W) float in [0, 1].
    The observation stores the window's map frames first, then (cursor_channel) its cursor frames."""
    x = frames.float() / 255.0
    return x.view(len(x), frame_channels, window, H, W).transpose(1, 2)


class FrameStackCNN(nn.Module):
    """(B, images, 96, 128) uint8 -> (B, features) float. `images` is the window (20), or twice that
    with the cursor on its own channels (data/dataset.py observation). `width` multiplies the number
    of channels of every layer (1 = the baseline; its state dict is unchanged)."""

    def __init__(self, images: int = 20, features: int = 512, width: float = 1.0):
        super().__init__()
        c1, c2, c3, c4 = (max(1, round(c * width)) for c in (32, 64, 128, 128))
        self.register_buffer("coords", _coords()[None])                      # (1, 2, H, W)
        self.convs = nn.Sequential(
            nn.Conv2d(images + 2, c1, 5, stride=2, padding=2), nn.ReLU(),     # -> 48 x 64
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),             # -> 24 x 32
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),             # -> 12 x 16
            nn.Conv2d(c3, c4, 3, stride=2, padding=1), nn.ReLU(),             # -> 6 x 8
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c4 * 6 * 8, features), nn.ReLU())
        self.map_channels = (c2, c4)     # channels of the 24 x 32 and 6 x 8 maps (forward_maps)

    def _input(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.float() / 255.0
        return torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.fc(self.convs(self._input(frames)))

    def forward_maps(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The 24 x 32 map (after the 2nd conv), the 6 x 8 map (after the 4th) and the features, for
        heads that point at a spot on the playfield (models/heads.py, aim_head: softargmax)."""
        x = self._input(frames)
        fine = self.convs[:4](x)
        coarse = self.convs[4:](fine)
        return fine, coarse, self.fc(coarse)


class Conv3DNet(nn.Module):
    """3D CNN: convolutions slide over time as well as over the image.

    (B, window, C + 2, H, W) is seen as a short video with C + 2 channels per frame (the frame, plus
    x / y). The first layer keeps every frame (the approach circle moves under a pixel per frame;
    shrinking time early would blur that away); the next three halve time, height and width:
    20 x 48 x 64 -> 10 x 24 x 32 -> 5 x 12 x 16 -> 3 x 6 x 8, then flatten and a linear layer as in
    the baseline. Channels (16, 32, 48, 48) keep the parameter count near the baseline's (~3.7M vs
    3.4M); it costs about 4x the baseline's compute per window."""

    def __init__(self, window: int = 20, frame_channels: int = 1, features: int = 512, width: float = 1.0):
        super().__init__()
        self.window, self.frame_channels = window, frame_channels
        c1, c2, c3, c4 = (max(1, round(c * width)) for c in (16, 32, 48, 48))
        self.register_buffer("coords", _coords()[None, :, None])             # (1, 2, 1, H, W)
        self.convs = nn.Sequential(
            nn.Conv3d(frame_channels + 2, c1, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)), nn.ReLU(),
            nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(c3, c4, 3, stride=2, padding=1), nn.ReLU(),
        )
        t = window
        for _ in range(3):
            t = (t + 1) // 2                     # stride 2, kernel 3, padding 1: ceil(t / 2)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c4 * t * 6 * 8, features), nn.ReLU())

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = _split_frames(frames, self.window, self.frame_channels).transpose(1, 2)   # (B, C, T, H, W)
        b, _, t = x.shape[:3]
        x = torch.cat([x, self.coords.expand(b, -1, t, -1, -1)], dim=1)
        return self.fc(self.convs(x))


class FrameGRU(nn.Module):
    """Per-frame CNN + GRU: look at each frame on its own, then read the frames in order.

    The same small 2D CNN (the frame plus x / y channels; 16, 32, 64, 64 channels, the baseline's
    strides) turns each frame into a vector of `embed` numbers; a GRU reads the window's vectors
    oldest first, and its last hidden state (`features` numbers) is the feature vector. The CNN never
    mixes frames, so everything about time (what moved, what was there and has faded) is up to the
    GRU's memory. About 2.6M parameters; about 3x the baseline's compute per window, since the CNN
    runs once per frame. It fits a tiny batch more slowly than the baseline (it must carry what it
    saw through 20 steps; tests/test_backbones.py), and a linear + ReLU after the GRU made that slower."""

    def __init__(self, window: int = 20, frame_channels: int = 1, features: int = 512, embed: int = 384,
                 width: float = 1.0):
        super().__init__()
        self.window, self.frame_channels = window, frame_channels
        c1, c2, c3, c4 = (max(1, round(c * width)) for c in (16, 32, 64, 64))
        self.register_buffer("coords", _coords()[None])                      # (1, 2, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_channels + 2, c1, 5, stride=2, padding=2), nn.ReLU(),   # -> 48 x 64
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),                   # -> 24 x 32
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),                   # -> 12 x 16
            nn.Conv2d(c3, c4, 3, stride=2, padding=1), nn.ReLU(),                   # -> 6 x 8
            nn.Flatten(), nn.Linear(c4 * 6 * 8, embed), nn.ReLU(),
        )
        self.gru = nn.GRU(embed, features, batch_first=True)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = _split_frames(frames, self.window, self.frame_channels)        # (B, T, C, H, W)
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        x = torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], dim=1)
        per_frame = self.cnn(x).view(b, t, -1)                             # (B, T, embed)
        _, last = self.gru(per_frame)                                      # (1, B, features)
        return last[0]


BACKBONES = ("framestack", "cnn3d", "cnn_gru")


def build_backbone(name: str, window: int = 20, frame_channels: int = 1, features: int = 512,
                   width: float = 1.0) -> nn.Module:
    """A backbone by its config name (model.backbone). frame_channels: 2 with cursor_channel, else 1."""
    if name == "framestack":
        return FrameStackCNN(window * frame_channels, features, width)
    if name == "cnn3d":
        return Conv3DNet(window, frame_channels, features, width)
    if name == "cnn_gru":
        return FrameGRU(window, frame_channels, features, width=width)
    raise ValueError(f"unknown backbone {name!r}; one of {', '.join(BACKBONES)}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
