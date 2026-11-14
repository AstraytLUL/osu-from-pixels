"""Policies: anything that plays a map one 60 fps frame at a time.

Each step a policy gets the last `window` map layers and its own cursor positions in them, builds what
it looks at from those (a model: data/dataset.py's `observation`, exactly as in training), and returns
the cursor and keys for the next frame: x, y in osu! pixels and press / held bitfields (bit 1 = left,
bit 2 = right, as in data/resample.py). eval/rollout.py runs every policy the same way, so the model,
Auto and baselines are judged alike.
"""
import numpy as np
import torch

from collections import deque

from data.dataset import observation, prev_actions
from data.resample import PLAYFIELD_HEIGHT, resample
from models.heads import CENTRE, MOVE_SCALE, BCPolicy, build_policy
from osu_sim.auto import auto_replay
from osu_sim.mods import HR
from osu_sim.playable import Playable
from osu_sim.key_decoder import KeyDecoder

Action = tuple[float, float, int, int]         # next frame's x, y, press bits, held bits


class Policy:
    window = 20

    def reset(self, pl: Playable, times: np.ndarray) -> None:
        """Called before playing a map; `times` is its frame grid (map ms)."""

    def act(self, k: int, layers: np.ndarray, cursors: np.ndarray) -> Action:
        """layers: (window, 96, 128) uint8, the map layers of frames k-window+1 .. k (no cursor);
        cursors: (window, 2), the policy's cursor in those frames (the last row is frame k)."""
        raise NotImplementedError


class AutoPolicy(Policy):
    """Auto (osu_sim/auto.py) on the frame grid: the upper bound, and a check of the rollout itself.
    It reads the beatmap, not the frames."""

    def reset(self, pl, times):
        self.g = resample(auto_replay(pl), times)

    def act(self, k, layers, cursors):
        g = self.g
        return g.x[k + 1], g.y[k + 1], int(g.press[k + 1]), int(g.held[k + 1])


class ModelPolicy(Policy):
    """A trained BCPolicy network. With canonical_hr (how it was trained, see data/dataset.py) an HR
    map is shown to the network flipped back to NoMod geometry, and its moves are flipped back.

    How the key outputs become clicks (edge_trigger, the default):
    - one click: "any key pressed" = 1 - (1 - p_left)(1 - p_right); "any key held" = the larger held
      probability. Which key a player uses is habit the frames cannot show, so the network hedges
      between the two; we only ask it *whether* to click.
    - a click happens when the press probability crosses the threshold from below, not on every frame
      above it: the network is unsure by a frame or two, and would otherwise click 2-3 times per note.
    - keys alternate left / right like a player's; while "held" is on, the last pressed key stays down.
    - release_below: a held key is let go once "held" is at or below this (below 0.5: hysteresis).
      In slider bodies "held" hovers just above 0.5, and one frame's dip would drop the slider.
    - hold_starts_key: "held" above 0.5 with no key down presses one. A spinner rarely starts with a
      click the press head is sure of, yet "held" is high all through it; without this no key ever
      goes down there.
    The defaults (0.3, on) were chosen on NoMod val: bc_abs_nochan 90.8% -> 98.6%, sliders 79 -> 98%,
    spinners 0 -> 11 of 12 hit (release_below=0.5, hold_starts_key=False is the first rule).
    edge_trigger=False is the raw reading (each key on its own, every frame above the threshold).
    """

    def __init__(self, model: BCPolicy, window: int, device: str = "cpu", canonical_hr: bool = True,
                 press_threshold: float = 0.5, edge_trigger: bool = True, cursor_channel: bool = False,
                 release_below: float = 0.3, hold_starts_key: bool = True):
        self.model, self.window, self.device = model, window, device
        self.canonical_hr, self.press_threshold, self.edge_trigger = canonical_hr, press_threshold, edge_trigger
        self.cursor_channel = cursor_channel
        self.release_below, self.hold_starts_key = release_below, hold_starts_key

    def reset(self, pl, times):
        self.flip = self.canonical_hr and bool(pl.mods & HR)
        self.model.eval()
        # its own key bits for the last `window` frames (nothing pressed before the map), for prev_action
        self.key_history = deque([(0, 0)] * self.window, maxlen=self.window)
        self.keys = KeyDecoder(self.press_threshold, self.release_below, self.hold_starts_key)

    def _predict(self, frames, prev=None):
        return self.model(frames, prev)

    @torch.no_grad()
    def act(self, k, layers, cursors):
        cursor = cursors[-1]
        if self.flip:                               # show the network NoMod geometry, as in training
            layers = layers[:, ::-1, :]
            cursors = np.stack([cursors[:, 0], PLAYFIELD_HEIGHT - cursors[:, 1]], axis=1)
        frames = observation(np.ascontiguousarray(layers), cursors, self.cursor_channel)
        x = torch.from_numpy(frames)[None].to(self.device)
        prev = None
        if getattr(self.model, "prev_action", False):    # its own actions, as the player's in training
            keys = np.array(self.key_history)
            prev = torch.from_numpy(prev_actions(cursors, keys[:, 0], keys[:, 1]))[None].to(self.device)
        out = self._predict(x, prev)
        if "position" in out:                       # absolute aim: the spot itself, from the centre
            px, py = (out["position"][0] * MOVE_SCALE).tolist()
            if self.flip:                           # the flip mirrors y about the centre
                py = -py
            dx, dy = CENTRE[0] + px - cursor[0], CENTRE[1] + py - cursor[1]
        else:                                       # relative aim: a move from the cursor
            dx, dy = (out["move"][0] * MOVE_SCALE).tolist()
            if self.flip:
                dy = -dy
        p_press = torch.sigmoid(out["press"][0])
        p_held = torch.sigmoid(out["held"][0])
        if self.edge_trigger:
            press, held = self._click(float(1 - (1 - p_press[0]) * (1 - p_press[1])), float(p_held.max()))
        else:
            press, held = _bits(p_press > self.press_threshold), _bits(p_held > 0.5)
        # keep the cursor on the playfield, where the policy can still see it
        x_next = float(np.clip(cursor[0] + dx, 0, 512))
        y_next = float(np.clip(cursor[1] + dy, 0, PLAYFIELD_HEIGHT))
        self.key_history.append((press, held))
        return x_next, y_next, press, held

    def _click(self, p_press: float, p_held: float) -> tuple[int, int]:
        """(press bits, held bits) for the next frame, from the "any key" probabilities."""
        return self.keys.step(p_press, p_held)


def _bits(left_right: torch.Tensor) -> int:
    return int(left_right[0]) | (int(left_right[1]) << 1)


def load_model_policy(checkpoint: str, device: str = "cpu", press_threshold: float | None = None,
                      edge_trigger: bool = True) -> ModelPolicy:
    """A ModelPolicy from a checkpoint written by train/bc.py."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = state["config"]
    d = cfg["data"]
    cursor_channel = d.get("cursor_channel", False)
    model = build_policy(cfg).to(device)
    model.load_state_dict(state["model"])
    threshold = press_threshold if press_threshold is not None else cfg["eval"]["press_threshold"]
    return ModelPolicy(model, d["window"], device, d["canonical_hr"], threshold, edge_trigger, cursor_channel)
