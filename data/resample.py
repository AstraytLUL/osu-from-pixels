"""Put a replay on the fixed frame grid the policy sees: 60 frames per REAL second.

The grid is the contract shared by the dataset (data/dataset.py) and closed-loop play:
  - grid times are map time: t_k = start + k * step, with step = 1000 / 60 * rate ms. So NoMod frames
    are 16.67 map ms apart and DT frames 25 map ms apart: under DT the bot sees 1.5x more motion per
    frame, which is the shift hypothesis 2 is about.
  - at frame k the policy sees the playfield and its own cursor at t_k, and outputs the cursor at
    t_{k+1} (as a displacement) and which keys are held at t_{k+1}.

Cursor: linear interpolation between replay frames (the game interpolates replay cursors too).
Keys, per key and frame, over the centred window [t_k - step/2, t_k + step/2):
  - press: the key went down at some moment in the window
  - held: the key is down at the end of the window
Both are needed. Sampling only "held at t_k" drops a tap that starts and ends between two grid
points; "held at any moment in the window" keeps taps but erases a release followed by a re-press
within one step, so the second click vanishes. On 300 v1 replays that last version cost 0.4%
accuracy on average and added misses to a third of them. With press + held, as_frames() turns a
press while already held into a release and a new press, and every click survives, moved by at
most half a step (8 ms NoMod, 12.5 map ms DT), in either direction rather than always late.

Cursor on press frames: moving a click in time also moves where the cursor is at the click, and a
player sweeping across a circle often clicks a few pixels inside its edge; a few ms later the cursor
is outside, the click misses, and note lock then blocks the next objects too (one replay lost 18
objects that way). So on a frame with a press, the cursor is taken at the moment of the (first)
press instead of at t_k: the click lands exactly where the player clicked.

Fidelity (judge the recorded replay vs its grid version, first 300 v1 replays): accuracy changes by
-0.37% on average (5th percentile -1.1%, worst -2.6%); 18 of 300 gain a miss. Part of that was a
rounding bug, fixed 9 Nov: a key event on a window boundary could count in both windows or neither
(200 train replays: -0.49 -> -0.32 pp on average, worst -2.7 -> -2.25; Auto on the grid went from
88.7% to 100% on one stream map, and is now 100% on every test map). What remains is the
±half-step timing shift turning some 300s near the window edge into 100s: the cost of a 60 fps
action grid, which the policy plays on too.
"""
from dataclasses import dataclass

import numpy as np

from data.replay import M1, M2, ReplayFrames
from osu_sim.mods import rate as mod_rate
from osu_sim.playable import Playable

FPS = 60
LEAD_MS = 2000                 # grid starts this long (map ms) before the first object: covers any
                               # approach time (<= 1800 ms) plus some history for the frame window
TAIL_MS = 500                  # and ends this long after the last object ends
PLAYFIELD_HEIGHT = 384


def grid_step(mods: int) -> float:
    """Map ms between frames: 1000 / 60 real ms, times the playback rate."""
    return 1000.0 / FPS * mod_rate(mods)


def map_grid(pl: Playable) -> np.ndarray:
    """Grid times (map ms) for a playable map: from LEAD_MS before its first object to TAIL_MS after its end."""
    step = grid_step(pl.mods)
    start = pl.objects[0].time - LEAD_MS
    end = max(o.end_time for o in pl.objects) + TAIL_MS
    return start + step * np.arange(int(np.ceil((end - start) / step)) + 1)


@dataclass
class GridReplay:
    t: np.ndarray              # (N,) grid times, map ms
    x: np.ndarray              # (N,) cursor, osu! pixels
    y: np.ndarray
    press: np.ndarray          # (N,) uint8: bit 1 = left pressed in this frame's window, bit 2 = right
    held: np.ndarray           # (N,) uint8: bit 1 = left down at the window's end, bit 2 = right
    mods: int

    def as_frames(self, gap_ms: float = 0.5) -> ReplayFrames:
        """The grid as a replay (e.g. to judge it). A key is down at t_k if pressed or held there; a
        press while the key was already down at t_{k-1} gets a release frame gap_ms before t_k."""
        down = self.press | self.held
        prev = np.concatenate([[0], down[:-1]]).astype(np.uint8)
        repress = self.press & prev                              # needs a release just before t_k
        t, x, y, k = [], [], [], []
        for i in range(len(self.t)):
            if repress[i]:
                t.append(self.t[i] - gap_ms)
                x.append(self.x[i])
                y.append(self.y[i])
                k.append(int(prev[i] & ~repress[i]))
            t.append(self.t[i])
            x.append(self.x[i])
            y.append(self.y[i])
            k.append(int(down[i]))
        return ReplayFrames(t=np.array(t), x=np.array(x), y=np.array(y), keys=np.array(k), mods=self.mods)


def resample(frames: ReplayFrames, times: np.ndarray) -> GridReplay:
    """Sample a replay on grid `times`: cursor interpolated; press and held per key over a centred
    window (see module doc). Before the replay's first frame: first cursor position, no keys."""
    t = frames.t.astype(float)
    times = np.asarray(times, dtype=float)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    x = np.interp(times, t, frames.x)
    y = np.interp(times, t, frames.y)
    # window k ends exactly where window k+1 starts (one shared array): with lo and hi computed
    # separately, an event on a boundary could fall in both after rounding and count twice
    bounds = np.concatenate([[times[0] - step / 2], (times[:-1] + times[1:]) / 2, [times[-1] + step / 2]])
    lo, hi = bounds[:-1], bounds[1:]
    press = np.zeros(len(times), dtype=np.uint8)
    held = np.zeros(len(times), dtype=np.uint8)
    first_press = np.full(len(times), np.inf)                    # time of the window's first press
    for bit in (M1, M2):
        down = (frames.keys & bit) > 0
        edge = down & ~np.concatenate([[False], down[:-1]])      # frames where the key goes down
        csum = np.concatenate([[0], np.cumsum(edge)])
        i_lo, i_hi = np.searchsorted(t, lo, side="left"), np.searchsorted(t, hi, side="left")
        pressed = csum[i_hi] - csum[i_lo] > 0
        last = i_hi - 1                                          # last frame before the window's end
        down_at_end = np.where(last >= 0, down[np.clip(last, 0, None)], False)
        press |= np.where(pressed, bit, 0).astype(np.uint8)
        held |= np.where(down_at_end, bit, 0).astype(np.uint8)
        edge_times = t[edge]                                     # sorted
        if len(edge_times):
            # where a window has a press, its first edge is the first one at or after the window start
            j = np.minimum(np.searchsorted(edge_times, lo, side="left"), len(edge_times) - 1)
            first_press = np.where(pressed, np.minimum(first_press, edge_times[j]), first_press)
    # on press frames, the cursor where the player actually clicked (see module doc)
    on = np.isfinite(first_press)
    x[on] = np.interp(first_press[on], t, frames.x)
    y[on] = np.interp(first_press[on], t, frames.y)
    return GridReplay(t=times, x=x, y=y, press=press, held=held, mods=frames.mods)


def flip_vertical(g: GridReplay) -> GridReplay:
    """Mirror the cursor top-to-bottom (y -> 384 - y). With the frames flipped the same way, an HR
    replay becomes a replay of the unflipped map (HR's smaller circles and faster approach remain)."""
    return GridReplay(t=g.t, x=g.x, y=PLAYFIELD_HEIGHT - g.y, press=g.press, held=g.held, mods=g.mods)
