"""Renderer: deterministic low-resolution grayscale frames of the playfield at any map time.

128x96 pixels covering the whole 512x384 playfield, so 1 px = 4 osu! pixels (the project proposal,
"Decisions fixed before the first commit"). What is drawn, and when:

- hit circle: fades in over `fade_in` from (time - preempt); a bright ring and a dim fill;
  approach circle shrinks from 4x to 1x the radius over preempt. Disappears when hit (see below).
- slider: body (dim fill, bright border) from (time - preempt) to its end; head drawn like a circle
  until it is hit; ball from start to end; a dot on the end where the next reverse happens, from
  when the slider appears (so a kick slider's reverse is visible before it is hit).
- spinner: a large ring from (time - preempt) to its end, and an inner ring shrinking to 0.
- cursor: a small white dot in a black ring, always on top (visible even on the slider ball).
- Hidden (HD): no approach circles; circles and heads fade in over 0.4 preempt and fade out over the
  next 0.3 preempt; slider bodies fade out from (time - preempt + fade_in) to their end
  (lazer OsuModHidden.cs).

When does a circle disappear? In the game, when it is hit or missed. The renderer takes optional
`hit_times` (object index -> map time it was judged, e.g. from the judge on a human replay); without
it, objects disappear at their own time, which is what a perfect player sees.

Times are map time (DT does not change preempt in map time; it only changes how far apart in map
time the 60 fps frames are). NumPy on the CPU; a batched GPU version can follow the same maths.
"""
import numpy as np

from osu_sim.mods import HD
from osu_sim.playable import Playable, PlayObject

RENDER_VERSION = 2             # bump when what is drawn changes: data/dataset.py's frame cache is keyed on it
                               # (2: reverse markers from when the slider appears, not from its start)
W, H = 128, 96
PX = 4.0                                     # osu! pixels per rendered pixel
CURSOR_RADIUS = 6.0                          # osu! pixels
SPINNER_RADIUS = 150.0
_xs = (np.arange(W) + 0.5) * PX              # pixel centres in osu! pixels
_ys = (np.arange(H) + 0.5) * PX
GX, GY = np.meshgrid(_xs, _ys)               # (H, W)


def _soft(d: np.ndarray) -> np.ndarray:
    """1 inside (d < 0), 0 outside, linear across one rendered pixel: cheap anti-aliasing."""
    return np.clip(0.5 - d / PX, 0.0, 1.0)


def _disc(cx, cy, r):
    return _soft(np.hypot(GX - cx, GY - cy) - r)


def _ring(cx, cy, r, width):
    return _soft(np.abs(np.hypot(GX - cx, GY - cy) - r) - width / 2)


def _segments_distance(pts: np.ndarray) -> np.ndarray:
    """Distance from every pixel centre to a polyline (N, 2), shape (H, W)."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    len2 = np.maximum((ab ** 2).sum(1), 1e-9)
    px = GX[..., None] - a[:, 0]
    py = GY[..., None] - a[:, 1]
    tt = np.clip((px * ab[:, 0] + py * ab[:, 1]) / len2, 0, 1)
    dx = px - tt * ab[:, 0]
    dy = py - tt * ab[:, 1]
    return np.sqrt(dx * dx + dy * dy).min(-1)


class Renderer:
    def __init__(self, pl: Playable, hit_times: dict[int, float] | None = None):
        self.pl = pl
        self.hidden = bool(pl.mods & HD)
        self.hit_times = hit_times or {}
        self.starts = np.array([o.time for o in pl.objects])
        self._body_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ visibility
    def _gone_at(self, o: PlayObject) -> float:
        """When the circle / head stops being drawn."""
        return self.hit_times.get(o.index, o.time)

    def _alpha(self, o: PlayObject, t: float) -> float:
        """Opacity of a circle or slider head."""
        pl = self.pl
        appear = o.time - pl.preempt
        if self.hidden:
            fade_in = pl.preempt * 0.4
            if t < appear + fade_in:
                return (t - appear) / fade_in
            return float(np.clip(1 - (t - appear - fade_in) / (pl.preempt * 0.3), 0, 1))
        return float(np.clip((t - appear) / pl.fade_in, 0, 1))

    def body_alpha(self, o: PlayObject, t: float) -> float:
        """Opacity of a slider body (eval/video.py also fades the slider ticks with it)."""
        pl = self.pl
        appear = o.time - pl.preempt
        a = float(np.clip((t - appear) / pl.fade_in, 0, 1))
        if self.hidden:
            start = appear + pl.fade_in
            if t > start:
                u = min((t - start) / max(o.end_time - start, 1e-9), 1)
                a *= 1 - (1 - (1 - u) ** 2)   # Easing.Out fade to 0 at the slider end
        return a

    def _body(self, o: PlayObject) -> np.ndarray:
        """Slider body distance field, cached per slider (it never moves)."""
        if o.index not in self._body_cache:
            pts = np.array([o.path_point(p) for p in np.linspace(0, 1, max(2, int(o.path.distance / 4) + 2))])
            self._body_cache[o.index] = _segments_distance(pts)
        return self._body_cache[o.index]

    # ------------------------------------------------------------ drawing
    # A frame is two layers: the map layer (everything but the cursor; the same for every replay of
    # a map under the same mods, so data/dataset.py caches it) and the cursor, drawn on top.
    def frame(self, t: float, cursor: tuple[float, float] | None = None) -> np.ndarray:
        """One (H, W) uint8 frame at map time t."""
        layer = self.map_layer(t)
        if cursor is None:
            return layer
        return add_cursor(layer[None], np.asarray([cursor], dtype=float))[0]

    def map_layer(self, t: float) -> np.ndarray:
        """(H, W) uint8: the playfield at map time t without the cursor."""
        pl = self.pl
        r = pl.radius
        img = np.zeros((H, W))
        # objects on screen at t: appeared already, not yet finished (later objects are drawn first,
        # earlier ones on top, as in the game)
        lo = np.searchsorted(self.starts, t - 60000)     # nothing lasts a minute
        hi = np.searchsorted(self.starts, t + pl.preempt, side="right")
        for o in reversed(pl.objects[lo:hi]):
            if t > o.end_time and t > self._gone_at(o):
                continue
            if o.kind == "spinner":
                if t <= o.end_time:
                    a = float(np.clip((t - (o.time - pl.preempt)) / pl.fade_in, 0, 1))
                    img = np.maximum(img, 0.5 * a * _ring(o.x, o.y, SPINNER_RADIUS, 4))
                    if t >= o.time:
                        left = 1 - (t - o.time) / max(o.end_time - o.time, 1)
                        img = np.maximum(img, 0.8 * a * _ring(o.x, o.y, SPINNER_RADIUS * left, 4))
                continue
            if o.kind == "slider" and t <= o.end_time:
                a = self.body_alpha(o, t)
                d = self._body(o) - r
                img = np.maximum(img, a * np.maximum(0.25 * _soft(d), 0.7 * _soft(np.abs(d) - 3)))
                if o.timing.repeats:
                    # the next reverse, shown from when the slider appears (as osu!'s reverse arrow):
                    # before the slider starts that is the first one, at the path's end
                    span = int(max(t - o.time, 0) / max(o.timing.span_duration, 1e-9))
                    if span < o.timing.spans - 1:            # a reverse is still coming
                        ex, ey = o.path_point(1.0 if span % 2 == 0 else 0.0)
                        img = np.maximum(img, a * _disc(ex, ey, r * 0.3))
                if t >= o.time:
                    bx, by = o.ball_position(t)
                    img = np.maximum(img, 0.6 * _disc(bx, by, r * 0.85))
            if t <= self._gone_at(o):                         # circle, or slider head
                a = self._alpha(o, t)
                if a > 0:
                    img = np.maximum(img, a * np.maximum(0.35 * _disc(o.x, o.y, r), _ring(o.x, o.y, r - 2, 4)))
                    if not self.hidden and t <= o.time:
                        scale = 1 + 3 * (o.time - t) / pl.preempt
                        ac = float(np.clip((t - (o.time - pl.preempt)) / min(2 * pl.fade_in, pl.preempt), 0, 1))
                        img = np.maximum(img, 0.8 * ac * _ring(o.x, o.y, r * scale, 3))
        return _to_u8(img)

    def frames(self, times, cursors=None) -> np.ndarray:
        """(N, H, W) uint8 frames; `cursors` is (N, 2) or None."""
        layers = np.stack([self.map_layer(t) for t in times])
        return layers if cursors is None else add_cursor(layers, np.asarray(cursors, dtype=float))


def _to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def add_cursor(layers: np.ndarray, cursors: np.ndarray) -> np.ndarray:
    """Draw one cursor per frame: layers (N, H, W) uint8, cursors (N, 2) osu! pixels -> (N, H, W) uint8.

    A white dot inside a black ring, so the cursor stays visible on a bright slider ball.
    Only a small patch around each cursor is redrawn (10x faster than whole frames, same bytes:
    the ring leaves every pixel whose centre is `reach` or more away unchanged).
    """
    out = layers.copy()
    reach = CURSOR_RADIUS + 1.5 * PX
    for img, (cx, cy) in zip(out, cursors):
        x0, x1 = max(0, int((cx - reach) // PX)), min(W, int((cx + reach) // PX) + 1)
        y0, y1 = max(0, int((cy - reach) // PX)), min(H, int((cy + reach) // PX) + 1)
        if x0 >= x1 or y0 >= y1:
            continue                             # cursor far off screen
        d = np.hypot(GX[y0:y1, x0:x1] - cx, GY[y0:y1, x0:x1] - cy)
        patch = img[y0:y1, x0:x1] / 255.0
        patch = patch * (1 - _soft(d - (CURSOR_RADIUS + PX)))
        patch = np.maximum(patch, _soft(d - CURSOR_RADIUS))
        img[y0:y1, x0:x1] = _to_u8(patch)
    return out
