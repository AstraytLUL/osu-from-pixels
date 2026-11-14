"""Motor limits: how a policy's chosen cursor position is carried out, so a bot can't move in ways a
player's hand can't.

A policy names where the cursor goes next (its intent); without limits the rollout puts it there
exactly, however far that is. A player's hand has a top speed and lands less precisely the faster it
moves. Motor models both, per 60 fps frame:
  max_step  the cursor moves at most this many osu! pixels toward the target (None = no cap)
  noise     then lands off by a Gaussian of standard deviation noise * step length, per axis
            (signal-dependent noise: still = exact, fast = imprecise)

Measured on 600 train-split replays (NoMod and HR, 5M frames in play): players' steps are 8 px
median, 50 px at the 99th percentile, 73-76 px at the 99.9th; the rest are mostly cursor teleports
(tablet pen lifts). Players also slow down before they click: 88% of clicks arrive at under 10 px
per frame, 1.3% at over 20 px; their click error grows with that arrival speed (rms 11.6 px at
0-2 px, 16.7 px at 20-35 px). HUMAN's numbers come from those: the 99.9th percentile step, and
noise that roughly reproduces the extra error of a fast arrival (sqrt(16.7^2 - 11.6^2) = 12 px rms
at ~27 px per frame, ~0.3 of the step per axis).

The rollout (eval/rollout.py) applies the same Motor to every policy, Auto included, and the policy
sees where its cursor actually went, so it has to correct its own landing errors.

What it costs the BC baseline (NoMod val): 98.48% free; with the 75 px cap alone, identical, map for
map (the bot's own steps: 8.8 px median, 64 px at the 99.9th percentile; 1.3% of its clicks arrive
at over 20 px per frame, as for players); cap and noise 0.1 / 0.2 / 0.3: 98.51 / 98.50 / 98.37%.
BC already moves like the players it copies. The limits are for RL, whose reward would otherwise
pay for snapping onto every object.
"""
from dataclasses import dataclass

import numpy as np

from data.resample import PLAYFIELD_HEIGHT

PLAYFIELD_WIDTH = 512


@dataclass(frozen=True)
class Motor:
    max_step: float | None = None              # osu! px per 60 fps frame
    noise: float = 0.0                         # per-axis sd, as a fraction of the step length

    def execute(self, x: float, y: float, tx: float, ty: float, rng: np.random.Generator) -> tuple[float, float]:
        """Where the cursor ends up this frame, going from (x, y) toward the target (tx, ty)."""
        dx, dy = tx - x, ty - y
        step = float(np.hypot(dx, dy))
        if self.max_step is not None and step > self.max_step:
            dx, dy = dx * self.max_step / step, dy * self.max_step / step
            step = self.max_step
        if self.noise and step:
            ex, ey = rng.normal(0.0, self.noise * step, 2)
            dx, dy = dx + ex, dy + ey
        return float(np.clip(x + dx, 0, PLAYFIELD_WIDTH)), float(np.clip(y + dy, 0, PLAYFIELD_HEIGHT))

    @property
    def free(self) -> bool:
        return self.max_step is None and not self.noise


FREE = Motor()                                 # the target exactly: the old behaviour
HUMAN = Motor(max_step=75.0, noise=0.3)        # see the module doc
CAP = Motor(max_step=75.0)                     # HUMAN's speed cap alone: the default everywhere (the report, rl/)
MOTORS = {"free": FREE, "cap": CAP, "human": HUMAN}   # by name, for --motor


def motor_from(cfg: dict | None) -> Motor:
    """A Motor from a config's `eval` section (keys max_step, aim_noise); a missing key takes HUMAN's
    value, so `max_step: null` and `aim_noise: 0` are needed for FREE."""
    cfg = cfg or {}
    return Motor(cfg.get("max_step", HUMAN.max_step), float(cfg.get("aim_noise", HUMAN.noise)))
