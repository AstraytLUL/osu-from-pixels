"""60 Hz pixel environment. Actions are six raw Gaussian latents, before motor noise.

The environment records what it did (`trace`, the same arrays as eval.rollout's GridReplay), and
can be saved and restored at any frame (`snapshot` / `restore`), so an environment can restart a
stretch of a map from a saved state (rl/env.py's segment resets).
"""
import copy
from collections import deque

import numpy as np

from data.dataset import observation
from data.resample import GridReplay
from eval.motor import HUMAN
from models.heads import CENTRE, MOVE_SCALE
from osu_sim.key_decoder import KeyDecoder
from osu_sim.streaming_judge import StreamingJudge


class OsuEnv:
    def __init__(self, pl, layers, window=20, motor=HUMAN, press_threshold=0.5):
        if pl.mods:
            raise ValueError('RL training uses NoMod maps only')
        self.pl, self.layers, self.window, self.motor = pl, layers, window, motor
        self.press_threshold = press_threshold

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.k, self.x, self.y, self.prev_down = 0, *CENTRE, 0
        self.keys = KeyDecoder(self.press_threshold)
        self.cursors = deque([(-1000., -1000.)] * (self.window - 1) + [CENTRE], maxlen=self.window)
        self.judge = StreamingJudge(self.pl)
        self.judge.advance(float(self.layers.times[0]), self.x, self.y, 0)
        self.done = False
        n = len(self.layers)
        self.trace = {'x': np.full(n, CENTRE[0]), 'y': np.full(n, CENTRE[1]),
                      'press': np.zeros(n, np.uint8), 'held': np.zeros(n, np.uint8)}
        return self.observe()

    # the map, its frames and the motor are shared between copies; everything else is the state
    _SHARED = ('pl', 'layers', 'motor')

    def snapshot(self):
        """The state at this frame, to `restore` later (any number of times)."""
        state = {k: v for k, v in self.__dict__.items() if k not in self._SHARED}
        return copy.deepcopy(state, self._memo())

    def restore(self, snapshot):
        self.__dict__.update(copy.deepcopy(snapshot, self._memo()))

    def _memo(self):
        # objects reached through the judge's per-object states must not be copied either
        memo = {id(o): o for o in self.pl.objects}
        memo[id(self.pl)] = self.pl
        return memo

    def replay(self) -> GridReplay:
        """What was played so far (frames after the current one are left as at reset)."""
        return GridReplay(t=self.layers.times, mods=self.pl.mods, **{k: v.copy() for k, v in self.trace.items()})

    def observe(self):
        return observation(self.layers.window(self.k, self.window), np.array(self.cursors)), self.keys.state()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError('expected six finite action latents')
        probs = 1 / (1 + np.exp(-np.clip(a[2:], -60, 60)))
        press, held = self.keys.step(float(1 - (1 - probs[0]) * (1 - probs[1])), float(probs[2:].max()))
        target = np.clip(np.array(CENTRE) + MOVE_SCALE * a[:2].astype(np.float32), 0, [512, 384])
        return self.step_control(float(target[0]), float(target[1]), press, held)

    def step_control(self, tx, ty, press, held):
        if self.done:
            raise RuntimeError('reset required after termination')
        self.x, self.y = self.motor.execute(self.x, self.y, tx, ty, self.rng)
        self.k += 1
        t = float(self.layers.times[self.k])
        new = []
        repress = int(press) & self.prev_down
        if repress:
            new.extend(self.judge.advance(t - .5, self.x, self.y, self.prev_down & ~repress))
        self.prev_down = int(press) | int(held)
        new.extend(self.judge.advance(t, self.x, self.y, self.prev_down))
        self.cursors.append((self.x, self.y))
        tr = self.trace
        tr['x'][self.k], tr['y'][self.k], tr['press'][self.k], tr['held'][self.k] = self.x, self.y, press, held
        self.done = self.k == len(self.layers) - 1
        if self.done:
            new.extend(self.judge.finish())
        reward = sum(j.result / 300 for j in new)
        info = {'judged': len(new), 'misses': sum(j.result == 0 for j in new)}
        if self.done:
            js = self.judge.judgements()
            info['accuracy'] = sum(j.result for j in js) / (300 * len(js))
        return self.observe(), reward, self.done, False, info
