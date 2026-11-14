"""Training environment for PPO: segments of a map, started at a random point, with the brief's reward.

Built on osu_sim/rl_env.py's OsuEnv (same observation, key decoder, motor and streaming judge as the
closed-loop evaluation, eval/rollout.py), with three changes:
- HD is allowed (it changes only what is drawn; the cached map layers are rendered with it).
- Segments: an episode starts at a random frame of the map. Auto plays the map up to that frame
  (no reward, no policy), so the judge, the cursor history and the frames are those of a clean play
  so far; then the policy plays `segment_frames` frames. The cut at the end is a truncation (PPO
  bootstraps from the value there); only the map's own end is terminal. No failing (no HP).
- Reward per judged object (rl/README brief, "Reward"): 300 +1, 100 +0.33, 50 +0.17, a missed
  circle / slider head / spinner -1, a dropped slider tick or repeat (a slider break) -1, each slider
  tick, repeat or tail hit +0.1. A dropped tail costs nothing extra: it does not break combo
  (osu_sim/score.py combo_breaks). A slider's reward arrives when the slider ends.
"""
from collections import deque

import numpy as np

from data.dataset import observation
from data.resample import resample
from eval.motor import CAP
from osu_sim.auto import auto_replay
from osu_sim.key_decoder import KeyDecoder
from osu_sim.mods import HD
from osu_sim.rl_env import OsuEnv

RESULT_REWARD = {300: 1.0, 100: 0.33, 50: 0.17}
MISS = -1.0
PART_HIT = 0.1


def object_reward(j) -> float:
    """The brief's reward for one judged object (module doc)."""
    if j.kind == "slider":
        r = RESULT_REWARD.get(j.result, 0.0)
        if j.offset is None:                          # head missed
            r += MISS
        for _, kind, hit in j.parts:
            if hit:
                r += PART_HIT
            elif kind != "tail":                      # tick or repeat dropped: a slider break
                r += MISS
        return r
    return RESULT_REWARD.get(j.result, MISS)


class MapEnv(OsuEnv):
    """OsuEnv that accepts HD, keeps the decoded frame window (no re-decompressing 20 frames a
    step) and pays the brief's reward."""

    def __init__(self, pl, layers, window=20, motor=CAP, press_threshold=0.5):
        if pl.mods & ~HD:
            raise ValueError("RL environments use NoMod or HD maps only")
        self.pl, self.layers, self.window, self.motor = pl, layers, window, motor
        self.press_threshold = press_threshold

    def reset(self, seed=0):
        self.frames = deque(self.layers.window(0, self.window), maxlen=self.window)
        return super().reset(seed)

    def observe(self):
        return observation(np.stack(self.frames), np.array(self.cursors)), self.keys.state()

    def step_control(self, tx, ty, press, held):
        if self.done:
            raise RuntimeError("reset required after termination")
        self.x, self.y = self.motor.execute(self.x, self.y, tx, ty, self.rng)
        self.k += 1
        t = float(self.layers.times[self.k])
        new = []
        repress = int(press) & self.prev_down
        if repress:                                   # as GridReplay.as_frames: release 0.5 ms before
            new.extend(self.judge.advance(t - .5, self.x, self.y, self.prev_down & ~repress))
        self.prev_down = int(press) | int(held)
        new.extend(self.judge.advance(t, self.x, self.y, self.prev_down))
        self.cursors.append((self.x, self.y))
        self.frames.append(self.layers.frame(self.k))
        tr = self.trace
        tr["x"][self.k], tr["y"][self.k], tr["press"][self.k], tr["held"][self.k] = self.x, self.y, press, held
        self.done = self.k == len(self.layers) - 1
        if self.done:
            new.extend(self.judge.finish())
        info = {"judged": len(new), "misses": sum(j.offset is None if j.kind == "slider" else j.result == 0
                                                  for j in new),
                "score": sum(j.result for j in new), "score_max": 300 * len(new)}
        return self.observe(), float(sum(object_reward(j) for j in new)), self.done, False, info


class SegmentEnv:
    """One environment slot for PPO: picks a map, starts a segment of it (module doc), steps it.

    maps: [(beatmap_id, playable, MapFrames)]. step() returns (obs, keys, reward, terminated,
    truncated, info); after either flag the next step() starts a new segment by itself and returns its
    first observation as the next obs."""

    # shared by every slot: (beatmap_id, mods) -> (Auto on the frame grid, [(frame, env snapshot)])
    AUTO: dict = {}
    SNAPSHOT_EVERY = 240                               # frames (4 s): a reset replays at most this many

    def __init__(self, maps, rng: np.random.Generator, segment_frames=900, window=20, motor=CAP,
                 press_threshold=0.5):
        self.maps, self.rng, self.segment = maps, rng, segment_frames
        self.window, self.motor, self.threshold = window, motor, press_threshold

    def _auto(self, bid, pl, layers):
        """Auto's play of the map and snapshots of the environment along it, made once per map."""
        key = (bid, pl.mods)
        if key not in self.AUTO:
            g = resample(auto_replay(pl), layers.times)
            env = MapEnv(pl, layers, self.window, self.motor, self.threshold)
            env.reset(0)
            snaps = [(0, env.snapshot())]
            while env.k < len(layers) - 2:
                k = env.k + 1
                env.step_control(float(g.x[k]), float(g.y[k]), int(g.press[k]), int(g.held[k]))
                if env.k % self.SNAPSHOT_EVERY == 0:
                    snaps.append((env.k, env.snapshot()))
            self.AUTO[key] = (g, snaps)
        return self.AUTO[key]

    def reset(self):
        bid, pl, layers = self.maps[self.rng.integers(len(self.maps))]
        times = layers.times
        first = int(np.searchsorted(times, pl.objects[0].time - pl.preempt))
        last = max(first, len(times) - 1 - self.segment)
        k0 = int(self.rng.integers(first, last + 1)) if last > first else first
        g, snaps = self._auto(bid, pl, layers)
        env = MapEnv(pl, layers, self.window, self.motor, self.threshold)
        env.reset(0)
        env.restore([s for k, s in snaps if k <= k0][-1])
        env.rng = np.random.default_rng(int(self.rng.integers(2 ** 31)))
        while env.k < k0:                             # Auto plays up to the start, unrewarded
            k = env.k + 1
            env.step_control(float(g.x[k]), float(g.y[k]), int(g.press[k]), int(g.held[k]))
        env.keys = KeyDecoder(self.threshold)         # the policy's own clicks start here; a key
        held = int(g.held[env.k]) & 3                 # Auto holds (a slider, a spinner) stays down
        if held:
            env.keys.key = env.keys.down = 1 if held & 1 else 2
        self.env, self.bid, self.start = env, bid, env.k
        return env.observe()

    def step(self, action):
        (obs, keys), r, done, _, info = self.env.step(action)
        truncated = not done and self.env.k - self.start >= self.segment
        info["beatmap_id"] = self.bid
        return obs, keys, r, done, truncated, info
