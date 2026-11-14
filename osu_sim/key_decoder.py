"""The BC click rule, shared by replay evaluation and the RL environment."""
import numpy as np


class KeyDecoder:
    def __init__(self, threshold=0.5, release_below=0.3, hold_starts_key=True):
        self.threshold, self.release_below, self.hold_starts_key = threshold, release_below, hold_starts_key
        self.was_above, self.key, self.down = False, 2, 0

    def state(self):
        return np.array([self.was_above, self.key == 1, self.down == 1, self.down == 2], dtype=np.float32)

    def step(self, p_press, p_held):
        above = p_press > self.threshold
        rising = above and not self.was_above
        self.was_above = above
        starts_hold = self.hold_starts_key and not self.down and p_held > 0.5
        press = 0
        if rising or starts_hold:
            self.key = 1 if self.key == 2 else 2
            press = self.key
            self.down = self.key
        elif p_held <= self.release_below:
            self.down = 0
        return press, self.down
