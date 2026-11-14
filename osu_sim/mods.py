"""Mod transforms applied to a parsed Beatmap.

Sources: osu! lazer, osu.Game.Rulesets.Osu/Mods/OsuModHardRock.cs and
Utils/OsuHitObjectGenerationUtils.cs (ReflectVerticallyAlongPlayfield); osu! wiki, Gameplay/Game_modifier.

Replay frames are in map time (a DT replay of an 88 s map ends near 88 s, not 59 s), so DT/HT do
not change the beatmap as the judge sees it; `rate` is for the renderer, which works in real time.
HD changes only what is drawn, so it is handled by the renderer too.
"""
import copy

import numpy as np

from osu_sim.beatmap import Beatmap

# legacy mod bits (the .osr header's bitfield)
EZ, HD, HR, DT, HT, NC = 2, 8, 16, 64, 256, 512
PLAYFIELD_HEIGHT = 384


def apply_mods(beatmap: Beatmap, mods: int) -> Beatmap:
    """Return a new Beatmap with the difficulty and geometry changes of `mods`; the input is not changed."""
    bm = copy.deepcopy(beatmap)
    d = bm.difficulty
    if mods & EZ:
        d.cs, d.od, d.ar, d.hp = d.cs * 0.5, d.od * 0.5, d.ar * 0.5, d.hp * 0.5
    if mods & HR:
        # osu! does this in single precision (float), so do the same: an OD that lands a hair
        # above or below a whole number changes the truncated hit window by 1 ms
        f32 = np.float32
        d.cs = float(min(f32(d.cs) * f32(1.3), f32(10)))
        d.od = float(min(f32(d.od) * f32(1.4), f32(10)))
        d.ar = float(min(f32(d.ar) * f32(1.4), f32(10)))
        d.hp = float(min(f32(d.hp) * f32(1.4), f32(10)))
        for o in bm.hit_objects:
            o.y = PLAYFIELD_HEIGHT - o.y
            o.curve_points = [(px, PLAYFIELD_HEIGHT - py) for px, py in o.curve_points]
    return bm


def rate(mods: int) -> float:
    """Playback speed: map time advances `rate` ms per real ms."""
    if mods & (DT | NC):       # NC also sets the DT bit in stable replays; checking both is harmless
        return 1.5
    if mods & HT:
        return 0.75
    return 1.0
