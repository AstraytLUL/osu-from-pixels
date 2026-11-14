"""Incremental adapter for the existing judge; events have identical replay semantics."""
from osu_sim.judge import (
    LEFT, RIGHT, _Part, _State, _alive, _checkpoints, _click, _judge_head,
    _judgement, _update, hit_windows,
)
from osu_sim.mods import rate as mod_rate


class StreamingJudge:
    def __init__(self, pl):
        self.pl, self.win, self.rate = pl, hit_windows(pl.od), mod_rate(pl.mods)
        self.states = [_State(o) for o in pl.objects]
        for s in self.states:
            if s.obj.kind == 'slider':
                tm = s.obj.timing
                s.parts = sorted([_Part(t, 'tick') for t, _ in tm.ticks]
                                 + [_Part(t, 'repeat') for t, _ in tm.repeats], key=lambda p: p.time)
                s.parts.append(_Part(tm.legacy_last_tick, 'tail'))
        self.checkpoints = sorted(_checkpoints(self.states, self.win))
        self.c, self.first, self.prev, self.prev_held = 0, [0], None, 0
        self.results, self.closed = {}, False

    def advance(self, t, x, y, keys):
        if self.closed:
            raise RuntimeError('judge already finished')
        if self.prev is not None and t < self.prev[0]:
            raise ValueError('replay times must be nondecreasing')
        held = int(keys) & (LEFT | RIGHT)
        touched = {}

        def update(ct, cx, cy, ck):
            alive = _alive(self.states, self.first, ct, self.pl)
            touched.update((s.obj.index, s) for s in alive)
            _update(alive, ct, cx, cy, ck, self.pl, self.win, self.rate)

        while self.c < len(self.checkpoints) and self.checkpoints[self.c] < t:
            ct = self.checkpoints[self.c]
            self.c += 1
            if self.prev is None:
                update(ct, x, y, 0)
            elif ct > self.prev[0]:
                pt, px, py = self.prev
                w = (ct - pt) / (t - pt)
                update(ct, px + (x - px) * w, py + (y - py) * w, self.prev_held)
        alive = _alive(self.states, self.first, t, self.pl)
        touched.update((s.obj.index, s) for s in alive)
        new = held & ~self.prev_held
        for action in (LEFT, RIGHT):
            if new & action:
                _click(alive, self.states, action, t, x, y, held, self.pl, self.win)
        _update(alive, t, x, y, held, self.pl, self.win, self.rate)
        self.prev_held, self.prev = held, (t, x, y)
        return self._emit(touched.values())

    def _emit(self, states):
        new = []
        for s in states:
            if s.done and s.obj.index not in self.results:
                j = _judgement(s, self.pl)
                self.results[j.index] = j
                new.append(j)
        return new

    def finish(self):
        if self.closed:
            return []
        for s in self.states:
            if not s.head_judged and s.obj.kind != 'spinner':
                _judge_head(s, 0, None)
            for part in s.parts:
                if part.hit is None:
                    part.hit = False
            s.done = True
        self.closed = True
        return self._emit(self.states)

    def judgements(self):
        return [self.results[i] for i in sorted(self.results)]
