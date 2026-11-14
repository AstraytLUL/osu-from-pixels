from dataclasses import asdict

import numpy as np
import pytest

from data.dataset import MapFrames, build_map_frames
from data.replay import ReplayFrames
from eval.motor import FREE, HUMAN
from eval.policy import AutoPolicy
from eval.rollout import rollout
from osu_sim.auto import auto_replay
from osu_sim.beatmap import Beatmap, Difficulty, HitObject, TimingPoint
from osu_sim.judge import judge
from osu_sim.mods import DT, HR, HD
from osu_sim.playable import make_playable
from osu_sim.rl_env import OsuEnv
from osu_sim.streaming_judge import StreamingJudge


def mixed_map(mods=0):
    bm = Beatmap(Difficulty(5, 4, 8, 9, 1., 1.), .7, [TimingPoint(0, 500., True)], [
        HitObject(100, 100, 1000, 'circle', True),
        HitObject(300, 120, 1300, 'slider', False, 'L', [(300, 120), (400, 120)], 2, 100),
        HitObject(200, 300, 2600, 'circle', False),
        HitObject(256, 192, 3000, 'spinner', False, end_time=5000),
    ])
    return make_playable(bm, mods)


@pytest.mark.parametrize('mods', [0, HR, DT, HD | DT])
@pytest.mark.parametrize('random_play', [False, True])
def test_streaming_matches_offline_with_sliders_spinners_and_checkpoints(mods, random_play):
    pl = mixed_map(mods)
    frames = auto_replay(pl)
    if random_play:
        rng = np.random.default_rng(7)
        n = 211
        frames = ReplayFrames(np.sort(rng.uniform(0, 5500, n)), rng.uniform(0, 512, n),
                              rng.uniform(0, 384, n), rng.integers(0, 4, n), mods)
    stream = StreamingJudge(pl)
    emitted = []
    for row in zip(frames.t, frames.x, frames.y, frames.keys):
        emitted.extend(stream.advance(*row))
    emitted.extend(stream.finish())
    assert len({j.index for j in emitted}) == len(emitted) == len(pl.objects)
    assert [asdict(j) for j in stream.judgements()] == [asdict(j) for j in judge(pl, frames)]
    assert stream.finish() == []


def test_partial_replay_flush_matches_original_judge():
    pl = mixed_map()
    frames = ReplayFrames(np.array([900., 1000.]), np.array([100., 100.]),
                          np.array([100., 100.]), np.array([0, 1]), 0)
    stream = StreamingJudge(pl)
    for row in zip(frames.t, frames.x, frames.y, frames.keys):
        stream.advance(*row)
    stream.finish()
    assert stream.judgements() == judge(pl, frames)


@pytest.mark.parametrize('motor', [FREE, HUMAN])
def test_env_reproduces_rollout_and_reward_exactly(tmp_path, motor):
    pl = mixed_map()
    layers = MapFrames(build_map_frames(pl, 42, tmp_path))
    policy = AutoPolicy()
    g = rollout(policy, pl, layers, motor, seed=19)
    env = OsuEnv(pl, layers, motor=motor)
    env.reset(19)
    policy.reset(pl, layers.times)
    rewards, xs, ys = [], [env.x], [env.y]
    for k in range(len(layers) - 1):
        action = policy.act(k, None, None)
        obs, reward, done, truncated, info = env.step_control(*action)
        rewards.append(reward)
        xs.append(env.x)
        ys.append(env.y)
        assert not truncated and done == (k == len(layers) - 2)
    assert np.array_equal(xs, g.x) and np.array_equal(ys, g.y)
    expected = judge(pl, g.as_frames())
    assert env.judge.judgements() == expected
    assert sum(rewards) == pytest.approx(sum(j.result / 300 for j in expected))
    with pytest.raises(RuntimeError):
        env.step_control(*action)
    reset_obs = env.reset(19)
    fresh = OsuEnv(pl, layers, motor=motor).reset(19)
    assert all(np.array_equal(a, b) for a, b in zip(reset_obs, fresh))


def test_snapshot_restore_replays_exactly(tmp_path):
    pl = mixed_map()
    layers = MapFrames(build_map_frames(pl, 42, tmp_path))
    policy = AutoPolicy()
    policy.reset(pl, layers.times)
    env = OsuEnv(pl, layers, motor=HUMAN)
    env.reset(3)
    for k in range(60):
        env.step_control(*policy.act(k, None, None))
    snap = env.snapshot()
    for k in range(60, len(layers) - 1):
        env.step_control(*policy.act(k, None, None))
    first = env.judge.judgements(), env.replay()
    env.restore(snap)
    assert env.k == 60
    for k in range(60, len(layers) - 1):
        env.step_control(*policy.act(k, None, None))
    assert env.judge.judgements() == first[0]
    assert all(np.array_equal(getattr(env.replay(), f), getattr(first[1], f)) for f in ("x", "y", "press", "held"))
    assert env.judge.pl is pl                    # the map is shared, not copied


def test_trace_matches_rollout(tmp_path):
    pl = mixed_map()
    layers = MapFrames(build_map_frames(pl, 42, tmp_path))
    policy = AutoPolicy()
    g = rollout(policy, pl, layers, HUMAN, seed=5)
    env = OsuEnv(pl, layers, motor=HUMAN)
    env.reset(5)
    policy.reset(pl, layers.times)
    for k in range(len(layers) - 1):
        env.step_control(*policy.act(k, None, None))
    r = env.replay()
    assert all(np.array_equal(getattr(r, f), getattr(g, f)) for f in ("x", "y", "press", "held"))
