import numpy as np

from eval.motor import FREE, HUMAN, Motor, motor_from
from eval.policy import AutoPolicy
from eval.rollout import play
from osu_sim.judge import accuracy
from test_train import small_map


def test_free_goes_exactly_to_the_target():
    rng = np.random.default_rng(0)
    assert FREE.free
    assert Motor(None, 0.0).execute(0, 0, 400, 300, rng) == (400.0, 300.0)


def test_max_step_caps_the_move_along_its_direction():
    x, y = Motor(max_step=50).execute(100, 100, 400, 500, np.random.default_rng(0))
    assert np.hypot(x - 100, y - 100) == np.float64(50.0)
    assert np.isclose((x - 100) / (y - 100), 300 / 400)
    assert Motor(max_step=50).execute(100, 100, 120, 100, np.random.default_rng(0)) == (120.0, 100.0)


def test_noise_grows_with_the_step_and_vanishes_when_still():
    m = Motor(noise=0.3)
    rng = np.random.default_rng(0)
    assert m.execute(200, 200, 200, 200, rng) == (200.0, 200.0)
    small = np.array([m.execute(200, 200, 210, 200, rng) for _ in range(2000)])
    large = np.array([m.execute(200, 200, 260, 200, rng) for _ in range(2000)])
    assert np.isclose(small[:, 1].std(), 3.0, rtol=0.1)
    assert np.isclose(large[:, 1].std(), 18.0, rtol=0.1)


def test_the_cursor_stays_on_the_playfield():
    x, y = Motor(noise=1.0).execute(510, 2, 512, 0, np.random.default_rng(1))
    assert 0 <= x <= 512 and 0 <= y <= 384


def test_motor_from_config():
    assert motor_from(None) == HUMAN
    assert motor_from({"press_threshold": 0.5}) == HUMAN
    assert motor_from({"max_step": None, "aim_noise": 0}).free
    assert motor_from({"max_step": 75, "aim_noise": 0.3}) == Motor(75, 0.3)


def test_rollout_applies_the_motor_and_is_seeded(tmp_path):
    pl = small_map()
    g_free, _ = play(AutoPolicy(), 42, pl, root=tmp_path, motor=FREE)
    assert np.hypot(np.diff(g_free.x), np.diff(g_free.y)).max() > 20
    g, js = play(AutoPolicy(), 42, pl, root=tmp_path, motor=Motor(max_step=20))
    assert np.hypot(np.diff(g.x), np.diff(g.y)).max() <= 20 + 1e-9
    a, _ = play(AutoPolicy(), 42, pl, root=tmp_path, motor=HUMAN, seed=3)
    b, _ = play(AutoPolicy(), 42, pl, root=tmp_path, motor=HUMAN, seed=3)
    c, _ = play(AutoPolicy(), 42, pl, root=tmp_path, motor=HUMAN, seed=4)
    assert np.array_equal(a.x, b.x) and not np.array_equal(a.x, c.x)
    # Auto has time to reach every object on this map under a human top speed
    assert accuracy(play(AutoPolicy(), 42, pl, root=tmp_path, motor=Motor(max_step=75))[1]) == 1.0
