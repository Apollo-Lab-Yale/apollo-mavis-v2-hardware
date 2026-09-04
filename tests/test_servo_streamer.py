"""_ServoStreamer: per-tick clamps, stall re-anchor, soft-realtime (§3.3)."""

import time

import numpy as np
import pytest
from conftest import FakeClock
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.config import ServoLimits
from apollo_mavis_v2_hardware.driver import _ServoStreamer

DT = 0.01
LIMITS = ServoLimits()  # vel 1.0 rad/s, acc 20 rad/s^2, cart step 9 mm
VEL_STEP = 1.0 * DT
ACC_STEP = 20.0 * DT * DT
LEVER = np.array(LIMITS.lever_arm_m)


def _ready_api(clock: FakeClock) -> FakeXArmAPI:
    api = FakeXArmAPI(clock=clock.now)
    api.motion_enable(True)
    api.set_mode(1)
    api.set_state(0)
    return api


def _run_ticks(api, streamer, n: int, timeout_s: float = 5.0) -> None:
    streamer.start()
    deadline = time.monotonic() + timeout_s
    while len(api.sent_joints) < n and time.monotonic() < deadline:
        time.sleep(0.001)
    streamer.stop()
    assert len(api.sent_joints) >= n, f"only {len(api.sent_joints)} ticks"


def test_per_tick_clamps_on_step_target(fake_clock: FakeClock) -> None:
    api = _ready_api(fake_clock)
    streamer = _ServoStreamer(api, LIMITS, lambda *a: None, fake_clock.now,
                              fake_clock.sleep)
    streamer.reseed(np.zeros(7))
    streamer.resume()
    streamer.set_target(np.full(7, 1.0))  # huge step: every clamp must engage
    _run_ticks(api, streamer, 300)
    sent = [np.asarray(q) for _, q in api.sent_joints]
    prev = np.zeros(7)
    prev_dq = np.zeros(7)
    for q in sent:
        dq = q - prev
        assert np.all(np.abs(dq) <= VEL_STEP + 1e-12)  # velocity limit
        # accel limit on the speed-up side (lever scaling may shrink faster)
        assert np.all(np.abs(dq) <= np.abs(prev_dq) + ACC_STEP + 1e-12)
        assert float(np.sum(np.abs(dq) * LEVER)) <= LIMITS.max_cart_step_m + 1e-12
        prev, prev_dq = q, dq
    # it actually makes progress toward the target
    assert sent[-1][0] > sent[0][0]


def test_single_joint_acc_ramp_then_lever_plateau(fake_clock: FakeClock) -> None:
    api = _ready_api(fake_clock)
    streamer = _ServoStreamer(api, LIMITS, lambda *a: None, fake_clock.now,
                              fake_clock.sleep)
    streamer.reseed(np.zeros(7))
    streamer.resume()
    tgt = np.zeros(7)
    tgt[0] = 0.5
    streamer.set_target(tgt)
    _run_ticks(api, streamer, 30)
    steps = [q[0] for _, q in api.sent_joints[:30]]
    deltas = np.diff([0.0, *steps])
    # accel-limited ramp: 0.002, 0.004, 0.006, then lever-arm cap 0.009/1.2
    assert deltas[0] == pytest.approx(ACC_STEP, abs=1e-12)
    assert deltas[1] == pytest.approx(2 * ACC_STEP, abs=1e-12)
    plateau = LIMITS.max_cart_step_m / LEVER[0]  # 0.0075 rad < vel_step
    assert deltas[10] == pytest.approx(plateau, abs=1e-9)


def test_stall_reanchors_never_bursts(fake_clock: FakeClock) -> None:
    api = _ready_api(fake_clock)
    streamer = _ServoStreamer(api, LIMITS, lambda *a: None, fake_clock.now,
                              fake_clock.sleep)
    streamer.reseed(np.zeros(7))
    streamer.resume()
    streamer.set_target(np.full(7, 1.0))  # keep it moving through the stall
    streamer.start()
    deadline = time.monotonic() + 5.0
    while len(api.sent_joints) < 50 and time.monotonic() < deadline:
        time.sleep(0.001)
    fake_clock.inject_stall(after_sleeps=1, extra=0.05)  # 50 ms scheduler stall
    n0 = len(api.sent_joints)
    while (
        fake_clock.stall_pending or len(api.sent_joints) < n0 + 100
    ) and time.monotonic() < deadline:
        time.sleep(0.001)
    streamer.stop()
    times = [t for t, _ in api.sent_joints]
    intervals = np.diff(times)
    # no catch-up burst: sends never closer than one period, even after the gap
    assert float(np.min(intervals)) >= DT - 1e-9
    assert float(np.max(intervals)) >= 0.05  # the stall is visible exactly once
    # and no velocity spike right after the gap
    sent = [np.asarray(q) for _, q in api.sent_joints]
    for i in range(1, len(sent)):
        assert np.all(np.abs(sent[i] - sent[i - 1]) <= VEL_STEP + 1e-12)
    assert streamer.stats.late_ticks >= 1


def test_paused_streamer_sends_nothing(fake_clock: FakeClock) -> None:
    api = _ready_api(fake_clock)
    streamer = _ServoStreamer(api, LIMITS, lambda *a: None, fake_clock.now,
                              fake_clock.sleep)
    streamer.reseed(np.zeros(7))  # paused by default
    _run_ticks_paused_check(api, streamer)


def _run_ticks_paused_check(api, streamer) -> None:
    streamer.start()
    time.sleep(0.05)
    streamer.stop()
    assert api.sent_joints == []


def test_fault_pauses_and_reports(fake_clock: FakeClock) -> None:
    api = _ready_api(fake_clock)
    faults: list[tuple[str, int]] = []
    from fakes.fake_xarm_api import FaultScript

    api.fault_script = FaultScript(fault_at_tick=5, error_code=24, servo_return=1)
    streamer = _ServoStreamer(api, LIMITS, lambda src, code: faults.append((src, code)),
                              fake_clock.now, fake_clock.sleep)
    streamer.reseed(np.zeros(7))
    streamer.resume()
    streamer.set_target(np.full(7, 0.5))
    streamer.start()
    deadline = time.monotonic() + 5.0
    while not faults and time.monotonic() < deadline:
        time.sleep(0.001)
    streamer.stop()
    assert faults == [("servo", 1)]
    assert streamer.paused
    assert len(api.sent_joints) == 4  # tick 5 was rejected, nothing after


def test_realtime_1000_ticks_p99_under_12ms() -> None:
    """Soft-realtime: 1000 ticks at 100 Hz on the real clock, p99 < 12 ms."""
    api = FakeXArmAPI(clock=time.monotonic)
    api.motion_enable(True)
    api.set_mode(1)
    api.set_state(0)
    streamer = _ServoStreamer(api, LIMITS, lambda *a: None)  # real clock/sleep
    streamer.reseed(np.zeros(7))
    streamer.resume()
    streamer.set_target(np.full(7, 0.5))
    streamer.start()
    deadline = time.monotonic() + 30.0
    while streamer.stats.ticks < 1001 and time.monotonic() < deadline:
        time.sleep(0.05)
    stats = streamer.stats
    streamer.stop()
    assert stats.ticks >= 1001
    periods = np.asarray(stats.periods_s[:1000])
    p99 = float(np.percentile(periods, 99))
    assert p99 < 0.012, f"tick period p99 {p99 * 1000:.2f} ms"
