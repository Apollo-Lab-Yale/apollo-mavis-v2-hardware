"""ArmStateMonitor (02-hardware §8.5, phase-09a): read-only polling, ZERO writes,
rail/gripper semantics + units, stale detection, reconnect backoff, hand-over."""

from __future__ import annotations

import math
import time

import pytest
from conftest import FakeClock
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import wait_until

from apollo_mavis_v2_hardware.monitor import (
    MAX_RECONNECT_S,
    READ_ONLY_SDK_ATTRS,
    READ_ONLY_SDK_METHODS,
    ArmMonitorSample,
    ArmStateMonitor,
    controller_error_title,
)

FORBIDDEN_NAMES = {
    "motion_enable",
    "set_mode",
    "set_state",
    "clean_error",
    "clean_warn",
    "emergency_stop",
    "save_conf",
    "set_servo_angle_j",
    "register_report_callback",
    "set_linear_track_back_origin",
    "set_linear_track_enable",
    "set_linear_track_pos",
    "clean_linear_track_error",
    "set_gripper_enable",
    "set_gripper_position",
    "set_gripper_g2_position",
    "clean_gripper_error",
}


class CallLoggingApi:
    """Transparent proxy over the fake: logs EVERY method call and attribute read."""

    def __init__(self, inner: FakeXArmAPI) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "methods", [])
        object.__setattr__(self, "attrs", [])

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if callable(attr):

            def call(*args, **kwargs):
                self.methods.append(name)
                return attr(*args, **kwargs)

            return call
        self.attrs.append(name)
        return attr

    def __setattr__(self, name: str, value) -> None:
        setattr(self._inner, name, value)


def make_monitor(
    fake_kwargs=None,
    *,
    gripper="xarm_g2",
    expect_rail=True,
    proxy=False,
    clock=time.monotonic,
    arm_id="grip",
    connect_fail_times=0,
    **mon_kwargs,
):
    holder: dict = {"factory_calls": 0}

    def factory(ip, **kw):
        # the monitor builds a FRESH client per attempt, so connect failures are
        # scripted here (factory level), not on one fake instance
        holder["factory_calls"] += 1
        if holder["factory_calls"] <= connect_fail_times:
            raise Exception("connect socket failed")  # what the SDK raises
        raw = FakeXArmAPI(ip, **kw, **(fake_kwargs or {}))
        holder["ctor"] = {"ip": ip, **kw}
        holder["raw"] = raw
        holder["api"] = CallLoggingApi(raw) if proxy else raw
        return holder["api"]

    defaults = dict(poll_hz=200.0, stale_s=0.5, reconnect_s=0.01)
    defaults.update(mon_kwargs)
    mon = ArmStateMonitor(
        arm_id,
        "192.168.1.201",
        gripper=gripper,
        expect_rail=expect_rail,
        api_factory=factory,
        clock=clock,
        **defaults,
    )
    return mon, holder


def wait_sample(mon, pred=lambda s: True, timeout=3.0, msg="no matching sample"):
    wait_until(lambda: mon.snapshot() is not None and pred(mon.snapshot()), timeout, msg)
    return mon.snapshot()


# -- connect / sample contents ---------------------------------------------------


def test_connects_read_only_and_samples_in_core_units() -> None:
    q = [0.1, -0.5, 0.2, 0.3, -0.1, 0.2, math.pi]
    tcp_mm = [300.0, -50.0, 220.0, math.pi, 0.0, -0.5]
    mon, h = make_monitor({"initial_q": q, "initial_tcp": tcp_mm, "has_rail": True})
    try:
        mon.start()
        wait_until(lambda: mon.status == "running", msg=f"status {mon.status}: {mon.detail}")
        assert h["ctor"]["is_radian"] is True
        assert h["ctor"]["do_not_open"] is True  # XArmAPI(ip, is_radian=True, do_not_open=True)
        assert h["raw"].call_names().count("connect") == 1
        s = wait_sample(mon)
        assert isinstance(s, ArmMonitorSample)
        assert s.arm_id == "grip" and s.seq >= 1
        assert s.q == pytest.approx(tuple(q))  # rad, unchanged, controller order
        assert s.tcp_pose[:3] == pytest.approx((0.3, -0.05, 0.22))  # mm -> m once
        assert s.tcp_pose[3:] == pytest.approx((math.pi, 0.0, -0.5))  # rad pass-through
        assert s.error_code == 0 and s.warn_code == 0
        assert s.state == 2 and s.mode == 0  # fresh box: standby, position mode
        assert mon.detail == ""
        assert mon.age_s is not None and mon.age_s < 0.5
        s2 = wait_sample(mon, lambda x: x.seq > s.seq)
        assert s2.seq > s.seq and s2.t_mono >= s.t_mono
    finally:
        mon.stop()
    assert mon.status == "off"


def test_controller_error_reported_with_sdk_title() -> None:
    mon, h = make_monitor({"has_rail": True}, gripper="none", arm_id="view")
    try:
        mon.start()
        wait_sample(mon)
        h["raw"].error_code = 19  # Perception Arm's live C19
        s = wait_sample(mon, lambda x: x.error_code == 19)
        assert s.error_code == 19
        assert mon.status == "running"  # reads keep working while an error is latched
        assert mon.detail == "controller error 19: End Effector Communication Error"
        assert controller_error_title(0) == ""
        assert controller_error_title(28).startswith("controller error 28: End Module")
    finally:
        mon.stop()


# -- rail semantics --------------------------------------------------------------


def test_rail_not_homed_gives_raw_mm_but_no_pos_m() -> None:
    # the lab tracks at power-on: {pos: 0, status: 2, error: 0, is_enabled: 0, on_zero: 0}
    mon, h = make_monitor({"has_rail": True, "rail_homed": False, "rail_enabled": False})
    try:
        mon.start()
        s = wait_sample(mon, lambda x: x.rail_present is not None)
        assert s.rail_present is True
        assert s.rail_homed is False and s.rail_enabled is False
        assert s.rail_pos_m is None  # meaningless until homed AND enabled
        assert s.rail_raw_mm == 0.0  # raw register always reported
        # homed + enabled -> position becomes meaningful (mm -> m, clamped)
        raw = h["raw"]
        raw._rail_pos_mm = 325
        raw._rail_on_zero = 1
        raw._rail_enabled = True
        s = wait_sample(mon, lambda x: x.rail_pos_m is not None)
        assert s.rail_homed is True and s.rail_enabled is True
        assert s.rail_pos_m == pytest.approx(0.325)
        assert s.rail_raw_mm == 325.0
        # homed but NOT enabled -> back to None
        raw._rail_enabled = False
        s = wait_sample(mon, lambda x: x.rail_enabled is False)
        assert s.rail_pos_m is None and s.rail_raw_mm == 325.0
    finally:
        mon.stop()


def test_rail_absent_and_rail_not_expected() -> None:
    mon, _ = make_monitor({"has_rail": False})
    try:
        mon.start()
        s = wait_sample(mon, lambda x: x.rail_present is not None)
        assert s.rail_present is False  # registers time out (code 3)
        assert s.rail_homed is None and s.rail_pos_m is None and s.rail_raw_mm is None
    finally:
        mon.stop()
    mon2, h2 = make_monitor({"has_rail": True}, expect_rail=False)
    try:
        mon2.start()
        s = wait_sample(mon2)
        assert s.rail_present is None and s.rail_raw_mm is None  # never polled
        time.sleep(0.05)
        assert "get_linear_track_registers" not in h2["raw"].call_names()
    finally:
        mon2.stop()


def test_simulation_mode_controller_counts_as_no_rail() -> None:
    mon, _ = make_monitor({"has_rail": True, "simulation_robot": True})
    try:
        mon.start()
        s = wait_sample(mon, lambda x: x.rail_present is not None)
        assert s.rail_present is False  # SDK answered (0, []) without touching the bus
    finally:
        mon.stop()


# -- gripper semantics -----------------------------------------------------------


def test_g2_gripper_uses_grippers_py_read_path_and_conversion() -> None:
    mon, h = make_monitor({}, gripper="xarm_g2")
    try:
        h_raw = None
        mon.start()
        s = wait_sample(mon, lambda x: x.gripper_open_frac is not None)
        h_raw = h["raw"]
        assert s.gripper_open_frac == 1.0 and s.gripper_raw == 84.0  # fake starts open
        h_raw._g2_mm = 42.0
        s = wait_sample(mon, lambda x: x.gripper_raw == 42.0)
        assert s.gripper_open_frac == pytest.approx(0.5)  # units.g2_mm_to_frac
        assert "get_gripper_g2_position" in READ_ONLY_SDK_METHODS
    finally:
        mon.stop()


def test_classic_gripper_and_no_gripper() -> None:
    mon, h = make_monitor({}, gripper="xarm")
    try:
        mon.start()
        wait_sample(mon)
        h["raw"]._gripper_pulse = 425
        s = wait_sample(mon, lambda x: x.gripper_raw == 425.0)
        assert s.gripper_open_frac == pytest.approx(0.5)  # units.pulse_to_frac
        assert "set_gripper_enable" not in h["raw"].call_names()  # read-only: no enable
    finally:
        mon.stop()
    mon2, h2 = make_monitor({}, gripper="none")
    try:
        mon2.start()
        s = wait_sample(mon2)
        assert s.gripper_open_frac is None and s.gripper_raw is None
    finally:
        mon2.stop()


# -- ZERO WRITES ------------------------------------------------------------------


def test_zero_writes_only_allowlisted_sdk_members_are_touched() -> None:
    mon, h = make_monitor({"has_rail": True, "rail_homed": False}, gripper="xarm_g2", proxy=True)
    try:
        mon.start()
        wait_sample(mon, lambda x: x.seq >= 20 and x.rail_present is not None)
        raw = h["raw"]
        before = (raw.state, raw.mode, raw.error_code, raw.warn_code, raw.motion_enabled)
        time.sleep(0.05)
    finally:
        mon.stop()
    api = h["api"]
    assert api.methods, "proxy saw no calls"
    unexpected = set(api.methods) - READ_ONLY_SDK_METHODS
    assert not unexpected, f"non-allowlisted SDK calls: {sorted(unexpected)}"
    assert set(api.attrs) <= READ_ONLY_SDK_ATTRS, sorted(set(api.attrs) - READ_ONLY_SDK_ATTRS)
    assert not (set(api.methods) & FORBIDDEN_NAMES)
    assert not any(n.startswith(("set_", "clean_", "motion_", "register_")) for n in api.methods)
    # the fake's own mutating-call log agrees, and the box state is untouched
    mutating = [n for n in raw.call_names() if n not in READ_ONLY_SDK_METHODS]
    assert mutating == []
    assert (raw.state, raw.mode, raw.error_code, raw.warn_code, raw.motion_enabled) == before
    assert before[:2] == (2, 0)  # never enabled, never left mode 0
    assert raw.rail_pos_commands == [] and raw._rail_on_zero == 0  # rail never homed/moved


# -- lifecycle: stop / disconnect (hand-over) / restart ---------------------------------


def test_stop_calls_sdk_disconnect_exactly_once() -> None:
    mon, h = make_monitor({})
    try:
        mon.start()
        wait_sample(mon)
    finally:
        assert mon.stop() is True  # thread joined: the box is released on return
        mon.stop()  # idempotent
    assert h["raw"].call_names().count("disconnect") == 1
    assert h["raw"].connected is False
    assert mon.status == "off" and not mon.connected


def test_disconnect_hands_over_and_start_reconnects() -> None:
    mon, h = make_monitor({})
    mon.start()
    wait_sample(mon)
    first = h["raw"]
    assert mon.disconnect() is True  # released on return (the thread was between polls)
    mon.disconnect()  # idempotent
    assert first.call_names().count("disconnect") == 1
    assert first.connected is False  # the box is free for a session driver
    assert mon.status == "paused" and not mon.connected
    assert "hand-over" in mon.detail
    assert mon.snapshot() is not None  # last reading kept (age tells its story)
    mon.start()  # session over: reconnect
    try:
        wait_until(lambda: mon.status == "running" and h["raw"] is not first)
        wait_sample(mon)
        assert h["raw"].call_names().count("connect") == 1
    finally:
        mon.stop()
    assert h["raw"].call_names().count("disconnect") == 1


def test_disconnect_during_a_slow_connect_never_adopts_the_client() -> None:
    """Hand-over race (SDK 1.18.5 connect() = two sockets with their own timeouts +
    the version handshake, i.e. > the 2 s join budget on a slow link): disconnect()
    returns while the thread is still inside connect(). The client must then be
    released BY THE THREAD, never published, and the status must stay 'paused' —
    not flip back to 'running' with a live second client on the box."""
    mon, h = make_monitor({"connect_delay_s": 0.4})
    mon.start()
    wait_until(lambda: "raw" in h, msg="factory not called")  # thread is inside connect()
    released = mon.disconnect(timeout=0.05)
    assert released is False  # join timed out: the thread still holds the connecting client
    assert mon.status == "paused" and not mon.connected
    assert mon.join(timeout=3.0) is True  # connect() returned; the thread is gone
    raw = h["raw"]
    assert raw.call_names() == ["connect", "disconnect"]  # released once, never polled
    assert raw.connected is False
    assert mon.status == "paused" and not mon.connected and mon.snapshot() is None
    assert "hand-over" in mon.detail
    mon.disconnect()  # idempotent after the late release
    assert raw.call_names().count("disconnect") == 1
    # start() after the hand-over reconnects with a fresh client
    mon.start()
    try:
        wait_until(lambda: mon.status == "running" and h["raw"] is not raw)
        wait_sample(mon)
    finally:
        assert mon.stop() is True
    assert h["raw"].call_names().count("disconnect") == 1


def test_start_while_the_previous_thread_is_still_connecting_waits_for_it() -> None:
    """disconnect() + immediate start() (supervisor edge or a phase-09 resume) while
    the old thread is blocked in connect(): start() waits for the old thread to
    release its client, then connects once with a fresh one — never two clients
    on the box, and the new thread is not a no-op."""
    mon, h = make_monitor({"connect_delay_s": 0.3})
    mon.start()
    wait_until(lambda: "raw" in h, msg="factory not called")
    first = h["raw"]
    assert mon.disconnect(timeout=0.02) is False
    t0 = time.monotonic()
    mon.start()  # blocks until the old thread has left connect() and released
    assert time.monotonic() - t0 >= 0.1
    try:
        assert first.call_names() == ["connect", "disconnect"] and first.connected is False
        wait_until(lambda: mon.status == "running" and h["raw"] is not first)
        assert h["factory_calls"] == 2  # exactly one reconnect
        assert mon.connected and h["raw"].connected is True
    finally:
        mon.stop()
    assert h["raw"].call_names().count("disconnect") == 1


def test_stop_during_a_slow_connect_keeps_status_off() -> None:
    mon, h = make_monitor({"connect_delay_s": 0.3})
    mon.start()
    wait_until(lambda: "raw" in h, msg="factory not called")
    assert mon.stop(timeout=0.02) is False
    assert mon.status == "off"
    assert mon.join(3.0)
    assert mon.status == "off" and not mon.connected
    assert h["raw"].call_names() == ["connect", "disconnect"] and h["raw"].connected is False


# -- failure paths: backoff, link loss, stale ----------------------------------------------


def test_connect_failures_back_off_exponentially_then_recover() -> None:
    mon, h = make_monitor({}, connect_fail_times=3, reconnect_s=0.01)
    try:
        mon.start()
        wait_until(lambda: mon.status == "running", timeout=3.0, msg="never connected")
        assert mon.connect_attempts == 4 and h["factory_calls"] == 4
        assert mon._backoff_log == pytest.approx([0.01, 0.02, 0.04])
        assert h["raw"].call_names().count("connect") == 1
    finally:
        mon.stop()


def test_client_whose_connect_raises_is_released_and_retried() -> None:
    """SDK connect() raising on an already-constructed client: the monitor still
    calls disconnect() on it (best effort) and retries on the next attempt."""
    shared = FakeXArmAPI("192.168.1.201", is_radian=True, do_not_open=True, connect_fails=1)
    mon = ArmStateMonitor(
        "grip",
        "192.168.1.201",
        gripper="none",
        expect_rail=False,
        poll_hz=200.0,
        reconnect_s=0.01,
        api_factory=lambda ip, **kw: shared,
    )
    try:
        mon.start()
        wait_until(lambda: mon.status == "running", timeout=3.0, msg="never connected")
        assert mon.connect_attempts == 2
        assert shared.call_names()[:3] == ["connect", "disconnect", "connect"]
    finally:
        mon.stop()
    assert shared.call_names().count("disconnect") == 2


def test_backoff_caps_at_10_s() -> None:
    assert ArmStateMonitor.next_backoff(2.0) == 4.0
    assert ArmStateMonitor.next_backoff(6.0) == MAX_RECONNECT_S == 10.0
    assert ArmStateMonitor.next_backoff(10.0) == 10.0


def test_error_status_and_detail_while_box_unreachable() -> None:
    def factory(ip, **kw):
        raise Exception("connect socket failed")  # what the SDK raises

    mon = ArmStateMonitor(
        "grip",
        "192.168.1.201",
        gripper="none",
        expect_rail=False,
        poll_hz=100.0,
        reconnect_s=0.01,
        api_factory=factory,
    )
    try:
        mon.start()
        wait_until(lambda: mon.status == "error" and mon.connect_attempts >= 3)
        assert "connect socket failed" in mon.detail
        assert mon.snapshot() is None
    finally:
        mon.stop()
    assert mon.status == "off"


def test_link_loss_reconnects_with_a_fresh_client() -> None:
    mon, h = make_monitor({})
    try:
        mon.start()
        wait_sample(mon)
        first = h["raw"]
        first.connected = False  # SDK .connected flips on a dropped socket
        wait_until(lambda: h["raw"] is not first and mon.status == "running")
        assert first.call_names().count("disconnect") == 1  # old client released
        assert mon.connect_attempts == 2
    finally:
        mon.stop()


def test_poll_exception_sets_error_then_reconnects() -> None:
    mon, h = make_monitor({})
    try:
        mon.start()
        wait_sample(mon)
        first = h["raw"]

        def boom(*a, **k):
            raise Exception("random SDK failure")

        first.get_position = boom
        wait_until(lambda: h["raw"] is not first and mon.status == "running")
        assert first.connected is False
        s = wait_sample(mon, lambda x: x.seq > 0)
        assert s.tcp_pose  # the fresh client reads pose again
    finally:
        mon.stop()


def test_stale_when_no_fresh_sample_and_recovers() -> None:
    fc = FakeClock()
    mon, h = make_monitor({}, clock=fc.now, stale_s=0.5)
    try:
        mon.start()
        wait_sample(mon)
        assert mon.status == "running"
        raw = h["raw"]
        good = raw.get_servo_angle
        raw.get_servo_angle = lambda *a, **k: (3, [])  # box answers timeouts: no sample
        wait_until(lambda: "get_servo_angle returned code 3" in mon.detail)
        assert mon.status == "running"  # fresh enough on the fake clock...
        fc.jump(1.0)  # ...until the last sample ages past stale_s
        assert mon.status == "stale"
        assert mon.age_s is not None and mon.age_s >= 1.0
        raw.get_servo_angle = good
        wait_until(lambda: mon.status == "running" and mon.detail == "")
    finally:
        mon.stop()


def test_ctor_validation() -> None:
    with pytest.raises(ValueError):
        ArmStateMonitor("a", "1.2.3.4", gripper="none", expect_rail=False, poll_hz=0)
    with pytest.raises(ValueError):
        ArmStateMonitor("a", "1.2.3.4", gripper="robotiq", expect_rail=False)  # type: ignore[arg-type]
