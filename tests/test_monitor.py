"""ArmStateMonitor (02-hardware §8.5, phase-09a/09b): read-only polling, ZERO writes
unless an explicit maintenance request, rail/gripper semantics + units, safety
read-backs, stale detection, reconnect backoff, hand-over, maintenance channel."""

from __future__ import annotations

import math
import threading
import time

import pytest
from conftest import FakeClock
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import wait_until

from apollo_mavis_v2_hardware import monitor as monitor_mod
from apollo_mavis_v2_hardware.backstops import apply_backstops, expected_backstop_sequence
from apollo_mavis_v2_hardware.config import XArmDriverConfig
from apollo_mavis_v2_hardware.monitor import (
    HOME_RAIL_Q_TOL_RAD,
    HOME_RAIL_SDK_WAIT_S,
    HOME_RAIL_TIMEOUT_S,
    MAINTENANCE_OPS,
    MAINTENANCE_SDK_METHODS,
    MAX_RECONNECT_S,
    READ_ONLY_SDK_ATTRS,
    READ_ONLY_SDK_METHODS,
    STALE_THREAD_JOIN_S,
    STATUS_ECHO_CODES,
    ArmMonitorSample,
    ArmStateMonitor,
    MaintenanceOutcome,
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
    # the safety read-backs are PROPERTY reads (no get_* exists in SDK 1.18.5), never set_*
    assert {"collision_sensitivity", "tcp_load"} <= set(api.attrs)
    assert not (MAINTENANCE_SDK_METHODS["clear_errors"] & set(api.methods))
    assert not (MAINTENANCE_SDK_METHODS["apply_backstops"] & set(api.methods))
    assert not (MAINTENANCE_SDK_METHODS["home_rail"] & set(api.methods))
    assert raw.homing_started == 0  # the poller never homes
    assert mon.maintenance_busy is False


def test_slow_round_reads_back_sensitivity_and_tcp_load() -> None:
    # the Perception Arm as found 2026-09-04: sensitivity 1, payload 0 kg
    mon, h = make_monitor({"collision_sensitivity": 1}, gripper="none", arm_id="view")
    try:
        mon.start()
        s = wait_sample(mon, lambda x: x.collision_sensitivity is not None)
        assert s.collision_sensitivity == 1
        assert s.tcp_load_kg == 0.0 and s.tcp_load_cog_mm == (0.0, 0.0, 0.0)
        raw = h["raw"]
        raw.collision_sensitivity = 3  # what the rich report frame would carry after set_*
        raw.tcp_load = [0.55, [0.0, 0.0, 90.0]]
        s = wait_sample(mon, lambda x: x.collision_sensitivity == 3)
        assert s.tcp_load_kg == pytest.approx(0.55) and s.tcp_load_cog_mm == (0.0, 0.0, 90.0)
        raw.tcp_load = None  # unreadable -> None / ()
        s = wait_sample(mon, lambda x: x.tcp_load_kg is None)
        assert s.tcp_load_cog_mm == () and s.collision_sensitivity == 3
    finally:
        mon.stop()


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


# -- maintenance channel (phase-09b) -------------------------------------------------


def record_threads(raw, *names: str) -> dict[str, str]:
    """Wrap fake methods so the test learns WHICH thread called them."""
    seen: dict[str, str] = {}
    for name in names:
        orig = getattr(raw, name)

        def wrapper(*a, _orig=orig, _name=name, **k):
            seen[_name] = threading.current_thread().name
            return _orig(*a, **k)

        setattr(raw, name, wrapper)
    return seen


def _backstop_cfg(**overrides) -> XArmDriverConfig:
    base = dict(
        arm_id="grip",
        ip="192.168.1.201",
        gripper="xarm_g2",
        tcp_load_kg=0.95,
        tcp_load_cog_mm=(0.0, 0.0, 60.0),
        collision_sensitivity=3,
    )
    base.update(overrides)
    return XArmDriverConfig(**base)


def test_maintenance_clear_errors_writes_exactly_clean_error_and_clean_warn() -> None:
    mon, h = make_monitor({"has_rail": True}, gripper="none", proxy=True, arm_id="view")
    try:
        mon.start()
        wait_sample(mon, lambda x: x.rail_present is not None)
        raw = h["raw"]
        raw.error_code, raw.warn_code = 19, 11  # the Perception Arm's live C19 + a warning
        wait_sample(mon, lambda x: x.error_code == 19)
        threads = record_threads(raw, "clean_error", "clean_warn")
        before_state = (raw.state, raw.mode, raw.motion_enabled)
        out = mon.maintenance("clear_errors", timeout_s=3.0)
        assert isinstance(out, MaintenanceOutcome)
        assert out.ok, out.detail
        assert out.arm_id == "view" and out.op == "clear_errors"
        assert out.before is not None and out.before.error_code == 19 and out.before.warn_code == 11
        assert out.after is not None and out.after.error_code == 0 and out.after.warn_code == 0
        assert out.after.seq > out.before.seq
        assert out.after.rail_present is True  # slow fields refreshed right after the op
        assert list(out.sdk_codes.items()) == [("clean_error", 0), ("clean_warn", 0)]
        assert out.warnings == ()
        assert out.detail == (
            "cleared controller error 19: End Effector Communication Error "
            "and controller warning 11"
        )
        # executed ON THE POLL THREAD, never on the caller's
        assert set(threads.values()) == {"hw.view.monitor-ro"}
        assert (raw.state, raw.mode, raw.motion_enabled) == before_state  # no enable, no mode
        assert mon.maintenance_busy is False
        assert mon.status == "running"
        wait_sample(mon, lambda x: x.seq > out.after.seq)  # polling continues
    finally:
        mon.stop()
    api = h["api"]
    mutating = [n for n in api.methods if n not in READ_ONLY_SDK_METHODS]
    assert mutating == ["clean_error", "clean_warn"]  # the write set, exactly, in order
    assert set(mutating) == MAINTENANCE_SDK_METHODS["clear_errors"]
    assert set(api.attrs) <= READ_ONLY_SDK_ATTRS
    assert not any(n.startswith(("set_", "motion_", "register_")) for n in api.methods)


def test_maintenance_clear_errors_outcomes_without_error_and_when_it_relatches() -> None:
    mon, h = make_monitor({}, gripper="none", arm_id="view")
    try:
        mon.start()
        wait_sample(mon)
        out = mon.maintenance("clear_errors", timeout_s=3.0)
        assert out.ok and out.detail.startswith("no controller error or warning was latched")
        raw = h["raw"]
        # clean_error answers 0 but the controller re-latches immediately (hardware fault)
        raw.error_code = 19
        wait_sample(mon, lambda x: x.error_code == 19)
        raw.clean_error = lambda: 0  # type: ignore[method-assign]
        out = mon.maintenance("clear_errors", timeout_s=3.0)
        assert not out.ok
        assert out.detail == (
            "controller error 19: End Effector Communication Error re-latched right after clearing"
        )
        assert out.after is not None and out.after.error_code == 19
        # a GENUINE SDK failure (transport, not a status echo) still fails the op
        raw.error_code = 0
        wait_sample(mon, lambda x: x.error_code == 0)
        raw.clean_error = lambda: 3  # type: ignore[method-assign]  # ERR_TOUT
        out = mon.maintenance("clear_errors", timeout_s=3.0)
        assert not out.ok and out.detail == "clean_error returned 3"
        assert out.sdk_codes == {"clean_error": 3, "clean_warn": 0}
    finally:
        mon.stop()


def test_clear_errors_status_echo_codes_are_not_failures() -> None:
    """SDK 1.18.5 returns ``clean_error``/``clean_warn`` RAW (no ``_check_code``):
    a box with something latched answers ERR_CODE 1 / WAR_CODE 2 / STATE_NOT_READY 9.
    Those are status echoes, not failures — judging on them reported
    "FAILED - clean_error returned 2" on the Perception Arm on 2026-09-05 while the
    controller error HAD been cleared (the after-sample read error_code 0)."""
    for echo in sorted(STATUS_ECHO_CODES):
        mon, h = make_monitor({}, gripper="none", arm_id="view")
        try:
            mon.start()
            wait_sample(mon)
            raw = h["raw"]
            raw.error_code = 19  # C19 latched, as found on the Perception Arm
            wait_sample(mon, lambda x: x.error_code == 19)

            def clean(raw=raw, echo=echo):  # clears the error AND echoes the status
                raw.error_code = 0
                return echo

            raw.clean_error = clean  # type: ignore[method-assign]
            out = mon.maintenance("clear_errors", timeout_s=3.0)
            assert out.ok, f"echo {echo} judged a failure: {out.detail}"
            assert out.detail.startswith("cleared controller error 19")
            assert out.sdk_codes["clean_error"] == echo  # kept for diagnosis
            assert out.after is not None and out.after.error_code == 0
        finally:
            mon.stop()


def test_maintenance_apply_backstops_matches_backstops_py_sequence() -> None:
    cfg = _backstop_cfg(reduced_tcp_boundary_mm=(700, -700, 600, -600, 800, 0))
    mon, h = make_monitor({"has_rail": True}, gripper="xarm_g2", proxy=True)
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.collision_sensitivity is not None)
        assert s0.collision_sensitivity == 0 and s0.tcp_load_kg == 0.0  # as found on the boxes
        raw = h["raw"]
        threads = record_threads(raw, "set_tcp_load", "set_collision_rebound")
        out = mon.maintenance("apply_backstops", cfg, timeout_s=3.0)
        assert out.ok, out.detail
        assert out.warnings == ()
        assert out.detail == (
            "safety settings applied: sensitivity 3, payload 0.95 kg at (0, 0, 60) mm, "
            "reduced-mode boundary on"
        )
        # the write set == backstops.py, same order, all codes 0
        ref = FakeXArmAPI()
        apply_backstops(ref, cfg)
        assert list(out.sdk_codes) == ref.call_names() == list(expected_backstop_sequence(cfg))
        assert set(out.sdk_codes.values()) == {0}
        # read-back right after the op reflects the config
        assert out.after is not None
        assert out.after.collision_sensitivity == 3
        assert out.after.tcp_load_kg == pytest.approx(0.95)
        assert out.after.tcp_load_cog_mm == (0.0, 0.0, 60.0)
        assert out.before is not None and out.before.collision_sensitivity == 0
        assert set(threads.values()) == {"hw.grip.monitor-ro"}
        assert raw.motion_enabled is False and raw.mode == 0  # still nothing enabled / moved
        assert "save_conf" not in raw.call_names()
        wait_sample(mon, lambda x: x.seq > out.after.seq and x.collision_sensitivity == 3)
    finally:
        mon.stop()
    api = h["api"]
    mutating = [n for n in api.methods if n not in READ_ONLY_SDK_METHODS]
    assert mutating == ref.call_names()
    assert set(mutating) <= MAINTENANCE_SDK_METHODS["apply_backstops"]
    assert set(api.attrs) <= READ_ONLY_SDK_ATTRS
    # without a reduced boundary the two reduced-mode calls are absent
    ref2 = FakeXArmAPI()
    apply_backstops(ref2, _backstop_cfg())
    assert ref2.call_names() == list(expected_backstop_sequence(_backstop_cfg()))
    assert "set_reduced_mode" not in ref2.call_names()


def test_maintenance_apply_backstops_reports_nonzero_codes_as_failure() -> None:
    mon, h = make_monitor({}, gripper="xarm_g2")
    try:
        mon.start()
        wait_sample(mon)
        raw = h["raw"]
        raw.set_collision_sensitivity = lambda value, wait=True: 1  # type: ignore[method-assign]
        out = mon.maintenance("apply_backstops", _backstop_cfg(), timeout_s=3.0)
        assert not out.ok
        assert out.warnings == ("set_collision_sensitivity returned 1",)
        assert out.detail == "set_collision_sensitivity returned 1"
        assert out.sdk_codes["set_collision_sensitivity"] == 1
        assert out.sdk_codes["set_tcp_load"] == 0  # the rest was still applied
        assert out.after is not None and out.after.tcp_load_kg == pytest.approx(0.95)
    finally:
        mon.stop()


def test_maintenance_apply_backstops_tolerates_state_not_ready_when_read_back_matches() -> None:
    """Live 2026-09-04: with the arm stopped (state 4/5) set_tcp_load returns APIState 9
    although the controller stores the value; the read-back decides, the op is ok."""
    mon, h = make_monitor({}, gripper="xarm_g2")
    try:
        mon.start()
        wait_sample(mon)
        raw = h["raw"]
        real_set = raw.set_tcp_load

        def stopped_set_tcp_load(weight, cog, wait=False, **kw):
            real_set(weight, cog, wait=wait, **kw)  # stored anyway
            return 9

        raw.set_tcp_load = stopped_set_tcp_load  # type: ignore[method-assign]
        out = mon.maintenance("apply_backstops", _backstop_cfg(), timeout_s=3.0)
        assert out.ok and out.sdk_codes["set_tcp_load"] == 9
        assert "verified by read-back" in out.detail
        assert out.after is not None and out.after.tcp_load_kg == pytest.approx(0.95)
    finally:
        mon.stop()


def test_maintenance_refusals_never_touch_the_sdk() -> None:
    mon, h = make_monitor({}, gripper="none")
    with pytest.raises(ValueError):
        mon.maintenance("reboot")  # type: ignore[arg-type]
    assert MAINTENANCE_OPS == (
        "clear_errors",
        "apply_backstops",
        "recover",
        "home_rail",
        "set_collision_sensitivity",
    )
    assert set(MAINTENANCE_SDK_METHODS) == set(MAINTENANCE_OPS)
    assert MAINTENANCE_SDK_METHODS["set_collision_sensitivity"] == {"set_collision_sensitivity"}
    # set_collision_sensitivity needs a level of 1..3, refused (not raised) otherwise
    for bad in (None, 0, 4, 5, 2.5, True, "2"):
        out = mon.maintenance("set_collision_sensitivity", level=bad, timeout_s=0.5)  # type: ignore[arg-type]
        assert not out.ok and out.detail.startswith("set_collision_sensitivity needs a level")
    # not started: off
    out = mon.maintenance("clear_errors", timeout_s=0.5)
    assert not out.ok and "monitor off" in out.detail and "raw" not in h
    # recover needs a session driver (enable + servo mode), never the monitor
    out = mon.maintenance("recover", timeout_s=0.5)
    assert not out.ok and out.detail == "recover needs a session"
    out = mon.maintenance("apply_backstops", None, timeout_s=0.5)
    assert not out.ok and "driver config" in out.detail
    try:
        mon.start()
        wait_sample(mon)
        out = mon.maintenance("recover", timeout_s=0.5)
        assert not out.ok and out.detail == "recover needs a session"
        mon.disconnect()  # hand-over: paused
        out = mon.maintenance("clear_errors", timeout_s=0.5)
        assert not out.ok and "monitor paused" in out.detail
        mutating = [n for n in h["raw"].call_names() if n not in READ_ONLY_SDK_METHODS]
        assert mutating == []
    finally:
        mon.stop()
    # unreachable box: status error
    mon2, _ = make_monitor({}, gripper="none", connect_fail_times=10**6)
    try:
        mon2.start()
        wait_until(lambda: mon2.status == "error")
        out = mon2.maintenance("clear_errors", timeout_s=0.5)
        assert not out.ok and "monitor error" in out.detail and "connect" in out.detail
    finally:
        mon2.stop()


def test_maintenance_timeout_drops_the_late_result_and_polling_continues() -> None:
    mon, h = make_monitor({}, gripper="none")
    try:
        mon.start()
        wait_sample(mon)
        raw = h["raw"]

        def slow_clean_error() -> int:
            time.sleep(0.3)
            raw.error_code = 0
            return 0

        raw.clean_error = slow_clean_error  # type: ignore[method-assign]
        t0 = time.monotonic()
        out = mon.maintenance("clear_errors", timeout_s=0.05)
        assert time.monotonic() - t0 < 0.25
        assert not out.ok and "clear_errors timed out after 0.05 s" in out.detail
        assert mon.maintenance_busy is True  # still executing on the poll thread
        wait_until(lambda: not mon.maintenance_busy, timeout=3.0)
        s = mon.snapshot()
        assert s is not None
        wait_sample(mon, lambda x: x.seq > s.seq)  # the monitor kept polling afterwards
        assert mon.status == "running"
        assert raw.call_names().count("clean_warn") == 1  # the op ran exactly once
    finally:
        mon.stop()


def test_maintenance_sdk_exception_fails_the_request_and_reconnects() -> None:
    mon, h = make_monitor({}, gripper="none")
    try:
        mon.start()
        wait_sample(mon)
        first = h["raw"]

        def boom() -> int:
            raise Exception("socket closed")  # what the SDK raises

        first.clean_error = boom  # type: ignore[method-assign]
        out = mon.maintenance("clear_errors", timeout_s=3.0)
        assert not out.ok
        assert out.detail == "clear_errors failed: Exception: socket closed"
        assert out.sdk_codes == {} and out.after is None and out.before is not None
        # the box counts as lost: fresh client, polling resumes
        wait_until(lambda: h["raw"] is not first and mon.status == "running")
        assert first.connected is False
        assert mon.maintenance_busy is False
    finally:
        mon.stop()


def test_maintenance_request_queued_during_an_sdk_call_fails_on_hand_over() -> None:
    """The request is still QUEUED (never popped) when disconnect() runs: it fails
    with the hand-over text and no write is ever issued."""
    mon, h = make_monitor({}, gripper="none")
    mon.start()
    wait_sample(mon)
    raw = h["raw"]
    good = raw.get_position
    inside = threading.Event()

    def slow_get_position(*a, **k):
        inside.set()  # the poll thread is now blocked inside the SDK
        time.sleep(0.6)
        return good(*a, **k)

    raw.get_position = slow_get_position  # type: ignore[method-assign]
    wait_until(inside.is_set, msg="poll never entered the slow SDK call")
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(out=mon.maintenance("clear_errors", timeout_s=5.0))
    )
    t.start()
    wait_until(lambda: mon.maintenance_busy, msg="request never queued")
    assert mon._maintenance_active is False  # queued, not popped: the thread is polling
    assert mon.disconnect(timeout=3.0) is True
    t.join(3.0)
    out = result["out"]
    assert not out.ok and out.detail == "monitor paused: released for hand-over"
    assert "clean_error" not in raw.call_names()  # never ran
    assert mon.maintenance_busy is False
    assert raw.connected is False


def test_maintenance_in_flight_request_refuses_before_its_first_write_on_hand_over() -> None:
    """The request was POPPED and its before-sample is inside the SDK when
    disconnect() runs: the op re-checks the hand-over before the first write and
    refuses; disconnect() waits for that and returns True (box released)."""
    mon, h = make_monitor({}, gripper="none")
    mon.start()
    wait_sample(mon)
    raw = h["raw"]
    good = raw.get_position

    def slow_get_position(*a, **k):
        time.sleep(1.0)  # every poll (incl. the before-sample) sits in the SDK for 1 s
        return good(*a, **k)

    raw.get_position = slow_get_position  # type: ignore[method-assign]
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(out=mon.maintenance("clear_errors", timeout_s=5.0))
    )
    t.start()
    # popped: the poll thread is executing the request (before-sample, >= 1 s in the SDK)
    wait_until(lambda: mon._maintenance_active, timeout=5.0, msg="request never popped")
    t0 = time.monotonic()
    assert mon.disconnect(timeout=3.0) is True
    assert time.monotonic() - t0 < 2.5  # only the before-sample's SDK call was waited for
    t.join(3.0)
    out = result["out"]
    assert not out.ok and out.detail == "monitor paused: released for hand-over"
    assert out.sdk_codes == {} and out.after is None
    assert "clean_error" not in raw.call_names() and "clean_warn" not in raw.call_names()
    assert mon.maintenance_busy is False
    assert mon.status == "paused"
    assert raw.connected is False


def test_maintenance_hand_over_mid_write_waits_for_the_op_and_reports_the_writes() -> None:
    """disconnect() lands AFTER the first write went out: the monitor waits for the
    op instead of disconnecting the client under it, the read-back is skipped and
    the outcome still says what was written."""
    mon, h = make_monitor({}, gripper="none")
    mon.start()
    wait_sample(mon)
    raw = h["raw"]
    raw.error_code = 19
    wait_sample(mon, lambda x: x.error_code == 19)
    good_warn = raw.clean_warn

    def slow_clean_warn() -> int:
        time.sleep(0.6)
        return good_warn()

    raw.clean_warn = slow_clean_warn  # type: ignore[method-assign]
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(out=mon.maintenance("clear_errors", timeout_s=5.0))
    )
    t.start()
    wait_until(lambda: "clean_error" in raw.call_names(), msg="first write never issued")
    # A short join timeout: the op is mid-write, so the shutdown extends its wait.
    assert mon.disconnect(timeout=0.05) is True
    t.join(3.0)
    out = result["out"]
    assert out.ok, out.detail
    assert out.detail == (
        "cleared controller error 19: End Effector Communication Error "
        "(monitor paused: released for hand-over before the read-back)"
    )
    assert list(out.sdk_codes) == ["clean_error", "clean_warn"]
    assert out.before is not None and out.after is None
    names = raw.call_names()
    assert names.index("clean_warn") < names.index("disconnect")  # released AFTER the op
    assert mon.maintenance_busy is False
    assert mon.status == "paused" and raw.connected is False


# -- home_rail: THE one motion op (phase-09c) ------------------------------------------


HOME_Q = (0.1, -0.5, 0.2, 0.3, -0.1, 0.2, math.pi)


def _rail_monitor(fake_kwargs=None, **kw):
    base = {"has_rail": True, "rail_homed": False, "rail_enabled": False, "initial_q": list(HOME_Q)}
    base.update(fake_kwargs or {})
    return make_monitor(base, gripper="xarm_g2", **kw)


def test_home_rail_constants_and_write_set() -> None:
    assert HOME_RAIL_TIMEOUT_S == 45.0 and HOME_RAIL_SDK_WAIT_S == 30.0
    assert HOME_RAIL_Q_TOL_RAD == 0.02
    assert HOME_RAIL_TIMEOUT_S > HOME_RAIL_SDK_WAIT_S > 0
    assert MAINTENANCE_SDK_METHODS["home_rail"] == {
        "set_linear_track_back_origin",
        "set_linear_track_enable",
        "set_linear_track_speed",
    }
    assert "motion_enable" not in MAINTENANCE_SDK_METHODS["home_rail"]  # arm stays braked


def test_home_rail_writes_exactly_the_three_track_calls_and_judges_from_registers() -> None:
    """Happy path on the lab's power-on state (on_zero 0, is_enabled 0): exactly
    back_origin(wait=True, timeout=30, auto_enable=False) -> enable(True) ->
    speed(cfg) on the poll thread; nothing to the arm; ok from the after-sample."""
    mon, h = _rail_monitor({"homing_duration_s": 0.15}, proxy=True)
    cfg = _backstop_cfg(rail_speed_mm_s=50)
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        assert s0.rail_homed is False and s0.rail_enabled is False and s0.rail_error == 0
        raw = h["raw"]
        threads = record_threads(raw, "set_linear_track_back_origin", "set_linear_track_speed")
        arm_before = (raw.state, raw.mode, raw.motion_enabled, raw.error_code)
        t0 = time.monotonic()
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert time.monotonic() - t0 >= 0.15  # blocked for the homing travel
        assert isinstance(out, MaintenanceOutcome) and out.ok, out.detail
        assert out.op == "home_rail" and out.arm_id == "grip"
        assert list(out.sdk_codes.items()) == [
            ("set_linear_track_back_origin", 0),
            ("set_linear_track_enable", 0),
            ("set_linear_track_speed", 0),
        ]
        call = [c for c in raw.calls if c[0] == "set_linear_track_back_origin"][0]
        assert call[2] == {"wait": True, "timeout": HOME_RAIL_SDK_WAIT_S, "auto_enable": False}
        assert ("set_linear_track_enable", (True,), {}) in raw.calls
        assert ("set_linear_track_speed", (50,), {}) in raw.calls
        assert raw.homing_started == 1 and raw.homing_completed == 1
        # judged from the after-sample REGISTERS
        assert out.after is not None and out.before is not None
        assert out.before.rail_homed is False
        assert out.after.rail_homed is True and out.after.rail_enabled is True
        assert out.after.rail_error == 0 and out.after.rail_pos_m == 0.0
        assert out.after.seq > out.before.seq
        assert out.detail == (
            "rail homed: carriage at 0.000 m (register 0 mm), track enabled, "
            "positioning speed 50 mm/s"
        )
        # on the poll thread, never the caller's; the ARM was not touched
        assert set(threads.values()) == {"hw.grip.monitor-ro"}
        assert (raw.state, raw.mode, raw.motion_enabled, raw.error_code) == arm_before
        assert mon.maintenance_busy is False and mon.status == "running"
        wait_sample(mon, lambda x: x.seq > out.after.seq and x.rail_pos_m == 0.0)
    finally:
        mon.stop()
    api = h["api"]
    mutating = [n for n in api.methods if n not in READ_ONLY_SDK_METHODS]
    assert mutating == [
        "set_linear_track_back_origin",
        "set_linear_track_enable",
        "set_linear_track_speed",
    ]
    assert set(mutating) == MAINTENANCE_SDK_METHODS["home_rail"]
    assert set(api.attrs) <= READ_ONLY_SDK_ATTRS
    assert not any(n in FORBIDDEN_NAMES - MAINTENANCE_SDK_METHODS["home_rail"] for n in api.methods)
    assert "motion_enable" not in api.methods and "set_state" not in api.methods


def test_home_rail_refused_on_posture_mismatch_before_any_write() -> None:
    mon, h = _rail_monitor()
    cfg = _backstop_cfg()
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        moved = list(s0.q)
        moved[3] += 0.05  # > 0.02 rad: the sweep was checked at a different posture
        out = mon.maintenance("home_rail", cfg, expected_q=moved)
        assert not out.ok
        assert out.detail.startswith("home_rail refused: the arm moved since the sweep was checked")
        assert "joint 4" in out.detail and "0.050 rad" in out.detail and "0.02 rad" in out.detail
        assert out.sdk_codes == {} and out.after is None
        raw = h["raw"]
        assert raw.homing_started == 0 and raw.rail_homed is False and raw.rail_enabled is False
        assert [n for n in raw.call_names() if n.startswith("set_")] == []
        # within tolerance -> accepted (and a wider caller tolerance is honoured)
        near = list(s0.q)
        near[3] += 0.019
        out = mon.maintenance("home_rail", cfg, expected_q=near)
        assert out.ok, out.detail
        assert raw.homing_started == 1
        out = mon.maintenance("home_rail", cfg, expected_q=moved, q_tol_rad=0.1)
        assert out.ok and raw.homing_started == 2  # re-homing an already homed track is allowed
        assert out.detail.startswith("rail re-homed")
    finally:
        mon.stop()


def test_home_rail_refused_when_no_fresh_sample_could_be_taken() -> None:
    """The posture re-check must run on a FRESH read. `_poll` publishes nothing when
    get_servo_angle fails, so `before` would silently be the PREVIOUS sample (equal to
    expected_q by construction) although the arm moved -> refuse, zero writes."""
    mon, h = _rail_monitor()
    cfg = _backstop_cfg()
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        raw = h["raw"]
        raw._q = [v + 0.5 for v in raw._q]  # the arm moved on every joint ...
        raw.get_servo_angle = lambda *a, **k: (1, [])  # ... and the joint read now fails
        wait_until(lambda: "get_servo_angle returned code 1" in mon.detail)
        seq_before = mon.snapshot().seq
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert not out.ok
        assert out.detail.startswith("home_rail refused: could not take a fresh sample")
        assert "get_servo_angle failed" in out.detail
        assert out.sdk_codes == {} and out.after is None
        assert out.before is not None and out.before.seq == seq_before  # no fresh read
        assert raw.homing_started == 0 and raw.rail_homed is False and raw.rail_enabled is False
        assert [n for n in raw.call_names() if n.startswith("set_")] == []
    finally:
        mon.stop()


def test_home_rail_refused_when_a_controller_or_track_error_is_latched() -> None:
    mon, h = _rail_monitor()
    cfg = _backstop_cfg()
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        raw = h["raw"]
        raw.error_code = 19  # the Perception Arm's live C19
        wait_sample(mon, lambda x: x.error_code == 19)
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert not out.ok
        assert out.detail == (
            "home_rail refused: controller error 19: End Effector Communication Error "
            "is latched; clear errors first"
        )
        raw.error_code = 0
        raw.inject_track_error(25)  # e.g. over-travel latched on the track itself
        s1 = wait_sample(mon, lambda x: x.error_code == 0 and x.rail_error == 25)
        assert "linear track error 25" in mon.detail
        out = mon.maintenance("home_rail", cfg, expected_q=s1.q)
        assert not out.ok and out.detail.startswith("home_rail refused: linear track error 25")
        assert raw.homing_started == 0
        assert [n for n in raw.call_names() if n.startswith("set_")] == []
    finally:
        mon.stop()


def test_home_rail_refusals_without_config_posture_or_track_never_queue() -> None:
    mon, h = _rail_monitor()
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        out = mon.maintenance("home_rail", None, expected_q=s0.q)
        assert not out.ok and "driver config" in out.detail
        out = mon.maintenance("home_rail", _backstop_cfg())
        assert not out.ok and "expected_q" in out.detail
        out = mon.maintenance("home_rail", _backstop_cfg(), expected_q=(0.0,) * 6)
        assert not out.ok and "expected_q" in out.detail
        out = mon.maintenance("home_rail", _backstop_cfg(), expected_q=s0.q, q_tol_rad=0.0)
        assert not out.ok and "q_tol_rad" in out.detail
        assert mon.maintenance_busy is False
        assert [n for n in h["raw"].call_names() if n.startswith("set_")] == []
    finally:
        mon.stop()
    # a monitor that does not poll a rail refuses too
    mon2, h2 = make_monitor({"has_rail": True}, expect_rail=False)
    try:
        mon2.start()
        s = wait_sample(mon2)
        out = mon2.maintenance("home_rail", _backstop_cfg(), expected_q=s.q)
        assert not out.ok and "does not poll a linear track" in out.detail
    finally:
        mon2.stop()
    # no track on the bus: refused from the before-sample, nothing written
    mon3, h3 = make_monitor({"has_rail": False, "initial_q": list(HOME_Q)})
    try:
        mon3.start()
        s = wait_sample(mon3, lambda x: x.rail_present is not None)
        out = mon3.maintenance("home_rail", _backstop_cfg(), expected_q=s.q)
        assert (
            not out.ok and out.detail == "home_rail refused: no linear track detected on this arm"
        )
        assert [n for n in h3["raw"].call_names() if n.startswith("set_")] == []
    finally:
        mon3.stop()


def test_home_rail_judges_only_the_registers_never_the_sdk_return_code(monkeypatch) -> None:
    """SDK 1.18.5 can return nonzero for a homing that finished (101 = register
    reads flaky) and 0 for one that did not (auto_enable masking); the monitor
    passes auto_enable=False and decides from on_zero / is_enabled / error."""
    monkeypatch.setattr(monitor_mod, "HOME_RAIL_SDK_WAIT_S", 0.2)  # keep the timeout case fast
    cfg = _backstop_cfg()
    # (a) code 101 but the carriage reached zero -> ok, code reported for diagnosis
    mon, h = _rail_monitor({"homing_result_code": 101})
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert out.ok, out.detail
        assert out.sdk_codes["set_linear_track_back_origin"] == 101
        assert out.detail.endswith(
            "(registers are authoritative; set_linear_track_back_origin returned 101)"
        )
        assert out.after is not None and out.after.rail_homed and out.after.rail_enabled
    finally:
        mon.stop()
    # (b) code 0 (the SDK lies) but on_zero still 0: the carriage never got there -> not ok
    mon, h = _rail_monitor({"homing_result_code": 0, "homing_duration_s": 10.0})
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        t0 = time.monotonic()
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert 0.2 <= time.monotonic() - t0 < 2.0  # waited the (patched) SDK timeout, no more
        call = [c for c in h["raw"].calls if c[0] == "set_linear_track_back_origin"][0]
        assert call[2]["timeout"] == 0.2 and call[2]["auto_enable"] is False
        assert not out.ok
        assert out.sdk_codes == {
            "set_linear_track_back_origin": 0,
            "set_linear_track_enable": 0,
            "set_linear_track_speed": 0,
        }
        assert (
            out.detail
            == "rail homing failed: on_zero still 0 (carriage did not reach the zero end)"
        )
        assert out.after is not None and out.after.rail_homed is False
        assert out.after.rail_enabled is True  # the follow-up writes still went out (non-motion)
    finally:
        mon.stop()
    # (c) an honest timeout (100) reads the same from the registers; the code is a hint only
    mon, h = _rail_monitor({"homing_duration_s": 10.0})
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert not out.ok and out.sdk_codes["set_linear_track_back_origin"] == 100
        assert out.detail == (
            "rail homing failed: on_zero still 0 (carriage did not reach the zero end) "
            "(set_linear_track_back_origin returned 100)"
        )
    finally:
        mon.stop()
    # (d) a track error during the travel: registers say error 26 (and the enable
    #     could not take) -> not ok, both problems named
    mon, h = _rail_monitor({"homing_duration_s": 10.0})
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        raw = h["raw"]
        threading.Timer(0.05, raw.inject_track_error, args=(26,)).start()
        out = mon.maintenance("home_rail", cfg, expected_q=s0.q)
        assert not out.ok
        assert out.sdk_codes["set_linear_track_back_origin"] == 80
        assert out.sdk_codes["set_linear_track_enable"] == 80
        assert "linear track error 26" in out.detail and "track not enabled" in out.detail
        assert out.after is not None and out.after.rail_error == 26
    finally:
        mon.stop()


def test_fake_back_origin_mirrors_the_sdk_auto_enable_masking() -> None:
    """Pins the SDK 1.18.5 behaviour the monitor works around: with the default
    auto_enable=True the enable's return code overwrites the wait result, so a
    homing that timed out comes back as 0; with auto_enable=False the truth (100)
    survives. Registers tell the real story either way."""
    api = FakeXArmAPI(has_rail=True, rail_homed=False, homing_duration_s=10.0)
    t0 = time.monotonic()
    assert api.set_linear_track_back_origin(wait=True, timeout=0.05) == 0  # masked!
    assert time.monotonic() - t0 < 1.0
    assert api.rail_homed is False and api.rail_enabled is True  # enabled but NOT homed
    api.rail_enabled = False
    assert api.set_linear_track_back_origin(wait=True, timeout=0.05, auto_enable=False) == 100
    assert api.rail_homed is False and api.rail_enabled is False
    # link loss during the wait ends it with 100 too (the SDK loop checks .connected)
    threading.Timer(0.05, api.disconnect).start()
    assert api.set_linear_track_back_origin(wait=True, timeout=5.0, auto_enable=False) == 100
    assert api.rail_homed is False
    # a fast track: the wait result is 0 and the registers agree
    api2 = FakeXArmAPI(has_rail=True, rail_homed=False, homing_duration_s=0.02)
    assert api2.set_linear_track_back_origin(wait=True, timeout=1.0, auto_enable=False) == 0
    assert api2.rail_homed is True and api2.rail_enabled is False and api2.homing_completed == 1
    # a track error aborts with 80 and the enable does not take
    api3 = FakeXArmAPI(has_rail=True, rail_homed=False)
    api3.inject_track_error(25)
    assert api3.set_linear_track_back_origin(wait=True, timeout=1.0) == 80
    assert api3.rail_homed is False and api3.rail_enabled is False
    assert api3.set_linear_track_enable(True) == 80
    api3.clean_linear_track_error()
    assert api3.set_linear_track_enable(True) == 0 and api3.rail_enabled is True


def test_home_rail_hand_over_waits_for_the_homing_to_finish() -> None:
    """disconnect() lands while the carriage is travelling: the monitor must NOT
    pull the SDK client (the SDK wait loop would end early with the track still
    moving); it waits for the op, releases afterwards, and the outcome is ok."""
    mon, h = _rail_monitor({"homing_duration_s": 0.6})
    cfg = _backstop_cfg()
    mon.start()
    s0 = wait_sample(mon, lambda x: x.rail_present is not None)
    raw = h["raw"]
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(out=mon.maintenance("home_rail", cfg, expected_q=s0.q))
    )
    t.start()
    wait_until(lambda: raw.homing_started == 1, msg="homing never started")
    assert mon.maintenance_busy is True
    t0 = time.monotonic()
    assert mon.disconnect(timeout=0.05) is True  # waited for the op, then released
    assert time.monotonic() - t0 >= 0.4
    t.join(3.0)
    out = result["out"]
    assert out.ok, out.detail
    assert out.after is None  # the hand-over pre-empted the read-back...
    assert "monitor paused: released for hand-over before the read-back" in out.detail
    assert list(out.sdk_codes) == [
        "set_linear_track_back_origin",
        "set_linear_track_enable",
        "set_linear_track_speed",
    ]
    names = raw.call_names()
    assert names.index("set_linear_track_speed") < names.index("disconnect")  # released AFTER
    assert raw.rail_homed is True and raw.homing_completed == 1  # ...but the homing finished
    assert raw.rail_enabled is True and raw.connected is False
    assert mon.status == "paused" and mon.maintenance_busy is False


def test_home_rail_hand_over_budget_is_45_s_not_the_generic_15_s(monkeypatch) -> None:
    """The generic mid-write wait is STALE_THREAD_JOIN_S; a homing in flight gets
    HOME_RAIL_TIMEOUT_S. Scaled down: generic 0.1 s, home_rail 1.0 s, homing 0.5 s."""
    monkeypatch.setattr(monitor_mod, "STALE_THREAD_JOIN_S", 0.1)
    monkeypatch.setattr(monitor_mod, "HOME_RAIL_TIMEOUT_S", 1.0)
    assert STALE_THREAD_JOIN_S == 15.0  # the real constants
    mon, h = _rail_monitor({"homing_duration_s": 0.5})
    mon.start()
    s0 = wait_sample(mon, lambda x: x.rail_present is not None)
    raw = h["raw"]
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(
            out=mon.maintenance("home_rail", _backstop_cfg(), expected_q=s0.q, timeout_s=3.0)
        )
    )
    t.start()
    wait_until(lambda: raw.homing_started == 1, msg="homing never started")
    assert mon.stop(timeout=0.02) is True  # 0.5 s > the generic 0.1 s budget, < 1.0 s
    t.join(3.0)
    assert result["out"].ok and raw.rail_homed is True
    assert raw.call_names().index("set_linear_track_speed") < raw.call_names().index("disconnect")


def test_home_rail_status_is_stale_and_busy_while_homing() -> None:
    """Documented: the poll thread sits in the SDK wait, no sample is published,
    so the arm reads 'stale' + maintenance_busy until the op completes."""
    mon, h = _rail_monitor({"homing_duration_s": 0.5}, stale_s=0.1)
    cfg = _backstop_cfg()
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        raw = h["raw"]
        result: dict = {}
        t = threading.Thread(
            target=lambda: result.update(out=mon.maintenance("home_rail", cfg, expected_q=s0.q))
        )
        t.start()
        wait_until(lambda: raw.homing_started == 1, msg="homing never started")
        wait_until(lambda: mon.status == "stale", timeout=1.0, msg=f"status {mon.status}")
        assert mon.maintenance_busy is True
        assert mon.snapshot().rail_homed is False  # last sample predates the homing
        t.join(3.0)
        assert result["out"].ok
        wait_until(lambda: mon.status == "running")
        assert mon.maintenance_busy is False
        wait_sample(mon, lambda x: x.rail_homed is True and x.rail_pos_m == 0.0)
    finally:
        mon.stop()


def test_home_rail_caller_timeout_abandons_but_the_homing_still_completes() -> None:
    mon, h = _rail_monitor({"homing_duration_s": 0.3})
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.rail_present is not None)
        out = mon.maintenance("home_rail", _backstop_cfg(), expected_q=s0.q, timeout_s=0.05)
        assert not out.ok and "home_rail timed out after 0.05 s" in out.detail
        assert mon.maintenance_busy is True  # the op is still running on the poll thread
        wait_until(lambda: not mon.maintenance_busy, timeout=3.0)
        raw = h["raw"]
        assert raw.rail_homed is True and raw.rail_enabled is True and raw.homing_completed == 1
        wait_sample(mon, lambda x: x.rail_homed is True)
    finally:
        mon.stop()


# -- set_collision_sensitivity (2026-09-11): the operator's level override --------------


def test_maintenance_set_collision_sensitivity_writes_one_call_and_judges_from_the_read_back():
    mon, h = make_monitor({"collision_sensitivity": 3, "has_rail": True}, proxy=True)
    try:
        mon.start()
        s0 = wait_sample(mon, lambda x: x.collision_sensitivity == 3)
        raw = h["raw"]
        threads = record_threads(raw, "set_collision_sensitivity")
        before_state = (raw.state, raw.mode, raw.motion_enabled)
        out = mon.maintenance("set_collision_sensitivity", _backstop_cfg(), timeout_s=3.0, level=2)
        assert isinstance(out, MaintenanceOutcome)
        assert out.ok, out.detail
        assert out.op == "set_collision_sensitivity" and out.arm_id == "grip"
        assert out.detail == (
            "collision sensitivity set to 2 (was 3; the config value 3 is re-applied at the "
            "next connect)"
        )
        assert out.sdk_codes == {"set_collision_sensitivity": 0} and out.warnings == ()
        assert out.before is not None and out.before.collision_sensitivity == 3
        assert out.before.seq > s0.seq  # a fresh before-sample
        assert out.after is not None and out.after.collision_sensitivity == 2
        assert out.after.seq > out.before.seq
        assert out.after.rail_present is True  # slow fields refreshed right after the op
        assert set(threads.values()) == {"hw.grip.monitor-ro"}  # ON THE POLL THREAD
        assert (raw.state, raw.mode, raw.motion_enabled) == before_state  # no enable, no mode
        assert raw.homing_started == 0 and raw.rail_pos_commands == []  # nothing moved
        assert "save_conf" not in raw.call_names()
        assert mon.maintenance_busy is False and mon.status == "running"
        wait_sample(mon, lambda x: x.seq > out.after.seq and x.collision_sensitivity == 2)
        # without a driver config the detail still says the value is volatile
        out2 = mon.maintenance("set_collision_sensitivity", timeout_s=3.0, level=1)
        assert out2.ok and out2.detail == (
            "collision sensitivity set to 1 (was 2; the config value is re-applied at the "
            "next connect)"
        )
    finally:
        mon.stop()
    api = h["api"]
    mutating = [n for n in api.methods if n not in READ_ONLY_SDK_METHODS]
    assert mutating == ["set_collision_sensitivity", "set_collision_sensitivity"]  # ONE per op
    assert set(mutating) == MAINTENANCE_SDK_METHODS["set_collision_sensitivity"]
    assert set(api.attrs) <= READ_ONLY_SDK_ATTRS
    assert [c for c in h["raw"].calls if c[0] == "set_collision_sensitivity"] == [
        ("set_collision_sensitivity", (2,), {}),
        ("set_collision_sensitivity", (1,), {}),
    ]


def test_maintenance_set_collision_sensitivity_not_ok_when_the_read_back_disagrees():
    mon, h = make_monitor({"collision_sensitivity": 3}, gripper="none", arm_id="view")
    try:
        mon.start()
        wait_sample(mon, lambda x: x.collision_sensitivity == 3)
        raw = h["raw"]
        # the SDK answers 0 but the controller does not take the value (rich frame keeps 3)
        raw.set_collision_sensitivity = lambda value, wait=True: 0  # type: ignore[method-assign]
        t0 = time.monotonic()
        out = mon.maintenance("set_collision_sensitivity", _backstop_cfg(), timeout_s=3.0, level=2)
        assert time.monotonic() - t0 >= monitor_mod.BACKSTOP_READBACK_SETTLE_S  # it waited
        assert not out.ok
        assert out.detail == "collision sensitivity still reads 3 after writing 2"
        assert out.sdk_codes == {"set_collision_sensitivity": 0}
        assert out.after is not None and out.after.collision_sensitivity == 3
        # a transport code is a failure whatever the read-back says
        real = FakeXArmAPI.set_collision_sensitivity

        def timeout(value, wait=True):
            real(raw, value, wait=wait)
            return 3  # ERR_TOUT

        raw.set_collision_sensitivity = timeout  # type: ignore[method-assign]
        out = mon.maintenance("set_collision_sensitivity", timeout_s=3.0, level=1)
        assert not out.ok and out.detail == "set_collision_sensitivity returned 3"
        assert out.warnings == ("set_collision_sensitivity returned 3",)
    finally:
        mon.stop()


def test_maintenance_set_collision_sensitivity_status_echo_is_diagnosis_not_failure():
    """set_collision_sensitivity returns the RAW reply (x3/xarm.py:964, no _check_code): a
    box with C19 latched echoes ERR_CODE 1 for a write that went through. Like
    clear_errors, the read-back decides; the code stays in sdk_codes."""
    for echo in sorted(STATUS_ECHO_CODES):
        mon, h = make_monitor({"collision_sensitivity": 3}, gripper="none", arm_id="view")
        try:
            mon.start()
            wait_sample(mon, lambda x: x.collision_sensitivity == 3)
            raw = h["raw"]
            raw.error_code = 19
            wait_sample(mon, lambda x: x.error_code == 19)
            real = FakeXArmAPI.set_collision_sensitivity

            def stored_with_echo(value, wait=True, raw=raw, echo=echo, real=real):
                real(raw, value, wait=wait)
                return echo

            raw.set_collision_sensitivity = stored_with_echo  # type: ignore[method-assign]
            out = mon.maintenance(
                "set_collision_sensitivity", _backstop_cfg(), timeout_s=3.0, level=2
            )
            assert out.ok, f"echo {echo} judged a failure: {out.detail}"
            assert out.detail == (
                "collision sensitivity set to 2 (was 3; the config value 3 is re-applied at "
                f"the next connect) (set_collision_sensitivity returned {echo}: status echo, "
                "value verified by read-back)"
            )
            assert out.sdk_codes == {"set_collision_sensitivity": echo}
            assert out.after is not None and out.after.collision_sensitivity == 2
            assert out.after.error_code == 19  # the op clears nothing
        finally:
            mon.stop()


def test_maintenance_set_collision_sensitivity_refusals_never_touch_the_sdk():
    mon, h = make_monitor({}, gripper="none")
    # not started
    out = mon.maintenance("set_collision_sensitivity", timeout_s=0.5, level=2)
    assert not out.ok and "monitor off" in out.detail and "raw" not in h
    try:
        mon.start()
        wait_sample(mon)
        raw = h["raw"]
        for bad in (None, 0, 4, 5):
            out = mon.maintenance("set_collision_sensitivity", timeout_s=0.5, level=bad)
            assert not out.ok
            assert out.detail.startswith(
                f"set_collision_sensitivity needs a level of 1, 2 or 3 (got {bad!r})"
            )
        mon.disconnect()  # hand-over
        out = mon.maintenance("set_collision_sensitivity", timeout_s=0.5, level=2)
        assert not out.ok and "monitor paused" in out.detail
        assert "set_collision_sensitivity" not in raw.call_names()
        assert mon.maintenance_busy is False
    finally:
        mon.stop()
