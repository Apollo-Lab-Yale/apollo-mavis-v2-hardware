"""XArmDriver.connect(readonly=True) — phase-12 state-only connection (02-hardware §4
additive; 14-dora §4.2 "Arm states without a session"): zero writes, rail / gripper
bookkeeping from register reads, report-stream q + staleness, commands refused."""

import threading
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import CommandError, GripperCommand, RailExpectedError
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import wait_until
from test_monitor import CallLoggingApi

from apollo_mavis_v2_hardware import units
from apollo_mavis_v2_hardware.config import XArmDriverConfig
from apollo_mavis_v2_hardware.driver import (
    READONLY_ALLOWED_SDK_ATTRS,
    READONLY_ALLOWED_SDK_METHODS,
    DriverPhase,
    XArmDriver,
)
from apollo_mavis_v2_hardware.events import StaleEvent

# every write a driving session performs — none may appear on a read-only connection
WRITE_NAMES = frozenset(
    {
        "clean_error",
        "clean_warn",
        "motion_enable",
        "set_mode",
        "set_state",
        "set_servo_angle_j",
        "set_tcp_load",
        "set_collision_sensitivity",
        "set_gripper_enable",
        "set_gripper_mode",
        "set_gripper_speed",
        "set_gripper_position",
        "set_gripper_g2_position",
        "set_linear_track_enable",
        "set_linear_track_speed",
        "set_linear_track_back_origin",
        "set_linear_track_pos",
        "save_conf",
    }
)


def make_readonly(fake_kwargs=None, cfg_kwargs=None, *, proxy=False, connect=True):
    holder: dict = {}

    def factory(ip, **kw):
        raw = FakeXArmAPI(ip, auto_report_hz=200.0, **(fake_kwargs or {}))
        holder["raw"] = raw
        holder["api"] = CallLoggingApi(raw) if proxy else raw
        return holder["api"]

    cfg = XArmDriverConfig(
        arm_id="a1", ip="192.168.1.235", monitor_rate_hz=50.0, **(cfg_kwargs or {})
    )
    drv = XArmDriver(cfg, api_factory=factory)
    if connect:
        drv.connect(readonly=True)
    return drv, holder


def _box_state(raw: FakeXArmAPI):
    return (
        raw.state,
        raw.mode,
        raw.error_code,
        raw.warn_code,
        raw.motion_enabled,
        raw.ready_to_move,
        raw.rail_homed,
        raw.rail_enabled,
        raw.rail_speed,
        raw._gripper_enabled,
    )


def test_readonly_connect_poll_state_stop_disconnect_perform_zero_writes() -> None:
    """The whole read-only life cycle stays inside READONLY_ALLOWED_SDK_METHODS and the
    box is left exactly as found (never enabled, never left mode 0, track and gripper
    untouched)."""
    drv, h = make_readonly(
        {"has_rail": True, "rail_homed": True, "rail_enabled": True},
        {"gripper": "xarm_g2"},
        proxy=True,
    )
    raw, api = h["raw"], h["api"]
    before = _box_state(raw)
    assert drv.phase is DriverPhase.READONLY and drv.readonly is True
    time.sleep(0.5)  # ~25 polls at 50 Hz
    state = drv.get_state()
    drv.stop()
    drv.disconnect()
    assert drv.phase is DriverPhase.IDLE
    # (a) the fake's own mutating-call log
    names = raw.call_names()
    assert names, "no SDK calls recorded"
    assert set(names) <= READONLY_ALLOWED_SDK_METHODS, sorted(
        set(names) - READONLY_ALLOWED_SDK_METHODS
    )
    assert not (set(names) & WRITE_NAMES)
    # (b) the transparent proxy sees EVERY method call and attribute read
    assert api.methods and set(api.methods) <= READONLY_ALLOWED_SDK_METHODS, sorted(
        set(api.methods) - READONLY_ALLOWED_SDK_METHODS
    )
    assert not (set(api.methods) & WRITE_NAMES)
    assert not any(n.startswith(("set_", "clean_", "motion_")) for n in api.methods)
    assert set(api.attrs) <= READONLY_ALLOWED_SDK_ATTRS, sorted(
        set(api.attrs) - READONLY_ALLOWED_SDK_ATTRS
    )
    # the reads it must make: codes + track registers + gripper, and the stream subscription
    assert api.methods.count("get_err_warn_code") >= 10
    assert api.methods.count("get_linear_track_registers") >= 10
    assert api.methods.count("get_gripper_g2_position") >= 10
    assert "get_gripper_position" not in api.methods  # G2 configured: classic read unused
    assert api.methods.count("register_report_callback") == 1
    assert api.methods[-1] == "disconnect" and api.methods.count("disconnect") == 1
    # (c) the box: state/mode/enable/track/gripper unchanged, nothing sent, nothing moved
    assert _box_state(raw) == before
    assert before[:2] == (2, 0) and raw.motion_enabled is False
    assert raw.sent_joints == [] and raw.servo_calls == 0
    assert raw.rail_pos_commands == [] and raw.homing_started == 0
    assert raw.connected is False  # released
    # (d) what it published
    assert state.q.shape == (8,) and state.error_code == 0 and not state.stale
    assert drv._streamer is None and drv._monitor is None  # never created


def test_readonly_rail_detected_and_position_known_only_when_homed_and_enabled() -> None:
    drv, h = make_readonly(
        {"has_rail": True, "rail_homed": True, "rail_enabled": True}, connect=False
    )
    orig = drv._api_factory

    def parked(ip, **kw):
        api = orig(ip, **kw)
        api._rail_pos_mm = 325  # homed carriage parked mid-travel
        return api

    drv._api_factory = parked
    drv.connect(readonly=True)
    try:
        assert drv.has_rail and drv.dof == 8
        assert drv.rail_position_known is True
        assert drv.rail_phase == "DETECTED"  # never READY: a read-only client never enables
        state = drv.get_state()
        assert state.q.shape == (8,) and state.q[7] == pytest.approx(0.325)
        assert state.rail_pos_m == pytest.approx(0.325)
        names = h["raw"].call_names()
        assert "set_linear_track_enable" not in names and "set_linear_track_speed" not in names
        assert "set_linear_track_back_origin" not in names
        assert not any("position UNKNOWN" in w for w in drv.connect_warnings)
    finally:
        drv.disconnect()


@pytest.mark.parametrize(
    "fake_kwargs",
    [
        {"has_rail": True, "rail_homed": False, "rail_enabled": False},  # as at power-on
        {"has_rail": True, "rail_homed": False, "rail_enabled": True},
        {"has_rail": True, "rail_homed": True, "rail_enabled": False},  # homed, not enabled
    ],
)
def test_readonly_unhomed_or_disabled_rail_publishes_the_placeholder(fake_kwargs) -> None:
    """No RailNotHomedError on a read-only connect: the track is detected (dof 8), the
    slot is the 0.0 placeholder flagged rail_position_known == False (phase-09d rule)."""
    drv, h = make_readonly(fake_kwargs)
    try:
        h["raw"]._rail_pos_mm = 400  # a register value that must NOT be trusted
        time.sleep(0.1)
        assert drv.has_rail and drv.dof == 8
        assert drv.rail_position_known is False
        state = drv.get_state()
        assert state.q[7] == 0.0 and state.rail_pos_m == 0.0
        assert any("position UNKNOWN" in w for w in drv.connect_warnings)
        assert h["raw"].homing_started == 0
    finally:
        drv.disconnect()


def test_readonly_picks_up_a_track_homed_by_someone_else() -> None:
    """The operator homes the track (monitor's home_rail op) while the reader holds its
    connection: the next poll flips rail_position_known and publishes the register."""
    drv, h = make_readonly({"has_rail": True, "rail_homed": False, "rail_enabled": False})
    try:
        assert drv.rail_position_known is False
        raw = h["raw"]
        raw.rail_homed = True
        raw.rail_enabled = True
        raw._rail_pos_mm = 0
        wait_until(lambda: drv.rail_position_known, msg="poll never saw the homed track")
        assert drv.get_state().rail_pos_m == 0.0
        raw._rail_pos_mm = 650
        wait_until(lambda: drv.get_state().rail_pos_m == pytest.approx(0.65))
    finally:
        drv.disconnect()


def test_readonly_without_rail_and_expect_rail_rules() -> None:
    drv, h = make_readonly()  # nothing on the bus
    try:
        assert not drv.has_rail and drv.dof == 7 and drv.rail_position_known is False
        assert drv.get_state().q.shape == (7,) and drv.get_state().rail_pos_m is None
    finally:
        drv.disconnect()
    drv2, h2 = make_readonly({"has_rail": True, "rail_homed": True}, {"expect_rail": "no"})
    try:
        assert not drv2.has_rail
        assert "get_linear_track_registers" not in h2["raw"].call_names()
    finally:
        drv2.disconnect()
    with pytest.raises(RailExpectedError):
        make_readonly(cfg_kwargs={"expect_rail": "yes"})


def test_readonly_q_follows_the_report_stream_and_goes_stale_when_it_stops() -> None:
    drv, h = make_readonly({"initial_q": [0.1] * 7})
    try:
        raw = h["raw"]
        wait_until(lambda: not drv.get_state().stale, msg="stream never became fresh")
        assert drv.get_state().q == pytest.approx([0.1] * 7)
        raw._q = [0.3] * 7  # somebody else moves the arm (Studio); we only watch
        wait_until(lambda: drv.get_state().q[0] == pytest.approx(0.3), msg="q never followed")
        assert drv.get_state().mode == 0 and drv.get_state().state == 2  # as the box reports
        drv.drain_events()
        raw.stop_fakes()  # 30003 silence
        t0 = time.monotonic()
        wait_until(lambda: drv.get_state().stale, timeout=1.0, msg="never went stale")
        assert time.monotonic() - t0 <= 0.5
        wait_until(
            lambda: any(isinstance(e, StaleEvent) and e.stale for e in drv.drain_events()),
            timeout=1.0,
            msg="no StaleEvent(stale=True)",
        )
        assert drv.phase is DriverPhase.READONLY  # staleness is reported, never "recovered"
        assert not (set(raw.call_names()) & WRITE_NAMES)
    finally:
        drv.disconnect()


def test_readonly_reports_controller_errors_without_recovering() -> None:
    drv, h = make_readonly()
    try:
        raw = h["raw"]
        assert drv.get_state().error_code == 0
        raw.inject_error(24, warn=11)
        wait_until(lambda: drv.get_state().error_code == 24, msg="poll never saw the error")
        assert drv.get_state().warn_code == 11
        time.sleep(0.1)
        assert drv.phase is DriverPhase.READONLY  # no FAULT / RECOVERING / LATCHED
        assert raw.error_code == 24  # NOT cleared by us
        assert "clean_error" not in raw.call_names() and "motion_enable" not in raw.call_names()
        raw.error_code, raw.warn_code = 0, 0  # cleared elsewhere (Studio / the monitor op)
        wait_until(lambda: drv.get_state().error_code == 0)
        assert drv.recovery_result() is None
    finally:
        drv.disconnect()


def test_readonly_gripper_opening_maps_like_the_backends_without_enable() -> None:
    drv, h = make_readonly(cfg_kwargs={"gripper": "xarm_g2"})
    try:
        raw = h["raw"]
        assert drv.gripper_kind == "xarm_g2" and drv.gripper_force_capable is True
        assert drv.get_state().gripper.open_frac == pytest.approx(units.g2_mm_to_frac(84))
        raw._g2_mm = 42.0
        wait_until(
            lambda: drv.get_state().gripper.open_frac == pytest.approx(units.g2_mm_to_frac(42))
        )
        assert "set_gripper_enable" not in raw.call_names()
    finally:
        drv.disconnect()
    drv2, h2 = make_readonly(cfg_kwargs={"gripper": "xarm"})
    try:
        raw2 = h2["raw"]
        raw2._gripper_pulse = 425
        wait_until(lambda: drv2.get_state().gripper.open_frac == pytest.approx(0.5))
        assert drv2.gripper_force_capable is False
        names = raw2.call_names()
        assert "set_gripper_enable" not in names and "set_gripper_mode" not in names
        assert "set_gripper_speed" not in names  # ClassicGripper.init() never ran
        assert "set_external_device_monitor_params" not in names
    finally:
        drv2.disconnect()
    drv3, h3 = make_readonly(cfg_kwargs={"gripper": "none"}, proxy=True)
    try:
        time.sleep(0.1)
        assert drv3.get_state().gripper.open_frac == 1.0
        assert not {"get_gripper_position", "get_gripper_g2_position"} & set(h3["api"].methods)
    finally:
        drv3.disconnect()


def test_readonly_commands_raise_command_error_and_send_nothing() -> None:
    drv, h = make_readonly({"has_rail": True, "rail_homed": True, "rail_enabled": True})
    try:
        with pytest.raises(CommandError, match="a1: read-only connection"):
            drv.command_joints(np.zeros(8))
        with pytest.raises(CommandError, match="read-only connection"):
            drv.command_gripper(GripperCommand(open_frac=0.5))
        with pytest.raises(CommandError, match="read-only connection"):
            drv.command_rail(0.3)
        with pytest.raises(CommandError, match="read-only connection"):
            drv.home_rail()
        with pytest.raises(CommandError, match="read-only connection"):
            drv.request_recovery()
        drv.clear_errors()  # only meaningful from LATCHED / FAULT: a no-op here
        time.sleep(0.1)
        assert drv.phase is DriverPhase.READONLY
        raw = h["raw"]
        assert raw.sent_joints == [] and raw.rail_pos_commands == []
        assert not (set(raw.call_names()) & WRITE_NAMES)
    finally:
        drv.disconnect()


def test_readonly_stop_and_disconnect_write_nothing_and_release_the_client() -> None:
    drv, h = make_readonly({"has_rail": True, "rail_homed": True, "rail_enabled": True})
    raw = h["raw"]
    poller = drv._poller
    assert poller is not None and poller.alive
    n_before = len(raw.calls)
    drv.stop()  # no set_state(4), no latch
    assert drv.phase is DriverPhase.READONLY
    assert {c[0] for c in raw.calls[n_before:]} <= {"get_linear_track_registers"}  # polls only
    n_before = len(raw.calls)
    drv.disconnect()
    tail = [c[0] for c in raw.calls[n_before:] if c[0] != "get_linear_track_registers"]
    assert tail == ["disconnect"]  # no set_mode(0) / set_state(4) / motion_enable(False)
    assert raw.connected is False and raw.motion_enabled is False and raw.mode == 0
    assert raw.rail_homed is True and raw.rail_enabled is True  # track untouched
    assert drv.phase is DriverPhase.IDLE and not poller.alive and drv._poller is None
    drv.disconnect()  # idempotent
    assert raw.call_names().count("disconnect") == 1


class _LinkDrops(FakeXArmAPI):
    """get_err_warn_code raises (socket gone) after the Nth poll, like the SDK does."""

    def __init__(self, *args, fail_after: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_after = fail_after
        self.err_reads = 0

    def get_err_warn_code(self, show: bool = False, **kwargs):
        self.err_reads += 1
        if self.err_reads > self.fail_after:
            raise Exception("socket closed")  # SDK raises bare Exception
        return super().get_err_warn_code(show, **kwargs)


def test_readonly_poll_failure_marks_the_state_stale_and_never_writes() -> None:
    holder: dict = {}

    def factory(ip, **kw):
        api = _LinkDrops(ip, auto_report_hz=200.0, fail_after=5)
        holder["api"] = api
        return api

    cfg = XArmDriverConfig(arm_id="a1", ip="192.168.1.235", monitor_rate_hz=50.0)
    drv = XArmDriver(cfg, api_factory=factory)
    drv.connect(readonly=True)
    try:
        api = holder["api"]
        wait_until(lambda: not drv.get_state().stale)
        wait_until(lambda: api.err_reads > 5 and drv.get_state().stale, msg="never stale")
        # the report stream is still flowing — staleness comes from the failed poll
        assert drv._snap is not None and (time.monotonic() - drv._snap.mono_ts) < cfg.stale_after_s
        assert drv._poller is not None and drv._poller.alive  # keeps trying (read-only)
        assert drv.phase is DriverPhase.READONLY
        assert not (set(api.call_names()) & WRITE_NAMES)
        api.fail_after = 10**9  # link back
        wait_until(lambda: not drv.get_state().stale, msg="never recovered from a failed poll")
    finally:
        drv.disconnect()


def test_readonly_link_loss_is_stale() -> None:
    drv, h = make_readonly()
    try:
        raw = h["raw"]
        wait_until(lambda: not drv.get_state().stale)
        raw.connected = False  # SDK link dropped (its report thread stops too)
        wait_until(lambda: drv.get_state().stale, msg="link loss never reported")
        assert drv.phase is DriverPhase.READONLY  # no latch: nothing to protect
        assert "clean_error" not in raw.call_names()
    finally:
        drv.disconnect()


def test_default_connect_is_still_the_full_bring_up() -> None:
    """connect() without the flag is byte-for-byte the driving path: enables, enters
    servo mode, streams — and readonly reads False."""
    holder: dict = {}

    def factory(ip, **kw):
        api = FakeXArmAPI(ip, auto_report_hz=200.0, has_rail=True, rail_homed=True)
        holder["api"] = api
        return api

    drv = XArmDriver(XArmDriverConfig(arm_id="a1", ip="192.168.1.235"), api_factory=factory)
    drv.connect()
    try:
        assert drv.readonly is False and drv.phase is DriverPhase.STREAMING
        names = holder["api"].call_names()
        assert {"clean_error", "clean_warn", "motion_enable", "set_mode", "set_state"} <= set(
            names
        )
        assert ("set_linear_track_enable", (True,), {}) in holder["api"].calls
        assert drv._poller is None and drv._monitor is not None and drv._streamer is not None
        assert drv.rail_phase == "READY"
    finally:
        drv.disconnect()
    assert not any(t.name == "hw.a1.readonly" for t in threading.enumerate())  # never started
