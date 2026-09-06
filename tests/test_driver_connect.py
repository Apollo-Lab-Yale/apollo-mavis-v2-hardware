"""XArmDriver bring-up, unit boundary, commands (02-hardware §3.1-§3.2)."""

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import (
    ArmIdentityError,
    BringupError,
    CommandError,
    GripperCommand,
    RailExpectedError,
    RailUnavailableError,
)
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.config import ServoLimits, XArmDriverConfig
from apollo_mavis_v2_hardware.driver import ArmFaultedError, DriverPhase, XArmDriver
from apollo_mavis_v2_hardware.events import StudioConflictWarning
from apollo_mavis_v2_hardware.rail import RailNotHomedError


def make_driver(fake_kwargs=None, cfg_kwargs=None, connect=True):
    holder = {}

    def factory(ip, **kw):
        api = FakeXArmAPI(ip, auto_report_hz=200.0, **(fake_kwargs or {}))
        holder["api"] = api
        holder["ctor"] = {"ip": ip, **kw}
        return api

    cfg = XArmDriverConfig(
        arm_id="a1", ip="192.168.1.235", monitor_rate_hz=100.0, **(cfg_kwargs or {})
    )
    drv = XArmDriver(cfg, api_factory=factory)
    if connect:
        drv.connect()
    return drv, holder


def wait_until(cond, timeout=3.0, msg=""):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError(msg or "condition never became true")


def test_api_constructed_with_radians_and_real_report() -> None:
    drv, h = make_driver()
    try:
        assert h["ctor"]["is_radian"] is True  # SDK default is DEGREES
        assert h["ctor"]["report_type"] == "real"  # 30003, 100 Hz push
        assert h["ctor"]["check_joint_limit"] is True
        assert drv.phase is DriverPhase.STREAMING
    finally:
        drv.disconnect()


def test_bringup_order_backstops_before_enable() -> None:
    drv, h = make_driver()
    try:
        names = h["api"].call_names()
        assert names.index("clean_error") < names.index("set_tcp_load")
        assert names.index("set_tcp_load") < names.index("motion_enable")
        # the rail gate (register read) runs BEFORE the arm is enabled, so an unhomed /
        # unverifiable track refuses the connect with the brakes still engaged
        assert names.index("get_linear_track_registers") < names.index("motion_enable")
        assert names.index("motion_enable") < names.index("set_mode")
        # every set_mode is followed by set_state(0); final mode is 1 (servo)
        mode_calls = [c for c in h["api"].calls if c[0] == "set_mode"]
        assert [c[1][0] for c in mode_calls] == [0, 1]
        assert "save_conf" not in names
    finally:
        drv.disconnect()


def test_identity_mismatch_raises() -> None:
    with pytest.raises(ArmIdentityError):
        make_driver(cfg_kwargs={"expected_sn": "XA7-REAL-9999"})


def test_command_joints_rad_reaches_fake_unchanged() -> None:
    """Units boundary: rad in -> rad on the wire (is_radian=True, no scaling)."""
    seed = [0.1] * 7
    drv, h = make_driver(fake_kwargs={"initial_q": list(seed)})
    try:
        target = np.array(seed) + 0.004  # reachable within a few ticks
        drv.command_joints(target)
        wait_until(
            lambda: (
                h["api"].sent_joints
                and np.allclose(h["api"].sent_joints[-1][1], target, atol=1e-12)
            ),
            msg="target never reached the fake in radians",
        )
    finally:
        drv.disconnect()


def test_command_joints_validation() -> None:
    drv, _ = make_driver()
    try:
        with pytest.raises(CommandError):
            drv.command_joints(np.zeros(8))  # dof is 7 without a rail
        with pytest.raises(CommandError):
            drv.command_joints(np.array([np.nan] * 7))
    finally:
        drv.disconnect()


def test_rail_detection_fixes_dof_and_seeds_the_rail_slot() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    try:
        assert drv.has_rail and drv.dof == 8
        names = h["api"].call_names()
        assert "set_linear_track_back_origin" not in names  # connect NEVER homes
        state = drv.get_state()
        assert state.q.shape == (8,)
        assert state.rail_pos_m == 0.0  # homed carriage at zero
        assert drv.rail_phase == "READY"
    finally:
        drv.disconnect()

    drv2, h2 = make_driver()  # no rail on the bus
    try:
        assert not drv2.has_rail and drv2.dof == 7
        with pytest.raises(RailUnavailableError):
            drv2.command_rail(0.3)
    finally:
        drv2.disconnect()


def test_connect_refuses_an_unhomed_rail_and_never_homes() -> None:
    """Phase-09c user rule: no implicit motion. The lab tracks are unhomed at
    power-on; connect() raises RailNotHomedError (step 'rail') and the SDK never
    sees set_linear_track_back_origin — the operator homes from the UI."""
    with pytest.raises(RailNotHomedError) as exc_info:
        make_driver(fake_kwargs={"has_rail": True, "rail_homed": False})
    assert exc_info.value.step == "rail"
    assert "a1" in str(exc_info.value) and "not homed" in str(exc_info.value)
    # the driver instance is reachable through a second construction with connect=False
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": False}, connect=False)
    with pytest.raises(RailNotHomedError):
        drv.connect()
    api = h["api"]
    names = api.call_names()
    assert "set_linear_track_back_origin" not in names
    assert "set_linear_track_enable" not in names and "set_linear_track_speed" not in names
    assert api.homing_started == 0 and api.rail_homed is False
    assert "set_servo_angle_j" not in names and api.sent_joints == []  # no streaming started
    # the refusal comes BEFORE motion_enable / set_mode / set_state: the arm is left
    # exactly as found (brakes engaged, never enabled) - zero writes to the arm
    assert ("motion_enable", (True,), {}) not in api.calls and api.motion_enabled is False
    assert "set_mode" not in names and "set_state" not in names
    assert drv.phase is DriverPhase.IDLE and not drv.has_rail or drv.has_rail  # dof fixed
    assert drv._streamer is None and drv._monitor is None
    drv.disconnect()  # teardown after a refused connect works and hands the arm back (D6)
    tail = [c for c in api.calls if c[0] in ("set_state", "motion_enable", "disconnect")][-3:]
    assert tail == [
        ("set_state", (4,), {}),
        ("motion_enable", (False,), {}),
        ("disconnect", (), {}),
    ]
    assert api.motion_enabled is False


class _TrackDropsAfterDetect(FakeXArmAPI):
    """Registers answer once (detect), then the bus times out (the gate read)."""

    def get_linear_track_registers(self, **kwargs):
        code, regs = super().get_linear_track_registers(**kwargs)
        if self.call_names().count("get_linear_track_registers") >= 2:
            return 3, {}
        return code, regs


def test_connect_refuses_an_unverifiable_track_and_never_enables_the_arm() -> None:
    """A failed gate register read used to latch RAIL_ERROR and let connect() CONTINUE
    with dof 8 and q[7] == 0.0 while the carriage sat at 0.65 m. Now it is a
    BringupError(step 'rail') refusal (not RailNotHomedError: the track may be homed),
    issued before motion_enable."""
    holder = {}

    def factory(ip, **kw):
        api = _TrackDropsAfterDetect(ip, auto_report_hz=200.0, has_rail=True, rail_homed=True)
        api._rail_pos_mm = 650
        holder["api"] = api
        return api

    drv = XArmDriver(XArmDriverConfig(arm_id="a1", ip="192.168.1.235"), api_factory=factory)
    with pytest.raises(BringupError) as exc_info:
        drv.connect()
    assert exc_info.value.step == "rail" and not isinstance(exc_info.value, RailNotHomedError)
    assert "a1: get_linear_track_registers failed (code 3)" in str(exc_info.value)
    api = holder["api"]
    names = api.call_names()
    assert names.count("get_linear_track_registers") == 2  # detect + gate
    assert ("motion_enable", (True,), {}) not in api.calls and api.motion_enabled is False
    assert "set_mode" not in names and "set_servo_angle_j" not in names
    assert [n for n in names if n.startswith("set_linear_track")] == []
    assert drv.phase is DriverPhase.IDLE and drv._rail is None and drv._streamer is None
    drv.disconnect()


def test_connect_seeds_rail_pos_from_the_register_before_the_first_monitor_tick() -> None:
    # homed carriage parked at 0.325 m: the very first get_state() must say so
    # (before phase-09c pos_m was 0.0 until the first 5 Hz step())
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True}, connect=False)
    h_api_holder = h
    # the fake is constructed inside connect(); patch the parking position via the factory
    orig_factory = drv._api_factory

    def parked_factory(ip, **kw):
        api = orig_factory(ip, **kw)
        api._rail_pos_mm = 325
        return api

    drv._api_factory = parked_factory
    drv.connect()
    try:
        api = h_api_holder["api"]
        assert api.call_names().count("set_linear_track_pos") == 0  # never commanded
        assert api.homing_started == 0
        assert drv.get_state().rail_pos_m == pytest.approx(0.325)
        assert drv.get_state().q[7] == pytest.approx(0.325)
    finally:
        drv.disconnect()


def test_expect_rail_yes_but_absent_raises() -> None:
    with pytest.raises(RailExpectedError):
        make_driver(cfg_kwargs={"expect_rail": "yes"})


def test_expect_rail_no_skips_detection() -> None:
    drv, h = make_driver(
        fake_kwargs={"has_rail": True, "rail_homed": True},
        cfg_kwargs={"expect_rail": "no"},
    )
    try:
        assert not drv.has_rail
        assert "get_linear_track_registers" not in h["api"].call_names()
    finally:
        drv.disconnect()


def test_command_rail_clamped_to_650mm() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    try:
        drv.command_rail(0.7)  # over max travel
        wait_until(lambda: h["api"].rail_pos_commands, msg="rail command never sent")
        assert h["api"].rail_pos_commands == [650]  # clamped to 0.650 m
    finally:
        drv.disconnect()


def test_rail_slot_via_command_joints_and_state_roundtrip() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    try:
        q = np.zeros(8)
        q[7] = 0.325
        drv.command_joints(q)
        wait_until(lambda: h["api"].rail_pos_commands == [325])
        wait_until(lambda: drv.get_state().rail_pos_m == pytest.approx(0.325))
        state = drv.get_state()
        assert state.q[7] == pytest.approx(0.325)
        assert state.dq[7] == 0.0  # rail velocity unobservable
    finally:
        drv.disconnect()


def test_gripper_capability_and_command_path() -> None:
    drv, h = make_driver(cfg_kwargs={"gripper": "xarm"})
    try:
        assert drv.gripper_force_capable is False
        drv.command_gripper(GripperCommand(open_frac=0.5, force=1.0))
        wait_until(
            lambda: any(c[0] == "set_gripper_position" for c in h["api"].calls),
            msg="gripper command never drained",
        )
        call = [c for c in h["api"].calls if c[0] == "set_gripper_position"][0]
        assert call[1] == (425,)  # position only; force ignored
    finally:
        drv.disconnect()

    drv2, h2 = make_driver(cfg_kwargs={"gripper": "xarm_g2"})
    try:
        assert drv2.gripper_force_capable is True
        drv2.command_gripper(GripperCommand(open_frac=1.0, force=0.7))
        wait_until(lambda: any(c[0] == "set_gripper_g2_position" for c in h2["api"].calls))
        call = [c for c in h2["api"].calls if c[0] == "set_gripper_g2_position"][0]
        assert call[2]["force"] == 70  # G2 honors force
    finally:
        drv2.disconnect()


def test_stop_latches_and_commands_raise() -> None:
    drv, h = make_driver()
    try:
        drv.stop()
        assert drv.phase is DriverPhase.LATCHED
        assert ("set_state", (4,), {}) in h["api"].calls  # software stop, not STO
        with pytest.raises(ArmFaultedError):
            drv.command_joints(np.zeros(7))
        drv.stop()  # safe twice
    finally:
        drv.disconnect()


def test_disconnect_idempotent() -> None:
    drv, h = make_driver()
    drv.disconnect()
    drv.disconnect()  # never raises
    assert drv.phase is DriverPhase.IDLE


def test_disconnect_hands_the_arm_back_stopped_and_braked_and_leaves_the_track_alone() -> None:
    """Phase-09c D6: after a session the cell is as found at power-on — mode 0,
    state 4 (stopped), motion disabled (brakes engaged). The track keeps its
    homed flag and its enable: no set_linear_track_enable(False), no homing."""
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    api = h["api"]
    assert api.motion_enabled is True and api.mode == 1  # streaming
    n_before = len(api.calls)
    drv.disconnect()
    tail = [(c[0], c[1]) for c in api.calls[n_before:]]
    assert tail == [
        ("set_mode", (0,)),
        ("set_state", (4,)),
        ("motion_enable", (False,)),
        ("disconnect", ()),
    ]
    assert api.motion_enabled is False and api.state == 4 and api.mode == 0
    assert api.connected is False
    assert api.rail_homed is True and api.rail_enabled is True  # track untouched
    assert ("set_linear_track_enable", (False,), {}) not in api.calls
    assert api.homing_started == 0
    assert drv.phase is DriverPhase.IDLE


def test_default_driver_caps_are_the_phase_09c_first_run_values() -> None:
    """D2: hardware caps at speed_scale 1.0 — 0.3 rad/s, 2 mm/tick (0.2 m/s), rail 50 mm/s.
    The runtime scales these per session; the streamer must honour them on the wire."""
    cfg = XArmDriverConfig(arm_id="a1", ip="192.168.1.235")
    assert cfg.servo.max_joint_vel == (0.3,) * 7
    assert cfg.servo.max_cart_step_m == 0.002
    assert cfg.rail_speed_mm_s == 50
    assert ServoLimits().max_joint_vel == (0.3,) * 7 and ServoLimits().max_cart_step_m == 0.002
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    try:
        api = h["api"]
        assert ("set_linear_track_speed", (50,), {}) in api.calls
        drv.command_joints(np.array([0.5] * 7 + [0.0]))  # far target: streamer slews at the cap
        wait_until(lambda: len(api.sent_joints) >= 12, msg="streamer never ticked")
        sent = [np.asarray(q) for _, q in api.sent_joints[:12]]
        dq = np.diff(np.stack(sent), axis=0)
        assert np.all(np.abs(dq) <= 0.3 / cfg.servo.rate_hz + 1e-9)  # per-joint vel cap
        lever = np.asarray(cfg.servo.lever_arm_m)
        assert np.all(np.abs(dq) @ lever <= cfg.servo.max_cart_step_m + 1e-9)  # TCP cap
    finally:
        drv.disconnect()


# -- SDK 1.18.5 surface regressions (first real-box contact, fixed 2026-09-04) ----


def test_fw_gates_come_from_version_number_not_raw_version_string() -> None:
    drv, h = make_driver(fake_kwargs={"version": "7,7,XS1305,MC1303,v1.12.10"})
    try:
        assert drv._fw == (1, 12, 10)  # parse_fw(api.version) gave a bogus major
        assert drv.fw_version == "1.12.10"
    finally:
        drv.disconnect()


def test_report_callback_uses_sdk_1_18_5_keywords_only() -> None:
    # the fake's register_report_callback has the exact SDK signature: a stray
    # ``report_mode=True`` would already have raised TypeError inside connect()
    drv, h = make_driver()
    try:
        call = [c for c in h["api"].calls if c[0] == "register_report_callback"][0]
        assert "report_mode" not in call[2]
        assert set(call[2]) <= FakeXArmAPI.REPORT_CALLBACK_KEYWORDS
        assert call[2]["report_joints"] and call[2]["report_cartesian"]
        assert call[2]["report_state"] and call[2]["report_cmd_num"]
    finally:
        drv.disconnect()


def test_snapshot_mode_comes_from_api_property_not_payload() -> None:
    """The 30003 payload has no ``mode``; reading it as 0 used to trip the
    Studio-conflict detector ~1.2 s after every connect and LATCH the arm."""
    drv, h = make_driver()
    try:
        wait_until(lambda: drv._snap is not None and drv._snap.mode == 1)
        time.sleep(0.7)  # past the 0.5 s set_mode grace window
        assert drv.phase is DriverPhase.STREAMING
        assert not any(isinstance(e, StudioConflictWarning) for e in drv.drain_events())
        assert drv.get_state().mode == 1
    finally:
        drv.disconnect()


def test_rail_without_sn_api_is_detected_and_warns_via_connect_warnings() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    try:
        assert drv.has_rail and drv.dof == 8
        assert any("get_linear_track_sn" in w for w in drv.connect_warnings)
    finally:
        drv.disconnect()
    drv2, _ = make_driver()  # no rail: no rail warning either
    try:
        assert not any("get_linear_track_sn" in w for w in drv2.connect_warnings)
    finally:
        drv2.disconnect()
