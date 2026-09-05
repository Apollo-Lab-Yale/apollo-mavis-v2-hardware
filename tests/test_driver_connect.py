"""XArmDriver bring-up, unit boundary, commands (02-hardware §3.1-§3.2)."""

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import (
    ArmIdentityError,
    CommandError,
    GripperCommand,
    RailExpectedError,
    RailUnavailableError,
)
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.config import XArmDriverConfig
from apollo_mavis_v2_hardware.driver import ArmFaultedError, DriverPhase, XArmDriver
from apollo_mavis_v2_hardware.events import StudioConflictWarning


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


def test_rail_detection_fixes_dof() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": False})
    try:
        assert drv.has_rail and drv.dof == 8
        # unhomed at power-on: bring-up homes BEFORE any position command
        names = h["api"].call_names()
        assert "set_linear_track_back_origin" in names
        state = drv.get_state()
        assert state.q.shape == (8,)
        assert state.rail_pos_m == 0.0
    finally:
        drv.disconnect()

    drv2, h2 = make_driver()  # no rail on the bus
    try:
        assert not drv2.has_rail and drv2.dof == 7
        with pytest.raises(RailUnavailableError):
            drv2.command_rail(0.3)
    finally:
        drv2.disconnect()


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
