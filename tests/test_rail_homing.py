"""Phase-09d: ``XArmDriver.home_rail()`` on a CONNECTED driver + the
``rail_homing: "allow_unhomed"`` connect path (02-hardware §3.2 / §5 / §9).

The runtime's rail-homing maintenance job connects ONE arm with an unhomed track
(``XArmDriverConfig.rail_homing == "allow_unhomed"``), pre-positions it along a
twin-planned path while the servo stream holds the joints, then calls
``home_rail()`` from its own thread. Normal sessions keep the default
``"require_homed"`` and are still refused with ``RailNotHomedError``.
"""

import threading
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import CommandError, RailUnavailableError
from apollo_mavis_v2_core.schemas import ArmConfig, PoseModel, WorkcellConfig
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import make_driver, wait_until

from apollo_mavis_v2_hardware.config import XArmDriverConfig
from apollo_mavis_v2_hardware.driver import DriverPhase, XArmDriver
from apollo_mavis_v2_hardware.events import RailEvent
from apollo_mavis_v2_hardware.rail import HOME_RAIL_SDK_WAIT_S, RailHomeOutcome, RailNotHomedError
from apollo_mavis_v2_hardware.workcell import HardwareWorkcell

UNHOMED = {"has_rail": True, "rail_homed": False}
ALLOW = {"rail_homing": "allow_unhomed"}


def _track_writes(api: FakeXArmAPI) -> list[str]:
    return [n for n in api.call_names() if n.startswith("set_linear_track")]


def _arm_writes(api: FakeXArmAPI) -> list[tuple]:
    return [(c[0], c[1]) for c in api.calls if c[0] in ("motion_enable", "set_mode", "set_state")]


# -- config default + the normal session path is unchanged ----------------------------------


def test_rail_homing_defaults_to_require_homed_and_connect_still_refuses_an_unhomed_rail():
    cfg = XArmDriverConfig(arm_id="a1", ip="192.168.1.235")
    assert cfg.rail_homing == "require_homed"
    assert XArmDriverConfig.model_fields["rail_homing"].default == "require_homed"
    with pytest.raises(ValueError):
        XArmDriverConfig(arm_id="a1", ip="192.168.1.235", rail_homing="home_on_connect")
    with pytest.raises(RailNotHomedError):
        make_driver(fake_kwargs=UNHOMED)  # default config: refused, exactly as in phase-09c
    with pytest.raises(RailNotHomedError):
        make_driver(fake_kwargs=UNHOMED, cfg_kwargs={"rail_homing": "require_homed"})
    # the runtime's speed-scaled copy keeps the literal (model_copy carries every field)
    scaled = cfg.model_copy(update={"rail_speed_mm_s": 5})
    assert scaled.rail_homing == "require_homed"
    assert cfg.model_copy(update=ALLOW).rail_homing == "allow_unhomed"


# -- allow_unhomed connect --------------------------------------------------------------------


def test_allow_unhomed_connect_proceeds_with_dof_8_a_placeholder_slot_and_the_unknown_flag():
    drv, h = make_driver(fake_kwargs=UNHOMED, cfg_kwargs=ALLOW)
    api = h["api"]
    try:
        assert drv.phase is DriverPhase.STREAMING and drv.has_rail and drv.dof == 8
        assert drv.rail_phase == "DETECTED" and drv.rail_position_known is False
        # zero track writes: no homing, no enable, no speed, no positioning
        assert _track_writes(api) == [] and api.homing_started == 0
        assert api.rail_homed is False and api.rail_enabled is False
        # the arm itself is brought up normally and streams the hold posture
        assert api.motion_enabled and api.mode == 1
        wait_until(lambda: len(api.sent_joints) >= 3, msg="streamer never ticked")
        # core ArmState cannot carry NaN/None in q[7]: the slot is a 0.0 PLACEHOLDER
        state = drv.get_state()
        assert state.q.shape == (8,) and state.dof == 8 and state.has_rail
        assert state.q[7] == 0.0 and state.rail_pos_m == 0.0 and np.all(np.isfinite(state.q))
        assert any("position UNKNOWN" in w for w in drv.connect_warnings)
        events = [e for e in drv.drain_events() if isinstance(e, RailEvent)]
        assert [e.phase for e in events] == ["DETECTED"] and "UNKNOWN" in events[0].detail
        # the rail slot of command_joints is DROPPED silently (the loop's hold keeps
        # flowing), an explicit command_rail is a programming error
        drv.command_joints(np.r_[np.zeros(7), 0.3])
        time.sleep(0.1)  # > several monitor periods at 100 Hz
        assert api.rail_pos_commands == [] and "set_linear_track_pos" not in api.call_names()
        with pytest.raises(CommandError, match="not homed"):
            drv.command_rail(0.3)
    finally:
        drv.disconnect()
    # D6 hand-back untouched; the track was never enabled so nothing to leave alone
    assert api.motion_enabled is False and api.state == 4 and api.homing_started == 0


# -- home_rail() on a connected driver ------------------------------------------------------------


def test_home_rail_keeps_streaming_writes_the_exact_track_set_and_judges_from_registers():
    drv, h = make_driver(
        fake_kwargs={**UNHOMED, "homing_duration_s": 0.3},
        cfg_kwargs={**ALLOW, "rail_speed_mm_s": 5},  # the job connects at speed_scale 0.1
    )
    api = h["api"]
    try:
        wait_until(lambda: len(api.sent_joints) >= 3)
        drv.drain_events()
        seed = list(api.sent_joints[-1][1])
        arm_before = _arm_writes(api)
        n_sent0 = len(api.sent_joints)
        caller = threading.current_thread().name
        homing_threads: set[str] = set()
        orig = api.set_linear_track_back_origin

        def spy(*a, **kw):
            homing_threads.add(threading.current_thread().name)
            return orig(*a, **kw)

        api.set_linear_track_back_origin = spy
        t0 = time.monotonic()
        out = drv.home_rail()
        dt = time.monotonic() - t0
        assert dt >= 0.3, dt  # blocked on the caller's thread for the travel
        assert isinstance(out, RailHomeOutcome) and out.ok, out.detail
        assert homing_threads == {caller}  # runs where it is called, not on a driver thread
        # the servo stream HELD the joints throughout: ticks kept coming, posture unchanged
        n_sent1 = len(api.sent_joints)
        assert n_sent1 - n_sent0 >= 10, (n_sent0, n_sent1)
        assert all(np.allclose(q, seed) for _, q in api.sent_joints[n_sent0:n_sent1])
        # exact linear-track write set, SDK 1.18.5 kwargs; nothing to the ARM
        assert _track_writes(api) == [
            "set_linear_track_back_origin",
            "set_linear_track_enable",
            "set_linear_track_speed",
        ]
        call = [c for c in api.calls if c[0] == "set_linear_track_back_origin"][0]
        assert call[2] == {"wait": True, "timeout": HOME_RAIL_SDK_WAIT_S, "auto_enable": False}
        assert ("set_linear_track_speed", (5,), {}) in api.calls
        assert _arm_writes(api) == arm_before
        assert api.homing_started == 1 and api.homing_completed == 1
        # judged from the registers; state now a MEASUREMENT
        assert (out.on_zero, out.is_enabled, out.error) == (1, 1, 0) and out.written
        assert out.pos_m == 0.0 and out.phase == "READY"
        assert drv.rail_phase == "READY" and drv.rail_position_known is True
        assert drv.phase is DriverPhase.STREAMING and drv.dof == 8
        state = drv.get_state()
        assert state.q[7] == 0.0 and state.rail_pos_m == 0.0 and not state.stale
        events = [e for e in drv.drain_events() if isinstance(e, RailEvent)]
        assert [(e.phase, e.code) for e in events] == [("READY", 0)]
        assert "rail homed: carriage at 0.000 m" in events[0].detail
        # the hold target the loop derives from the published state (0.0) is motionless;
        # the track is commandable again for a FRESH target
        drv.command_joints(np.r_[seed, 0.0])
        time.sleep(0.05)
        assert api.rail_pos_commands == []
        drv.command_joints(np.r_[seed, 0.3])
        wait_until(lambda: api.rail_pos_commands == [300], msg="fresh rail target never sent")
    finally:
        drv.disconnect()
    assert api.rail_homed and api.rail_enabled  # D6: track left homed + enabled


def test_home_rail_judges_registers_not_the_sdk_code_on_the_driver_path():
    drv, h = make_driver(
        fake_kwargs={**UNHOMED, "homing_result_code": 100},  # SDK claims timeout
        cfg_kwargs=ALLOW,
    )
    try:
        out = drv.home_rail()
        assert out.ok and out.sdk_codes["set_linear_track_back_origin"] == 100
        assert drv.rail_position_known and drv.rail_phase == "READY"
    finally:
        drv.disconnect()


def test_home_rail_failure_latches_only_the_rail_and_the_position_stays_unknown():
    drv, h = make_driver(fake_kwargs={**UNHOMED, "homing_track_error": 26}, cfg_kwargs=ALLOW)
    api = h["api"]
    try:
        wait_until(lambda: len(api.sent_joints) >= 3)
        drv.drain_events()
        out = drv.home_rail()
        assert out.ok is False and out.written is True
        assert out.error == 26 and out.on_zero == 0 and out.pos_m is None
        assert "linear track error 26" in out.detail
        assert drv.rail_phase == "RAIL_ERROR" and drv.rail_position_known is False
        events = [e for e in drv.drain_events() if isinstance(e, RailEvent)]
        assert [(e.phase, e.code) for e in events] == [("RAIL_ERROR", 80)]
        # the ARM is unaffected: still streaming, still holding
        assert drv.phase is DriverPhase.STREAMING
        n = len(api.sent_joints)
        wait_until(lambda: len(api.sent_joints) > n + 3)
        state = drv.get_state()
        assert state.q[7] == 0.0 and state.rail_pos_m == 0.0  # placeholder, flagged unknown
        with pytest.raises(CommandError, match="not homed"):
            drv.command_rail(0.1)
        drv.command_joints(np.r_[np.zeros(7), 0.4])  # dropped: faulted track never commanded
        time.sleep(0.05)
        assert api.rail_pos_commands == []
    finally:
        drv.disconnect()


def test_home_rail_refusals_never_write():
    # not connected
    drv, _ = make_driver(fake_kwargs=UNHOMED, cfg_kwargs=ALLOW, connect=False)
    with pytest.raises(CommandError, match="not connected"):
        drv.home_rail()
    # no track on the bus
    drv, h = make_driver()
    try:
        with pytest.raises(RailUnavailableError):
            drv.home_rail()
    finally:
        drv.disconnect()
    # LATCHED (software stop): the stream does not hold the joints -> refused, zero writes
    drv, h = make_driver(fake_kwargs=UNHOMED, cfg_kwargs=ALLOW)
    api = h["api"]
    try:
        drv.stop()
        assert drv.phase is DriverPhase.LATCHED
        out = drv.home_rail()
        assert out.ok is False and out.written is False and "latched" in out.detail
        assert "nothing written" in out.detail and out.phase == "DETECTED"
        assert _track_writes(api) == [] and api.homing_started == 0
        assert drv.rail_position_known is False
    finally:
        drv.disconnect()
    # a controller error cached by the monitor poll -> refused before any write
    drv, h = make_driver(fake_kwargs=UNHOMED, cfg_kwargs=ALLOW)
    api = h["api"]
    try:
        drv._err_code = 19  # white-box: what the 5 Hz err/warn poll caches
        out = drv.home_rail()
        assert out.ok is False and out.written is False
        assert "controller error 19" in out.detail
        assert _track_writes(api) == [] and api.homing_started == 0
    finally:
        drv.disconnect()


def test_home_rail_from_a_job_thread_while_the_loop_keeps_commanding_the_rail_slot():
    """The runtime's job thread homes; the control loop keeps calling command_joints
    (rail slot included) at its own rate; the driver's monitor thread keeps stepping.
    Nothing may reach set_linear_track_pos during the homing, and the target set
    during it is cleared afterwards (only a FRESH one moves the carriage)."""
    drv, h = make_driver(fake_kwargs={**UNHOMED, "homing_duration_s": 0.3}, cfg_kwargs=ALLOW)
    api = h["api"]
    try:
        wait_until(lambda: len(api.sent_joints) >= 3)
        seed = np.asarray(api.sent_joints[-1][1])
        result: list[RailHomeOutcome] = []
        job = threading.Thread(target=lambda: result.append(drv.home_rail()), name="rail-job")
        job.start()
        deadline = time.monotonic() + 3.0
        while not drv._rail.homing and time.monotonic() < deadline:
            time.sleep(0.001)
        assert drv._rail.homing
        n_cmds = 0
        while job.is_alive() and time.monotonic() < deadline:
            drv.command_joints(np.r_[seed, 0.5])  # the loop "wants" 0.5 m: dropped
            n_cmds += 1
            time.sleep(0.005)
        job.join(timeout=5.0)
        assert not job.is_alive() and n_cmds > 5
        assert result and result[0].ok, result[0].detail if result else "no outcome"
        time.sleep(0.05)  # several monitor periods: the cleared target sends nothing
        assert api.rail_pos_commands == []
        assert drv.rail_phase == "READY" and drv.rail_position_known
        assert drv.get_state().q[7] == 0.0
        drv.command_joints(np.r_[seed, 0.5])  # fresh target after the homing: honoured
        wait_until(lambda: api.rail_pos_commands == [500])
    finally:
        drv.disconnect()


# -- HardwareWorkcell mapping ---------------------------------------------------------------------


def _wc_config() -> WorkcellConfig:
    return WorkcellConfig(
        kind="hardware",
        arms=[ArmConfig(id="grip", ip="192.168.1.201", base_in_world=PoseModel())],
        cameras=[],
        digital_twin_scene="mavis_v2",
    )


def test_workcell_allow_unhomed_reports_rail_unhomed_connected_and_no_error():
    apis: dict[str, FakeXArmAPI] = {}

    def driver_factory(cfg):  # what the runtime's homing job does: flip the literal
        def api_factory(ip, **kw):
            api = FakeXArmAPI(ip, auto_report_hz=100.0, has_rail=True, rail_homed=False, **kw)
            apis[cfg.arm_id] = api
            return api

        return XArmDriver(
            cfg.model_copy(update={"rail_homing": "allow_unhomed", "rail_speed_mm_s": 5}),
            api_factory=api_factory,
            sleep=lambda s: time.sleep(min(s, 0.01)),
        )

    wc = HardwareWorkcell(_wc_config(), driver_factory=driver_factory)
    try:
        statuses = wc.bring_up(timeout_s=5.0)
        s = statuses["grip"]
        assert s.connected is True and s.error is None and s.rail == "unhomed"
        assert any("position UNKNOWN" in w for w in s.warnings)
        drv = wc.arms["grip"]
        assert drv.dof == 8 and drv.rail_position_known is False
        assert wc.states()["grip"].q[7] == 0.0
        out = drv.home_rail()
        assert out.ok, out.detail
        assert drv.rail_position_known and wc.states()["grip"].rail_pos_m == 0.0
        assert apis["grip"].rail_homed and apis["grip"].rail_speed == 5
    finally:
        wc.shutdown()
    api = apis["grip"]
    assert api.motion_enabled is False and api.state == 4  # D6 hand-back
    assert api.rail_homed and api.rail_enabled  # track left homed + enabled


def test_workcell_default_config_still_maps_an_unhomed_rail_to_a_refused_connect():
    def driver_factory(cfg):
        return XArmDriver(
            cfg,
            api_factory=lambda ip, **kw: FakeXArmAPI(
                ip, auto_report_hz=100.0, has_rail=True, rail_homed=False, **kw
            ),
            sleep=lambda s: time.sleep(min(s, 0.01)),
        )

    wc = HardwareWorkcell(_wc_config(), driver_factory=driver_factory)
    try:
        s = wc.bring_up(timeout_s=5.0)["grip"]
        assert s.connected is False and s.rail == "unhomed" and s.error and "not homed" in s.error
    finally:
        wc.shutdown()
