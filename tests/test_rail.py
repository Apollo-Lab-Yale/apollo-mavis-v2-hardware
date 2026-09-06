"""RailController + driver-level rail behavior (02-hardware §5).

Phase-09c: connect NEVER homes — ``require_homed()`` refuses an unhomed track
with ``RailNotHomedError`` and zero writes; homing is the monitor's
operator-triggered ``home_rail`` op (tests/test_monitor.py).
Phase-09d: ``require_homed(allow_unhomed=True)`` accepts an unhomed track for the
runtime's maintenance motion (position UNKNOWN, 0.0 placeholder, never commanded)
and ``home()`` homes it on the caller's thread, judged from the registers only
(driver-level tests: tests/test_rail_homing.py)."""

import threading
import time

import pytest
from apollo_mavis_v2_core.errors import BringupError
from apollo_mavis_v2_core.errors import RailNotHomedError as CoreRailNotHomedError
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware import rail as rail_mod
from apollo_mavis_v2_hardware.rail import (
    HOME_RAIL_SDK_WAIT_S,
    UNKNOWN_RAIL_POS_M,
    RailController,
    RailHomeOutcome,
    RailNotHomedError,
    RailPhase,
)


class FakeWithTrackSn(FakeXArmAPI):
    """A hypothetical SDK that DOES expose get_linear_track_sn (1.18.5 does not)."""

    def get_linear_track_sn(self) -> tuple[int, str]:
        self._rec("get_linear_track_sn")
        if not self._rail_present:
            return 3, ""
        return 0, self._rail_sn


def test_absent_rail_registers_timeout() -> None:
    api = FakeXArmAPI(has_rail=False)  # nothing on the RS-485 bus -> code 3
    rail = RailController(api)
    assert rail.detect() is False
    assert rail.phase is RailPhase.ABSENT


def test_sdk_1_18_5_has_no_track_sn_api_detect_by_registers_with_warning() -> None:
    api = FakeXArmAPI(has_rail=True)
    assert not hasattr(api, "get_linear_track_sn")  # mirrors XArmAPI 1.18.5
    rail = RailController(api)
    assert rail.detect() is True  # rail.py:77 used to raise AttributeError here
    assert rail.phase is RailPhase.DETECTED
    assert len(rail.warnings) == 1
    assert "get_linear_track_sn" in rail.warnings[0]


def test_sn_prefix_verified_only_when_the_sdk_exposes_it() -> None:
    api = FakeWithTrackSn(has_rail=True, rail_sn="AL1300FAKE1234")
    rail = RailController(api)
    assert rail.detect() is True
    assert rail.warnings == []
    assert "get_linear_track_sn" in api.call_names()
    # bogus SN (sim-mode controller answering track calls) -> absent
    for bogus in ("", "XXXX00000000"):
        assert RailController(FakeWithTrackSn(has_rail=True, rail_sn=bogus)).detect() is False


def test_simulation_mode_controller_registers_noop_means_absent() -> None:
    # @xarm_is_not_simulation_mode returns (0, []) without touching the bus
    api = FakeXArmAPI(has_rail=True, simulation_robot=True)
    rail = RailController(api)
    assert rail.detect() is False
    assert rail.phase is RailPhase.ABSENT


def test_rail_phase_has_no_homing_value() -> None:
    # the driver never homes, so there is nothing to be "in" while homing
    assert not hasattr(RailPhase, "HOMING")
    assert {p.name for p in RailPhase} == {"ABSENT", "DETECTED", "READY", "RAIL_ERROR"}
    assert not hasattr(RailController, "ensure_homed")  # the homing entry point is gone


def test_rail_not_homed_error_is_the_core_class_with_step_rail() -> None:
    assert RailNotHomedError is CoreRailNotHomedError  # core >= phase-09c ships it
    exc = RailNotHomedError("rail", "x")
    assert exc.step == "rail" and str(exc) == "[rail] x"


def test_require_homed_refuses_unhomed_track_and_never_homes() -> None:
    """The lab tracks at power-on: on_zero 0. connect() must raise and write NOTHING —
    in particular never set_linear_track_back_origin (motion) and never enable."""
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, speed_mm_s=50, arm_id="grip")
    rail.detect()
    with pytest.raises(RailNotHomedError) as exc_info:
        rail.require_homed()
    assert exc_info.value.step == "rail"
    assert "grip" in str(exc_info.value) and "not homed" in str(exc_info.value)
    mutating = [n for n in api.call_names() if n != "get_linear_track_registers"]
    assert mutating == []  # zero writes
    assert api.homing_started == 0 and api.rail_homed is False and api.rail_enabled is False
    assert rail.phase is RailPhase.DETECTED  # not a fault: the operator homes and retries
    assert rail.drain_events() == []
    rail.set_target(0.2)
    rail.step()  # a DETECTED (un-enabled) track is never commanded
    assert api.rail_pos_commands == [] and "get_linear_track_pos" not in api.call_names()


def test_require_homed_enables_sets_speed_and_seeds_pos_from_the_register() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    api._rail_pos_mm = 325  # homed carriage parked mid-track
    rail = RailController(api, speed_mm_s=50)
    rail.detect()
    assert rail.pos_m == 0.0  # nothing known before the gate
    rail.require_homed()
    names = api.call_names()
    assert "set_linear_track_back_origin" not in names
    assert api.homing_started == 0
    mutating = [n for n in names if n.startswith("set_")]
    assert mutating == ["set_linear_track_enable", "set_linear_track_speed"]  # exactly, in order
    assert ("set_linear_track_enable", (True,), {}) in api.calls
    assert ("set_linear_track_speed", (50,), {}) in api.calls
    assert api.rail_enabled is True and api.rail_speed == 50
    assert rail.phase is RailPhase.READY
    # pos_m is seeded BEFORE the first 5 Hz step(): the gate twin never sees 0
    assert rail.pos_m == pytest.approx(0.325)
    assert rail.drain_events() == [("READY", 0, "rail ready at 0.325 m")]


def test_require_homed_is_a_noop_without_a_track() -> None:
    api = FakeXArmAPI(has_rail=False)
    rail = RailController(api)
    assert rail.detect() is False
    rail.require_homed()  # absent: nothing to require
    assert rail.phase is RailPhase.ABSENT
    assert [n for n in api.call_names() if n.startswith("set_")] == []


def test_require_homed_register_read_failure_refuses_the_connect() -> None:
    """A transient modbus failure of the gate read must NOT let the connect continue
    with ``pos_m == 0.0`` for a carriage that may sit at 0.65 m (the gate twin would
    be off by the full travel): BringupError(step 'rail') - not RailNotHomedError,
    the track may well be homed - phase RAIL_ERROR, zero writes."""
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    api._rail_pos_mm = 650
    rail = RailController(api, arm_id="grip")
    rail.detect()
    api._rail_present = False  # the track dropped off the bus between detect and gate
    with pytest.raises(BringupError) as exc_info:
        rail.require_homed()
    assert not isinstance(exc_info.value, RailNotHomedError)
    assert exc_info.value.step == "rail"
    assert "grip: get_linear_track_registers failed (code 3)" in str(exc_info.value)
    assert "carriage position unverifiable" in str(exc_info.value)
    assert rail.phase is RailPhase.RAIL_ERROR
    assert [n for n in api.call_names() if n.startswith("set_")] == []
    assert rail.drain_events() == [("RAIL_ERROR", 3, "get_linear_track_registers failed")]
    assert rail.pos_m == 0.0  # never seeded - and never reported as a position either


class _SpeedWriteFails(FakeXArmAPI):
    def set_linear_track_speed(self, speed: int) -> int:
        self._rec("set_linear_track_speed", speed)
        return 9  # e.g. modbus write rejected


def test_require_homed_refuses_when_the_enable_or_speed_write_fails() -> None:
    """The two setup writes' return codes are checked (they were ignored before):
    a latched track error makes the enable return 80 (SDK semantics) -> refused;
    a failing speed write would leave the D2 positioning cap unapplied -> refused."""
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    api.inject_track_error(3)
    rail = RailController(api, speed_mm_s=50, arm_id="grip")
    rail.detect()
    with pytest.raises(BringupError) as exc_info:
        rail.require_homed()
    assert exc_info.value.step == "rail" and not isinstance(exc_info.value, RailNotHomedError)
    assert "set_linear_track_enable(True) returned 80" in str(exc_info.value)
    assert rail.phase is RailPhase.RAIL_ERROR
    assert [n for n in api.call_names() if n.startswith("set_")] == ["set_linear_track_enable"]
    assert rail.drain_events() == [("RAIL_ERROR", 80, "set_linear_track_enable failed")]

    api = _SpeedWriteFails(has_rail=True, rail_homed=True)
    rail = RailController(api, speed_mm_s=50, arm_id="grip")
    rail.detect()
    with pytest.raises(BringupError) as exc_info:
        rail.require_homed()
    assert exc_info.value.step == "rail"
    assert "set_linear_track_speed(50) returned 9" in str(exc_info.value)
    assert rail.phase is RailPhase.RAIL_ERROR
    assert [n for n in api.call_names() if n.startswith("set_")] == [
        "set_linear_track_enable",
        "set_linear_track_speed",
    ]
    assert api.homing_started == 0  # never homes, whatever fails


def test_target_clamped_to_650_mm_and_wait_false() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.require_homed()
    rail.set_target(0.7)  # over-travel: SDK does not clamp; error 25/26 if sent
    rail.step()
    assert api.rail_pos_commands == [650]
    call = [c for c in api.calls if c[0] == "set_linear_track_pos"][0]
    assert call[2]["wait"] is False
    assert isinstance(call[1][0], int)  # absolute int mm


def test_small_delta_skipped() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.require_homed()
    rail.set_target(0.100)
    rail.step()
    rail.set_target(0.1004)  # |100.4 - 100| < 1 mm -> skip
    rail.step()
    assert api.rail_pos_commands == [100]
    rail.set_target(0.102)
    rail.step()
    assert api.rail_pos_commands == [100, 102]


def test_measured_position_feeds_pos_m() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.require_homed()
    api._rail_pos_mm = 325
    rail.step()
    assert rail.pos_m == 0.325


def test_latch_error_freezes_rail_and_clears_target() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.require_homed()
    rail.set_target(0.2)
    rail.latch_error(111)  # RS-485 drop: only the rail latches
    rail.step()  # frozen: no command goes out
    assert api.rail_pos_commands == []
    assert rail.phase is RailPhase.RAIL_ERROR
    events = rail.drain_events()
    assert any(code == 111 for _, code, _ in events)


# -- phase-09d: allow_unhomed connect gate + home() on the caller's thread ------------


def _track_writes(api: FakeXArmAPI) -> list[str]:
    return [n for n in api.call_names() if n.startswith("set_linear_track")]


def test_require_homed_allow_unhomed_accepts_with_zero_writes_and_a_placeholder() -> None:
    """The maintenance-motion connect: an unhomed track is accepted, NOTHING is written
    (no homing, no enable, no speed), the phase stays DETECTED, the position is flagged
    unknown and reads the 0.0 placeholder, one warning + one DETECTED event say so, and
    the track is still never commanded (targets dropped)."""
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, speed_mm_s=50, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)  # no raise
    assert rail.phase is RailPhase.DETECTED
    assert rail.pos_known is False and rail.pos_m == UNKNOWN_RAIL_POS_M == 0.0
    assert _track_writes(api) == [] and api.homing_started == 0
    assert api.rail_homed is False and api.rail_enabled is False
    assert any("position UNKNOWN" in w and w.startswith("grip: ") for w in rail.warnings)
    events = rail.drain_events()
    assert len(events) == 1 and events[0][0] == "DETECTED" and events[0][1] == 0
    assert "position UNKNOWN" in events[0][2]
    rail.set_target(0.3)  # dropped, not queued
    rail.step()
    assert api.rail_pos_commands == [] and "get_linear_track_pos" not in api.call_names()
    assert rail._target_m is None


def test_require_homed_default_still_refuses_and_allow_unhomed_still_refuses_unreadable() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    with pytest.raises(RailNotHomedError):
        rail.require_homed()  # default: require_homed
    with pytest.raises(RailNotHomedError):
        rail.require_homed(allow_unhomed=False)
    # a track that does not answer cannot be homed either: refused even when allowed
    api._rail_present = False
    with pytest.raises(BringupError) as exc_info:
        rail.require_homed(allow_unhomed=True)
    assert not isinstance(exc_info.value, RailNotHomedError)
    assert rail.phase is RailPhase.RAIL_ERROR and rail.pos_known is False


def test_home_writes_exactly_the_three_track_calls_and_judges_from_the_registers() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False, homing_duration_s=0.05)
    rail = RailController(api, speed_mm_s=5, arm_id="grip")  # 10 % of the 50 mm/s cap
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    rail.drain_events()
    t0 = time.monotonic()
    out = rail.home()
    assert time.monotonic() - t0 >= 0.05  # blocked for the travel
    assert isinstance(out, RailHomeOutcome) and out.ok, out.detail
    assert out.written is True and out.phase == "READY"
    assert (out.on_zero, out.is_enabled, out.error) == (1, 1, 0)
    assert out.pos_m == 0.0 and out.duration_s >= 0.05
    assert out.sdk_codes == {
        "set_linear_track_back_origin": 0,
        "set_linear_track_enable": 0,
        "set_linear_track_speed": 0,
    }
    assert _track_writes(api) == [
        "set_linear_track_back_origin",
        "set_linear_track_enable",
        "set_linear_track_speed",
    ]
    call = [c for c in api.calls if c[0] == "set_linear_track_back_origin"][0]
    assert call[2] == {"wait": True, "timeout": HOME_RAIL_SDK_WAIT_S, "auto_enable": False}
    assert HOME_RAIL_SDK_WAIT_S == 30.0
    assert ("set_linear_track_enable", (True,), {}) in api.calls
    assert ("set_linear_track_speed", (5,), {}) in api.calls
    assert api.homing_started == 1 and api.homing_completed == 1
    assert api.rail_homed and api.rail_enabled and api.rail_speed == 5
    assert rail.phase is RailPhase.READY and rail.pos_known is True and rail.pos_m == 0.0
    assert rail.homing is False
    assert rail.drain_events() == [
        (
            "READY",
            0,
            "grip: rail homed: carriage at 0.000 m, track enabled, positioning speed 5 mm/s",
        )
    ]
    assert out.detail == (
        "grip: rail homed: carriage at 0.000 m, track enabled, positioning speed 5 mm/s"
    )


def test_home_judges_the_registers_never_the_sdk_return_code(monkeypatch) -> None:
    """SDK 1.18.5 can return nonzero for a homing that finished (registers say homed ->
    ok, code kept for diagnosis) and 0 for one that did not (registers say on_zero 0 ->
    failed, RAIL_ERROR, position still unknown)."""
    api = FakeXArmAPI(has_rail=True, rail_homed=False, homing_result_code=100)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    out = rail.home()
    assert out.ok is True and out.sdk_codes["set_linear_track_back_origin"] == 100
    assert "registers are authoritative; set_linear_track_back_origin returned 100" in out.detail
    assert rail.phase is RailPhase.READY and rail.pos_known

    # the carriage is still travelling when the SDK gives up, but the SDK says 0
    monkeypatch.setattr(rail_mod, "HOME_RAIL_SDK_WAIT_S", 0.05)
    api = FakeXArmAPI(has_rail=True, rail_homed=False, homing_duration_s=5.0, homing_result_code=0)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    rail.drain_events()
    out = rail.home()
    assert out.ok is False and out.written is True
    assert out.sdk_codes["set_linear_track_back_origin"] == 0  # lied
    assert out.on_zero == 0 and out.pos_m is None
    assert "on_zero still 0" in out.detail and "grip: rail homing failed" in out.detail
    call = [c for c in api.calls if c[0] == "set_linear_track_back_origin"][0]
    assert call[2]["timeout"] == 0.05  # read at call time
    assert rail.phase is RailPhase.RAIL_ERROR and rail.pos_known is False
    assert rail.pos_m == UNKNOWN_RAIL_POS_M
    events = rail.drain_events()
    assert len(events) == 1 and events[0][0] == "RAIL_ERROR" and "on_zero still 0" in events[0][2]
    assert api.rail_homed is False


def test_home_refused_on_a_latched_track_error_with_zero_writes() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    rail.drain_events()
    api.inject_track_error(25)
    out = rail.home()
    assert out.ok is False and out.written is False
    assert "refused" in out.detail and "linear track error 25" in out.detail
    assert out.error == 25 and out.sdk_codes is None
    assert _track_writes(api) == [] and api.homing_started == 0
    assert rail.phase is RailPhase.DETECTED  # unchanged: nothing happened
    assert rail.drain_events() == [] and rail.homing is False


def test_home_failure_mid_travel_latches_rail_error_and_leaves_the_position_unknown() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False, homing_track_error=26)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    rail.drain_events()
    out = rail.home()
    assert out.ok is False and out.written is True
    assert out.on_zero == 0 and out.is_enabled == 0 and out.error == 26
    assert out.sdk_codes["set_linear_track_back_origin"] == 80
    assert out.sdk_codes["set_linear_track_enable"] == 80
    assert "linear track error 26" in out.detail and "on_zero still 0" in out.detail
    assert rail.phase is RailPhase.RAIL_ERROR and rail.pos_known is False and out.pos_m is None
    events = rail.drain_events()
    assert len(events) == 1 and events[0][0] == "RAIL_ERROR" and events[0][1] == 80
    assert api.homing_started == 1 and api.homing_completed == 0
    rail.set_target(0.2)  # a faulted track is never commanded
    rail.step()
    assert api.rail_pos_commands == []


def test_home_unreadable_registers_fail_before_and_after_the_writes() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed(allow_unhomed=True)
    rail.drain_events()
    api._rail_present = False  # dropped off the bus
    out = rail.home()
    assert out.ok is False and out.written is False and "nothing written" in out.detail
    assert api.homing_started == 0 and rail.phase is RailPhase.RAIL_ERROR
    assert rail.drain_events()[0][2] == "get_linear_track_registers failed before homing"


def test_home_drops_the_pending_target_and_rehoming_a_ready_track_is_allowed() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    api._rail_pos_mm = 325
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed()
    assert rail.phase is RailPhase.READY and rail.pos_m == pytest.approx(0.325)
    rail.set_target(0.3)  # parked before the homing starts
    out = rail.home()
    assert out.ok and rail.pos_m == 0.0 and rail.pos_known
    assert rail._target_m is None and rail._last_sent_mm is None
    rail.step()  # nothing queued may move the carriage away from the zero end
    assert api.rail_pos_commands == []
    assert api.homing_started == 1
    rail.set_target(0.3)  # a FRESH target after the homing is honoured
    rail.step()
    assert api.rail_pos_commands == [300]


def test_home_latch_makes_step_a_noop_and_drops_targets_while_the_carriage_travels() -> None:
    """Threading: home() on a caller thread, step()/set_target() on the monitor thread."""
    api = FakeXArmAPI(has_rail=True, rail_homed=True, homing_duration_s=0.3)
    rail = RailController(api, arm_id="grip")
    rail.detect()
    rail.require_homed()
    rail.set_target(0.3)
    rail.step()
    assert api.rail_pos_commands == [300]
    result: list[RailHomeOutcome] = []
    t = threading.Thread(target=lambda: result.append(rail.home()), daemon=True)
    t.start()
    deadline = time.monotonic() + 2.0
    while not rail.homing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert rail.homing is True
    steps = 0
    while rail.homing and time.monotonic() < deadline:
        rail.set_target(0.5)  # the control loop keeps re-targeting: dropped
        rail.step()  # no bus traffic while homing
        steps += 1
        time.sleep(0.005)
    t.join(timeout=5.0)
    assert not t.is_alive() and steps > 5
    assert result and result[0].ok, result[0].detail if result else "home() never returned"
    assert api.rail_pos_commands == [300]  # nothing sent during or right after the homing
    assert rail.phase is RailPhase.READY and rail.pos_m == 0.0 and rail._target_m is None
    rail.step()  # the cleared target sends nothing ...
    assert api.rail_pos_commands == [300]
    rail.set_target(0.2)  # ... a FRESH target after the homing is honoured again
    rail.step()
    assert api.rail_pos_commands == [300, 200]
