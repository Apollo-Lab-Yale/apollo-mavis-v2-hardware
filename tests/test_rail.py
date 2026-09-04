"""RailController + driver-level rail behavior (02-hardware §5)."""

from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.rail import RailController, RailPhase


def test_absent_rail_registers_timeout() -> None:
    api = FakeXArmAPI(has_rail=False)  # nothing on the RS-485 bus -> code 3
    rail = RailController(api)
    assert rail.detect() is False
    assert rail.phase is RailPhase.ABSENT


def test_sim_mode_bogus_sn_means_absent() -> None:
    # sim-mode controllers silently no-op track calls (registers code 0!)
    # but the SN is not a real AL13x track
    api = FakeXArmAPI(has_rail=True, rail_sn="")
    rail = RailController(api)
    assert rail.detect() is False
    api2 = FakeXArmAPI(has_rail=True, rail_sn="XXXX00000000")
    assert RailController(api2).detect() is False


def test_present_rail_detected_by_registers_and_sn() -> None:
    api = FakeXArmAPI(has_rail=True, rail_sn="AL1300FAKE1234")
    rail = RailController(api)
    assert rail.detect() is True
    assert rail.phase is RailPhase.DETECTED


def test_ensure_homed_homes_unhomed_rail_before_anything() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, speed_mm_s=200)
    rail.detect()
    rail.ensure_homed()
    names = api.call_names()
    # back_origin BEFORE enable/speed; commanding unhomed would return 82
    assert "set_linear_track_back_origin" in names
    assert names.index("set_linear_track_back_origin") < names.index(
        "set_linear_track_enable"
    )
    assert rail.phase is RailPhase.READY
    call = [c for c in api.calls if c[0] == "set_linear_track_back_origin"][0]
    assert call[2]["wait"] is True


def test_ensure_homed_skips_homing_when_already_on_zero() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.ensure_homed()
    assert "set_linear_track_back_origin" not in api.call_names()
    assert rail.phase is RailPhase.READY


def test_target_clamped_to_650_mm_and_wait_false() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.ensure_homed()
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
    rail.ensure_homed()
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
    rail.ensure_homed()
    api._rail_pos_mm = 325
    rail.step()
    assert rail.pos_m == 0.325


def test_latch_error_freezes_rail_and_clears_target() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=True)
    rail = RailController(api)
    rail.detect()
    rail.ensure_homed()
    rail.set_target(0.2)
    rail.latch_error(111)  # RS-485 drop: only the rail latches
    rail.step()  # frozen: no command goes out
    assert api.rail_pos_commands == []
    assert rail.phase is RailPhase.RAIL_ERROR
    events = rail.drain_events()
    assert any(code == 111 for _, code, _ in events)
