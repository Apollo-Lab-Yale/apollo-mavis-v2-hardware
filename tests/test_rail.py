"""RailController + driver-level rail behavior (02-hardware §5)."""

from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.rail import RailController, RailPhase


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


def test_ensure_homed_homes_unhomed_rail_before_anything() -> None:
    api = FakeXArmAPI(has_rail=True, rail_homed=False)
    rail = RailController(api, speed_mm_s=200)
    rail.detect()
    rail.ensure_homed()
    names = api.call_names()
    # back_origin BEFORE enable/speed; commanding unhomed would return 82
    assert "set_linear_track_back_origin" in names
    assert names.index("set_linear_track_back_origin") < names.index("set_linear_track_enable")
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
