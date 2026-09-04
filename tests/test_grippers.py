"""Gripper backends (02-hardware §4): classic pulses, G2 force, fw gates."""

from apollo_mavis_v2_core import GripperCommand
from conftest import FakeClock
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.grippers import (
    ClassicGripper,
    G2Gripper,
    NoGripper,
    make_gripper,
    parse_fw,
)

FW_OLD = (2, 6, 107)
FW_MONITOR = (2, 7, 100)


def _classic(api: FakeXArmAPI, clock: FakeClock, fw=FW_OLD) -> ClassicGripper:
    g = ClassicGripper(clock.now)
    g.init(api, fw)
    return g


def test_parse_fw_tolerates_prefixes() -> None:
    assert parse_fw("v2.6.107") == (2, 6, 107)
    assert parse_fw("2.7.100-beta") == (2, 7, 100)
    assert parse_fw("3.4") == (3, 4, 0)


def test_classic_init_sequence() -> None:
    api = FakeXArmAPI()
    _classic(api, FakeClock())
    names = api.call_names()
    assert names.index("set_gripper_enable") < names.index("set_gripper_mode")
    assert "set_gripper_speed" in names


def test_classic_half_open_is_425_pulses_and_force_ignored() -> None:
    api = FakeXArmAPI()
    g = _classic(api, FakeClock())
    g.command(GripperCommand(open_frac=0.5, force=0.9))  # force must be ignored
    calls = [c for c in api.calls if c[0] == "set_gripper_position"]
    assert len(calls) == 1
    assert calls[0][1] == (425,)  # 0.5 open frac <-> 425 pulses
    assert calls[0][2] == {"wait": False}  # NEVER wait=True in-session
    assert "force" not in calls[0][2]


def test_classic_rate_limit_and_min_delta() -> None:
    api = FakeXArmAPI()
    clock = FakeClock()
    g = _classic(api, clock)
    g.command(GripperCommand(open_frac=0.5))
    g.command(GripperCommand(open_frac=1.0))  # < 0.1 s later: rate-limited
    assert len([c for c in api.calls if c[0] == "set_gripper_position"]) == 1
    clock.jump(0.2)
    g.command(GripperCommand(open_frac=0.504))  # |425 -> 428| < 5 pulses: skipped
    assert len([c for c in api.calls if c[0] == "set_gripper_position"]) == 1
    g.command(GripperCommand(open_frac=1.0))
    assert len([c for c in api.calls if c[0] == "set_gripper_position"]) == 2


def test_classic_current_monitor_fw_gate() -> None:
    api_old = FakeXArmAPI()
    _classic(api_old, FakeClock(), fw=FW_OLD)
    assert "set_external_device_monitor_params" not in api_old.call_names()

    api_new = FakeXArmAPI()
    _classic(api_new, FakeClock(), fw=FW_MONITOR)
    call = [c for c in api_new.calls if c[0] == "set_external_device_monitor_params"][0]
    assert call[2] == {"dev_type": 1, "frequency": 10}


def test_classic_poll_and_injected_current_source() -> None:
    api = FakeXArmAPI()
    api._gripper_pulse = 425
    g = ClassicGripper(FakeClock().now, current_source=lambda: 0.42)
    g.init(api, FW_MONITOR)
    state = g.poll()
    assert state.open_frac == 425 / 850
    assert state.current == 0.42
    assert not g.force_capable


def test_classic_gripper_error_survives_one_clean_cycle() -> None:
    api = FakeXArmAPI()
    g = _classic(api, FakeClock())
    api._gripper_err = 7
    api.clean_gripper_error = lambda: 0  # keep the error latched (fake override)
    g.poll()  # first: clean + re-enable, no fault yet
    assert g.drain_faults() == []
    g.poll()  # error survived the retry -> fault (arm unaffected)
    assert g.drain_faults() == [7]


def test_g2_passes_force_through() -> None:
    api = FakeXArmAPI()
    g = G2Gripper(FakeClock().now)
    g.init(api, FW_OLD)
    g.command(GripperCommand(open_frac=0.5, force=0.8))
    call = [c for c in api.calls if c[0] == "set_gripper_g2_position"][0]
    assert call[1] == (42.0,)  # 0.5 -> 42 mm of the 84 mm span
    assert call[2]["force"] == 80  # normalized [0,1] -> percent
    assert call[2]["wait"] is False
    assert g.force_capable


def test_g2_default_force_and_speed() -> None:
    api = FakeXArmAPI()
    g = G2Gripper(FakeClock().now)
    g.init(api, FW_OLD)
    g.command(GripperCommand(open_frac=1.0))
    call = [c for c in api.calls if c[0] == "set_gripper_g2_position"][0]
    assert call[2]["force"] == 50
    assert call[2]["speed"] == 150


def test_force_capable_flags_via_factory() -> None:
    assert make_gripper("xarm").force_capable is False
    assert make_gripper("xarm_g2").force_capable is True
    assert make_gripper("none").force_capable is False
    assert isinstance(make_gripper("none"), NoGripper)
