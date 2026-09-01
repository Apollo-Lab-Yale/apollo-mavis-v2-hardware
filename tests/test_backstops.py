"""apply_backstops ordering + volatility rules (02-hardware §6)."""

from fakes.fake_xarm_api import FakeXArmAPI

from apollo_xarm7_hardware.backstops import apply_backstops
from apollo_xarm7_hardware.config import XArmDriverConfig


def _cfg(**overrides) -> XArmDriverConfig:
    return XArmDriverConfig(arm_id="arm1", ip="192.168.1.235", **overrides)


def test_order_payload_first_rebound_last() -> None:
    api = FakeXArmAPI()
    warnings = apply_backstops(api, _cfg())
    names = api.call_names()
    assert warnings == []
    assert names[0] == "set_tcp_load"  # torque-based collision detection needs payload
    assert names[1] == "set_gravity_direction"
    assert names.index("set_collision_sensitivity") < names.index(
        "set_self_collision_detection"
    )
    assert names[-1] == "set_collision_rebound"
    rebound_call = [c for c in api.calls if c[0] == "set_collision_rebound"][0]
    assert rebound_call[1] == (False,)  # stop-and-latch, never bounce


def test_never_persists_controller_config() -> None:
    api = FakeXArmAPI()
    apply_backstops(api, _cfg())
    assert "save_conf" not in api.call_names()  # volatile by design


def test_tool_model_matches_gripper_kind() -> None:
    for kind, model in (("xarm", 1), ("xarm_g2", 9), ("none", 0)):
        api = FakeXArmAPI()
        apply_backstops(api, _cfg(gripper=kind))
        call = [c for c in api.calls if c[0] == "set_collision_tool_model"][0]
        assert call[1] == (model,)


def test_reduced_mode_off_by_default_on_when_boundary_set() -> None:
    api = FakeXArmAPI()
    apply_backstops(api, _cfg())
    assert "set_reduced_mode" not in api.call_names()

    api = FakeXArmAPI()
    apply_backstops(api, _cfg(reduced_tcp_boundary_mm=(700, -700, 600, -600, 800, 0)))
    names = api.call_names()
    # params before the master switch, master switch LAST of the reduced pair
    assert names.index("set_reduced_tcp_boundary") < names.index("set_reduced_mode")


def test_nonzero_codes_become_warnings_not_failures() -> None:
    api = FakeXArmAPI()
    api.set_collision_sensitivity = lambda value, wait=True: 1  # type: ignore[assignment]
    warnings = apply_backstops(api, _cfg())
    assert any("set_collision_sensitivity" in w for w in warnings)


def test_sensitivity_value_from_config() -> None:
    api = FakeXArmAPI()
    apply_backstops(api, _cfg(collision_sensitivity=4))
    call = [c for c in api.calls if c[0] == "set_collision_sensitivity"][0]
    assert call[1] == (4,)
