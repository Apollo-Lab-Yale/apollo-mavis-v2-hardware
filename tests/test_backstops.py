"""apply_backstops ordering + volatility rules (02-hardware §6)."""

from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.backstops import (
    BACKSTOP_SDK_METHODS,
    apply_backstops,
    expected_backstop_sequence,
)
from apollo_mavis_v2_hardware.config import XArmDriverConfig


def _cfg(**overrides) -> XArmDriverConfig:
    return XArmDriverConfig(arm_id="arm1", ip="192.168.1.235", **overrides)


def test_order_payload_first_rebound_last() -> None:
    api = FakeXArmAPI()
    warnings = apply_backstops(api, _cfg())
    names = api.call_names()
    assert warnings == []
    assert names[0] == "set_tcp_load"  # torque-based collision detection needs payload
    assert names[1] == "set_gravity_direction"
    assert names.index("set_collision_sensitivity") < names.index("set_self_collision_detection")
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


def test_codes_records_every_return_code_in_call_order_and_fake_echoes_readback() -> None:
    api = FakeXArmAPI()
    codes: dict[str, int] = {}
    cfg = _cfg(reduced_tcp_boundary_mm=(700, -700, 600, -600, 800, 0))
    warnings = apply_backstops(api, cfg, codes)
    assert warnings == []
    assert list(codes) == api.call_names() == list(expected_backstop_sequence(cfg))
    assert list(codes) == list(BACKSTOP_SDK_METHODS)
    assert set(codes.values()) == {0}
    # without a boundary the two reduced-mode calls are skipped, order otherwise identical
    assert expected_backstop_sequence(_cfg()) == tuple(
        n for n in BACKSTOP_SDK_METHODS if not n.startswith("set_reduced_")
    )
    # the fake mirrors the controller: the report-frame properties echo the new values
    assert api.collision_sensitivity == 3
    assert api.tcp_load == [0.82, [0.0, 0.0, 48.0]]
    api2 = FakeXArmAPI()
    api2.set_collision_sensitivity = lambda value, wait=True: 1  # type: ignore[method-assign]
    codes2: dict[str, int] = {}
    warnings2 = apply_backstops(api2, _cfg(), codes2)
    assert codes2["set_collision_sensitivity"] == 1 and warnings2 == [
        "set_collision_sensitivity returned 1"
    ]


def test_check_normalises_list_replies_and_explains_state_not_ready():
    """set_collision_rebound returns the raw reply list in SDK 1.18.5, and set_tcp_load
    returns APIState 9 while the arm is stopped although the value is stored (live
    2026-09-04): neither may raise, both are recorded as plain int codes."""
    api = FakeXArmAPI()
    api.set_tcp_load = lambda w, c, wait=False, **kw: 9  # type: ignore[method-assign]
    codes: dict[str, int] = {}
    warnings = apply_backstops(api, XArmDriverConfig(arm_id="grip", ip="1.2.3.4"), codes)
    assert codes["set_collision_rebound"] == 0 and codes["set_tcp_load"] == 9
    assert all(isinstance(v, int) for v in codes.values())
    assert any("set_tcp_load returned 9 (arm stopped" in w for w in warnings)
