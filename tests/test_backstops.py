"""apply_backstops ordering + volatility rules (02-hardware §6)."""

import pytest
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.backstops import (
    BACKSTOP_SDK_METHODS,
    COLLISION_SENSITIVITY_LEVELS,
    STATUS_ECHO_CODES,
    apply_backstops,
    expected_backstop_sequence,
    set_collision_sensitivity,
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


# -- set_collision_sensitivity: the operator's level override (2026-09-11) -------------


def test_set_collision_sensitivity_is_one_write_with_the_apply_backstops_arguments():
    """Exactly one SDK call, the same call + arguments as step (2) of apply_backstops
    (``wait=False``), the operator's level instead of the config value; the fake
    echoes it into the rich-frame property; the code lands in ``codes``."""
    api = FakeXArmAPI(collision_sensitivity=3)
    codes: dict[str, int] = {}
    warnings = set_collision_sensitivity(api, 2, codes)
    assert warnings == [] and codes == {"set_collision_sensitivity": 0}
    assert api.calls == [("set_collision_sensitivity", (2,), {})]
    assert api.collision_sensitivity == 2
    assert "save_conf" not in api.call_names()  # volatile, like the rest
    # the same call apply_backstops issues, argument for argument
    ref = FakeXArmAPI()
    apply_backstops(ref, XArmDriverConfig(arm_id="a", ip="1.2.3.4", collision_sensitivity=2))
    ours = [c for c in api.calls if c[0] == "set_collision_sensitivity"]
    theirs = [c for c in ref.calls if c[0] == "set_collision_sensitivity"]
    assert ours == theirs
    # codes is optional
    assert set_collision_sensitivity(FakeXArmAPI(), 1) == []


def test_set_collision_sensitivity_refuses_everything_but_1_2_3_before_any_call():
    assert COLLISION_SENSITIVITY_LEVELS == frozenset({1, 2, 3})
    for bad in (0, 4, 5, -1, 2.5, True, False, None, "2"):
        api = FakeXArmAPI()
        with pytest.raises((ValueError, TypeError)):
            set_collision_sensitivity(api, bad)  # type: ignore[arg-type]
        assert api.calls == []  # nothing reached the SDK
    # ints in float clothing are accepted as their int (2.0 -> 2), never written as a float
    api = FakeXArmAPI()
    set_collision_sensitivity(api, 2.0)  # type: ignore[arg-type]
    assert api.calls == [("set_collision_sensitivity", (2,), {})]


def test_set_collision_sensitivity_reports_nonzero_codes_as_the_one_warning():
    """The raw uxbus reply (x3/xarm.py:964 skips _check_code): a latched box answers a
    STATUS_ECHO (1 / 2 / 9) for a write that went through; a transport code (3) is a
    real failure. The helper reports both the same way - the caller decides."""
    assert STATUS_ECHO_CODES == frozenset({1, 2, 9})
    for code in (1, 2, 9, 3):
        api = FakeXArmAPI()
        real = api.set_collision_sensitivity

        def echo(value, wait=True, _real=real, _code=code):
            _real(value, wait=wait)  # the controller stored it
            return _code

        api.set_collision_sensitivity = echo  # type: ignore[method-assign]
        codes: dict[str, int] = {}
        warnings = set_collision_sensitivity(api, 3, codes)
        assert codes == {"set_collision_sensitivity": code}
        assert warnings == [f"set_collision_sensitivity returned {code}"]
        assert api.collision_sensitivity == 3
    # list replies are normalised like set_collision_rebound's
    api = FakeXArmAPI()
    api.set_collision_sensitivity = lambda value, wait=True: [0, 7]  # type: ignore[method-assign]
    codes2: dict[str, int] = {}
    assert set_collision_sensitivity(api, 1, codes2) == [] and codes2 == {
        "set_collision_sensitivity": 0
    }
