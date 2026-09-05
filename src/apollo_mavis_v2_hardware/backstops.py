"""Controller-side safety backstops (02-hardware §6; overview §6 layer 3).

Volatile, re-applied on every connect; never ``save_conf()`` (controllers
stay config-clean). Non-fatal failures are returned as warning strings —
the arm still comes up, the UI surfaces them.

The parameters come from the core ``ArmConfig`` (``tcp_load_kg``,
``tcp_load_cog_mm``, ``collision_sensitivity``, ``reduced_tcp_boundary_mm``)
mapped into ``XArmDriverConfig`` by ``workcell._driver_cfg``; phase-09b also
lets the session-less read-only monitor apply them on an explicit operator
request (``ArmStateMonitor.maintenance("apply_backstops", cfg)``).
"""

from __future__ import annotations

from typing import Any

from .config import XArmDriverConfig

# set_collision_tool_model tool types (SDK doc)
TOOL_MODEL_NONE = 0
TOOL_MODEL_XARM_GRIPPER = 1
TOOL_MODEL_XARM_GRIPPER_G2 = 9

_GRIPPER_TOOL_MODEL = {
    "xarm": TOOL_MODEL_XARM_GRIPPER,
    "xarm_g2": TOOL_MODEL_XARM_GRIPPER_G2,
    "none": TOOL_MODEL_NONE,
}

# The complete, ordered XArmAPI write sequence of apply_backstops(). The two
# reduced-mode calls are issued only when cfg.reduced_tcp_boundary_mm is set.
BACKSTOP_SDK_METHODS: tuple[str, ...] = (
    "set_tcp_load",
    "set_gravity_direction",
    "set_collision_sensitivity",
    "set_self_collision_detection",
    "set_collision_tool_model",
    "set_reduced_tcp_boundary",  # only with a boundary configured
    "set_reduced_mode",  # only with a boundary configured
    "set_collision_rebound",
)


def expected_backstop_sequence(cfg: XArmDriverConfig) -> tuple[str, ...]:
    """The exact ``XArmAPI`` call order :func:`apply_backstops` issues for ``cfg``."""
    if cfg.reduced_tcp_boundary_mm is not None:
        return BACKSTOP_SDK_METHODS
    return tuple(n for n in BACKSTOP_SDK_METHODS if not n.startswith("set_reduced_"))


def apply_backstops(
    api: Any, cfg: XArmDriverConfig, codes: dict[str, int] | None = None
) -> list[str]:
    """Apply controller-enforced limits, in the §6 binding order.

    (1) payload first — collision detection is torque-estimate based, a wrong
    payload means false positives/negatives; (2) collision sensitivity;
    (3) self-collision detection + tool model; (4) optional reduced-mode TCP
    boundary (master switch LAST); (5) stop-and-latch, never bounce.

    ``codes`` (optional) receives every SDK return code keyed by call name, in
    call order (the maintenance channel reports them to the operator).
    """
    warnings: list[str] = []

    def _check(name: str, code: Any) -> None:
        # SDK 1.18.5 quirk (live, 2026-09-04): set_collision_rebound returns the raw
        # reply LIST [code, ...] instead of ret[0] like every other setter.
        if isinstance(code, (list, tuple)):
            code = code[0] if code else -1
        code = int(code)
        if codes is not None:
            codes[name] = code
        if code != 0:
            note = ""
            if code == 9 and name == "set_tcp_load":
                # APIState 9 = STATE_NOT_READY: the SDK reports it while the arm is
                # stopped (state 4/5) although the controller stores the value —
                # verified by the tcp_load read-back (live, 2026-09-04).
                note = " (arm stopped; value verified by read-back)"
            warnings.append(f"{name} returned {code}{note}")

    # (1) payload + gravity FIRST (wait=False: the SDK defaults would wait_move(),
    #     which returns 9 while the arm is stopped)
    _check("set_tcp_load", api.set_tcp_load(cfg.tcp_load_kg, list(cfg.tcp_load_cog_mm), wait=False))
    _check("set_gravity_direction", api.set_gravity_direction([0, 0, -1], wait=False))
    # (2) collision sensitivity (volatile; re-applied every connect)
    _check(
        "set_collision_sensitivity",
        api.set_collision_sensitivity(cfg.collision_sensitivity, wait=False),
    )
    # (3) self-collision model with the correct end-tool geometry
    _check("set_self_collision_detection", api.set_self_collision_detection(True))
    _check(
        "set_collision_tool_model",
        api.set_collision_tool_model(_GRIPPER_TOOL_MODEL[cfg.gripper]),
    )
    # (4) optional reduced mode: params first, master switch LAST
    if cfg.reduced_tcp_boundary_mm is not None:
        _check(
            "set_reduced_tcp_boundary",
            api.set_reduced_tcp_boundary(list(cfg.reduced_tcp_boundary_mm)),
        )
        _check("set_reduced_mode", api.set_reduced_mode(True))
    # (5) stop-and-latch, not bounce (C22/C31/C35 = RECOVERABLE with budget)
    _check("set_collision_rebound", api.set_collision_rebound(False))
    return warnings
