"""Controller-side safety backstops (02-hardware §6; overview §6 layer 3).

Volatile, re-applied on every connect; never ``save_conf()`` (controllers
stay config-clean). Non-fatal failures are returned as warning strings —
the arm still comes up, the UI surfaces them.
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


def apply_backstops(api: Any, cfg: XArmDriverConfig) -> list[str]:
    """Apply controller-enforced limits, in the §6 binding order.

    (1) payload first — collision detection is torque-estimate based, a wrong
    payload means false positives/negatives; (2) collision sensitivity;
    (3) self-collision detection + tool model; (4) optional reduced-mode TCP
    boundary (master switch LAST); (5) stop-and-latch, never bounce.
    """
    warnings: list[str] = []

    def _check(name: str, code: int) -> None:
        if code != 0:
            warnings.append(f"{name} returned {code}")

    # (1) payload + gravity FIRST
    _check("set_tcp_load", api.set_tcp_load(cfg.tcp_load_kg, list(cfg.tcp_load_cog_mm)))
    _check("set_gravity_direction", api.set_gravity_direction([0, 0, -1]))
    # (2) collision sensitivity (volatile; re-applied every connect)
    _check(
        "set_collision_sensitivity",
        api.set_collision_sensitivity(cfg.collision_sensitivity),
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
