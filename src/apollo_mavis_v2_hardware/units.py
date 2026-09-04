"""THE single m/rad <-> mm / pulses / rail-mm conversion boundary (02-hardware §2).

Core is m/rad/quat-wxyz; the xArm SDK is mm + rad (drivers always construct
``XArmAPI(is_radian=True)`` — degrees never appear anywhere). Conversion
happens exactly once, here; nothing else in the stack multiplies by 1000.
Joints and torques pass through unchanged (rad, N·m). RPY convention =
intrinsic XYZ per ``core.se3.rpy_to_quat`` (the xArm firmware convention).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from apollo_mavis_v2_core import Pose, se3

GRIPPER_PULSE_MAX = 850  # classic gripper pulses <-> 0.085 m opening span
GRIPPER_G2_MM_MAX = 84.0  # G2 gripper opening, mm
RAIL_MM_MAX = 650  # SDK does NOT clamp; over-travel raises rail error 25/26


def m_to_mm(x: float) -> float:
    return float(x) * 1000.0


def mm_to_m(x: float) -> float:
    return float(x) / 1000.0


def pose_to_sdk(pose: Pose) -> list[float]:
    """core Pose (m, wxyz) -> SDK ``[x_mm, y_mm, z_mm, roll, pitch, yaw]`` (mm + rad)."""
    rpy = se3.quat_to_rpy(pose.orientation)
    return [
        m_to_mm(pose.position[0]),
        m_to_mm(pose.position[1]),
        m_to_mm(pose.position[2]),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    ]


def sdk_to_pose(p: Sequence[float]) -> Pose:
    """SDK ``[x_mm, y_mm, z_mm, r, p, y]`` -> core Pose; quat normalized, w >= 0."""
    if len(p) < 6:
        raise ValueError(f"SDK pose needs 6 values, got {len(p)}")
    position = np.array([mm_to_m(p[0]), mm_to_m(p[1]), mm_to_m(p[2])])
    quat = se3.quat_normalize(se3.rpy_to_quat(p[3:6]))
    return Pose(position, quat)


def frac_to_pulse(f: float) -> int:
    """[0, 1] open fraction -> classic gripper pulses [0, 850] (clamped)."""
    f = min(max(float(f), 0.0), 1.0)
    return round(f * GRIPPER_PULSE_MAX)


def pulse_to_frac(pulse: float) -> float:
    """Classic gripper pulses -> [0, 1] open fraction (clamped)."""
    return min(max(float(pulse) / GRIPPER_PULSE_MAX, 0.0), 1.0)


def frac_to_g2_mm(f: float) -> float:
    """[0, 1] open fraction -> G2 opening [0, 84.0] mm (clamped)."""
    f = min(max(float(f), 0.0), 1.0)
    return f * GRIPPER_G2_MM_MAX


def g2_mm_to_frac(mm: float) -> float:
    """G2 opening mm -> [0, 1] open fraction (clamped)."""
    return min(max(float(mm) / GRIPPER_G2_MM_MAX, 0.0), 1.0)


def rail_m_to_mm(x: float) -> int:
    """Rail m -> absolute int mm, clamped to [0, 650].

    The SDK does not clamp; commanding over-travel raises linear-motor
    error 25/26 — every rail command must go through here.
    """
    return int(min(max(m_to_mm(x), 0.0), float(RAIL_MM_MAX)))


def rail_mm_to_m(mm: float) -> float:
    """Rail mm -> m, clamped to [0, 0.650]."""
    return min(max(mm_to_m(mm), 0.0), RAIL_MM_MAX / 1000.0)
