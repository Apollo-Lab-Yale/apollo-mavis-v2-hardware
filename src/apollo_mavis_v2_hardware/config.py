"""Driver configuration models (design doc 02-hardware §3.1)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# xArm7 factory joint limits, rad (UFACTORY user manual); the streamer clamps
# commands `joint_limit_margin_rad` inside these to avoid SDK -8 OUT_OF_RANGE.
XARM7_JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-2.0 * math.pi, 2.0 * math.pi),
    (math.radians(-118.0), math.radians(120.0)),
    (-2.0 * math.pi, 2.0 * math.pi),
    (math.radians(-11.0), math.radians(225.0)),
    (-2.0 * math.pi, 2.0 * math.pi),
    (math.radians(-97.0), math.radians(180.0)),
    (-2.0 * math.pi, 2.0 * math.pi),
)


class ServoLimits(BaseModel):
    """Per-tick limits owned by the mode-1 servo streamer (no firmware smoothing).

    The defaults are the HARDWARE CAPS at ``SessionSpec.speed_scale == 1.0``
    (phase-09c D2): **0.6 rad/s per joint and 4 mm per 10 ms tick (0.4 m/s TCP)**
    since 2026-09-07 - the first live runs used 0.3 rad/s / 2 mm (0.2 m/s) and the
    operator found 100 % "still very slow, over-conservative". The Cartesian step
    is deliberately kept at HALF the gate's 8 mm geometry inflation
    (``SafetyConfig.geom_inflation_m``): the twin gate checks the commanded
    posture once per tick, so one tick must never be able to cross the inflated
    shell - 10 mm/tick (the firmware's own limit) would. The runtime multiplies
    ``max_joint_vel`` and ``max_cart_step_m`` by the session's ``speed_scale``
    (default 0.1) inside its ``driver_factory`` closure.
    """

    rate_hz: float = 100.0
    max_joint_vel: tuple[float, ...] = tuple([0.6] * 7)  # rad/s cap (per-tick slew = vel*dt)
    max_joint_acc: tuple[float, ...] = tuple([20.0] * 7)  # rad/s^2 (prevents C24 on steps)
    lever_arm_m: tuple[float, ...] = (1.20, 1.20, 1.00, 0.75, 0.44, 0.30, 0.10)
    max_cart_step_m: float = 0.004  # 0.4 m/s TCP cap (D2); half the 8 mm gate inflation
    joint_limit_margin_rad: float = 0.0087  # 0.5 deg inside limits (avoids -8)
    joint_limits_rad: tuple[tuple[float, float], ...] = XARM7_JOINT_LIMITS_RAD

    @field_validator("max_joint_vel", "max_joint_acc", "lever_arm_m")
    @classmethod
    def _len7(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if len(v) != 7:
            raise ValueError(f"expected 7 entries, got {len(v)}")
        return v

    @field_validator("joint_limits_rad")
    @classmethod
    def _limits7(cls, v: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
        if len(v) != 7 or any(lo >= hi for lo, hi in v):
            raise ValueError("joint_limits_rad must be 7 (lo, hi) pairs with lo < hi")
        return v


class XArmDriverConfig(BaseModel):
    """One arm; one XArmAPI(ip, is_radian=True, report_type='real') per driver.

    ``servo`` and ``rail_speed_mm_s`` are the speed-scale-1.0 hardware caps
    (phase-09c D2); the runtime's ``driver_factory`` scales them per session.
    ``connect()`` never homes the rail. With the default ``rail_homing ==
    "require_homed"`` it requires ``on_zero == 1`` (``RailNotHomedError``
    otherwise) — a normal session homes beforehand through the monitor's
    operator-triggered ``home_rail`` maintenance op. ``"allow_unhomed"``
    (phase-09d) is for the runtime's rail-homing MAINTENANCE MOTION only: the
    connect proceeds with an unhomed track (``dof`` stays 8, the rail phase stays
    ``DETECTED``, ``XArmDriver.rail_position_known`` is False and the published
    rail slot reads 0.0 m as a placeholder) so the runtime can pre-position the
    arm along a twin-planned, rail-position-agnostic path and then call
    ``XArmDriver.home_rail()`` while the servo stream holds the joints.
    """

    arm_id: str
    ip: str
    expect_rail: Literal["auto", "yes", "no"] = "auto"
    gripper: Literal["xarm", "xarm_g2", "none"] = "xarm"
    tcp_load_kg: float = 0.82  # classic gripper mass; override per tool
    tcp_load_cog_mm: tuple[float, float, float] = (0.0, 0.0, 48.0)
    collision_sensitivity: int = Field(default=3, ge=0, le=5)  # 3..4 per overview §6
    reduced_tcp_boundary_mm: tuple[int, int, int, int, int, int] | None = None
    # optional [x_max, x_min, y_max, y_min, z_max, z_min] base-frame envelope
    # (11-safety §11); None = reduced mode off
    rail_speed_mm_s: int = 50  # positioning speed cap at scale 1.0 (D2); track max 200
    # phase-09d: "allow_unhomed" ONLY for the runtime's rail-homing maintenance motion
    # (see the class docstring); every normal session keeps "require_homed"
    rail_homing: Literal["require_homed", "allow_unhomed"] = "require_homed"
    servo: ServoLimits = ServoLimits()
    monitor_rate_hz: float = 5.0
    stale_after_s: float = 0.15  # 30003 silence => ArmState.stale
    expected_sn: str | None = None  # assert vs arm.sn (catches cabling swaps)
