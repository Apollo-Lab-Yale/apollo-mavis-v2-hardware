"""units.py: THE single m/rad <-> mm/pulse boundary (02-hardware §2)."""

import math

import numpy as np
import pytest
from apollo_xarm7_core import Pose, se3
from hypothesis import given
from hypothesis import strategies as st

from apollo_xarm7_hardware import units


def test_one_meter_is_exactly_1000_mm() -> None:
    # pins the boundary: conversion happens exactly once, here
    assert units.m_to_mm(1.0) == 1000.0
    assert units.mm_to_m(1000.0) == 1.0


@given(st.floats(-10, 10, allow_nan=False))
def test_m_mm_round_trip(x: float) -> None:
    assert units.mm_to_m(units.m_to_mm(x)) == pytest.approx(x, abs=1e-12)


def test_pose_to_sdk_units_and_convention() -> None:
    rpy = (0.1, -0.2, 0.3)
    pose = Pose(np.array([1.0, 0.2, 0.3]), se3.rpy_to_quat(rpy))
    sdk = units.pose_to_sdk(pose)
    assert sdk[:3] == [1000.0, 200.0, 300.0]  # m -> mm exactly once
    assert sdk[3:] == pytest.approx(list(rpy), abs=1e-12)  # rad passthrough


def test_sdk_to_pose_round_trip_and_canonical_quat() -> None:
    sdk = [500.0, -100.0, 250.0, math.pi / 4, -0.3, 1.2]
    pose = units.sdk_to_pose(sdk)
    assert pose.position == pytest.approx([0.5, -0.1, 0.25])
    assert pose.orientation[0] >= 0  # canonical w >= 0
    assert np.linalg.norm(pose.orientation) == pytest.approx(1.0)
    assert units.pose_to_sdk(pose) == pytest.approx(sdk, abs=1e-5)


def test_sdk_to_pose_rejects_short_input() -> None:
    with pytest.raises(ValueError):
        units.sdk_to_pose([1.0, 2.0, 3.0])


def test_gripper_pulse_conversions() -> None:
    assert units.frac_to_pulse(0.5) == 425  # acceptance-pinned
    assert units.frac_to_pulse(0.0) == 0
    assert units.frac_to_pulse(1.0) == 850
    assert units.frac_to_pulse(1.5) == 850  # clamped
    assert units.frac_to_pulse(-0.2) == 0
    assert units.pulse_to_frac(425) == pytest.approx(0.5)
    assert units.pulse_to_frac(9999) == 1.0


@given(st.floats(0, 1, allow_nan=False))
def test_pulse_round_trip(f: float) -> None:
    assert units.pulse_to_frac(units.frac_to_pulse(f)) == pytest.approx(f, abs=1 / 850)


def test_g2_mm_conversions() -> None:
    assert units.frac_to_g2_mm(1.0) == 84.0
    assert units.frac_to_g2_mm(0.5) == 42.0
    assert units.frac_to_g2_mm(2.0) == 84.0  # clamped
    assert units.g2_mm_to_frac(42.0) == pytest.approx(0.5)


def test_rail_clamp_edges() -> None:
    # SDK does NOT clamp; over-travel raises rail error 25/26 — we must
    assert units.rail_m_to_mm(0.7) == 650
    assert units.rail_m_to_mm(0.650) == 650
    assert units.rail_m_to_mm(-0.1) == 0
    assert units.rail_m_to_mm(0.3252) == 325
    assert isinstance(units.rail_m_to_mm(0.3), int)
    assert units.rail_mm_to_m(700) == 0.65
    assert units.rail_mm_to_m(-5) == 0.0


@given(st.floats(0, 0.65, allow_nan=False))
def test_rail_round_trip(x: float) -> None:
    assert units.rail_mm_to_m(units.rail_m_to_mm(x)) == pytest.approx(x, abs=1e-3)


def test_rpy_convention_matches_core_se3() -> None:
    """RPY = intrinsic XYZ per core.se3 (the xArm firmware convention)."""
    rpy = np.array([0.4, 0.5, -0.6])
    pose = units.sdk_to_pose([0, 0, 0, *rpy])
    assert se3.quat_to_rpy(pose.orientation) == pytest.approx(rpy, abs=1e-9)
