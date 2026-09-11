"""units.py: THE single m/rad <-> mm/pulse boundary (02-hardware §2)."""

import math

import numpy as np
import pytest
from apollo_mavis_v2_core import Pose, se3
from hypothesis import given
from hypothesis import strategies as st

from apollo_mavis_v2_hardware import units


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


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rpy_convention_is_extrinsic_xyz() -> None:
    """RPY = extrinsic XYZ, R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (the xArm firmware
    convention, verified against 15 hardware episodes 2026-09-11) - a numeric pin, not
    just the self-consistency the roundtrip above already has."""
    r, p, y = 0.4, 0.5, -0.6
    pose = units.sdk_to_pose([0, 0, 0, r, p, y])
    assert np.allclose(se3.quat_to_mat(pose.orientation), _rz(y) @ _ry(p) @ _rx(r), atol=1e-12)
    assert se3.quat_to_rpy(pose.orientation) == pytest.approx([r, p, y], abs=1e-9)
    # the pre-2026-09-11 composition is measurably different for two non-zero angles
    wrong = se3.mat_to_quat(_rx(r) @ _ry(p) @ _rz(y))
    assert se3.quat_geodesic(pose.orientation, wrong) > 0.1


def test_sdk_to_tcp_pose_is_flange_plus_rz_pi_plus_172mm() -> None:
    """Tool pointing DOWN (roll pi): the 0.172 m TCP offset lowers the pose and the
    Rz(pi) gripper mount turns Rx(pi) into Rx(pi).Rz(pi) = Ry(pi)."""
    sdk = [300.0, -50.0, 220.0, math.pi, 0.0, 0.0]
    tcp = units.sdk_to_tcp_pose(sdk, gripper=True)
    assert tcp.position == pytest.approx([0.3, -0.05, 0.048], abs=1e-9)
    assert np.abs(tcp.orientation) == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-9)
    assert np.allclose(se3.quat_to_mat(tcp.orientation), _ry(math.pi), atol=1e-9)
    # inverse recovers the SDK flange numbers
    assert units.tcp_pose_to_sdk(tcp, gripper=True)[:3] == pytest.approx(sdk[:3], abs=1e-6)
    flange = se3.tcp_to_flange(tcp, gripper=True)
    assert se3.quat_geodesic(flange.orientation, units.sdk_to_pose(sdk).orientation) < 1e-9


def test_sdk_to_tcp_pose_gripperless_is_the_flange() -> None:
    sdk = [500.0, -100.0, 250.0, math.pi / 4, -0.3, 1.2]
    tcp = units.sdk_to_tcp_pose(sdk, gripper=False)
    flange = units.sdk_to_pose(sdk)
    assert tcp.position == pytest.approx(flange.position)
    assert tcp.orientation == pytest.approx(flange.orientation)
    assert units.tcp_pose_to_sdk(tcp, gripper=False) == pytest.approx(sdk, abs=1e-5)


@given(
    st.floats(-math.pi, math.pi, allow_nan=False),
    st.floats(-1.4, 1.4, allow_nan=False),
    st.floats(-math.pi, math.pi, allow_nan=False),
)
def test_sdk_tcp_round_trip(r: float, p: float, y: float) -> None:
    sdk = [120.0, -30.0, 410.0, r, p, y]
    back = units.tcp_pose_to_sdk(units.sdk_to_tcp_pose(sdk, gripper=True), gripper=True)
    assert back[:3] == pytest.approx(sdk[:3], abs=1e-6)
    q0 = units.sdk_to_pose(sdk).orientation
    assert se3.quat_geodesic(units.sdk_to_pose(back).orientation, q0) < 1e-6
