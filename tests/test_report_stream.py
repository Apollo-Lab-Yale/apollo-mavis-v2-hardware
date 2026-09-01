"""30003 report path: replayer-driven staleness, frame parsing, dq EMA (§3.4)."""

import math
import time

import pytest
from conftest import FakeClock
from fakes.fake_xarm_api import FakeXArmAPI
from fakes.report_replayer import (
    FrameSplitter,
    ReportReplayer,
    build_real_frame,
    load_fixture,
    parse_real_frame,
)
from test_driver_connect import wait_until

from apollo_xarm7_hardware.config import XArmDriverConfig
from apollo_xarm7_hardware.driver import XArmDriver
from apollo_xarm7_hardware.events import StaleEvent


def test_fixture_frames_parse() -> None:
    frames = load_fixture("tests/fixtures/report/joint6_sine_100hz.bin")
    assert len(frames) == 200
    first = parse_real_frame(frames[0])
    assert first["mode"] == 1 and first["state"] == 0
    mid = parse_real_frame(frames[50])  # sine peak at t=0.5 s
    assert mid["joints"][5] == pytest.approx(0.3, abs=1e-5)
    assert mid["cartesian"][2] == pytest.approx(220.0, abs=1e-3)


def test_splitter_handles_torn_chunks_and_garbage() -> None:
    frames = [build_real_frame([i * 0.01] * 7, cmd_num=i) for i in range(5)]
    stream = (
        frames[0]
        + b"\xff\xff\xff\xff\x00\x99"  # bogus size prefix + junk
        + frames[1]
        + frames[2]
        + b"\x00"  # stray byte
        + frames[3]
        + frames[4]
    )
    splitter = FrameSplitter()
    out: list[bytes] = []
    for i in range(0, len(stream), 13):  # torn into odd-sized chunks
        out.extend(splitter.feed(stream[i : i + 13]))  # never raises
    cmd_nums = [parse_real_frame(f)["cmdnum"] for f in out]
    for expect in (0, 2, 4):  # frames adjacent to garbage may be sacrificed,
        assert expect in cmd_nums  # but the splitter always resyncs


def test_staleness_flips_at_150ms_and_recovers_via_replayer() -> None:
    frames = load_fixture("tests/fixtures/report/joint6_sine_100hz.bin")
    replayer = ReportReplayer(frames, rate_hz=100.0)
    replayer.start()
    holder = {}

    def factory(ip, **kw):
        api = FakeXArmAPI(ip, report_stream=replayer.address, **{})
        holder["api"] = api
        return api

    cfg = XArmDriverConfig(
        arm_id="a1", ip="192.168.1.235", monitor_rate_hz=100.0, stale_after_s=0.15
    )
    drv = XArmDriver(cfg, api_factory=factory)
    try:
        drv.connect()
        wait_until(lambda: not drv.get_state().stale, msg="stream never became fresh")
        drv.drain_events()
        replayer.pause()  # 30003 goes silent
        t0 = time.monotonic()
        wait_until(lambda: drv.get_state().stale, timeout=1.0, msg="never went stale")
        elapsed = time.monotonic() - t0
        assert 0.12 <= elapsed <= 0.20, f"stale flipped at {elapsed * 1000:.0f} ms"
        wait_until(
            lambda: any(
                isinstance(e, StaleEvent) and e.stale for e in drv.drain_events()
            ),
            timeout=1.0,
            msg="no StaleEvent(stale=True)",
        )
        replayer.resume()  # push resumes -> staleness clears automatically
        wait_until(lambda: not drv.get_state().stale, timeout=1.0,
                   msg="staleness never cleared")
        wait_until(
            lambda: any(
                isinstance(e, StaleEvent) and not e.stale for e in drv.drain_events()
            ),
            timeout=1.0,
            msg="no StaleEvent(stale=False)",
        )
    finally:
        drv.disconnect()
        replayer.stop()


def test_state_snapshot_carries_frame_values_in_meters() -> None:
    """Cartesian mm from the wire -> exactly one mm->m conversion in ArmState."""
    q = [0.0, -0.5, 0.0, 0.3, 0.0, 0.2, 0.0]
    tcp = [300.0, -50.0, 220.0, math.pi, 0.0, 0.0]
    frames = [build_real_frame(q, tcp, [0.1] * 7)]
    replayer = ReportReplayer(frames, rate_hz=100.0)
    replayer.start()
    drv = XArmDriver(
        XArmDriverConfig(arm_id="a1", ip="1.2.3.4", monitor_rate_hz=50.0),
        api_factory=lambda ip, **kw: FakeXArmAPI(ip, report_stream=replayer.address),
    )
    try:
        drv.connect()
        wait_until(lambda: not drv.get_state().stale)
        state = drv.get_state()
        assert state.q == pytest.approx(q, abs=1e-6)
        assert state.ee_pose.position == pytest.approx([0.3, -0.05, 0.22], abs=1e-6)
        assert abs(state.ee_pose.orientation[1]) == pytest.approx(1.0, abs=1e-6)
        assert state.mode == 1 and state.state == 0
    finally:
        drv.disconnect()
        replayer.stop()


def test_dq_finite_difference_with_ema() -> None:
    fc = FakeClock()
    drv = XArmDriver(
        XArmDriverConfig(arm_id="a1", ip="1.2.3.4"),
        api_factory=lambda ip, **kw: FakeXArmAPI(ip),
        clock=fc.now,
    )
    drv._api = FakeXArmAPI()  # state path only; no connect
    frame = {"cartesian": [0.0] * 6, "mode": 1, "state": 0, "cmdnum": 0}
    drv._on_report({**frame, "joints": [0.0] * 7})
    fc.jump(0.01)
    drv._on_report({**frame, "joints": [0.01] + [0.0] * 6})  # raw dq = 1.0 rad/s
    assert drv._snap.dq[0] == pytest.approx(0.5)  # EMA alpha=0.5 from 0
    fc.jump(0.01)
    drv._on_report({**frame, "joints": [0.02] + [0.0] * 6})
    assert drv._snap.dq[0] == pytest.approx(0.75)  # 0.5*1.0 + 0.5*0.5
