"""Recovery state machine (02-hardware §3.5): C24, budget, 111, Studio, e-stop."""

import threading
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import CommandError
from fakes.fake_xarm_api import FaultScript
from test_driver_connect import make_driver, wait_until

from apollo_mavis_v2_hardware.driver import ArmFaultedError, DriverPhase, RecoveryResult
from apollo_mavis_v2_hardware.events import (
    FaultEvent,
    RecoveredEvent,
    ReseedEvent,
    StudioConflictWarning,
)
from apollo_mavis_v2_hardware.rail import RailPhase


def collect_until(drv, predicate, timeout=5.0):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(drv.drain_events())
        if predicate(events):
            return events
        time.sleep(0.005)
    raise AssertionError(f"predicate never satisfied; events={events}")


def _last_index(calls, name, args=None):
    idx = -1
    for i, (n, a, _kw) in enumerate(calls):
        if n == name and (args is None or a == args):
            idx = i
    assert idx >= 0, f"{name}{args or ''} never called"
    return idx


def test_c24_recovery_sequence_reseed_from_measured() -> None:
    seed = [0.1] * 7
    drv, h = make_driver(
        fake_kwargs={
            "initial_q": list(seed),
            "fault_script": FaultScript(fault_at_tick=20, error_code=24),
        }
    )
    api = h["api"]
    try:
        drv.command_joints(np.array(seed) + 0.3)  # far target: keeps marching
        events = collect_until(
            drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs)
        )
        # binding order: clean_error -> motion_enable -> set_mode(1) -> set_state(0)
        calls = api.calls
        i_clean = _last_index(calls, "clean_error")
        i_enable = _last_index(calls, "motion_enable")
        i_mode = _last_index(calls, "set_mode", (1,))
        i_state = _last_index(calls, "set_state", (0,))
        assert i_clean < i_enable < i_mode < i_state
        # re-seeded from the MEASURED position, not the old target:
        # ticks 1..19 accepted, tick 20 rejected; first send after recovery
        # must equal the tick-19 value exactly (velocity zero).
        wait_until(lambda: len(api.sent_joints) >= 20, msg="stream never resumed")
        before = np.asarray(api.sent_joints[18][1])
        after = np.asarray(api.sent_joints[19][1])
        assert np.allclose(after, before, atol=1e-12)
        assert not np.allclose(after, np.array(seed) + 0.3)
        reseeds = [e for e in events if isinstance(e, ReseedEvent)]
        assert reseeds and np.allclose(reseeds[0].q, before, atol=1e-12)
        assert drv.phase is DriverPhase.STREAMING
    finally:
        drv.disconnect()


def test_c24_halves_limits_for_10s_then_restores() -> None:
    drv, h = make_driver(
        fake_kwargs={"fault_script": FaultScript(fault_at_tick=10, error_code=24)}
    )
    try:
        drv.command_joints(np.full(7, 0.3))
        collect_until(drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs))
        assert drv._streamer._scale == 0.5  # halved vel/acc
        remaining = drv._c24_backoff_until - time.monotonic()
        assert 9.0 < remaining <= 10.1  # 10 s backoff window
        drv._c24_backoff_until = time.monotonic() - 0.001  # fast-forward expiry
        wait_until(lambda: drv._streamer._scale == 1.0, msg="scale never restored")
    finally:
        drv.disconnect()


def test_second_c24_inside_backoff_window_latches() -> None:
    drv, h = make_driver(
        fake_kwargs={"fault_script": FaultScript(fault_at_tick=10, error_code=24)}
    )
    try:
        drv.command_joints(np.full(7, 0.3))
        collect_until(drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs))
        h["api"].inject_error(24)  # second C24 while the backoff is active
        wait_until(lambda: drv.phase is DriverPhase.LATCHED, msg="never latched")
        with pytest.raises(ArmFaultedError):
            drv.command_joints(np.zeros(7))
    finally:
        drv.disconnect()


def test_fourth_recoverable_error_in_30s_latches() -> None:
    drv, h = make_driver()
    try:
        for n in range(1, 4):  # three recoveries fit the budget
            h["api"].inject_error(22)  # self-collision: RECOVERABLE
            collect_until(
                drv,
                lambda evs, n=n: sum(isinstance(e, RecoveredEvent) for e in evs) >= 1,
            )
            wait_until(lambda: drv.phase is DriverPhase.STREAMING)
        h["api"].inject_error(22)  # 4th within the rolling 30 s window
        wait_until(lambda: drv.phase is DriverPhase.LATCHED, msg="budget never latched")
    finally:
        drv.disconnect()


def test_estop_variant_latches_immediately_and_clear_errors_recovers() -> None:
    drv, h = make_driver()
    try:
        h["api"].inject_error(1)  # e-stop: UNRECOVERABLE, never auto-resume
        wait_until(lambda: drv.phase is DriverPhase.LATCHED)
        with pytest.raises(ArmFaultedError):
            drv.command_joints(np.zeros(7))
        drv.clear_errors()  # explicit user recovery
        assert drv.phase is DriverPhase.STREAMING
    finally:
        drv.disconnect()


def test_clear_errors_stays_latched_while_estop_engaged() -> None:
    drv, h = make_driver()
    try:
        h["api"].inject_error(1)
        wait_until(lambda: drv.phase is DriverPhase.LATCHED)
        h["api"].motion_enable_fails = True  # physical e-stop still down
        drv.clear_errors()
        assert drv.phase is DriverPhase.LATCHED
        h["api"].motion_enable_fails = False  # button released
        drv.clear_errors()
        assert drv.phase is DriverPhase.STREAMING
    finally:
        drv.disconnect()


def test_error_111_latches_rail_only_stream_continues() -> None:
    drv, h = make_driver(fake_kwargs={"has_rail": True, "rail_homed": True})
    api = h["api"]
    try:
        drv.command_joints(np.concatenate([np.full(7, 0.2), [0.1]]))
        wait_until(lambda: len(api.sent_joints) > 10)
        api.inject_error(111, drop_mode=False)  # rail drop does not stop the arm
        wait_until(
            lambda: drv._rail.phase is RailPhase.RAIL_ERROR, msg="rail never latched"
        )
        n0 = len(api.sent_joints)
        time.sleep(0.15)
        assert drv.phase is DriverPhase.STREAMING  # arm keeps streaming
        assert len(api.sent_joints) > n0  # servo stream never stopped
        assert api.error_code == 0  # 111 cleaned so the arm can keep moving
    finally:
        drv.disconnect()


def test_studio_conflict_warns_retries_then_latches() -> None:
    drv, h = make_driver()
    api = h["api"]
    try:
        time.sleep(0.6)  # past the set_mode grace window
        api.mode = 0  # Studio Live control grabs mode/state; NO error code
        api.state = 2
        events = collect_until(
            drv, lambda evs: any(isinstance(e, StudioConflictWarning) for e in evs)
        )
        assert any(isinstance(e, StudioConflictWarning) for e in events)
        wait_until(lambda: api.mode == 1)  # driver retried mode 1 once
        wait_until(lambda: drv.phase is DriverPhase.STREAMING)
        time.sleep(0.6)  # second grab within 5 s of the first retry
        api.mode = 0
        api.state = 2
        wait_until(lambda: drv.phase is DriverPhase.LATCHED, timeout=5.0,
                   msg="second conflict never latched")
    finally:
        drv.disconnect()


# -- request_recovery() / recovery_result() (phase-09b) ---------------------------------


def _record_threads(api, *names):
    seen: dict[str, str] = {}
    for name in names:
        orig = getattr(api, name)

        def wrapper(*a, _orig=orig, _name=name, **k):
            seen[_name] = threading.current_thread().name
            return _orig(*a, **k)

        setattr(api, name, wrapper)
    return seen


def test_request_recovery_runs_on_the_monitor_thread_and_reaches_drain_events() -> None:
    drv, h = make_driver()
    api = h["api"]
    try:
        api.inject_error(1)  # e-stop variant: UNRECOVERABLE -> LATCHED, no auto-resume
        wait_until(lambda: drv.phase is DriverPhase.LATCHED)
        auto = drv.recovery_result()
        assert isinstance(auto, RecoveryResult) and not auto.ok and not auto.user_initiated
        assert auto.error_code == 1 and auto.detail == "controller error 1"
        drv.drain_events()  # discard the fault history
        seen = _record_threads(api, "clean_error", "clean_warn", "motion_enable", "set_mode")
        drv.request_recovery()  # returns at once; nothing ran on this thread
        assert "clean_error" not in seen
        events = collect_until(drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs))
        assert drv.phase is DriverPhase.STREAMING
        assert set(seen.values()) == {"hw.a1.monitor"}  # the driver's 5 Hz monitor thread
        kinds = [type(e) for e in events if type(e) in (FaultEvent, ReseedEvent, RecoveredEvent)]
        assert kinds == [FaultEvent, ReseedEvent, RecoveredEvent]
        fault = next(e for e in events if isinstance(e, FaultEvent))
        assert fault.source == "user" and fault.error_code == 1  # C1 still latched at capture
        recovered = next(e for e in events if isinstance(e, RecoveredEvent))
        assert recovered.arm_id == "a1"
        # binding order on the wire: clean_error -> clean_warn -> motion_enable -> mode 1 -> state 0
        calls = api.calls
        i_clean = _last_index(calls, "clean_error")
        i_warn = _last_index(calls, "clean_warn")
        i_enable = _last_index(calls, "motion_enable")
        i_mode = _last_index(calls, "set_mode", (1,))
        i_state = _last_index(calls, "set_state", (0,))
        assert i_clean < i_warn < i_enable < i_mode < i_state
        res = drv.recovery_result()
        assert res is not None and res.ok and res.user_initiated and res.seq == auto.seq + 1
        assert res.detail == ""
        with_rail_free = drv.get_state()
        assert with_rail_free.error_code == 0
    finally:
        drv.disconnect()


def test_request_recovery_with_estop_engaged_stays_latched_and_reports_it() -> None:
    drv, h = make_driver()
    api = h["api"]
    try:
        api.inject_error(1)
        wait_until(lambda: drv.phase is DriverPhase.LATCHED)
        seq0 = drv.recovery_result().seq
        api.motion_enable_fails = True  # the physical e-stop is still down
        drv.drain_events()
        drv.request_recovery()
        wait_until(lambda: (r := drv.recovery_result()) is not None and r.seq > seq0)
        res = drv.recovery_result()
        assert not res.ok and res.user_initiated
        assert res.detail.startswith("motion_enable failed")
        assert drv.phase is DriverPhase.LATCHED
        events = drv.drain_events()
        latch = [e for e in events if isinstance(e, FaultEvent) and e.source == "latch"]
        assert latch and latch[0].detail.startswith("motion_enable failed")
        assert not any(isinstance(e, RecoveredEvent) for e in events)
        with pytest.raises(ArmFaultedError):
            drv.command_joints(np.zeros(7))
        api.motion_enable_fails = False  # button released: try again
        drv.request_recovery()
        wait_until(lambda: drv.phase is DriverPhase.STREAMING)
        assert drv.recovery_result().ok
    finally:
        drv.disconnect()


def test_request_recovery_from_streaming_reseeds_without_moving() -> None:
    seed = [0.2] * 7
    drv, h = make_driver(fake_kwargs={"initial_q": list(seed)})
    api = h["api"]
    try:
        wait_until(lambda: len(api.sent_joints) > 5)
        drv.drain_events()
        drv.request_recovery()
        events = collect_until(drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs))
        reseed = next(e for e in events if isinstance(e, ReseedEvent))
        assert np.allclose(reseed.q, seed, atol=1e-12)  # re-seeded from the measured position
        assert drv.phase is DriverPhase.STREAMING
        wait_until(lambda: len(api.sent_joints) > 0 and np.allclose(api.sent_joints[-1][1], seed))
    finally:
        drv.disconnect()


def test_request_recovery_requires_a_connected_driver() -> None:
    drv, _ = make_driver(connect=False)
    with pytest.raises(CommandError):
        drv.request_recovery()
    assert drv.recovery_result() is None
    drv2, _ = make_driver()
    drv2.disconnect()
    with pytest.raises(CommandError):
        drv2.request_recovery()


def test_auto_recovery_records_its_result_too() -> None:
    drv, h = make_driver(fake_kwargs={"fault_script": FaultScript(fault_at_tick=10, error_code=24)})
    try:
        drv.command_joints(np.full(7, 0.3))
        collect_until(drv, lambda evs: any(isinstance(e, RecoveredEvent) for e in evs))
        res = drv.recovery_result()
        assert res is not None and res.ok and res.error_code == 24 and not res.user_initiated
    finally:
        drv.disconnect()
