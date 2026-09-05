"""HardwareWorkcell bring-up: parallel arms, partial failure, statuses (§9)."""

import time
from types import SimpleNamespace

import pytest
from apollo_mavis_v2_core import CommandError, WorkcellBringupError
from apollo_mavis_v2_core.schemas import ArmConfig, PoseModel, WorkcellConfig
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import wait_until

from apollo_mavis_v2_hardware.driver import DriverPhase, XArmDriver
from apollo_mavis_v2_hardware.events import FaultEvent, RecoveredEvent, ReseedEvent
from apollo_mavis_v2_hardware.netsetup.types import MatchResult
from apollo_mavis_v2_hardware.workcell import ArmBringupStatus, HardwareWorkcell, _driver_cfg


def _wc_config(n_arms=2) -> WorkcellConfig:
    return WorkcellConfig(
        kind="hardware",
        arms=[
            ArmConfig(id=f"arm{i}", ip=f"192.168.{i}.235", base_in_world=PoseModel())
            for i in range(1, n_arms + 1)
        ],
        cameras=[],
        digital_twin_scene="lab_twin",
    )


def _fast_sleep(s: float) -> None:
    time.sleep(min(s, 0.01))


def make_workcell(fail_arms=frozenset(), netsetup=None, n_arms=2, api_cls=FakeXArmAPI):
    apis: dict[str, FakeXArmAPI] = {}

    def driver_factory(cfg):
        def api_factory(ip, **kw):
            api = api_cls(
                ip,
                auto_report_hz=100.0,
                has_rail=(cfg.arm_id == "arm2"),
                rail_homed=True,
                **kw,
            )
            if cfg.arm_id in fail_arms:
                api.connected = False  # ArmConnectError after retries
            apis[cfg.arm_id] = api
            return api

        return XArmDriver(cfg, api_factory=api_factory, sleep=_fast_sleep)

    wc = HardwareWorkcell(_wc_config(n_arms), driver_factory=driver_factory, netsetup=netsetup)
    return wc, apis


def test_bring_up_all_arms_and_states():
    wc, apis = make_workcell()
    try:
        transitions: list[ArmBringupStatus] = []
        statuses = wc.bring_up(status_cb=transitions.append, timeout_s=20.0)
        assert statuses["arm1"].connected and statuses["arm2"].connected
        assert statuses["arm1"].rail == "none"
        assert statuses["arm2"].rail == "ready"  # detected + homed during connect
        assert statuses["arm1"].gripper == "xarm"
        assert statuses["arm1"].sn == "XA7-FAKE-0001"
        assert transitions, "status_cb never fired"
        assert {t.arm_id for t in transitions} == {"arm1", "arm2"}
        states = wc.states()
        assert set(states) == {"arm1", "arm2"}
        assert states["arm2"].q.shape == (8,)  # rail slot present
        assert not states["arm1"].stale  # fresh 30003 snapshot required
    finally:
        wc.shutdown()


def test_one_arm_failing_never_aborts_the_others():
    wc, apis = make_workcell(fail_arms={"arm2"})
    try:
        statuses = wc.bring_up(timeout_s=20.0)
        assert statuses["arm1"].connected  # sibling untouched by arm2's failure
        assert not statuses["arm2"].connected
        assert statuses["arm2"].error is not None
        with pytest.raises(WorkcellBringupError) as exc_info:
            wc.start()
        assert exc_info.value.statuses["arm2"] is not None
        assert exc_info.value.statuses["arm1"] is None
    finally:
        wc.shutdown()


def test_netsetup_none_skips_network_stage_with_warning():
    wc, _ = make_workcell(n_arms=1)
    try:
        statuses = wc.bring_up(timeout_s=20.0)
        assert statuses["arm1"].network == "ok"
        assert any("netsetup skipped" in w for w in statuses["arm1"].warnings)
    finally:
        wc.shutdown()


class StubNetSetup:
    """Duck-typed NetSetup: probe 'refused' first, then 502 opens."""

    def __init__(self, probe="open"):
        self.probe_result = probe
        self.install_problems: list[str] = []
        self.poll_calls = 0

    def verify(self):
        return {
            "arm1": MatchResult(
                "arm1", "enp36s0f0", "08:BF:B8:89:4F:3A", "uuid-1", self.probe_result, ""
            ),
        }

    def match(self):  # pragma: no cover - verify() succeeds in these tests
        return self.verify()

    def poll_502(self, arm_name):
        self.poll_calls += 1
        return "open" if self.poll_calls >= 2 else "refused"


def test_booting_arm_polls_502_not_nics(monkeypatch):
    import apollo_mavis_v2_hardware.workcell as workcell_mod

    monkeypatch.setattr(workcell_mod, "BOOT_POLL_PERIOD_S", 0.01)  # fast test
    ns = StubNetSetup(probe="refused")  # ping OK, 502 refused: arm still booting
    wc, _ = make_workcell(netsetup=ns, n_arms=1)
    try:
        transitions: list[ArmBringupStatus] = []
        statuses = wc.bring_up(status_cb=transitions.append, timeout_s=20.0)
        assert any(t.network == "booting" for t in transitions)
        assert ns.poll_calls >= 2  # polled the control port, never re-probed NICs
        assert statuses["arm1"].network == "ok"
        assert statuses["arm1"].connected
    finally:
        wc.shutdown()


def test_shutdown_idempotent_and_stop_alias():
    wc, _ = make_workcell(n_arms=1)
    wc.bring_up(timeout_s=20.0)
    wc.stop()
    wc.stop()  # never raises
    assert wc.kind == "hardware"


class _BackstopFailingApi(FakeXArmAPI):
    def set_collision_sensitivity(self, value, wait=True):
        self._rec("set_collision_sensitivity", value)
        return 1  # non-fatal: apply_backstops turns it into a warning


def test_driver_connect_warnings_reach_bringup_status():
    """Backstop + rail-detect warnings were dropped on the floor before 2026-09-04."""
    wc, apis = make_workcell(api_cls=_BackstopFailingApi)
    try:
        statuses = wc.bring_up(timeout_s=20.0)
        for arm_id in ("arm1", "arm2"):
            assert statuses[arm_id].connected
            assert any(
                "set_collision_sensitivity returned 1" in w for w in statuses[arm_id].warnings
            )
        # only arm2 has a rail; SDK 1.18.5 cannot verify its SN
        assert any("get_linear_track_sn" in w for w in statuses["arm2"].warnings)
        assert not any("get_linear_track_sn" in w for w in statuses["arm1"].warnings)
    finally:
        wc.shutdown()


# -- phase-09b: backstop parameters from ArmConfig, events, recovery ----------------------


def test_driver_cfg_maps_backstop_parameters_from_arm_config():
    arm = ArmConfig(
        id="grip",
        ip="192.168.1.201",
        base_in_world=PoseModel(),
        gripper="xarm_g2",
        tcp_load_kg=0.95,
        tcp_load_cog_mm=(0.0, 0.0, 60.0),
    )
    cfg = _driver_cfg(arm)
    assert cfg.arm_id == "grip" and cfg.ip == "192.168.1.201" and cfg.gripper == "xarm_g2"
    assert cfg.tcp_load_kg == 0.95 and cfg.tcp_load_cog_mm == (0.0, 0.0, 60.0)
    if not hasattr(arm, "collision_sensitivity"):
        # core without the phase-09b fields: the driver defaults apply
        assert cfg.collision_sensitivity == 3
        assert cfg.reduced_tcp_boundary_mm is None and cfg.expected_sn is None
    # the phase-09b ArmConfig fields are forwarded verbatim (duck-typed here so the
    # test holds both before and after core gains them)
    arm09b = SimpleNamespace(
        id="view",
        ip="192.168.2.219",
        expect_rail="auto",
        gripper="none",
        tcp_load_kg=0.55,
        tcp_load_cog_mm=(0.0, 0.0, 90.0),
        collision_sensitivity=4,
        reduced_tcp_boundary_mm=(700, -700, 600, -600, 800, 0),
        expected_sn="XS1305",
    )
    cfg = _driver_cfg(arm09b)
    assert cfg.collision_sensitivity == 4
    assert cfg.reduced_tcp_boundary_mm == (700, -700, 600, -600, 800, 0)
    assert cfg.expected_sn == "XS1305"
    assert cfg.tcp_load_kg == 0.55 and cfg.gripper == "none"
    with pytest.raises(ValueError):  # ArmConfig-style bound (0..5) enforced on the driver side
        _driver_cfg(SimpleNamespace(**{**vars(arm09b), "collision_sensitivity": 6}))


def _events_until(wc, pred, timeout=5.0):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(wc.drain_events())
        if pred(events):
            return events
        time.sleep(0.005)
    raise AssertionError(f"predicate never satisfied; events={events}")


def test_drain_events_aggregates_all_drivers_and_request_recovery_targets_one_arm():
    wc, apis = make_workcell()
    try:
        wc.bring_up(timeout_s=20.0)
        time.sleep(0.05)
        wc.drain_events()  # bring-up chatter (stale transitions, rail phases)
        apis["arm1"].inject_error(1)  # e-stop on arm1 only
        events = _events_until(
            wc, lambda evs: any(isinstance(e, FaultEvent) and e.arm_id == "arm1" for e in evs)
        )
        assert all(e.arm_id in ("arm1", "arm2") for e in events)
        assert not any(isinstance(e, FaultEvent) and e.arm_id == "arm2" for e in events)
        assert [e.t_mono for e in events] == sorted(e.t_mono for e in events)  # timeline order
        wait_until(lambda: wc.arms["arm1"].phase is DriverPhase.LATCHED)
        assert wc.arms["arm2"].phase is DriverPhase.STREAMING  # sibling untouched
        assert wc.recovery_result("arm1") is not None and not wc.recovery_result("arm1").ok
        wc.request_recovery("arm1")
        events = _events_until(
            wc, lambda evs: any(isinstance(e, RecoveredEvent) and e.arm_id == "arm1" for e in evs)
        )
        assert any(isinstance(e, ReseedEvent) and e.arm_id == "arm1" for e in events)
        assert not any(e.arm_id == "arm2" and isinstance(e, RecoveredEvent) for e in events)
        assert wc.arms["arm1"].phase is DriverPhase.STREAMING
        assert wc.recovery_result("arm1").ok and wc.recovery_result("arm1").user_initiated
        assert wc.recovery_result("arm2") is None  # never recovered
        with pytest.raises(KeyError):
            wc.request_recovery("nope")
    finally:
        wc.shutdown()


def test_request_recovery_on_a_driver_without_the_channel_raises_command_error():
    wc, _ = make_workcell(n_arms=1)
    wc.arms["arm1"] = SimpleNamespace(get_state=lambda: None)  # foreign ArmInterface impl
    assert wc.drain_events() == []
    assert wc.recovery_result("arm1") is None
    with pytest.raises(CommandError):
        wc.request_recovery("arm1")
