"""HardwareWorkcell bring-up: parallel arms, partial failure, statuses (§9)."""

import time
from types import SimpleNamespace

import pytest
from apollo_mavis_v2_core import BringupError, CommandError, WorkcellBringupError
from apollo_mavis_v2_core.schemas import ArmConfig, PoseModel, WorkcellConfig
from fakes.fake_xarm_api import FakeXArmAPI
from test_driver_connect import wait_until

from apollo_mavis_v2_hardware.driver import DriverPhase, SettingResult, XArmDriver
from apollo_mavis_v2_hardware.events import FaultEvent, RecoveredEvent, ReseedEvent
from apollo_mavis_v2_hardware.netsetup.types import MatchResult
from apollo_mavis_v2_hardware.rail import RailNotHomedError
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


def make_workcell(
    fail_arms=frozenset(),
    netsetup=None,
    n_arms=2,
    api_cls=FakeXArmAPI,
    unhomed_arms=frozenset(),
):
    apis: dict[str, FakeXArmAPI] = {}

    def driver_factory(cfg):
        def api_factory(ip, **kw):
            api = api_cls(
                ip,
                auto_report_hz=100.0,
                has_rail=(cfg.arm_id == "arm2"),
                rail_homed=cfg.arm_id not in unhomed_arms,
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
        assert statuses["arm2"].rail == "ready"  # detected + already homed (connect never homes)
        assert apis["arm2"].homing_started == 0
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


def test_unhomed_rail_maps_to_status_unhomed_and_bring_up_never_homes():
    """Phase-09c: the lab tracks are unhomed at power-on. The arm with the rail
    fails its connect with RailNotHomedError -> rail 'unhomed' (not 'error'),
    NOTHING is written to the track, the sibling arm is untouched, and start()
    surfaces the typed error so the session layer refuses."""
    wc, apis = make_workcell(unhomed_arms={"arm2"})
    try:
        transitions: list[ArmBringupStatus] = []
        statuses = wc.bring_up(status_cb=transitions.append, timeout_s=20.0)
        assert statuses["arm1"].connected and statuses["arm1"].rail == "none"
        s2 = statuses["arm2"]
        assert not s2.connected and s2.rail == "unhomed"
        assert s2.error is not None and "not homed" in s2.error and "arm2" in s2.error
        names = apis["arm2"].call_names()
        assert "set_linear_track_back_origin" not in names  # never homes
        assert "set_linear_track_enable" not in names and "set_linear_track_speed" not in names
        assert apis["arm2"].homing_started == 0 and apis["arm2"].rail_homed is False
        assert not any(t.rail == "homing" for t in transitions)  # the value no longer exists
        with pytest.raises(WorkcellBringupError) as exc_info:
            wc.start()
        assert isinstance(exc_info.value.statuses["arm2"], RailNotHomedError)
        assert exc_info.value.statuses["arm2"].step == "rail"
        assert exc_info.value.statuses["arm1"] is None
    finally:
        wc.shutdown()
    # shutdown handed arm1 back stopped + braked (D6) and left arm2's track alone
    assert apis["arm1"].motion_enabled is False and apis["arm1"].state == 4
    assert apis["arm2"].rail_homed is False and apis["arm2"].homing_started == 0


class _TrackDropsAfterDetect(FakeXArmAPI):
    """The gate register read (2nd call) times out: carriage position unverifiable."""

    def get_linear_track_registers(self, **kwargs):
        code, regs = super().get_linear_track_registers(**kwargs)
        if self.call_names().count("get_linear_track_registers") >= 2:
            return 3, {}
        return code, regs


def test_unverifiable_rail_maps_to_status_error_with_error_set_and_no_enable():
    """Phase-09c review fix: a failed gate register read is a per-arm bring-up
    ERROR (rail 'error' + error + not connected) - never a connected arm whose
    rail silently reports 0.0 m; and the arm was never enabled."""
    wc, apis = make_workcell(api_cls=_TrackDropsAfterDetect)
    try:
        statuses = wc.bring_up(timeout_s=20.0)
        assert statuses["arm1"].connected and statuses["arm1"].rail == "none"
        s2 = statuses["arm2"]
        assert not s2.connected and s2.rail == "error"
        assert s2.error is not None and "get_linear_track_registers failed" in s2.error
        assert "arm2" in s2.error and "carriage position unverifiable" in s2.error
        api2 = apis["arm2"]
        assert ("motion_enable", (True,), {}) not in api2.calls and api2.motion_enabled is False
        assert [n for n in api2.call_names() if n.startswith("set_linear_track")] == []
        with pytest.raises(WorkcellBringupError) as exc_info:
            wc.start()
        err = exc_info.value.statuses["arm2"]
        assert isinstance(err, BringupError) and not isinstance(err, RailNotHomedError)
        assert err.step == "rail" and exc_info.value.statuses["arm1"] is None
    finally:
        wc.shutdown()


def test_arm_bringup_status_rail_literal_has_unhomed_not_homing():
    assert ArmBringupStatus(arm_id="a", rail="unhomed").rail == "unhomed"
    with pytest.raises(ValueError):
        ArmBringupStatus(arm_id="a", rail="homing")


def test_subset_config_builds_only_the_selected_drivers():
    """A session WorkcellConfig restricted to one arm (phase-09c first live run:
    Manipulation Arm only) must not construct — let alone connect — the other."""
    full = _wc_config(2)
    subset = full.model_copy(update={"arms": [a for a in full.arms if a.id == "arm1"]})
    built: list[str] = []

    def driver_factory(cfg):
        built.append(cfg.arm_id)
        return XArmDriver(cfg, api_factory=lambda ip, **kw: FakeXArmAPI(ip, **kw))

    wc = HardwareWorkcell(subset, driver_factory=driver_factory)
    assert built == ["arm1"] and set(wc.arms) == {"arm1"}
    assert wc.cfg.arms[0].id == "arm1" and len(wc.cfg.arms) == 1


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


# -- 2026-09-11: the collision-sensitivity channel, the sibling of request_recovery ---------


def test_request_set_collision_sensitivity_targets_one_arm_and_publishes_a_setting_result():
    wc, apis = make_workcell()
    try:
        wc.bring_up(timeout_s=20.0)
        assert wc.setting_result("arm1") is None and wc.setting_result("arm2") is None
        assert apis["arm1"].collision_sensitivity == 3  # connect applied the config value
        wc.request_set_collision_sensitivity("arm1", 2)
        wait_until(lambda: wc.setting_result("arm1") is not None)
        res = wc.setting_result("arm1")
        assert isinstance(res, SettingResult) and res.ok and res.level == 2 and res.code == 0
        assert res.seq == 1 and res.detail == ""
        assert apis["arm1"].collision_sensitivity == 2
        assert apis["arm2"].collision_sensitivity == 3  # the sibling is untouched
        assert wc.setting_result("arm2") is None
        # the write ran on arm1's monitor thread, not here, and moved nothing
        calls = [c for c in apis["arm1"].calls if c[0] == "set_collision_sensitivity"]
        assert calls[-1] == ("set_collision_sensitivity", (2,), {})
        assert "motion_enable" not in [c[0] for c in apis["arm1"].calls[-3:]]
        assert wc.arms["arm1"].phase is DriverPhase.STREAMING
        with pytest.raises(KeyError):
            wc.request_set_collision_sensitivity("nope", 2)
        with pytest.raises(CommandError):
            wc.request_set_collision_sensitivity("arm1", 4)  # the driver refuses 4 / 5 / 0
        assert wc.setting_result("arm1").seq == 1  # nothing new was written
    finally:
        wc.shutdown()


def test_request_set_collision_sensitivity_on_a_driver_without_the_channel_raises():
    wc, _ = make_workcell(n_arms=1)
    wc.arms["arm1"] = SimpleNamespace(get_state=lambda: None)  # foreign ArmInterface impl
    assert wc.setting_result("arm1") is None
    with pytest.raises(CommandError):
        wc.request_set_collision_sensitivity("arm1", 2)
