"""HardwareWorkcell bring-up: parallel arms, partial failure, statuses (§9)."""

import time

import pytest
from apollo_mavis_v2_core import WorkcellBringupError
from apollo_mavis_v2_core.schemas import ArmConfig, PoseModel, WorkcellConfig
from fakes.fake_xarm_api import FakeXArmAPI

from apollo_mavis_v2_hardware.driver import XArmDriver
from apollo_mavis_v2_hardware.netsetup.types import MatchResult
from apollo_mavis_v2_hardware.workcell import ArmBringupStatus, HardwareWorkcell


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


def make_workcell(fail_arms=frozenset(), netsetup=None, n_arms=2):
    apis: dict[str, FakeXArmAPI] = {}

    def driver_factory(cfg):
        def api_factory(ip, **kw):
            api = FakeXArmAPI(
                ip, auto_report_hz=100.0, has_rail=(cfg.arm_id == "arm2"),
                rail_homed=True, **kw,
            )
            if cfg.arm_id in fail_arms:
                api.connected = False  # ArmConnectError after retries
            apis[cfg.arm_id] = api
            return api

        return XArmDriver(cfg, api_factory=api_factory, sleep=_fast_sleep)

    wc = HardwareWorkcell(_wc_config(n_arms), driver_factory=driver_factory,
                          netsetup=netsetup)
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
            "arm1": MatchResult("arm1", "enp36s0f0", "08:BF:B8:89:4F:3A",
                                "uuid-1", self.probe_result, ""),
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
