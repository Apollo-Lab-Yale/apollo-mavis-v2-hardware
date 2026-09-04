"""reconcile(): plan on the real polluted machine state + apply semantics (§7.3)."""

import shutil

from test_netsetup_match import make_sys_run

from apollo_mavis_v2_hardware.netsetup import ArmNet, NetSetup, TranscriptRunner
from apollo_mavis_v2_hardware.netsetup.__main__ import main as netsetup_main
from apollo_mavis_v2_hardware.netsetup.nmcli import is_mutating
from apollo_mavis_v2_hardware.netsetup.reconcile import apply_plan

ARMS = [
    ArmNet(name="arm1", ip="192.168.1.235"),
    ArmNet(name="arm2", ip="192.168.2.235"),
    ArmNet(name="arm3", ip="192.168.3.235"),
]
UUID1 = "11111111-aaaa-4aaa-8aaa-111111111111"
UUID1_DUP = "22222222-bbbb-4bbb-8bbb-222222222222"
UUID2 = "bd3e7da6-cccc-4ccc-8ccc-333333333333"
UUID3 = "6753457b-dddd-4ddd-8ddd-444444444444"
UUID_WIFI = "99999999-eeee-4eee-8eee-555555555555"
UUID_TS = "88888888-ffff-4fff-8fff-666666666666"


def make_netsetup(tmp_path, nmcli_fixtures):
    shutil.copy(nmcli_fixtures / "nic_map.json", tmp_path / "nic_map.json")
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt")
    ns = NetSetup(
        ARMS,
        state_path=tmp_path / "nic_map.json",
        run=run,
        run_sys=make_sys_run(),
        tcp_connect=lambda ip, port, timeout: "open",
        carrier_of=lambda dev: True,
    )
    return ns, run


def _step_for(plan, uuid):
    steps = [s for s in plan.steps if uuid in s.args]
    assert len(steps) <= 1
    return steps[0] if steps else None


def test_plan_is_exactly_dedupe_strip_pin_priority(tmp_path, nmcli_fixtures):
    ns, run = make_netsetup(tmp_path, nmcli_fixtures)
    plan = ns.reconcile(apply=False)
    assert not plan.applied
    # arm1's profile: already name-pinned + manual; needs MAC pin, priority,
    # never-default, gateway strip
    s1 = _step_for(plan, UUID1)
    args = list(s1.args)
    assert args[args.index("802-3-ethernet.mac-address") + 1] == "08:BF:B8:89:4F:3A"
    assert args[args.index("connection.autoconnect-priority") + 1] == "50"
    assert args[args.index("ipv4.never-default") + 1] == "yes"
    assert args[args.index("ipv4.gateway") + 1] == ""
    assert "connection.interface-name" not in args  # already pinned
    assert "ipv4.method" not in args  # already manual
    # arm2's profile carries a gateway OUTSIDE its own subnet (live bug) and
    # is unpinned: interface-name + MAC + priority + hygiene
    s2 = _step_for(plan, UUID2)
    args2 = list(s2.args)
    assert args2[args2.index("connection.interface-name") + 1] == "enp36s0f1"
    assert args2[args2.index("802-3-ethernet.mac-address") + 1] == "08:BF:B8:89:4F:3B"
    assert args2[args2.index("ipv4.gateway") + 1] == ""
    # arm3 pinned to the USB dongle by MAC (its name is already MAC-derived)
    s3 = _step_for(plan, UUID3)
    args3 = list(s3.args)
    assert args3[args3.index("connection.interface-name") + 1] == "enx00e04c683d97"
    # the duplicate xarm7_1 is disabled, not deleted
    sdup = _step_for(plan, UUID1_DUP)
    assert list(sdup.args[-2:]) == ["connection.autoconnect", "no"]
    assert "dedupe" in sdup.reason
    # exactly these four steps; internet profiles are untouched
    assert len(plan.steps) == 4
    assert _step_for(plan, UUID_WIFI) is None and _step_for(plan, UUID_TS) is None


def test_plan_only_executes_no_mutating_nmcli(tmp_path, nmcli_fixtures):
    ns, run = make_netsetup(tmp_path, nmcli_fixtures)
    ns.reconcile(apply=False)
    for call in run.calls:
        assert not is_mutating(call), f"plan-only reconcile mutated: {call}"


def test_apply_executes_exactly_the_plan(tmp_path, nmcli_fixtures):
    ns, _ = make_netsetup(tmp_path, nmcli_fixtures)
    plan = ns.reconcile(apply=False)
    executor = TranscriptRunner({step.args: (0, "") for step in plan.steps})
    apply_plan(plan, executor)
    assert plan.applied
    assert executor.calls == [step.args for step in plan.steps]


def test_apply_never_touches_profiles_carrying_sdk_traffic(tmp_path, nmcli_fixtures):
    ns, _ = make_netsetup(tmp_path, nmcli_fixtures)
    plan = ns.reconcile(apply=False, active_uuids=frozenset({UUID1}))
    assert _step_for(plan, UUID1) is None
    assert any(UUID1 in s for s in plan.skipped)
    assert len(plan.steps) == 3  # everything else still planned


def test_cli_reconcile_plan_only_with_fixture_runner(tmp_path, nmcli_fixtures, capsys):
    """python -m apollo_mavis_v2_hardware.netsetup reconcile (fixture-injected)."""
    shutil.copy(nmcli_fixtures / "nic_map.json", tmp_path / "nic_map.json")
    rc = netsetup_main(
        [
            "reconcile",
            "--fixtures", str(nmcli_fixtures / "machine_polluted.txt"),
            "--state", str(tmp_path / "nic_map.json"),
            "--arm", "arm1=192.168.1.235",
            "--arm", "arm2=192.168.2.235",
            "--arm", "arm3=192.168.3.235",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "plan only" in out and "--apply" in out
    assert "4 step(s)" in out
    # the TranscriptRunner contains NO mutating entries, so reaching here
    # proves no mutating nmcli command was executed without --apply
    assert UUID1 in out and UUID2 in out and UUID3 in out and UUID1_DUP in out
    assert "ipv4.gateway" in out and "802-3-ethernet.mac-address" in out


def test_plan_without_mapping_disables_nothing(tmp_path, nmcli_fixtures):
    """Live regression (2026-09-04): with no nic_map.json the old plan wanted to
    disable BOTH working arm profiles as 'stale duplicates'. No mapping -> no
    keeper -> nothing is a duplicate; --apply before match must be harmless."""
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt")
    ns = NetSetup(
        ARMS, state_path=tmp_path / "absent.json", run=run, run_sys=make_sys_run(),
        tcp_connect=lambda ip, port, timeout: "open", carrier_of=lambda dev: True,
    )
    plan = ns.reconcile(apply=False)
    assert plan.steps == []
    assert len(plan.skipped) == 3 and all("run match first" in s for s in plan.skipped)
    for call in run.calls:
        assert not is_mutating(call)


def test_plan_partial_mapping_dedupes_only_mapped_subnets(tmp_path, nmcli_fixtures):
    from apollo_mavis_v2_hardware.netsetup.state import save_nic_map
    from apollo_mavis_v2_hardware.netsetup.types import NicMapEntry

    save_nic_map(
        {"arm1": NicMapEntry("08:BF:B8:89:4F:3A", "enp36s0f0", UUID1, "192.168.1.235")},
        tmp_path / "nic_map.json",
    )
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt")
    ns = NetSetup(
        ARMS, state_path=tmp_path / "nic_map.json", run=run, run_sys=make_sys_run(),
        tcp_connect=lambda ip, port, timeout: "open", carrier_of=lambda dev: True,
    )
    plan = ns.reconcile(apply=False)
    assert _step_for(plan, UUID1) is not None  # mapped arm: pin + hygiene
    assert _step_for(plan, UUID1_DUP) is not None  # its duplicate: disabled
    assert _step_for(plan, UUID2) is None and _step_for(plan, UUID3) is None  # unmapped: kept
    assert len(plan.steps) == 2


def test_plan_clears_user_restricted_permissions(tmp_path, nmcli_fixtures):
    """connection.permissions set -> only one account can activate the profile;
    the root dispatcher and other users cannot. reconcile makes it system-wide."""
    shutil.copy(nmcli_fixtures / "nic_map.json", tmp_path / "nic_map.json")
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt",
                                      nmcli_fixtures / "permissions_restricted.txt")
    ns = NetSetup(
        ARMS, state_path=tmp_path / "nic_map.json", run=run, run_sys=make_sys_run(),
        tcp_connect=lambda ip, port, timeout: "open", carrier_of=lambda dev: True,
    )
    plan = ns.reconcile(apply=False)
    args = list(_step_for(plan, UUID1).args)
    assert args[args.index("connection.permissions") + 1] == ""
    assert "system-wide" in _step_for(plan, UUID1).reason
    # unrestricted profiles get no permissions arg
    assert "connection.permissions" not in _step_for(plan, UUID2).args
