"""verify()/match(): denylist, UUID addressing, serialized probing (§7.2)."""

import subprocess

import pytest

from apollo_xarm7_hardware.netsetup import ArmNet, NetSetup, TranscriptRunner
from apollo_xarm7_hardware.netsetup.match import create_profile
from apollo_xarm7_hardware.netsetup.nmcli import is_mutating
from apollo_xarm7_hardware.netsetup.probe import (
    candidate_pool,
    denylist,
    internet_devices,
    list_nics,
)
from apollo_xarm7_hardware.netsetup.state import save_nic_map
from apollo_xarm7_hardware.netsetup.types import NicMapEntry

ARM1 = ArmNet(name="arm1", ip="192.168.1.235")
UUID1 = "11111111-aaaa-4aaa-8aaa-111111111111"
ROUTES = (
    '[{"dst":"default","gateway":"192.168.0.1","dev":"wlp38s0","metric":600},'
    '{"dst":"default","gateway":"192.168.1.1","dev":"enp36s0f0","metric":20100}]'
)
CARRIER = {"enp36s0f0", "enp36s0f1"}


def make_sys_run(ping_ok=frozenset(), routes=ROUTES):
    calls = []

    def run(argv):
        calls.append(tuple(argv))
        if argv[0] == "ping":
            dev = argv[argv.index("-I") + 1]
            rc = 0 if dev in ping_ok else 1
            return subprocess.CompletedProcess(argv, rc, "", "")
        if argv[:3] == ["ip", "-j", "route"]:
            return subprocess.CompletedProcess(argv, 0, routes, "")
        if argv[:3] == ["ip", "neigh", "show"]:
            out = f"{argv[3]} dev enp36s0f1 lladdr de:ad:be:ef:00:01 REACHABLE"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] == "id":
            return subprocess.CompletedProcess(argv, 0, "user adm netdev\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    run.calls = calls
    return run


def make_netsetup(tmp_path, fixtures, files, tcp="open", ping_ok=frozenset(),
                  arms=(ARM1,)):
    run = TranscriptRunner.from_files(*[fixtures / f for f in files])
    ns = NetSetup(
        list(arms),
        state_path=tmp_path / "nic_map.json",
        run=run,
        run_sys=make_sys_run(ping_ok=ping_ok),
        tcp_connect=lambda ip, port, timeout: tcp,
        carrier_of=lambda dev: dev in CARRIER,
    )
    return ns, run


def test_internet_nic_is_lowest_metric_only() -> None:
    # the bogus metric-20100 default route via the arm NIC must NOT denylist it
    assert internet_devices(make_sys_run()) == {"wlp38s0"}


def test_internet_nic_never_in_candidate_pool(nmcli_fixtures) -> None:
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt")
    nics = list_nics(run, carrier_of=lambda dev: True)  # even with carrier on
    routes = '[{"dst":"default","dev":"enp36s0f0","metric":100}]'  # eth as internet
    deny = denylist(nics, make_sys_run(routes=routes))
    pool = candidate_pool(nics, deny, mapped=set())
    assert "enp36s0f0" in deny  # lowest-metric default-route device
    assert "wlp38s0" in deny and "tailscale0" in deny  # non-ethernet always
    assert [n.dev for n in pool] == ["enp36s0f1", "enx00e04c683d97"]


def test_match_selects_duplicate_named_profile_by_uuid(tmp_path, nmcli_fixtures):
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "match_extra.txt"],
        ping_ok={"enp36s0f1"},
    )
    results = ns.match(reconcile_after=False)
    res = results["arm1"]
    # two profiles are both named xarm7_1 — the right one by UUID, never name
    assert res.profile_uuid == UUID1
    for call in run.calls:
        assert "xarm7_1" not in call  # every profile op addressed by UUID
    assert res.probe == "open" and res.ifname == "enp36s0f1"
    assert res.mac == "08:BF:B8:89:4F:3B"


def test_match_probes_serially_and_releases_nics(tmp_path, nmcli_fixtures):
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "match_extra.txt"],
        ping_ok={"enp36s0f1"},
    )
    ns.match(reconcile_after=False)
    ups = [c for c in run.calls if "up" in c]
    downs = [c for c in run.calls if "down" in c]
    assert len(ups) == 2 and len(downs) == 1  # f0 tried, released, then f1
    assert run.calls.index(downs[0]) > run.calls.index(ups[0])
    assert run.calls.index(downs[0]) < run.calls.index(ups[1])
    # the pin was cleared before probing (ifname override would fail otherwise)
    assert ("connection", "modify", "uuid", UUID1,
            "connection.interface-name", "") in run.calls


def test_match_never_mutates_denylisted_devices(tmp_path, nmcli_fixtures):
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "match_extra.txt"],
        ping_ok={"enp36s0f1"},
    )
    ns.match(reconcile_after=False)
    for call in run.calls:  # global call-log assert
        if is_mutating(call):
            assert "wlp38s0" not in call and "tailscale0" not in call


def test_match_persists_mac_keyed_state(tmp_path, nmcli_fixtures):
    ns, _ = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "match_extra.txt"],
        ping_ok={"enp36s0f1"},
    )
    ns.match(reconcile_after=False)
    from apollo_xarm7_hardware.netsetup.state import load_nic_map

    entry = load_nic_map(tmp_path / "nic_map.json")["arm1"]
    assert entry.mac == "08:BF:B8:89:4F:3B"  # MAC is the stable key
    assert entry.ifname == "enp36s0f1"
    assert entry.profile_uuid == UUID1


def test_create_profile_is_route_inert(nmcli_fixtures):
    run = TranscriptRunner.from_files(nmcli_fixtures / "create_profile.txt")
    uuid = create_profile(run, ArmNet(name="arm4", ip="10.0.4.235"))
    assert uuid == "44444444-4444-4444-8444-444444444444"
    add = [c for c in run.calls if c[:2] == ("connection", "add")][0]
    args = list(add)
    i = args.index("ipv4.never-default")
    assert args[i + 1] == "yes"
    j = args.index("ipv4.gateway")
    assert args[j + 1] == ""  # NO gateway: the arm subnet is a stub
    assert args[args.index("ipv6.method") + 1] == "disabled"
    assert args[args.index("ipv4.addresses") + 1] == "10.0.4.12/24"  # .12 convention


def _seed_state(tmp_path) -> None:
    save_nic_map(
        {
            "arm1": NicMapEntry(
                mac="08:BF:B8:89:4F:3A", ifname="enp36s0f0",
                profile_uuid=UUID1, arm_ip="192.168.1.235",
            )
        },
        tmp_path / "nic_map.json",
    )


def test_verify_fast_path_ok(tmp_path, nmcli_fixtures):
    _seed_state(tmp_path)
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "verify_extra.txt"],
        ping_ok={"enp36s0f0"},
    )
    results = ns.verify()
    assert results["arm1"].ok and results["arm1"].probe == "open"
    for call in run.calls:
        assert not is_mutating(call)  # verify is read-only


def test_ping_ok_but_502_refused_means_arm_booting(tmp_path, nmcli_fixtures):
    _seed_state(tmp_path)
    ns, _ = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "verify_extra.txt"],
        tcp="refused", ping_ok={"enp36s0f0"},
    )
    res = ns.verify()["arm1"]
    assert res.probe == "refused"
    assert res.ok  # right NIC — poll 502, do NOT re-probe NICs
    assert "poll 502" in res.detail
    assert ns.poll_502("arm1") == "refused"  # the booting loop path


def test_verify_miss_without_state(tmp_path, nmcli_fixtures):
    ns, _ = make_netsetup(tmp_path, nmcli_fixtures, ["machine_polluted.txt"])
    res = ns.verify()["arm1"]
    assert not res.ok and "no persisted mapping" in res.detail


def test_ping_retried_at_least_three_times(tmp_path, nmcli_fixtures):
    # first ping is routinely lost to ARP: the prober must retry >= 3x
    _seed_state(tmp_path)
    ns, _ = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_polluted.txt", "verify_extra.txt"],
        ping_ok=frozenset(),  # ping never succeeds
    )
    sys_run = ns._run_sys
    res = ns.verify()["arm1"]
    assert res.probe == "unreachable"
    pings = [c for c in sys_run.calls if c and c[0] == "ping"]
    assert len(pings) >= 3


def test_install_check_reports_missing_grant(tmp_path):
    from apollo_xarm7_hardware.netsetup import install as install_mod

    problems = install_mod.check(
        run_sys=make_sys_run(), pkla_path=tmp_path / "nope.pkla"
    )
    assert any("polkit grant missing" in p for p in problems)
    # user IS in netdev per make_sys_run -> only the grant problem
    assert len(problems) == 1
    pkla = tmp_path / "46-apollo-networkmanager.pkla"
    pkla.write_text(install_mod.PKLA_CONTENT)
    assert install_mod.check(run_sys=make_sys_run(), pkla_path=pkla) == []


def test_pkla_content_targets_local_authority() -> None:
    # Ubuntu 22.04 polkit 0.105: LocalAuthority only, JS .rules are IGNORED
    from apollo_xarm7_hardware.netsetup.install import PKLA_CONTENT, PKLA_PATH

    assert str(PKLA_PATH).startswith("/etc/polkit-1/localauthority/")
    assert str(PKLA_PATH).endswith(".pkla")
    assert "Identity=unix-group:netdev" in PKLA_CONTENT
    assert "org.freedesktop.NetworkManager.network-control" in PKLA_CONTENT
    assert "org.freedesktop.NetworkManager.settings.modify.system" in PKLA_CONTENT


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
