"""repair(): the NM dispatcher's conservative convergence policy (02-hardware §7.4).

Fixtures model the MAVIS v2 cell in its pinned steady state (live UUIDs). The
invariants under test: a healthy system produces NO nmcli mutation (that is what
stops the NM-event -> dispatcher -> nmcli -> NM-event loop); a healthy arm's NIC
is never a candidate; unplugged arms are left to NM autoconnect; booting arms are
polled, not re-probed; swapped cables are re-matched after the deadline; failed
re-probes put the profile back; re-probes of merely unreachable arms are
rate-limited by the holdoff stamp.
"""

import shutil
import subprocess

from apollo_mavis_v2_hardware.netsetup import ArmNet, NetSetup, TranscriptRunner
from apollo_mavis_v2_hardware.netsetup.__main__ import main as netsetup_main
from apollo_mavis_v2_hardware.netsetup.match import classify_for_repair
from apollo_mavis_v2_hardware.netsetup.nmcli import is_mutating
from apollo_mavis_v2_hardware.netsetup.state import holdoff_path, load_nic_map, write_holdoff
from apollo_mavis_v2_hardware.netsetup.types import MatchResult, NicInfo, NicMapEntry

VIEW = ArmNet(name="view", ip="192.168.2.219")
GRIP = ArmNet(name="grip", ip="192.168.1.201")
ARMS = [VIEW, GRIP]
UUID_V = "6753457b-89fc-435a-a5a0-e94c728bc826"
UUID_G = "5a4846b2-2383-4f3f-8871-ce71a8f7f303"
F0, F1, DONGLE = "enp36s0f0", "enp36s0f1", "enx00e04c683d97"
MAC_F0, MAC_F1, MAC_DONGLE = "08:BF:B8:89:4F:3A", "08:BF:B8:89:4F:3B", "00:E0:4C:68:3D:97"
ROUTES = '[{"dst":"default","gateway":"192.168.0.1","dev":"wlp38s0","metric":600}]'


def make_sys_run(ping_ok):
    """ping keyed by (dev, ip): which arm answers behind which NIC."""
    calls = []

    def run(argv):
        calls.append(tuple(argv))
        if argv[0] == "ping":
            dev, ip = argv[argv.index("-I") + 1], argv[-1]
            return subprocess.CompletedProcess(argv, 0 if ping_ok(dev, ip) else 1, "", "")
        if argv[:3] == ["ip", "-j", "route"]:
            return subprocess.CompletedProcess(argv, 0, ROUTES, "")
        if argv[:3] == ["ip", "neigh", "show"]:
            out = f"{argv[3]} dev {F0} lladdr de:ad:be:ef:00:01 REACHABLE"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] == "id":
            return subprocess.CompletedProcess(argv, 0, "user netdev\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    run.calls = calls
    return run


def healthy(dev, ip):
    return (dev, ip) in {(F0, VIEW.ip), (F1, GRIP.ip)}


def swapped(dev, ip):
    return (dev, ip) in {(F1, VIEW.ip), (F0, GRIP.ip)}


def nobody(dev, ip):
    return False


def make_netsetup(tmp_path, nmcli_fixtures, files, ping_ok, carrier=frozenset({F0, F1})):
    shutil.copy(nmcli_fixtures / "nic_map_pinned.json", tmp_path / "nic_map.json")
    run = TranscriptRunner.from_files(*[nmcli_fixtures / f for f in files])
    ns = NetSetup(
        ARMS,
        state_path=tmp_path / "nic_map.json",
        run=run,
        run_sys=make_sys_run(ping_ok),
        tcp_connect=lambda ip, port, timeout: "open",
        carrier_of=lambda dev: dev in carrier,
    )
    return ns, run


def repair(ns, fake_clock, **kwargs):
    return ns.repair(sleep=fake_clock.sleep, clock=fake_clock.now, now=fake_clock.now,
                     **kwargs)


def mutating(run):
    return [c for c in run.calls if is_mutating(c)]


def test_healthy_system_is_quiet(tmp_path, nmcli_fixtures, fake_clock):
    ns, run = make_netsetup(tmp_path, nmcli_fixtures, ["machine_pinned.txt"], healthy)
    results = repair(ns, fake_clock)
    assert all(r.ok for r in results.values())
    assert mutating(run) == []  # the convergence guarantee: no NM change, no new events
    assert ns.notes == []
    assert not holdoff_path(tmp_path / "nic_map.json").exists()


def test_unplugged_arm_is_left_to_autoconnect(tmp_path, nmcli_fixtures, fake_clock):
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_pinned.txt", "pinned_f1_unavailable.txt"],
        healthy, carrier={F0},
    )
    results = repair(ns, fake_clock)
    assert results["view"].ok
    assert not results["grip"].ok and results["grip"].reason == "profile-inactive"
    assert mutating(run) == []  # nothing to match: cable is gone; pins stay
    assert any("no carrier" in n for n in ns.notes)
    assert load_nic_map(tmp_path / "nic_map.json")["grip"].mac == MAC_F1  # unchanged


def test_booting_arm_is_polled_not_reprobed(tmp_path, nmcli_fixtures, fake_clock):
    start = fake_clock.now()

    def ping_ok(dev, ip):  # grip's IP stack comes up 20 s after link
        if (dev, ip) == (F1, GRIP.ip):
            return fake_clock.now() >= start + 20.0
        return healthy(dev, ip)

    ns, run = make_netsetup(tmp_path, nmcli_fixtures, ["machine_pinned.txt"], ping_ok)
    results = repair(ns, fake_clock)
    assert results["grip"].ok and results["view"].ok
    assert mutating(run) == []
    assert 20.0 <= fake_clock.now() - start < 60.0  # waited, but not the full deadline


def test_swapped_cables_are_rematched_after_deadline(tmp_path, nmcli_fixtures, fake_clock):
    start = fake_clock.now()
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_pinned.txt", "pinned_swap_extra.txt"], swapped
    )
    results = repair(ns, fake_clock)
    assert fake_clock.now() - start >= 60.0  # gave the boxes the full boot deadline
    assert results["view"].ok and results["view"].ifname == F1
    assert results["grip"].ok and results["grip"].ifname == F0
    nic_map = load_nic_map(tmp_path / "nic_map.json")
    assert nic_map["view"].mac == MAC_F1 and nic_map["grip"].mac == MAC_F0
    # both pins were cleared before probing, then re-pinned to the new NICs
    assert ("connection", "modify", "uuid", UUID_V,
            "connection.interface-name", "", "802-3-ethernet.mac-address", "") in run.calls
    assert ("connection", "modify", "uuid", UUID_V, "connection.interface-name", F1,
            "802-3-ethernet.mac-address", MAC_F1) in run.calls
    assert ("connection", "modify", "uuid", UUID_G, "connection.interface-name", F0,
            "802-3-ethernet.mac-address", MAC_F0) in run.calls
    assert holdoff_path(tmp_path / "nic_map.json").exists()
    assert any("swapped" in n for n in ns.notes)


def test_unreachable_arms_within_holdoff_are_not_reprobed(tmp_path, nmcli_fixtures, fake_clock):
    ns, run = make_netsetup(tmp_path, nmcli_fixtures, ["machine_pinned.txt"], nobody)
    write_holdoff(tmp_path / "nic_map.json", fake_clock.now() - 30.0)  # re-probed 30 s ago
    results = repair(ns, fake_clock)
    assert not results["view"].ok and not results["grip"].ok
    assert mutating(run) == []  # no down/up churn while the boxes stay silent
    assert any("holdoff" in n for n in ns.notes)


def test_failed_reprobe_restores_profile_on_stored_nic(tmp_path, nmcli_fixtures, fake_clock):
    ns, run = make_netsetup(
        tmp_path, nmcli_fixtures, ["machine_pinned.txt", "pinned_f1_stolen.txt"],
        lambda dev, ip: (dev, ip) == (F0, VIEW.ip),
    )
    results = repair(ns, fake_clock)
    assert results["view"].ok
    assert not results["grip"].ok and results["grip"].reason == "no-candidate"
    up_f1 = ("-w", "15", "connection", "up", "uuid", UUID_G, "ifname", F1)
    down = ("-w", "10", "connection", "down", "uuid", UUID_G)
    assert run.calls.count(up_f1) == 2  # probe, then restore
    assert run.calls.index(down) > run.calls.index(up_f1)
    assert run.calls.index(down) < len(run.calls) - 1 - run.calls[::-1].index(up_f1)
    assert not any(UUID_V in c for c in mutating(run))  # healthy arm untouched
    assert not any(F0 in c for c in mutating(run))  # its NIC was frozen
    assert any("restored" in n for n in ns.notes)


def _nic(dev, mac, carrier, state="connected", conn=""):
    return NicInfo(dev=dev, type="ethernet", state=state, connection=conn, carrier=carrier,
                   mac=mac)


def _map():
    return {
        "view": NicMapEntry(MAC_F0, F0, UUID_V, VIEW.ip),
        "grip": NicMapEntry(MAC_F1, F1, UUID_G, GRIP.ip),
    }


def test_classify_cable_moved_to_free_nic_probes():
    nics = [_nic(F0, MAC_F0, True, conn="mavis_viewpoint_arm"),
            _nic(F1, MAC_F1, False, state="unavailable"),
            _nic(DONGLE, MAC_DONGLE, True, state="disconnected")]
    results = {
        "view": MatchResult("view", F0, MAC_F0, UUID_V, "open"),
        "grip": MatchResult("grip", F1, MAC_F1, UUID_G, "unreachable",
                            "stored profile not active on stored NIC", reason="profile-inactive"),
    }
    dec = classify_for_repair(ARMS, results, _map(), nics, deny=set())
    assert dec.frozen == {F0}
    assert dec.probe == ["grip"] and dec.pending == []
    assert any(DONGLE in n and "cable moved" in n for n in dec.notes)


def test_classify_first_run_and_missing_nic_probe_immediately():
    nics = [_nic(F0, MAC_F0, True), _nic(F1, MAC_F1, True)]
    results = {
        "view": MatchResult("view", "", "", "", "unreachable", "no persisted mapping",
                            reason="no-mapping"),
        "grip": MatchResult("grip", "", "AA:AA:AA:AA:AA:AA", UUID_G, "unreachable",
                            "MAC AA:AA:AA:AA:AA:AA not present", reason="nic-missing"),
    }
    dec = classify_for_repair(ARMS, results, {}, nics, deny=set())
    assert dec.probe == ["view", "grip"] and dec.pending == [] and dec.frozen == set()


def test_classify_active_but_unreachable_is_pending():
    nics = [_nic(F0, MAC_F0, True), _nic(F1, MAC_F1, True)]
    results = {
        "view": MatchResult("view", F0, MAC_F0, UUID_V, "refused",
                            "arm booting: poll 502, do not re-probe NICs"),
        "grip": MatchResult("grip", F1, MAC_F1, UUID_G, "unreachable", reason="unreachable"),
    }
    dec = classify_for_repair(ARMS, results, _map(), nics, deny=set())
    assert dec.frozen == {F0}  # "refused" is ok: right NIC, box booting
    assert dec.pending == ["grip"] and dec.probe == []


def test_cli_match_repair_with_fixtures_never_mutates(tmp_path, nmcli_fixtures, capsys):
    """python -m apollo_mavis_v2_hardware.netsetup match --repair (fixture-injected):
    the transcript holds no mutating entries, so completing proves nothing was
    executed against NetworkManager."""
    shutil.copy(nmcli_fixtures / "nic_map_pinned.json", tmp_path / "nic_map.json")
    rc = netsetup_main([
        "match", "--repair",
        "--fixtures", str(nmcli_fixtures / "machine_pinned.txt"),
        "--state", str(tmp_path / "nic_map.json"),
        "--arm", "view=192.168.2.219", "--arm", "grip=192.168.1.201",
    ])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert f"state: {tmp_path / 'nic_map.json'}" in out
    assert "view" in out and "grip" in out
