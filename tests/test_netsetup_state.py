"""nic_map.json location precedence (§7.1): explicit > user (exists) > system (exists) > user."""

import os
import shutil

import pytest
from test_netsetup_match import ARM1, UUID1, make_sys_run

from apollo_mavis_v2_hardware.netsetup import NetSetup, TranscriptRunner
from apollo_mavis_v2_hardware.netsetup import state as state_mod
from apollo_mavis_v2_hardware.netsetup.state import (
    load_nic_map,
    read_holdoff,
    resolve_state_path,
    save_nic_map,
    write_holdoff,
)
from apollo_mavis_v2_hardware.netsetup.types import NicMapEntry

CARRIER = {"enp36s0f0", "enp36s0f1"}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    user = tmp_path / "home" / ".config" / "apollo-mavis-v2" / "nic_map.json"
    system = tmp_path / "etc" / "apollo-mavis-v2" / "nic_map.json"
    monkeypatch.setattr(state_mod, "USER_STATE_PATH", user)
    monkeypatch.setattr(state_mod, "SYSTEM_STATE_PATH", system)
    return user, system


def test_resolve_precedence(paths, tmp_path):
    user, system = paths
    assert resolve_state_path() == user  # neither exists: user path is the write default
    system.parent.mkdir(parents=True)
    system.write_text("{}")
    assert resolve_state_path() == system  # only the root-written map exists
    user.parent.mkdir(parents=True)
    user.write_text("{}")
    assert resolve_state_path() == user  # a user map shadows the system map
    explicit = tmp_path / "elsewhere.json"
    assert resolve_state_path(explicit) == explicit  # --state wins even if missing


def test_netsetup_state_path_resolves_dynamically(paths):
    user, system = paths
    ns = NetSetup([ARM1])
    assert ns.state_path == user
    system.parent.mkdir(parents=True)
    system.write_text("{}")
    assert ns.state_path == system  # picked up without re-construction
    assert NetSetup([ARM1], state_path=user).state_path == user


def test_verify_reads_the_system_map_when_no_user_map(paths, nmcli_fixtures):
    """The runtime's session-start fast path consumes what root (dispatcher /
    install) wrote to /etc without any --state plumbing."""
    user, system = paths
    save_nic_map(
        {"arm1": NicMapEntry("08:BF:B8:89:4F:3A", "enp36s0f0", UUID1, ARM1.ip)}, system
    )
    run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt",
                                      nmcli_fixtures / "verify_extra.txt")
    ns = NetSetup([ARM1], run=run, run_sys=make_sys_run(ping_ok={"enp36s0f0"}),
                  tcp_connect=lambda ip, port, timeout: "open",
                  carrier_of=lambda dev: dev in CARRIER)
    assert ns.verify()["arm1"].ok
    assert not user.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_match_falls_back_to_user_map_when_system_map_is_read_only(paths, nmcli_fixtures):
    user, system = paths
    save_nic_map({}, system)
    system.parent.chmod(0o555)
    try:
        run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt",
                                          nmcli_fixtures / "match_extra.txt")
        ns = NetSetup([ARM1], run=run, run_sys=make_sys_run(ping_ok={"enp36s0f1"}),
                      tcp_connect=lambda ip, port, timeout: "open",
                      carrier_of=lambda dev: dev in CARRIER)
        assert ns.state_path == system
        res = ns.match(reconcile_after=False)
        assert res["arm1"].ok
        assert load_nic_map(user)["arm1"].profile_uuid == UUID1
        assert any("not writable" in w for w in ns.warnings)
        assert ns.state_path == user  # the user map now shadows the system map
    finally:
        system.parent.chmod(0o755)


def test_explicit_read_only_state_is_an_error(paths, nmcli_fixtures, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    from apollo_mavis_v2_hardware.netsetup import NetSetupError

    target = tmp_path / "ro" / "nic_map.json"
    save_nic_map({}, target)
    target.parent.chmod(0o555)
    try:
        run = TranscriptRunner.from_files(nmcli_fixtures / "machine_polluted.txt",
                                          nmcli_fixtures / "match_extra.txt")
        ns = NetSetup([ARM1], state_path=target, run=run,
                      run_sys=make_sys_run(ping_ok={"enp36s0f1"}),
                      tcp_connect=lambda ip, port, timeout: "open",
                      carrier_of=lambda dev: dev in CARRIER)
        with pytest.raises(NetSetupError, match="not writable"):
            ns.match(reconcile_after=False)
    finally:
        target.parent.chmod(0o755)


def test_holdoff_stamp_roundtrip(tmp_path):
    state = tmp_path / "nic_map.json"
    assert read_holdoff(state) is None  # never re-probed
    write_holdoff(state, 1234.5)
    assert read_holdoff(state) == 1234.5
    assert state_mod.holdoff_path(state) == tmp_path / "nic_map.holdoff"


def test_load_tolerates_missing_and_foreign_entries(tmp_path, nmcli_fixtures):
    assert load_nic_map(tmp_path / "absent.json") == {}
    shutil.copy(nmcli_fixtures / "nic_map.json", tmp_path / "nic_map.json")
    entries = load_nic_map(tmp_path / "nic_map.json")
    assert set(entries) == {"arm1", "arm2", "arm3"}
