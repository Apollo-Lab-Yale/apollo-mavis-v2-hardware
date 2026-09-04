"""nmcli terse parsing + transcript runner + mutation detection (§7.1)."""

import pytest

from apollo_mavis_v2_hardware.netsetup.nmcli import (
    TranscriptRunner,
    is_mutating,
    split_terse,
    unescape_value,
)


def test_split_terse_plain_fields() -> None:
    assert split_terse("enp36s0f0:ethernet:connected:xarm7_1") == [
        "enp36s0f0", "ethernet", "connected", "xarm7_1",
    ]


def test_split_terse_unescapes_mac_colons() -> None:
    # MACs arrive with escaped colons in -t/-g output (observed on the machine)
    line = r"xarm7_2:802-3-ethernet:08\:BF\:B8\:89\:4F\:3B"
    assert split_terse(line) == ["xarm7_2", "802-3-ethernet", "08:BF:B8:89:4F:3B"]


def test_split_terse_handles_escaped_backslash_and_empties() -> None:
    assert split_terse(r"a\\b:c") == ["a\\b", "c"]
    assert split_terse("::x") == ["", "", "x"]
    assert split_terse("") == [""]


def test_unescape_value() -> None:
    assert unescape_value(r"08\:BF\:B8\:89\:4F\:3A") == "08:BF:B8:89:4F:3A"
    assert unescape_value("plain") == "plain"


def test_transcript_runner_replays_and_rejects_unscripted(tmp_path) -> None:
    fixture = tmp_path / "t.txt"
    fixture.write_text(
        "$ nmcli -t -f DEVICE device status\n"
        "enp36s0f0\n"
        "$ rc=10 nmcli connection show uuid missing\n"
    )
    run = TranscriptRunner.from_files(fixture)
    ok = run("-t", "-f", "DEVICE", "device", "status")
    assert ok.returncode == 0 and ok.stdout == "enp36s0f0\n"
    fail = run("connection", "show", "uuid", "missing")
    assert fail.returncode == 10
    with pytest.raises(AssertionError, match="unscripted"):
        run("connection", "delete", "uuid", "whatever")
    assert len(run.calls) == 3


def test_is_mutating_classification() -> None:
    assert is_mutating(("connection", "modify", "uuid", "x", "ipv4.gateway", ""))
    assert is_mutating(("-w", "15", "connection", "up", "uuid", "x", "ifname", "eth0"))
    assert is_mutating(("connection", "add", "type", "ethernet"))
    assert is_mutating(("connection", "delete", "uuid", "x"))
    assert not is_mutating(("-t", "-f", "DEVICE", "device", "status"))
    assert not is_mutating(("-g", "connection.uuid", "connection", "show", "uuid", "x"))
    assert not is_mutating(("-g", "GENERAL.HWADDR", "device", "show", "enp36s0f0"))
