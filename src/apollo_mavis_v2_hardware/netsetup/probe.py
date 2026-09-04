"""NIC pools, denylist, and the arm probe (02-hardware §7.2).

Probing needs no privileges: ``ping`` (ping_group_range), ``ip neigh``, and a
plain TCP connect to port 502 (the live SDK control channel — connect, close,
**write nothing**). All subprocess/socket use goes through injectable seams so
tests never touch real network state.
"""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path

from .nmcli import NmcliRunner, unescape_value
from .types import NicInfo, ProbeResult

# seams
SysRunner = Callable[[list[str]], subprocess.CompletedProcess]
TcpConnect = Callable[[str, int, float], ProbeResult]
CarrierReader = Callable[[str], bool]

XARM_CONTROL_PORT = 502  # 30001-30003 are report streams; never probe those


def sys_run(argv: list[str]) -> subprocess.CompletedProcess:
    """Default runner for ping / ip (never nmcli — that goes via NmcliRunner)."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=10.0, check=False)


def read_carrier(dev: str) -> bool:
    path = Path(f"/sys/class/net/{dev}/carrier")
    try:
        return path.read_text().strip() == "1"
    except OSError:
        return False


def tcp_probe(ip: str, port: int = XARM_CONTROL_PORT, timeout: float = 1.0) -> ProbeResult:
    """Connect-and-close probe; writes nothing on the socket."""
    try:
        socket.create_connection((ip, port), timeout=timeout).close()
        return "open"
    except ConnectionRefusedError:
        return "refused"  # host up, control service not (yet) listening
    except OSError:
        return "unreachable"


def internet_devices(run_sys: SysRunner = sys_run) -> set[str]:
    """Devices of the lowest-metric default route — the hard denylist core.

    The machine currently has a bogus second default route via an arm NIC
    (metric 20100); only the lowest metric is the real internet path.
    """
    proc = run_sys(["ip", "-j", "route", "show", "default"])
    try:
        routes = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        routes = []
    if not routes:
        return set()
    best = min(r.get("metric", 0) for r in routes)
    return {r["dev"] for r in routes if r.get("metric", 0) == best and "dev" in r}


def list_nics(
    run: NmcliRunner,
    carrier_of: CarrierReader = read_carrier,
) -> list[NicInfo]:
    """All NM devices with carrier + MAC (nmcli terse; MACs arrive ``\\:``-escaped)."""
    from .nmcli import split_terse

    proc = run("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status")
    nics: list[NicInfo] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = split_terse(line)
        if len(fields) < 4:
            continue
        dev, typ, state, conn = fields[:4]
        mac_proc = run("-g", "GENERAL.HWADDR", "device", "show", dev)
        mac = unescape_value(mac_proc.stdout.strip()) if mac_proc.returncode == 0 else ""
        nics.append(
            NicInfo(
                dev=dev,
                type=typ,
                state=state,
                connection=conn,
                carrier=carrier_of(dev),
                mac=mac,
            )
        )
    return nics


def denylist(nics: list[NicInfo], run_sys: SysRunner = sys_run) -> set[str]:
    """Internet NIC(s) + every non-ethernet device: excluded from EVERY
    mutating path (never activated/deactivated/modified)."""
    deny = internet_devices(run_sys)
    deny |= {n.dev for n in nics if n.type != "ethernet"}
    return deny


def candidate_pool(nics: list[NicInfo], deny: set[str], mapped: set[str]) -> list[NicInfo]:
    """Ethernet, carrier on, not denylisted, not already mapped to another arm."""
    return [
        n
        for n in nics
        if n.type == "ethernet" and n.carrier and n.dev not in deny and n.dev not in mapped
    ]


def ping_via(dev: str, ip: str, run_sys: SysRunner = sys_run, tries: int = 3) -> bool:
    """Interface-bound ping; retried (the first packet is routinely lost to ARP)."""
    for _ in range(max(tries, 3)):
        proc = run_sys(["ping", "-c1", "-W1", "-I", dev, ip])
        if proc.returncode == 0:
            return True
    return False


def neigh_ok(ip: str, run_sys: SysRunner = sys_run) -> bool:
    """Cross-check ARP actually resolved (weak-host-model false positives)."""
    proc = run_sys(["ip", "neigh", "show", ip])
    out = proc.stdout.strip()
    if not out:
        return False
    return "FAILED" not in out and "INCOMPLETE" not in out


def probe_arm(
    dev: str,
    ip: str,
    run_sys: SysRunner = sys_run,
    tcp_connect: TcpConnect = tcp_probe,
) -> ProbeResult:
    """Probe P: ping-via-NIC (>=3 tries) + neigh cross-check + TCP 502.

    "refused" = ping OK but 502 closed: right NIC, control box still booting
    (~1-2 min) — callers must poll 502, NOT re-probe NICs.
    """
    if not ping_via(dev, ip, run_sys):
        return "unreachable"
    if not neigh_ok(ip, run_sys):
        return "unreachable"
    result = tcp_connect(ip, XARM_CONTROL_PORT, 1.0)
    if result == "open":
        return "open"
    if result == "refused":
        return "refused"
    return "unreachable"
