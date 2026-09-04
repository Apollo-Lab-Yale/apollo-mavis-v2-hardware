"""netsetup data types (02-hardware §7.1)."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Literal

from apollo_mavis_v2_core import ApolloError

ProbeResult = Literal["open", "refused", "unreachable"]


class NetSetupError(ApolloError):
    """netsetup verify/match/reconcile failure."""


@dataclass(frozen=True)
class ArmNet:
    """One arm's network identity from config."""

    name: str  # arm id
    ip: str  # controller IP, e.g. 192.168.1.235
    prefix: int = 24
    host_ip: str | None = None  # host address on that subnet; None -> .12

    @property
    def subnet(self) -> ipaddress.IPv4Network:
        return ipaddress.ip_network(f"{self.ip}/{self.prefix}", strict=False)

    def resolved_host_ip(self) -> str:
        if self.host_ip:
            return self.host_ip
        # lab convention: host at .12 in each arm subnet
        return str(self.subnet.network_address + 12)


@dataclass(frozen=True)
class NicInfo:
    """One NM device row (+ carrier + MAC)."""

    dev: str
    type: str  # nmcli TYPE (ethernet, wifi, tun, loopback, ...)
    state: str  # connected | disconnected | unavailable | unmanaged
    connection: str  # active profile name or ""
    carrier: bool = False
    mac: str = ""


@dataclass(frozen=True)
class ProfileInfo:
    """One NM connection profile (always addressed by UUID)."""

    uuid: str
    name: str
    type: str = "802-3-ethernet"
    device: str = ""  # active device, "" if inactive
    ifname: str = ""  # connection.interface-name
    autoconnect: str = "yes"
    autoconnect_priority: int = 0
    method: str = ""  # ipv4.method
    addresses: tuple[str, ...] = ()
    gateway: str = ""
    never_default: str = "no"
    mac_pin: str = ""  # 802-3-ethernet.mac-address

    @property
    def active(self) -> bool:
        return bool(self.device)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of verify()/match() for one arm."""

    arm: str
    ifname: str
    mac: str
    profile_uuid: str
    probe: ProbeResult
    detail: str = ""

    @property
    def ok(self) -> bool:
        # "refused" = right NIC, box still booting: poll 502, do NOT re-probe
        return self.probe in ("open", "refused")


@dataclass(frozen=True)
class NicMapEntry:
    """Persisted arm -> NIC binding (~/.config/apollo-mavis-v2/nic_map.json).

    Keyed by MAC (stable); ifname is resolved at runtime (names drift)."""

    mac: str
    ifname: str
    profile_uuid: str
    arm_ip: str
    ts: float = 0.0


@dataclass(frozen=True)
class PlanStep:
    """One reconcile action: an nmcli argv + the reason it is needed."""

    args: tuple[str, ...]  # nmcli argv (without the leading "nmcli")
    reason: str
    mutating: bool = True


@dataclass
class ReconcilePlan:
    """Ordered nmcli commands reconcile() would run (apply=False = plan only)."""

    steps: list[PlanStep] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # active/denylisted, with reason
    applied: bool = False

    def add(self, args: list[str], reason: str) -> None:
        self.steps.append(PlanStep(tuple(args), reason))

    def render(self) -> str:
        def fmt(arg: str) -> str:
            return arg if arg and " " not in arg else f'"{arg}"'

        lines = [
            f"nmcli {' '.join(fmt(a) for a in s.args)}    # {s.reason}" for s in self.steps
        ]
        lines += [f"(skipped) {s}" for s in self.skipped]
        return "\n".join(lines) if lines else "(nothing to do)"
