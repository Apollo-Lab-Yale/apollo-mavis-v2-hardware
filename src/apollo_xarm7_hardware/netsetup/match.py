"""Profile enumeration + the verify()/match() algorithms (02-hardware §7.2)."""

from __future__ import annotations

import time

from .nmcli import NmcliRunner, split_terse, unescape_value
from .probe import (
    CarrierReader,
    SysRunner,
    TcpConnect,
    candidate_pool,
    denylist,
    list_nics,
    probe_arm,
    read_carrier,
    sys_run,
    tcp_probe,
)
from .types import ArmNet, MatchResult, NetSetupError, NicMapEntry, ProfileInfo

_DETAIL_FIELDS = (
    "connection.interface-name,connection.autoconnect,connection.autoconnect-priority,"
    "ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.never-default,802-3-ethernet.mac-address"
)


def profile_detail(run: NmcliRunner, uuid: str, name: str = "", typ: str = "",
                   device: str = "") -> ProfileInfo:
    """Read one profile's static config — always addressed by UUID (detail
    views print one value per line)."""
    proc = run("-g", _DETAIL_FIELDS, "connection", "show", "uuid", uuid)
    lines = (proc.stdout.splitlines() + [""] * 8)[:8]
    ifname, autoconnect, priority, method, addresses, gateway, never_default, mac = (
        unescape_value(v.strip()) for v in lines
    )
    try:
        prio = int(priority or 0)
    except ValueError:
        prio = 0
    return ProfileInfo(
        uuid=uuid,
        name=name,
        type=typ,
        device=device,
        ifname=ifname,
        autoconnect=autoconnect or "yes",
        autoconnect_priority=prio,
        method=method,
        addresses=tuple(a.strip() for a in addresses.split(",") if a.strip()),
        gateway=gateway,
        never_default=never_default or "no",
        mac_pin=mac,
    )


def list_profiles(run: NmcliRunner) -> list[ProfileInfo]:
    """All profiles (list view) + per-UUID detail."""
    proc = run("-t", "-f", "NAME,UUID,TYPE,DEVICE", "connection", "show")
    profiles: list[ProfileInfo] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = split_terse(line)
        if len(fields) < 4:
            continue
        name, uuid, typ, device = fields[:4]
        profiles.append(profile_detail(run, uuid, name=name, typ=typ, device=device))
    return profiles


def address_in_subnet(profile: ProfileInfo, arm: ArmNet) -> str | None:
    """First static address of the profile inside the arm subnet whose host
    part is not the arm itself; None otherwise."""
    import ipaddress

    for addr in profile.addresses:
        ip_part = addr.split("/")[0]
        try:
            ip = ipaddress.ip_address(ip_part)
        except ValueError:
            continue
        if ip in arm.subnet and str(ip) != arm.ip:
            return addr
    return None


def find_profile_for_arm(profiles: list[ProfileInfo], arm: ArmNet) -> ProfileInfo | None:
    """Pick (by UUID) an ethernet manual-IPv4 profile with a host address in
    the arm subnet. Two profiles may share a NAME (live bug) — never by name."""
    for p in profiles:
        if p.type not in ("802-3-ethernet", "ethernet"):
            continue
        if p.method != "manual":
            continue
        if address_in_subnet(p, arm) is not None:
            return p
    return None


def create_profile(run: NmcliRunner, arm: ArmNet) -> str:
    """Create the canonical arm profile; returns its UUID. Route-inert by
    construction: never-default, no gateway, ipv6 disabled."""
    con_name = f"apollo-{arm.name}"
    proc = run(
        "connection", "add", "type", "ethernet",
        "con-name", con_name, "ifname", "*",
        "connection.autoconnect", "no",
        "ipv4.method", "manual",
        "ipv4.addresses", f"{arm.resolved_host_ip()}/{arm.prefix}",
        "ipv4.never-default", "yes",
        "ipv4.gateway", "",
        "ipv6.method", "disabled",
    )
    if proc.returncode != 0:
        raise NetSetupError(f"nmcli connection add failed for {arm.name}: {proc.stdout}")
    uuid_proc = run("-g", "connection.uuid", "connection", "show", con_name)
    uuid = uuid_proc.stdout.strip()
    if uuid_proc.returncode != 0 or not uuid:
        raise NetSetupError(f"could not read UUID of created profile {con_name}")
    return uuid


def verify(
    arms: list[ArmNet],
    nic_map: dict[str, NicMapEntry],
    run: NmcliRunner,
    run_sys: SysRunner = sys_run,
    tcp_connect: TcpConnect = tcp_probe,
    carrier_of: CarrierReader = read_carrier,
) -> dict[str, MatchResult]:
    """Fast path (every session start, ~1 s healthy): stored MAC -> current
    ifname, stored UUID active on it, then probe. Any miss -> caller runs match()."""
    nics = list_nics(run, carrier_of)
    by_mac = {n.mac.lower(): n for n in nics if n.mac}
    results: dict[str, MatchResult] = {}
    for arm in arms:
        entry = nic_map.get(arm.name)
        if entry is None:
            results[arm.name] = MatchResult(arm.name, "", "", "", "unreachable",
                                            "no persisted mapping")
            continue
        nic = by_mac.get(entry.mac.lower())
        if nic is None:
            results[arm.name] = MatchResult(arm.name, "", entry.mac, entry.profile_uuid,
                                            "unreachable", f"MAC {entry.mac} not present")
            continue
        proc = run("-g", "GENERAL.CON-UUID", "device", "show", nic.dev)
        active_uuid = unescape_value(proc.stdout.strip())
        if active_uuid != entry.profile_uuid or nic.state != "connected":
            results[arm.name] = MatchResult(arm.name, nic.dev, entry.mac,
                                            entry.profile_uuid, "unreachable",
                                            "stored profile not active on stored NIC")
            continue
        probe = probe_arm(nic.dev, arm.ip, run_sys, tcp_connect)
        detail = "arm booting: poll 502, do not re-probe NICs" if probe == "refused" else ""
        results[arm.name] = MatchResult(arm.name, nic.dev, entry.mac, entry.profile_uuid,
                                        probe, detail)
    return results


def match(
    arms: list[ArmNet],
    run: NmcliRunner,
    run_sys: SysRunner = sys_run,
    tcp_connect: TcpConnect = tcp_probe,
    carrier_of: CarrierReader = read_carrier,
    now: float | None = None,
) -> tuple[dict[str, MatchResult], dict[str, NicMapEntry]]:
    """Full serialized probe (profiles are single-active: probing a profile on
    NIC2 silently detaches it from NIC1). Returns (results, new nic_map)."""
    nics = list_nics(run, carrier_of)
    deny = denylist(nics, run_sys)
    profiles = list_profiles(run)
    results: dict[str, MatchResult] = {}
    mapping: dict[str, NicMapEntry] = {}
    mapped_devs: set[str] = set()
    for arm in arms:  # one arm at a time — serialized by design
        profile = find_profile_for_arm(profiles, arm)
        if profile is None:
            uuid = create_profile(run, arm)
        else:
            uuid = profile.uuid
            if profile.ifname:  # a profile pinned to devA cannot activate on devB
                run("connection", "modify", "uuid", uuid, "connection.interface-name", "")
        result = MatchResult(arm.name, "", "", uuid, "unreachable", "no NIC matched")
        for nic in candidate_pool(nics, deny, mapped_devs):
            if nic.dev in deny:  # hard denylist enforcement on the mutating path
                raise NetSetupError(f"denylisted device {nic.dev} in candidate pool")
            up = run("-w", "15", "connection", "up", "uuid", uuid, "ifname", nic.dev)
            if up.returncode != 0:
                continue
            probe = probe_arm(nic.dev, arm.ip, run_sys, tcp_connect)
            if probe in ("open", "refused"):
                detail = ("arm booting: poll 502, do not re-probe NICs"
                          if probe == "refused" else "")
                result = MatchResult(arm.name, nic.dev, nic.mac, uuid, probe, detail)
                mapped_devs.add(nic.dev)
                mapping[arm.name] = NicMapEntry(
                    mac=nic.mac, ifname=nic.dev, profile_uuid=uuid, arm_ip=arm.ip,
                    ts=now if now is not None else time.time(),
                )
                break
            run("-w", "10", "connection", "down", "uuid", uuid)  # release the NIC
        results[arm.name] = result
    return results, mapping
