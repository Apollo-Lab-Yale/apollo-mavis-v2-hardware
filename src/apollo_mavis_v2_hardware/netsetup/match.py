"""Profile enumeration + the verify()/match() algorithms (02-hardware §7.2) and
the repair classification used by the NM dispatcher path (§7.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
from .types import ArmNet, MatchResult, NetSetupError, NicInfo, NicMapEntry, ProfileInfo

_DETAIL_FIELDS = (
    "connection.interface-name,connection.autoconnect,connection.autoconnect-priority,"
    "ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.never-default,802-3-ethernet.mac-address,"
    "connection.permissions"
)
_N_DETAIL = 9


def profile_detail(run: NmcliRunner, uuid: str, name: str = "", typ: str = "",
                   device: str = "") -> ProfileInfo:
    """Read one profile's static config — always addressed by UUID (detail
    views print one value per line)."""
    proc = run("-g", _DETAIL_FIELDS, "connection", "show", "uuid", uuid)
    lines = (proc.stdout.splitlines() + [""] * _N_DETAIL)[:_N_DETAIL]
    (ifname, autoconnect, priority, method, addresses, gateway, never_default, mac,
     permissions) = (unescape_value(v.strip()) for v in lines)
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
        permissions=permissions,
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


def permission_problems(
    arms: list[ArmNet], nic_map: dict[str, NicMapEntry], run: NmcliRunner
) -> list[str]:
    """Warn when a mapped arm profile is user-restricted (``connection.permissions``
    set): NM keyfiles in /etc/NetworkManager/system-connections are shared by all
    accounts *unless* this property is set — then other users and the root
    dispatcher cannot activate the profile. reconcile clears it."""
    problems: list[str] = []
    for arm in arms:
        entry = nic_map.get(arm.name)
        if entry is None:
            continue
        profile = profile_detail(run, entry.profile_uuid)
        if profile.user_restricted:
            problems.append(
                f"{arm.name}: profile {entry.profile_uuid} is restricted to "
                f"{profile.permissions!r} (connection.permissions) — other accounts and "
                f"the NM dispatcher cannot use it; run: netsetup reconcile --apply "
                f"(or: nmcli connection modify uuid {entry.profile_uuid} "
                f"connection.permissions '')"
            )
    return problems


def verify(
    arms: list[ArmNet],
    nic_map: dict[str, NicMapEntry],
    run: NmcliRunner,
    run_sys: SysRunner = sys_run,
    tcp_connect: TcpConnect = tcp_probe,
    carrier_of: CarrierReader = read_carrier,
    nics: list[NicInfo] | None = None,
) -> dict[str, MatchResult]:
    """Fast path (every session start, ~1 s healthy): stored MAC -> current
    ifname, stored UUID active on it, then probe. Any miss -> caller runs match()."""
    if nics is None:
        nics = list_nics(run, carrier_of)
    by_mac = {n.mac.lower(): n for n in nics if n.mac}
    results: dict[str, MatchResult] = {}
    for arm in arms:
        entry = nic_map.get(arm.name)
        if entry is None:
            results[arm.name] = MatchResult(arm.name, "", "", "", "unreachable",
                                            "no persisted mapping", reason="no-mapping")
            continue
        nic = by_mac.get(entry.mac.lower())
        if nic is None:
            results[arm.name] = MatchResult(arm.name, "", entry.mac, entry.profile_uuid,
                                            "unreachable", f"MAC {entry.mac} not present",
                                            reason="nic-missing")
            continue
        proc = run("-g", "GENERAL.CON-UUID", "device", "show", nic.dev)
        active_uuid = unescape_value(proc.stdout.strip())
        if active_uuid != entry.profile_uuid or nic.state != "connected":
            results[arm.name] = MatchResult(arm.name, nic.dev, entry.mac,
                                            entry.profile_uuid, "unreachable",
                                            "stored profile not active on stored NIC",
                                            reason="profile-inactive")
            continue
        probe = probe_arm(nic.dev, arm.ip, run_sys, tcp_connect)
        detail = "arm booting: poll 502, do not re-probe NICs" if probe == "refused" else ""
        reason = "unreachable" if probe == "unreachable" else ""
        results[arm.name] = MatchResult(arm.name, nic.dev, entry.mac, entry.profile_uuid,
                                        probe, detail, reason=reason)
    return results


def match(
    arms: list[ArmNet],
    run: NmcliRunner,
    run_sys: SysRunner = sys_run,
    tcp_connect: TcpConnect = tcp_probe,
    carrier_of: CarrierReader = read_carrier,
    now: float | None = None,
    frozen: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, MatchResult], dict[str, NicMapEntry]]:
    """Full serialized probe (profiles are single-active: probing a profile on
    NIC2 silently detaches it from NIC1). Returns (results, new nic_map).

    ``frozen`` devices (NICs of arms that already verify OK, §7.4) are never
    candidates. Both pins — ``connection.interface-name`` AND
    ``802-3-ethernet.mac-address`` — are cleared before probing: a MAC-pinned
    profile cannot be activated on another NIC, so a cable swap could never be
    re-matched otherwise. reconcile() re-pins afterwards.
    """
    nics = list_nics(run, carrier_of)
    deny = denylist(nics, run_sys)
    profiles = list_profiles(run)
    results: dict[str, MatchResult] = {}
    mapping: dict[str, NicMapEntry] = {}
    mapped_devs: set[str] = set(frozen)
    for arm in arms:  # one arm at a time — serialized by design
        profile = find_profile_for_arm(profiles, arm)
        if profile is None:
            uuid = create_profile(run, arm)
        else:
            uuid = profile.uuid
        pool = candidate_pool(nics, deny, mapped_devs)
        if pool and profile is not None and (profile.ifname or profile.mac_pin):
            # pinned to devA -> cannot activate on devB; only unpin when there is
            # actually something to try (an unplugged arm keeps its pins)
            run("connection", "modify", "uuid", uuid,
                "connection.interface-name", "", "802-3-ethernet.mac-address", "")
        result = MatchResult(arm.name, "", "", uuid, "unreachable", "no NIC matched",
                             reason="no-candidate")
        for nic in pool:
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


@dataclass
class RepairDecision:
    """What repair() does with each verify() miss (02-hardware §7.4)."""

    probe: list[str] = field(default_factory=list)  # arms to re-match now
    pending: list[str] = field(default_factory=list)  # active-but-unreachable: wait first
    frozen: set[str] = field(default_factory=set)  # NICs of healthy arms: never candidates
    notes: list[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        return not self.probe and not self.pending


def classify_for_repair(
    arms: list[ArmNet],
    results: dict[str, MatchResult],
    nic_map: dict[str, NicMapEntry],
    nics: list[NicInfo],
    deny: set[str],
) -> RepairDecision:
    """Turn verify() misses into a conservative plan that never disturbs a
    healthy arm and produces no NM change when nothing can be gained (the
    dispatcher's convergence guarantee: quiet -> no events -> no re-entry).

    * ok                -> freeze its NIC
    * no-mapping / nic-missing -> probe
    * profile-inactive, stored NIC without carrier -> probe only if a *free* NIC
      (ethernet, carrier, not denylisted, not frozen, no arm profile mapped to it)
      exists (cable moved elsewhere); otherwise skip — NM autoconnect re-activates
      the pinned profile when the cable comes back
    * profile-inactive, stored NIC with carrier -> probe (something else grabbed it
      or autoconnect is blocked; match re-activates it)
    * unreachable (active, probe failed) -> pending: box may still be booting
    """
    dec = RepairDecision()
    by_mac = {n.mac.lower(): n for n in nics if n.mac}
    for arm in arms:
        res = results.get(arm.name)
        if res is not None and res.ok and res.ifname:
            dec.frozen.add(res.ifname)
    mapped_devs = {by_mac[e.mac.lower()].dev for e in nic_map.values()
                   if e.mac.lower() in by_mac}
    free_nics = [
        n.dev for n in candidate_pool(nics, deny, dec.frozen | mapped_devs)
    ]
    for arm in arms:
        res = results.get(arm.name)
        if res is None or res.ok:
            continue
        if res.reason in ("no-mapping", "nic-missing"):
            dec.probe.append(arm.name)
            dec.notes.append(f"{arm.name}: {res.detail} -> re-match")
        elif res.reason == "profile-inactive":
            entry = nic_map.get(arm.name)
            nic = by_mac.get(entry.mac.lower()) if entry else None
            if nic is not None and not nic.carrier:
                if free_nics:
                    dec.probe.append(arm.name)
                    dec.notes.append(
                        f"{arm.name}: NIC {nic.dev} has no carrier but free NIC(s) "
                        f"{','.join(free_nics)} have -> re-match (cable moved?)"
                    )
                else:
                    dec.notes.append(
                        f"{arm.name}: NIC {nic.dev} has no carrier (unplugged / box off) "
                        f"-> nothing to match; autoconnect handles re-plug"
                    )
            else:
                dec.probe.append(arm.name)
                dec.notes.append(f"{arm.name}: {res.detail} -> re-activate/re-match")
        elif res.reason == "unreachable":
            dec.pending.append(arm.name)
        else:
            dec.probe.append(arm.name)
            dec.notes.append(f"{arm.name}: {res.detail or res.probe} -> re-match")
    return dec
