"""One-time cleanup: ReconcilePlan builder + executor (02-hardware §7.3).

Plan first, execute only on ``apply=True``. NEVER touches a profile currently
carrying SDK traffic (``active_uuids``) nor any denylisted device. Fixes the
known pollution on the target machine: duplicate ``xarm7_1`` profiles,
``ipv4.gateway`` entries creating a bogus default route, missing MAC pins,
user-restricted ``connection.permissions``. Dedupe only happens inside the
subnet of an arm that HAS a persisted mapping (otherwise nothing is known to
be the keeper).
"""

from __future__ import annotations

from .match import address_in_subnet, list_profiles
from .nmcli import NmcliRunner, is_mutating
from .probe import CarrierReader, SysRunner, denylist, list_nics, read_carrier, sys_run
from .types import ArmNet, NetSetupError, NicMapEntry, ProfileInfo, ReconcilePlan

ARM_PROFILE_PRIORITY = 50


def _pin_args(profile: ProfileInfo, entry: NicMapEntry) -> tuple[list[str], list[str]]:
    """(argv fragment, reasons) for one mapped arm profile; empty = compliant."""
    args: list[str] = []
    reasons: list[str] = []

    def want(key: str, value: str, reason: str) -> None:
        args.extend([key, value])
        reasons.append(reason)

    if profile.ifname != entry.ifname:
        want("connection.interface-name", entry.ifname, "pin to matched NIC by name")
    if profile.mac_pin.lower() != entry.mac.lower():
        want("802-3-ethernet.mac-address", entry.mac.upper(),
             "pin by MAC (survives ifname renames)")
    if profile.autoconnect != "yes":
        want("connection.autoconnect", "yes", "deterministic boot via autoconnect")
    if profile.autoconnect_priority != ARM_PROFILE_PRIORITY:
        want("connection.autoconnect-priority", str(ARM_PROFILE_PRIORITY),
             "beat stale same-NIC profiles at boot")
    if profile.never_default != "yes":
        want("ipv4.never-default", "yes", "route hygiene: arm subnet is a stub")
    if profile.gateway:
        want("ipv4.gateway", "", "strip gateway pollution (bogus default route)")
    if profile.method != "manual":
        want("ipv4.method", "manual", "DHCP on an arm NIC hangs ~45 s per cycle")
    if profile.user_restricted:
        want("connection.permissions", "",
             "system-wide profile: every account + the root dispatcher must use it")
    return args, reasons


def build_plan(
    arms: list[ArmNet],
    nic_map: dict[str, NicMapEntry],
    run: NmcliRunner,
    run_sys: SysRunner = sys_run,
    carrier_of: CarrierReader = read_carrier,
    active_uuids: frozenset[str] = frozenset(),
) -> ReconcilePlan:
    plan = ReconcilePlan()
    nics = list_nics(run, carrier_of)
    deny = denylist(nics, run_sys)
    profiles = list_profiles(run)
    by_uuid = {p.uuid: p for p in profiles}
    matched_uuids = {e.profile_uuid for e in nic_map.values()}
    # 1. pin + route-hygiene every mapped arm profile
    for arm in arms:
        entry = nic_map.get(arm.name)
        if entry is None:
            plan.skipped.append(f"{arm.name}: no persisted mapping (run match first)")
            continue
        profile = by_uuid.get(entry.profile_uuid)
        if profile is None:
            plan.skipped.append(f"{arm.name}: mapped profile {entry.profile_uuid} missing")
            continue
        if entry.profile_uuid in active_uuids:
            plan.skipped.append(
                f"{arm.name}: profile {entry.profile_uuid} carries SDK traffic — untouched"
            )
            continue
        if entry.ifname in deny:
            plan.skipped.append(f"{arm.name}: mapped NIC {entry.ifname} is denylisted")
            continue
        args, reasons = _pin_args(profile, entry)
        if args:
            plan.add(
                ["connection", "modify", "uuid", entry.profile_uuid, *args],
                f"{arm.name}: " + "; ".join(reasons),
            )
    # 2. disable stale/duplicate profiles inside a MAPPED arm's subnet. Without a
    #    mapping there is no "matched" profile to keep, so nothing is a duplicate:
    #    an --apply before match must never knock out the working profiles.
    mapped_arms = [a for a in arms if a.name in nic_map]
    for profile in profiles:
        if profile.uuid in matched_uuids:
            continue
        if profile.type not in ("802-3-ethernet", "ethernet"):
            continue
        hit = next(
            (a for a in mapped_arms if address_in_subnet(profile, a) is not None), None
        )
        if hit is None:
            continue
        if profile.uuid in active_uuids:
            plan.skipped.append(
                f"duplicate {profile.name} ({profile.uuid}) is active with SDK traffic"
            )
            continue
        if profile.autoconnect != "no":
            plan.add(
                ["connection", "modify", "uuid", profile.uuid,
                 "connection.autoconnect", "no"],
                f"disable stale/duplicate profile {profile.name!r} in "
                f"{hit.name}'s subnet (dedupe)",
            )
    return plan


def apply_plan(plan: ReconcilePlan, run: NmcliRunner) -> ReconcilePlan:
    """Execute every step; refuses non-mutating nonsense and stops on failure."""
    for step in plan.steps:
        if not is_mutating(step.args):
            continue
        proc = run(*step.args)
        if proc.returncode != 0:
            raise NetSetupError(
                f"reconcile step failed (rc={proc.returncode}): nmcli {' '.join(step.args)}"
            )
    plan.applied = True
    return plan
