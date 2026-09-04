"""NetSetup facade (02-hardware §7): verify / match / repair / reconcile / install / status.

All NetworkManager access goes through the injected ``NmcliRunner`` and the
``probe`` seams — tests inject transcript runners and fake sockets; the real
runners are only wired by defaults used from the CLI / runtime paths.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from . import install as install_mod
from . import match as match_mod
from . import reconcile as reconcile_mod
from . import state as state_mod
from .nmcli import NmcliRunner, TranscriptRunner, nmcli, split_terse
from .probe import (
    CarrierReader,
    SysRunner,
    TcpConnect,
    denylist,
    list_nics,
    probe_arm,
    read_carrier,
    sys_run,
    tcp_probe,
)
from .state import (
    DEFAULT_STATE_PATH,
    SYSTEM_STATE_PATH,
    USER_STATE_PATH,
    load_nic_map,
    read_holdoff,
    resolve_state_path,
    save_nic_map,
    write_holdoff,
)
from .types import (
    ArmNet,
    MatchResult,
    NetSetupError,
    NicInfo,
    NicMapEntry,
    PlanStep,
    ProfileInfo,
    ReconcilePlan,
)

__all__ = [
    "DEFAULT_STATE_PATH",
    "SYSTEM_STATE_PATH",
    "USER_STATE_PATH",
    "ArmNet",
    "MatchResult",
    "NetSetup",
    "NetSetupError",
    "NicInfo",
    "NicMapEntry",
    "NmcliRunner",
    "PlanStep",
    "ProfileInfo",
    "ReconcilePlan",
    "TranscriptRunner",
    "nmcli",
    "resolve_state_path",
    "split_terse",
]

# repair() defaults (02-hardware §7.4)
REPAIR_DEADLINE_S = 60.0  # wait this long for an active-but-unreachable arm (box booting)
REPAIR_POLL_S = 5.0
REPAIR_HOLDOFF_S = 600.0  # min spacing between re-probes of merely unreachable arms


class NetSetup:
    """One workcell's network auto-matcher."""

    def __init__(
        self,
        arms: list[ArmNet],
        state_path: Path | None = None,
        run: NmcliRunner = nmcli,
        run_sys: SysRunner = sys_run,
        tcp_connect: TcpConnect = tcp_probe,
        carrier_of: CarrierReader = read_carrier,
    ) -> None:
        self.arms = list(arms)
        # None -> resolved on every access (explicit > user map > system map > user)
        self._state_path_arg = Path(state_path) if state_path is not None else None
        self._run = run
        self._run_sys = run_sys
        self._tcp_connect = tcp_connect
        self._carrier_of = carrier_of
        self.install_problems: list[str] = []
        self.warnings: list[str] = []  # non-fatal notes from match()/repair()
        self.notes: list[str] = []  # repair() decision trail

    @property
    def state_path(self) -> Path:
        return resolve_state_path(self._state_path_arg)

    def _arm(self, name: str) -> ArmNet:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise NetSetupError(f"unknown arm {name!r}")

    def _health(self, nic_map: dict[str, NicMapEntry]) -> list[str]:
        """install artefacts + profile permissions (both actionable warnings)."""
        problems = install_mod.check(self._run_sys)
        problems += match_mod.permission_problems(self.arms, nic_map, self._run)
        return problems

    def _save(self, mapping: dict[str, NicMapEntry]) -> None:
        """Merge into the resolved map; a user-run match that resolved to the
        (root-owned) system map falls back to the user map, which then shadows."""
        target = self.state_path
        merged = load_nic_map(target)
        merged.update(mapping)
        try:
            save_nic_map(merged, target)
        except PermissionError:
            if self._state_path_arg is not None:
                raise NetSetupError(f"state file {target} is not writable") from None
            fallback = state_mod.user_state_path()
            save_nic_map(merged, fallback)
            self.warnings.append(
                f"{target} not writable; wrote {fallback} instead (the user map now "
                f"takes precedence over the system map)"
            )

    def verify(self) -> dict[str, MatchResult]:
        """Fast path, every session start (~1 s healthy). Also refreshes the
        polkit/dispatcher/permissions health warnings (non-fatal)."""
        nic_map = load_nic_map(self.state_path)
        self.install_problems = self._health(nic_map)
        return match_mod.verify(
            self.arms, nic_map, self._run, self._run_sys, self._tcp_connect,
            self._carrier_of,
        )

    def match(self, reconcile_after: bool = True) -> dict[str, MatchResult]:
        """Full serialized probe; persists state and (by default) reconciles
        so the next boot is deterministic (autoconnect does the work)."""
        results, mapping = match_mod.match(
            self.arms, self._run, self._run_sys, self._tcp_connect, self._carrier_of
        )
        if mapping:
            self._save(mapping)
            if reconcile_after:
                self.reconcile(apply=True)
        return results

    def repair(
        self,
        deadline_s: float = REPAIR_DEADLINE_S,
        poll_s: float = REPAIR_POLL_S,
        holdoff_s: float = REPAIR_HOLDOFF_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ) -> dict[str, MatchResult]:
        """Dispatcher entry point (§7.4): converge without disturbing healthy arms.

        verify() first; arms that pass have their NICs frozen. Misses are
        classified (``match.classify_for_repair``): re-probe only where a probe
        can help; active-but-unreachable arms (box booting? cable swapped?) are
        polled up to ``deadline_s`` and then re-probed at most once per
        ``holdoff_s``. Arms that still find no NIC are put back on their stored
        NIC. Ends with reconcile(apply=True) so pins/permissions/route hygiene
        are restored. A healthy system yields NO nmcli mutation — that is what
        stops the NM event -> dispatcher -> nmcli -> NM event loop.
        """
        start = clock()
        while True:
            nics = list_nics(self._run, self._carrier_of)
            nic_map = load_nic_map(self.state_path)
            results = match_mod.verify(
                self.arms, nic_map, self._run, self._run_sys, self._tcp_connect,
                self._carrier_of, nics=nics,
            )
            deny = denylist(nics, self._run_sys)
            dec = match_mod.classify_for_repair(self.arms, results, nic_map, nics, deny)
            if not dec.pending or clock() - start >= deadline_s:
                break
            sleep(poll_s)
        self.install_problems = self._health(nic_map)
        notes = list(dec.notes)
        if dec.pending:
            names = ",".join(dec.pending)
            last = read_holdoff(self.state_path)
            since_last = float("inf") if last is None else now() - last
            if since_last >= holdoff_s:
                notes.append(
                    f"{names}: unreachable on pinned NIC after {deadline_s:.0f}s "
                    f"-> re-match (cables swapped?)"
                )
                dec.probe.extend(dec.pending)
                write_holdoff(self.state_path, now())
            else:
                notes.append(
                    f"{names}: still unreachable; last re-match {since_last:.0f}s ago "
                    f"(< holdoff {holdoff_s:.0f}s) -> leaving profiles as they are"
                )
        if dec.probe:
            targets = [a for a in self.arms if a.name in dec.probe]
            probed, mapping = match_mod.match(
                targets, self._run, self._run_sys, self._tcp_connect, self._carrier_of,
                frozen=dec.frozen,
            )
            results.update(probed)
            by_mac = {n.mac.lower(): n for n in nics if n.mac}
            for arm in targets:
                if probed[arm.name].ok:
                    continue
                entry = nic_map.get(arm.name)
                nic = by_mac.get(entry.mac.lower()) if entry else None
                if entry and nic is not None and nic.carrier and nic.dev not in dec.frozen:
                    # leave things as we found them: back on the stored NIC
                    self._run("-w", "15", "connection", "up", "uuid", entry.profile_uuid,
                              "ifname", nic.dev)
                    notes.append(f"{arm.name}: no NIC answered; profile restored on {nic.dev}")
            if mapping:
                self._save(mapping)
        self.reconcile(apply=True)
        self.notes = notes
        return results

    def reconcile(
        self, apply: bool = False, active_uuids: frozenset[str] = frozenset()
    ) -> ReconcilePlan:
        """Plan-only by default; ``apply=True`` executes. Profiles carrying
        SDK traffic (``active_uuids``) and denylisted devices are never touched."""
        nic_map = load_nic_map(self.state_path)
        plan = reconcile_mod.build_plan(
            self.arms, nic_map, self._run, self._run_sys, self._carrier_of, active_uuids
        )
        if apply:
            reconcile_mod.apply_plan(plan, self._run)
        return plan

    def status(self) -> list[NicInfo]:
        """Landing-page network health: all NM devices with carrier/MAC."""
        return list_nics(self._run, self._carrier_of)

    def poll_502(self, arm_name: str) -> str:
        """Booting loop for 'ping OK, 502 refused': re-check only the control
        port, never re-probe NICs."""
        arm = self._arm(arm_name)
        return self._tcp_connect(arm.ip, 502, 1.0)

    def probe(self, arm_name: str, dev: str) -> str:
        arm = self._arm(arm_name)
        return probe_arm(dev, arm.ip, self._run_sys, self._tcp_connect)
