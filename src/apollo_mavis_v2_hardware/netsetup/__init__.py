"""NetSetup facade (02-hardware §7): verify / match / reconcile / install / status.

All NetworkManager access goes through the injected ``NmcliRunner`` and the
``probe`` seams — tests inject transcript runners and fake sockets; the real
runners are only wired by defaults used from the CLI / runtime paths.
"""

from __future__ import annotations

from pathlib import Path

from . import install as install_mod
from . import match as match_mod
from . import reconcile as reconcile_mod
from .nmcli import NmcliRunner, TranscriptRunner, nmcli, split_terse
from .probe import (
    CarrierReader,
    SysRunner,
    TcpConnect,
    list_nics,
    probe_arm,
    read_carrier,
    sys_run,
    tcp_probe,
)
from .state import DEFAULT_STATE_PATH, load_nic_map, save_nic_map
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
    "split_terse",
]


class NetSetup:
    """One workcell's network auto-matcher."""

    def __init__(
        self,
        arms: list[ArmNet],
        state_path: Path = DEFAULT_STATE_PATH,
        run: NmcliRunner = nmcli,
        run_sys: SysRunner = sys_run,
        tcp_connect: TcpConnect = tcp_probe,
        carrier_of: CarrierReader = read_carrier,
    ) -> None:
        self.arms = list(arms)
        self.state_path = Path(state_path)
        self._run = run
        self._run_sys = run_sys
        self._tcp_connect = tcp_connect
        self._carrier_of = carrier_of
        self.install_problems: list[str] = []

    def _arm(self, name: str) -> ArmNet:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise NetSetupError(f"unknown arm {name!r}")

    def verify(self) -> dict[str, MatchResult]:
        """Fast path, every session start (~1 s healthy). Also refreshes the
        polkit/grant health warnings (non-fatal)."""
        self.install_problems = install_mod.check(self._run_sys)
        nic_map = load_nic_map(self.state_path)
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
            merged = load_nic_map(self.state_path)
            merged.update(mapping)
            save_nic_map(merged, self.state_path)
            if reconcile_after:
                self.reconcile(apply=True)
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
