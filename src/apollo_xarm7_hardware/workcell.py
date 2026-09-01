"""HardwareWorkcell: arms + cameras assembly and bring-up (02-hardware §9)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from apollo_xarm7_core import (
    ArmState,
    BringupError,
    WorkcellBringupError,
    WorkcellConfig,
)
from apollo_xarm7_core.interfaces import ArmInterface, CameraInterface, WorkcellInterface
from pydantic import BaseModel, Field

from .cameras import make_camera
from .config import XArmDriverConfig
from .driver import XArmDriver
from .netsetup import NetSetup

BOOT_POLL_PERIOD_S = 2.0  # 502 poll while the control box boots (~1-2 min)
FRESH_REPORT_TIMEOUT_S = 5.0  # one fresh 30003 snapshot required before "connected"


class ArmBringupStatus(BaseModel):
    """Streamed to the UI landing page via status_cb on every transition."""

    arm_id: str
    network: Literal["pending", "probing", "ok", "booting", "failed"] = "pending"
    connected: bool = False
    fw_version: str | None = None
    sn: str | None = None
    rail: Literal["unknown", "none", "detected", "homing", "ready", "error"] = "unknown"
    gripper: Literal["unknown", "xarm", "xarm_g2", "none", "error"] = "unknown"
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


def _driver_cfg(arm: Any) -> XArmDriverConfig:
    return XArmDriverConfig(
        arm_id=arm.id,
        ip=arm.ip,
        expect_rail=arm.expect_rail,
        gripper=arm.gripper,
        tcp_load_kg=arm.tcp_load_kg,
        tcp_load_cog_mm=arm.tcp_load_cog_mm,
    )


class HardwareWorkcell(WorkcellInterface):
    """Real workcell: XArmDrivers + V4L2/RealSense cameras.

    ``netsetup=None`` skips the network stage (tests / pre-wired machines);
    the runtime passes a configured NetSetup so bring-up verifies NIC
    matching first. Cameras start independently of arm bring-up (the landing
    page needs live previews before any session).
    """

    def __init__(
        self,
        cfg: WorkcellConfig,
        driver_factory: Callable[[XArmDriverConfig], ArmInterface] = XArmDriver,
        netsetup: NetSetup | None = None,
        camera_factory: Callable[..., CameraInterface] = make_camera,
    ) -> None:
        if cfg.kind != "hardware":
            raise ValueError(f"HardwareWorkcell got kind={cfg.kind!r}")
        self.cfg = cfg
        self._netsetup = netsetup
        self.arms: dict[str, ArmInterface] = {
            arm.id: driver_factory(_driver_cfg(arm)) for arm in cfg.arms
        }
        self.cameras: dict[str, CameraInterface] = {
            cam.id: camera_factory(cam) for cam in cfg.cameras
        }
        self._statuses: dict[str, ArmBringupStatus] = {}
        self._bringup_errors: dict[str, BringupError | None] = {}
        self._cameras_started = False
        self._stopped = False

    @property
    def kind(self) -> Literal["hardware", "sim"]:
        return "hardware"

    # -- core ABC ---------------------------------------------------------------
    def start(self) -> None:
        """core ABC — delegates to bring_up(); collects per-arm failures into
        WorkcellBringupError without aborting siblings."""
        self.bring_up()
        if any(err is not None for err in self._bringup_errors.values()):
            raise WorkcellBringupError(dict(self._bringup_errors))

    def stop(self) -> None:
        """core ABC — delegates to shutdown(); idempotent."""
        self.shutdown()

    def states(self) -> dict[str, ArmState]:
        return {arm_id: arm.get_state() for arm_id, arm in self.arms.items()}

    # -- cameras ------------------------------------------------------------------
    def start_cameras(self) -> None:
        """Pre-session landing previews; independent of arm bring-up; idempotent.
        A dead camera never blocks the workcell."""
        if self._cameras_started:
            return
        for cam in self.cameras.values():
            try:
                cam.start()
            except Exception:  # noqa: BLE001 — UI greys the tile instead
                continue
        self._cameras_started = True

    def stop_cameras(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.stop()
            except Exception:  # noqa: BLE001
                continue
        self._cameras_started = False

    # -- bring-up -------------------------------------------------------------------
    def bring_up(
        self,
        status_cb: Callable[[ArmBringupStatus], None] | None = None,
        timeout_s: float = 180.0,
    ) -> dict[str, ArmBringupStatus]:
        """Per-arm parallel bring-up; one arm failing never aborts the others.

        Sequence per §9: (1) netsetup verify -> match on miss; probe "refused"
        means the box is booting — poll TCP 502 every 2 s, never re-probe NICs;
        (2) connect each arm in its own thread; (3) rail detect+home inside
        connect; (4) require one fresh 30003 snapshot before "connected";
        (5) start cameras (independent, idempotent).
        """
        deadline = time.monotonic() + timeout_s
        self._stopped = False
        self._statuses = {
            arm_id: ArmBringupStatus(arm_id=arm_id) for arm_id in self.arms
        }
        self._bringup_errors = dict.fromkeys(self.arms)

        def push(status: ArmBringupStatus) -> None:
            if status_cb is not None:
                status_cb(status.model_copy(deep=True))

        net_results = self._network_stage(push)
        threads: list[threading.Thread] = []
        for arm_id in self.arms:
            t = threading.Thread(
                target=self._bring_up_arm,
                args=(arm_id, net_results.get(arm_id), push, deadline),
                name=f"hw.{arm_id}.bringup",
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()) + 5.0)
        self.start_cameras()
        return {k: v.model_copy(deep=True) for k, v in self._statuses.items()}

    def _network_stage(
        self, push: Callable[[ArmBringupStatus], None]
    ) -> dict[str, Any]:
        """verify() then match() on miss; returns per-arm MatchResults."""
        results: dict[str, Any] = {}
        if self._netsetup is None:
            for status in self._statuses.values():
                status.network = "ok"
                status.warnings.append("netsetup skipped (no NetSetup wired)")
                push(status)
            return results
        for status in self._statuses.values():
            status.network = "probing"
            push(status)
        try:
            results = self._netsetup.verify()
            if any(not r.ok for r in results.values()):
                results = self._netsetup.match()  # serialized full probe
        except Exception as exc:  # noqa: BLE001 — network failure is per-arm data
            for status in self._statuses.values():
                status.network = "failed"
                status.error = f"netsetup: {exc}"
                push(status)
            return {}
        for warning in self._netsetup.install_problems:
            for status in self._statuses.values():
                status.warnings.append(warning)
        for arm_id, res in results.items():
            if arm_id not in self._statuses:
                continue
            status = self._statuses[arm_id]
            if res.probe == "open":
                status.network = "ok"
            elif res.probe == "refused":
                status.network = "booting"  # right NIC, box booting: poll 502
            else:
                status.network = "failed"
                status.error = f"network: {res.detail or res.probe}"
            push(status)
        return results

    def _bring_up_arm(
        self,
        arm_id: str,
        net_result: Any,
        push: Callable[[ArmBringupStatus], None],
        deadline: float,
    ) -> None:
        status = self._statuses[arm_id]
        driver = self.arms[arm_id]
        # booting: poll only TCP 502 (never re-probe NICs) until open/deadline
        if status.network == "booting" and self._netsetup is not None:
            while time.monotonic() < deadline:
                if self._netsetup.poll_502(arm_id) == "open":
                    status.network = "ok"
                    push(status)
                    break
                time.sleep(BOOT_POLL_PERIOD_S)
            else:
                status.network = "failed"
                status.error = "network: 502 never opened (arm stuck booting?)"
                push(status)
        if status.network == "failed":
            self._bringup_errors[arm_id] = BringupError("network", status.error or "")
            return
        try:
            driver.connect()
        except BringupError as exc:
            status.error = str(exc)
            if exc.step == "gripper":
                status.gripper = "error"
            if exc.step == "rail":
                status.rail = "error"
            self._bringup_errors[arm_id] = exc
            push(status)
            return
        status.sn = getattr(driver, "sn", None)
        status.fw_version = getattr(driver, "fw_version", None)
        status.gripper = getattr(driver, "gripper_kind", "unknown")
        if driver.has_rail:
            rail_phase = getattr(driver, "rail_phase", None)
            status.rail = "ready" if rail_phase in (None, "READY") else "error"
        else:
            status.rail = "none"
        push(status)
        # require one fresh 30003 snapshot before declaring connected
        fresh_deadline = min(deadline, time.monotonic() + FRESH_REPORT_TIMEOUT_S)
        while time.monotonic() < fresh_deadline:
            if not driver.get_state().stale:
                status.connected = True
                break
            time.sleep(0.05)
        if not status.connected:
            status.error = "report stream never became fresh (30003 silent)"
            self._bringup_errors[arm_id] = BringupError("report", status.error)
        push(status)

    def shutdown(self) -> None:
        """Reverse order; never raises; idempotent."""
        if self._stopped:
            return
        self.stop_cameras()
        for arm in reversed(list(self.arms.values())):
            try:
                arm.disconnect()
            except Exception:  # noqa: BLE001
                continue
        self._stopped = True