"""Gripper backends (02-hardware §4): Classic (pulses, position-only), G2, None.

``command()`` is invoked from the driver's 5 Hz monitor thread (the driver
queues latest-wins); backends own rate limiting and unit conversion. Never
``wait=True`` in-session (latency jitter on the shared 502 socket).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from apollo_mavis_v2_core import GripperCommand, GripperInitError, GripperState

from . import units

# controller fw gate for the port-30000 external-device current monitor
CURRENT_MONITOR_FW = (2, 7, 100)
# gripper (not controller) fw gate for the grasp-status bit
GRASP_STATUS_FW = (3, 4, 3)


def parse_fw(version: str) -> tuple[int, int, int]:
    """Parse 'x.y.z' (tolerating suffixes like '2.7.100-beta') into a tuple."""
    parts: list[int] = []
    for chunk in str(version).split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


class GripperBackend(ABC):
    """One gripper on one arm; driven from the monitor thread."""

    force_capable: ClassVar[bool] = False
    kind: ClassVar[str] = "none"

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._api: Any = None
        self._faults: list[int] = []

    @abstractmethod
    def init(self, api: Any, fw: tuple[int, int, int]) -> None:
        """Enable/configure; raises GripperInitError on failure."""

    @abstractmethod
    def command(self, cmd: GripperCommand) -> None:
        """Send one (already latest-wins-queued) command; non-blocking."""

    @abstractmethod
    def poll(self) -> GripperState:
        """5 Hz state refresh on the monitor thread."""

    def close(self) -> None:  # best-effort teardown; default no-op
        return None

    def drain_faults(self) -> list[int]:
        """Gripper error codes that survived one clean+re-enable cycle."""
        out, self._faults = self._faults, []
        return out


class NoGripper(GripperBackend):
    force_capable: ClassVar[bool] = False
    kind: ClassVar[str] = "none"

    def init(self, api: Any, fw: tuple[int, int, int]) -> None:
        self._api = api

    def command(self, cmd: GripperCommand) -> None:
        return None

    def poll(self) -> GripperState:
        return GripperState(open_frac=1.0)


class ClassicGripper(GripperBackend):
    """RS-485 tool-modbus gripper: position-only, pulses 0-850, no force.

    ``cmd.force`` is silently ignored (core contract). Rate-limited to 10 Hz,
    latest-wins, skip when |delta pulse| < 5.
    """

    force_capable: ClassVar[bool] = False
    kind: ClassVar[str] = "xarm"
    MIN_SEND_PERIOD_S = 0.1  # 10 Hz
    MIN_DELTA_PULSE = 5

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        current_source: Callable[[], float | None] | None = None,
    ) -> None:
        # current_source: seam for the optional _Port30000Reader (fw >= 2.7.100
        # controller streams gripper current on port 30000); wired in phase-09.
        super().__init__(clock)
        self._current_source = current_source
        self._gripper_fw: tuple[int, int, int] = (0, 0, 0)
        self._monitor_enabled = False
        self._last_sent_pulse: int | None = None
        self._last_send_t = -1e9
        self._last_state = GripperState(open_frac=1.0)
        self._cleaned_once = False

    def init(self, api: Any, fw: tuple[int, int, int]) -> None:
        self._api = api
        code = api.set_gripper_enable(True)
        if code != 0:
            raise GripperInitError("gripper", f"set_gripper_enable returned {code}")
        code = api.set_gripper_mode(0)  # 0 = position mode (the only mode)
        if code != 0:
            raise GripperInitError("gripper", f"set_gripper_mode returned {code}")
        api.set_gripper_speed(3000)  # r/min, valid ~1000-5000
        code, ver = api.get_gripper_version()
        if code == 0 and ver:
            self._gripper_fw = parse_fw(ver)
        if fw >= CURRENT_MONITOR_FW:
            # controller streams gripper pos/speed/current in the 30000 report
            code = api.set_external_device_monitor_params(dev_type=1, frequency=10)
            self._monitor_enabled = code == 0

    def command(self, cmd: GripperCommand) -> None:
        pulse = units.frac_to_pulse(cmd.open_frac)  # cmd.force ignored: position-only
        now = self._clock()
        if now - self._last_send_t < self.MIN_SEND_PERIOD_S:
            return
        if (
            self._last_sent_pulse is not None
            and abs(pulse - self._last_sent_pulse) < self.MIN_DELTA_PULSE
        ):
            return
        code = self._api.set_gripper_position(pulse, wait=False)  # NEVER wait=True
        if code == 0:
            self._last_sent_pulse = pulse
            self._last_send_t = now

    def poll(self) -> GripperState:
        code, pulse = self._api.get_gripper_position()
        open_frac = self._last_state.open_frac
        if code == 0 and pulse is not None:
            open_frac = units.pulse_to_frac(pulse)
        moving: bool | None = None
        grasped: bool | None = None
        if self._gripper_fw >= GRASP_STATUS_FW:
            scode, status = self._api.get_gripper_status()
            if scode == 0 and status is not None:
                moving = (status & 0x03) == 1
                grasped = (status & 0x03) == 2
        current = self._current_source() if self._current_source else None
        self._poll_errors()
        self._last_state = GripperState(
            open_frac=open_frac, moving=moving, grasped=grasped, current=current
        )
        return self._last_state

    def _poll_errors(self) -> None:
        code, err = self._api.get_gripper_err_code()
        if code != 0 or not err:
            self._cleaned_once = False
            return
        if not self._cleaned_once:  # one clean + re-enable attempt
            self._api.clean_gripper_error()
            self._api.set_gripper_enable(True)
            self._cleaned_once = True
        else:  # error survived the retry -> fault (arm unaffected)
            self._faults.append(int(err))

    def close(self) -> None:
        return None


class G2Gripper(GripperBackend):
    """xArm Gripper G2: mm units + real force control (the only force-capable one).

    ``GripperCommand.force`` (normalized [0, 1]) is honored here and only here
    -> SDK percent 1-100. Position 0-84 mm, speed 15-225 mm/s.
    """

    force_capable: ClassVar[bool] = True
    kind: ClassVar[str] = "xarm_g2"
    MIN_SEND_PERIOD_S = 0.1  # 10 Hz
    DEFAULT_SPEED_MM_S = 150
    DEFAULT_FORCE_PCT = 50

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock)
        self._last_send_t = -1e9
        self._last_state = GripperState(open_frac=1.0)

    def init(self, api: Any, fw: tuple[int, int, int]) -> None:
        self._api = api

    def command(self, cmd: GripperCommand) -> None:
        now = self._clock()
        if now - self._last_send_t < self.MIN_SEND_PERIOD_S:
            return
        mm = units.frac_to_g2_mm(cmd.open_frac)
        force = (
            self.DEFAULT_FORCE_PCT
            if cmd.force is None
            else min(max(int(round(cmd.force * 100)), 1), 100)
        )
        speed = (
            self.DEFAULT_SPEED_MM_S
            if cmd.speed is None
            else int(round(15 + cmd.speed * (225 - 15)))
        )
        code = self._api.set_gripper_g2_position(mm, speed=speed, force=force, wait=False)
        if code == 0:
            self._last_send_t = now

    def poll(self) -> GripperState:
        code, mm = self._api.get_gripper_g2_position()
        open_frac = self._last_state.open_frac
        if code == 0 and mm is not None:
            open_frac = units.g2_mm_to_frac(mm)
        self._last_state = GripperState(open_frac=open_frac)
        return self._last_state


def make_gripper(
    kind: str,
    clock: Callable[[], float] = time.monotonic,
    current_source: Callable[[], float | None] | None = None,
) -> GripperBackend:
    """Dispatch on XArmDriverConfig.gripper."""
    if kind == "xarm":
        return ClassicGripper(clock, current_source)
    if kind == "xarm_g2":
        return G2Gripper(clock)
    if kind == "none":
        return NoGripper(clock)
    raise ValueError(f"unknown gripper kind {kind!r}")
