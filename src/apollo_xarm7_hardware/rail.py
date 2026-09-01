"""Linear track / rail controller (02-hardware §5).

The track is a modbus slave on the control-box RS-485 (proxied over 502):
slow, position-only, NO streaming interface, not in controller kinematics.
Treated as a slow axis: sparse absolute int-mm targets with ``wait=False``;
the measured position is folded into ``ArmState.q[7]`` by the driver.
Uses the ``*_linear_track_*`` SDK spelling throughout (alias of
``*_linear_motor_*`` since SDK 1.17.0).
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any

from . import units

# APIState 82: commanding an unhomed track (LINEAR_MOTOR_NOT_INIT)
RAIL_NOT_HOMED_CODE = 82
RAIL_SN_PREFIX = "AL13"  # AL1300 0.7m / AL1301 1.0m / AL1302 1.5m


class RailPhase(Enum):
    ABSENT = "absent"
    DETECTED = "detected"
    HOMING = "homing"
    READY = "ready"
    RAIL_ERROR = "rail_error"


class RailController:
    """One rail on one arm; stepped at 5 Hz from the driver's monitor thread."""

    MIN_DELTA_MM = 1.0  # skip re-targeting below this

    def __init__(self, api: Any, speed_mm_s: int = 200) -> None:
        self._api = api
        self._speed_mm_s = int(speed_mm_s)
        self._phase = RailPhase.ABSENT
        self._lock = threading.Lock()
        self._target_m: float | None = None  # latest-wins
        self._pos_m = 0.0
        self._last_sent_mm: int | None = None
        self._cleaned_once = False
        self._events: list[tuple[str, int, str]] = []  # (phase, code, detail)

    # -- properties ---------------------------------------------------------
    @property
    def phase(self) -> RailPhase:
        return self._phase

    @property
    def pos_m(self) -> float:
        return self._pos_m

    @property
    def present(self) -> bool:
        return self._phase not in (RailPhase.ABSENT,)

    def drain_events(self) -> list[tuple[str, int, str]]:
        out, self._events = self._events, []
        return out

    # -- bring-up -----------------------------------------------------------
    def detect(self) -> bool:
        """Present = registers read OK AND the SN looks like a real track.

        Absent tracks return code 3 (timeout) / 20 (host id) / 23 (modbus
        length). Sim-mode controllers silently no-op track calls and return
        success — requiring a valid AL13x SN catches that.
        """
        code, _registers = self._api.get_linear_track_registers()
        if code != 0:
            self._phase = RailPhase.ABSENT
            return False
        sn_code, sn = self._api.get_linear_track_sn()
        if sn_code != 0 or not str(sn or "").startswith(RAIL_SN_PREFIX):
            self._phase = RailPhase.ABSENT
            return False
        self._phase = RailPhase.DETECTED
        return True

    def ensure_homed(self) -> None:
        """Home once per power-on (commanding unhomed returns APIState 82)."""
        if self._phase == RailPhase.ABSENT:
            return
        code, on_zero = self._api.get_linear_track_on_zero()
        if code != 0:
            self._fault(code, "get_linear_track_on_zero failed")
            return
        if on_zero == 0:
            self._phase = RailPhase.HOMING
            code = self._api.set_linear_track_back_origin(wait=True, timeout=30)
            if code != 0:
                self._fault(code, "set_linear_track_back_origin failed")
                return
        self._api.set_linear_track_enable(True)
        self._api.set_linear_track_speed(self._speed_mm_s)
        self._phase = RailPhase.READY
        self._events.append((self._phase.name, 0, "rail ready"))

    # -- runtime ------------------------------------------------------------
    def set_target(self, pos_m: float) -> None:
        """Thread-safe latest-wins absolute target (m); clamped at send time."""
        with self._lock:
            self._target_m = float(pos_m)

    def step(self) -> None:
        """5 Hz on the monitor thread: refresh pos, send sparse targets."""
        if self._phase in (RailPhase.ABSENT, RailPhase.RAIL_ERROR, RailPhase.HOMING):
            return
        code, pos_mm = self._api.get_linear_track_pos()
        if code == 0 and pos_mm is not None:
            self._pos_m = units.rail_mm_to_m(float(pos_mm))
        with self._lock:
            target_m = self._target_m
        if target_m is None:
            return
        target_mm = units.rail_m_to_mm(target_m)  # clamped [0, 650]
        ref_mm = self._last_sent_mm if self._last_sent_mm is not None else self._pos_m * 1000.0
        if abs(target_mm - ref_mm) < self.MIN_DELTA_MM:
            return
        code = self._api.set_linear_track_pos(target_mm, wait=False)
        if code == 0:
            self._last_sent_mm = target_mm
            self._cleaned_once = False
        else:
            self._handle_track_error(code)

    def _handle_track_error(self, code: int) -> None:
        if not self._cleaned_once:  # one clean + re-enable attempt
            self._api.clean_linear_track_error()
            self._api.set_linear_track_enable(True)
            self._cleaned_once = True
        else:
            self._fault(code, "track command failed after clean+re-enable")

    def latch_error(self, code: int, detail: str = "") -> None:
        """Controller error 111 (RS-485 drop): latch ONLY the rail; the arm
        keeps streaming (02-hardware §3.5)."""
        self._fault(code, detail or "control-box external-485 comms error")

    def _fault(self, code: int, detail: str) -> None:
        self._phase = RailPhase.RAIL_ERROR
        with self._lock:
            self._target_m = None
        self._events.append((self._phase.name, int(code), detail))
