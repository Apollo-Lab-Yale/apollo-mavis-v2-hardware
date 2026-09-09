"""Linear track / rail controller (02-hardware §5).

The track is a modbus slave on the control-box RS-485 (proxied over 502):
slow, position-only, NO streaming interface, not in controller kinematics.
Treated as a slow axis: sparse absolute int-mm targets with ``wait=False``;
the measured position is folded into ``ArmState.q[7]`` by the driver.
Uses the ``*_linear_track_*`` SDK spelling throughout (alias of
``*_linear_motor_*`` since SDK 1.17.0).

**Connect never homes (phase-09c).** Homing (``set_linear_track_back_origin``)
drives the carriage to the zero end — MOTION — and is only ever issued on an
operator's explicit, twin-gated request: either by the ``home_rail``
maintenance op of the read-only monitor (``monitor.py``, 02-hardware §8.6,
session-less, joints braked) or — phase-09d — by :meth:`RailController.home`
through ``XArmDriver.home_rail()`` on a CONNECTED driver whose servo stream
holds the joints, after the runtime has pre-positioned the arm along a
rail-position-agnostic path. At connect the driver by default REQUIRES the
track to be homed (:meth:`RailController.require_homed`): an unhomed track
raises :class:`RailNotHomedError` and the session is refused, because a
carriage whose position is unknown cannot be gated by the twin. The
maintenance motion connects with ``allow_unhomed=True`` instead: the track
stays ``DETECTED`` (never commanded), ``pos_known`` is False and ``pos_m``
reads 0.0 — a PLACEHOLDER, not a measurement (core's ``ArmState`` requires a
finite ``q[7] == rail_pos_m``, so NaN/None cannot be published; 0.0 is the
zero end the homing drives to, so a hold target seeded from the published
state is motionless once the track is homed).

SDK 1.18.5 facts (verified in the pinned source, 2026-09-04): ``XArmAPI``
exposes ``get_linear_track_registers/pos/status/error/is_enabled/on_zero``,
``set_linear_track_*`` and ``clean_linear_track_error`` through its alias map
(``wrapper/xarm_api.py:120-133``) but NO ``get_linear_track_sn`` /
``get_linear_track_version`` (the ``x3`` layer has ``get_linear_motor_sn`` but
the wrapper never surfaces it; ``__getattr__`` raises AttributeError).
``registers['pos']`` is meaningless until the track is homed (``on_zero == 1``)
AND enabled — the lab tracks read ``{pos: 0, status: 2, error: 0,
is_enabled: 0, on_zero: 0}`` at power-on. ``set_linear_motor_back_origin(wait,
**kwargs)`` (``x3/linear_motor.py:131-148``) takes ``timeout`` (default 10 s)
and ``auto_enable`` (default True — and then OVERWRITES the wait result with
the enable's return code), so homing here passes ``auto_enable=False``,
enables + sets the speed itself and judges ONLY from the registers.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from apollo_mavis_v2_core.errors import BringupError

from . import units

try:  # core >= phase-09c ships it; an older core gets a same-named local subclass
    from apollo_mavis_v2_core.errors import RailNotHomedError
except ImportError:  # pragma: no cover - exercised only against an older core

    class RailNotHomedError(BringupError):  # type: ignore[no-redef]
        """Linear track detected but not homed (``on_zero == 0``): position unknown.

        Local stand-in for ``apollo_mavis_v2_core.errors.RailNotHomedError`` (same
        name, same ``(step="rail", message)`` signature) for a core that predates
        phase-09c.
        """

        def __init__(self, step: str = "rail", message: str = ""):
            super().__init__(step, message)


# APIState 82: commanding an unhomed track (LINEAR_MOTOR_NOT_INIT)
RAIL_NOT_HOMED_CODE = 82
RAIL_SN_PREFIX = "AL13"  # AL1300 0.7m / AL1301 1.0m / AL1302 1.5m
# How long the SDK's ``set_linear_track_back_origin(wait=True)`` polls for
# ``on_zero`` before giving up (100). Shared with the monitor's ``home_rail`` op.
# The lab tracks' homing duration is unmeasured ("~10 s" is folklore); 0.65 m at
# the track's own homing speed fits comfortably. Read at call time (tests patch it).
HOME_RAIL_SDK_WAIT_S = 30.0
UNKNOWN_RAIL_POS_M = 0.0  # placeholder published while the track is unhomed


class RailPhase(Enum):
    """Driver-side rail phases. There is deliberately NO ``HOMING``: homing is a
    latch (:attr:`RailController.homing`) inside :meth:`RailController.home`,
    never a resting phase — the track is ``DETECTED`` (unhomed, never commanded),
    ``READY`` (homed + enabled, position known) or ``RAIL_ERROR``."""

    ABSENT = "absent"
    DETECTED = "detected"
    READY = "ready"
    RAIL_ERROR = "rail_error"


@dataclass(frozen=True)
class RailHomeOutcome:
    """Result of :meth:`RailController.home` / ``XArmDriver.home_rail`` (phase-09d).

    ``ok`` is decided from the REGISTERS read back after the writes (``on_zero
    == 1 and is_enabled == 1 and error == 0``), never from the SDK return codes
    (kept in ``sdk_codes`` for diagnosis). ``written`` is False when the homing
    was refused before the first write (nothing moved). ``pos_m`` is the seeded
    carriage position when ``ok`` (0.0 by construction), else None: the position
    stays UNKNOWN after a failed homing.
    """

    ok: bool
    detail: str
    written: bool = False
    phase: str = RailPhase.ABSENT.name  # RailPhase.name after the attempt
    on_zero: int | None = None
    is_enabled: int | None = None
    error: int | None = None
    pos_m: float | None = None
    sdk_codes: dict[str, int] | None = None
    duration_s: float = 0.0


class RailController:
    """One rail on one arm; stepped at 5 Hz from the driver's monitor thread.

    Threading (phase-09d): :meth:`step` runs on the driver's ``_MonitorThread``;
    :meth:`set_target` on any thread (the runtime's control loop via
    ``command_joints``); :meth:`home` on the CALLER's thread (the runtime's
    rail-homing job) and blocks up to ``HOME_RAIL_SDK_WAIT_S`` + a few register
    round-trips. ``home`` raises the :attr:`homing` latch under ``_lock``, then
    takes ``_io_lock`` once as a barrier so a ``step()`` already inside its SDK
    calls finishes first; every later ``step()`` sees the latch and returns
    without touching the bus. Targets set while the track is not ``READY`` (or
    while homing) are DROPPED, not queued: a stale target must never move the
    carriage the moment the track becomes commandable.
    """

    MIN_DELTA_MM = 1.0  # skip re-targeting below this

    def __init__(self, api: Any, speed_mm_s: int = 50, arm_id: str = "") -> None:
        self._api = api
        self._speed_mm_s = int(speed_mm_s)
        self._arm_id = arm_id
        self._phase = RailPhase.ABSENT
        self._lock = threading.Lock()  # target / homing latch / events
        self._io_lock = threading.Lock()  # step() body vs the start of home()
        self._target_m: float | None = None  # latest-wins
        self._pos_m = UNKNOWN_RAIL_POS_M
        self._pos_known = False  # True once seeded from a homed + enabled track
        self._homing = False
        self._last_sent_mm: int | None = None
        self._cleaned_once = False
        self._events: list[tuple[str, int, str]] = []  # (phase, code, detail)
        self.warnings: list[str] = []  # non-fatal detect() notes -> driver.connect_warnings
        # the register dict the last successful detect() read (phase-12: lets the
        # read-only connect seed pos_m / pos_known without a second round-trip)
        self.last_registers: dict[str, Any] | None = None

    # -- properties ---------------------------------------------------------
    @property
    def phase(self) -> RailPhase:
        return self._phase

    @property
    def pos_m(self) -> float:
        """Measured carriage position (m) when :attr:`pos_known`; otherwise the
        ``UNKNOWN_RAIL_POS_M`` placeholder (0.0)."""
        return self._pos_m

    @property
    def pos_known(self) -> bool:
        """False until the track was seen homed AND enabled (connect gate or a
        successful :meth:`home`); a failed homing leaves it False."""
        return self._pos_known

    @property
    def homing(self) -> bool:
        return self._homing

    @property
    def present(self) -> bool:
        return self._phase not in (RailPhase.ABSENT,)

    def drain_events(self) -> list[tuple[str, int, str]]:
        with self._lock:
            out, self._events = self._events, []
        return out

    def _event(self, phase: RailPhase, code: int, detail: str) -> None:
        with self._lock:
            self._events.append((phase.name, int(code), detail))

    # -- bring-up -----------------------------------------------------------
    def detect(self) -> bool:
        """Present = the track registers read OK (a real modbus reply).

        Absent tracks return code 3 (timeout) / 20 (host id) / 23 (modbus
        length). A controller in simulation mode never touches the bus: the
        SDK's ``@xarm_is_not_simulation_mode`` returns ``(0, [])`` — an empty
        non-dict reply — which we treat as absent. The AL13x SN prefix is
        verified only when the SDK exposes ``get_linear_track_sn`` (1.18.5 does
        NOT — calling it raised AttributeError on the first real connect, fixed
        2026-09-04); otherwise one warning is recorded and the track is
        accepted on the registers alone.
        """
        code, registers = self._api.get_linear_track_registers()
        if code != 0 or not isinstance(registers, dict) or "pos" not in registers:
            self._phase = RailPhase.ABSENT
            return False
        self.last_registers = dict(registers)
        read_sn = getattr(self._api, "get_linear_track_sn", None)
        if callable(read_sn):
            sn_code, sn = read_sn()
            if sn_code != 0 or not str(sn or "").startswith(RAIL_SN_PREFIX):
                self._phase = RailPhase.ABSENT
                return False
        else:
            self.warnings.append(
                "rail SN not verified: SDK has no get_linear_track_sn "
                "(xarm-python-sdk 1.18.5); detected from registers only"
            )
        self._phase = RailPhase.DETECTED
        return True

    def require_homed(self, allow_unhomed: bool = False) -> None:
        """Connect-time gate — NEVER homes (phase-09c, user rule "no implicit motion").

        Reads ``get_linear_track_registers``: ``on_zero == 0`` raises
        :class:`RailNotHomedError` with NOTHING written (the phase stays
        ``DETECTED``; the operator homes from the UI via the monitor's
        ``home_rail`` op and retries). ``on_zero == 1`` runs the two non-motion
        setup writes — ``set_linear_track_enable(True)`` +
        ``set_linear_track_speed(speed_mm_s)`` — and SEEDS ``pos_m`` from the
        register (refreshed with one ``get_linear_track_pos`` after the enable,
        since ``pos`` is authoritative only when homed AND enabled) so the gate
        twin sees the true carriage position before the first 5 Hz ``step()``.

        ``allow_unhomed=True`` (phase-09d, ``XArmDriverConfig.rail_homing ==
        "allow_unhomed"``, the runtime's rail-homing maintenance motion ONLY):
        an unhomed track is ACCEPTED instead of raising — still nothing written,
        phase ``DETECTED`` (``step()`` never commands it, targets are dropped),
        ``pos_known`` False, ``pos_m`` the 0.0 placeholder, one warning + one
        ``DETECTED`` event so the operator sees "position unknown". The caller
        homes it later with :meth:`home`.

        Every other failure REFUSES the connect (:class:`BringupError`,
        ``step="rail"``; the workcell reports ``rail: error`` + ``error`` and the
        runtime refuses the session): a failing register read (the carriage
        position cannot be verified, so the gate twin could not place it — and a
        track that does not answer cannot be homed either), or a non-zero code
        from ``set_linear_track_enable`` / ``set_linear_track_speed`` (an
        un-enabled track would drop every rail command silently; an unset
        positioning speed would leave the D2 cap unapplied). The phase latches
        ``RAIL_ERROR`` and a ``RailEvent`` records the code, then the error
        propagates - the rail never silently reports ``pos_m == 0.0`` as a
        MEASUREMENT for a carriage of unknown position.
        """
        if self._phase == RailPhase.ABSENT:
            return
        who = f"{self._arm_id}: " if self._arm_id else ""
        code, registers = self._api.get_linear_track_registers()
        if code != 0 or not isinstance(registers, dict):
            self._fault(code, "get_linear_track_registers failed")
            raise BringupError(
                "rail",
                f"{who}get_linear_track_registers failed (code {code}): carriage position "
                "unverifiable - connect refused (the gate twin cannot place the carriage)",
            )
        on_zero = int(registers.get("on_zero", 0) or 0)
        if on_zero != 1:
            if not allow_unhomed:
                raise RailNotHomedError(
                    "rail",
                    f"{who}linear track not homed (on_zero == 0): carriage position unknown; "
                    "home it from the Hardware tab (Home rail) before starting a session",
                )
            self._pos_m = UNKNOWN_RAIL_POS_M
            self._pos_known = False
            note = (
                "linear track not homed (on_zero == 0): carriage position UNKNOWN - "
                f"rail slot reads {UNKNOWN_RAIL_POS_M:.3f} m as a placeholder; connected for "
                "the rail-homing maintenance motion only (home_rail)"
            )
            self.warnings.append(f"{who}{note}")
            self._event(RailPhase.DETECTED, 0, note)
            return
        self._pos_m = units.rail_mm_to_m(float(registers.get("pos", 0) or 0))
        self._setup_homed_track(who)
        self._event(self._phase, 0, f"rail ready at {self._pos_m:.3f} m")

    def observe_registers(self, registers: Mapping[str, Any] | None) -> None:
        """Read-only bookkeeping from one ``get_linear_track_registers`` reply
        (phase-12, ``XArmDriver.connect(readonly=True)``): NEVER writes and never
        changes the phase — the track stays ``DETECTED`` (a read-only client
        never enables, homes or commands it, and ``set_target`` keeps dropping).

        ``pos_known`` becomes True iff ``on_zero == 1 and is_enabled == 1`` (the
        same rule as the connect gate and the read-only monitor: ``pos`` is a
        measurement only on a homed AND enabled track) and ``pos_m`` is then the
        register; otherwise — unhomed, disabled, or ``registers`` None / not a
        register dict (a failed read) — the position is UNKNOWN and ``pos_m``
        reads the ``UNKNOWN_RAIL_POS_M`` placeholder (0.0), exactly like the
        phase-09d ``allow_unhomed`` case. Called on the driver's read-only poll
        thread; a track homed by the operator meanwhile (the monitor's
        ``home_rail`` op) is picked up on the next poll.
        """
        if not isinstance(registers, Mapping) or "pos" not in registers:
            self._pos_m = UNKNOWN_RAIL_POS_M
            self._pos_known = False
            return
        homed = int(registers.get("on_zero", 0) or 0) == 1
        enabled = int(registers.get("is_enabled", 0) or 0) == 1
        if homed and enabled:
            self._pos_m = units.rail_mm_to_m(float(registers.get("pos", 0) or 0))
            self._pos_known = True
        else:
            self._pos_m = UNKNOWN_RAIL_POS_M
            self._pos_known = False

    def _setup_homed_track(self, who: str) -> None:
        """Enable + positioning speed (non-motion) on a HOMED track, then refresh
        ``pos_m`` from ``get_linear_track_pos`` and go ``READY`` with the position
        known. Raises ``BringupError("rail")`` (phase ``RAIL_ERROR``) on a failing
        write — shared by :meth:`require_homed` and :meth:`home`."""
        code = int(self._api.set_linear_track_enable(True) or 0)
        if code != 0:
            self._fault(code, "set_linear_track_enable failed")
            raise BringupError(
                "rail",
                f"{who}set_linear_track_enable(True) returned {code}: linear track not "
                "enabled - connect refused (check the track in UFACTORY Studio)",
            )
        code = int(self._api.set_linear_track_speed(self._speed_mm_s) or 0)
        if code != 0:
            self._fault(code, "set_linear_track_speed failed")
            raise BringupError(
                "rail",
                f"{who}set_linear_track_speed({self._speed_mm_s}) returned {code}: positioning "
                "speed cap not applied - connect refused",
            )
        code, pos_mm = self._api.get_linear_track_pos()
        if code == 0 and pos_mm is not None:
            self._pos_m = units.rail_mm_to_m(float(pos_mm))
        self._pos_known = True
        self._last_sent_mm = None
        self._cleaned_once = False
        self._phase = RailPhase.READY

    # -- homing (phase-09d, caller's thread) ---------------------------------
    def home(self) -> RailHomeOutcome:
        """Home the track — MOTION: the carriage drives to the zero end. Caller's
        thread; blocks up to ``HOME_RAIL_SDK_WAIT_S`` plus a few register reads.

        Sequence: raise the :attr:`homing` latch (``step()`` stops sending,
        pending target dropped) → pre-read the registers (unreadable → fail; a
        latched track error → REFUSED, zero writes) →
        ``set_linear_track_back_origin(wait=True, timeout=HOME_RAIL_SDK_WAIT_S,
        auto_enable=False)`` → ``set_linear_track_enable(True)`` →
        ``set_linear_track_speed(speed_mm_s)`` → read the registers back and
        JUDGE FROM THEM ONLY (``on_zero == 1 and is_enabled == 1 and error ==
        0``; the SDK codes are diagnostic — 1.18.5 can return 0 for a homing
        that did not finish). Success seeds ``pos_m`` from the register (0.0),
        sets ``pos_known``, clears the target / last-sent bookkeeping and goes
        ``READY`` (+ event). Failure latches ``RAIL_ERROR`` (+ event) and leaves
        the position UNKNOWN. Re-homing a ``READY`` track is allowed (same path).
        Never raises for a track outcome; only ``ABSENT`` is answered with a
        refused outcome (the driver raises ``RailUnavailableError`` before that).
        """
        who = f"{self._arm_id}: " if self._arm_id else ""
        if self._phase == RailPhase.ABSENT:
            return RailHomeOutcome(False, f"{who}no linear track detected", phase=self._phase.name)
        t0 = time.monotonic()
        with self._lock:
            self._homing = True
            self._target_m = None
        try:
            with self._io_lock:  # barrier: a step() already on the bus finishes first
                pass
            return self._home_locked(who, t0)
        finally:
            with self._lock:
                self._homing = False

    def _home_locked(self, who: str, t0: float) -> RailHomeOutcome:
        codes: dict[str, int] = {}
        code, regs = self._api.get_linear_track_registers()
        if code != 0 or not isinstance(regs, dict):
            self._fault(code, "get_linear_track_registers failed before homing")
            return self._outcome(
                False,
                f"{who}rail homing failed: get_linear_track_registers returned {code} before "
                "the homing write (track unreachable) - nothing written",
                written=False,
                codes=codes,
                t0=t0,
            )
        track_err = int(regs.get("error", 0) or 0)
        if track_err:
            return self._outcome(
                False,
                f"{who}rail homing refused: linear track error {track_err} is latched - clear "
                "it first (nothing written)",
                written=False,
                regs=regs,
                codes=codes,
                t0=t0,
            )
        codes["set_linear_track_back_origin"] = int(
            self._api.set_linear_track_back_origin(
                wait=True, timeout=HOME_RAIL_SDK_WAIT_S, auto_enable=False
            )
            or 0
        )
        codes["set_linear_track_enable"] = int(self._api.set_linear_track_enable(True) or 0)
        codes["set_linear_track_speed"] = int(
            self._api.set_linear_track_speed(self._speed_mm_s) or 0
        )
        sdk_note = "; ".join(f"{n} returned {c}" for n, c in codes.items() if c != 0)
        code, regs = self._api.get_linear_track_registers()
        if code != 0 or not isinstance(regs, dict):
            self._fault(code, "get_linear_track_registers failed after homing")
            return self._outcome(
                False,
                f"{who}rail homing failed: register read-back returned {code} after the writes; "
                "carriage position unknown" + (f" ({sdk_note})" if sdk_note else ""),
                written=True,
                codes=codes,
                t0=t0,
            )
        on_zero = int(regs.get("on_zero", 0) or 0)
        enabled = int(regs.get("is_enabled", 0) or 0)
        track_err = int(regs.get("error", 0) or 0)
        problems: list[str] = []
        if on_zero != 1:
            problems.append("on_zero still 0 (carriage did not reach the zero end)")
        if enabled != 1:
            problems.append("track not enabled")
        if track_err:
            problems.append(f"linear track error {track_err}")
        if problems:
            detail = f"{who}rail homing failed: " + ", ".join(problems)
            if sdk_note:
                detail += f" ({sdk_note})"
            worst = next((c for c in codes.values() if c != 0), track_err)
            self._fault(worst, detail)
            return self._outcome(False, detail, written=True, regs=regs, codes=codes, t0=t0)
        self._pos_m = units.rail_mm_to_m(float(regs.get("pos", 0) or 0))
        code, pos_mm = self._api.get_linear_track_pos()
        if code == 0 and pos_mm is not None:
            self._pos_m = units.rail_mm_to_m(float(pos_mm))
        with self._lock:
            self._target_m = None  # a target set during the homing must not move it now
        self._pos_known = True
        self._last_sent_mm = None
        self._cleaned_once = False
        self._phase = RailPhase.READY
        detail = (
            f"{who}rail homed: carriage at {self._pos_m:.3f} m, track enabled, positioning "
            f"speed {self._speed_mm_s} mm/s"
        )
        if sdk_note:
            detail += f" (registers are authoritative; {sdk_note})"
        self._event(self._phase, 0, detail)
        return self._outcome(True, detail, written=True, regs=regs, codes=codes, t0=t0)

    def _outcome(
        self,
        ok: bool,
        detail: str,
        *,
        written: bool,
        t0: float,
        regs: dict[str, Any] | None = None,
        codes: dict[str, int] | None = None,
    ) -> RailHomeOutcome:
        def reg(name: str) -> int | None:
            if regs is None or name not in regs:
                return None
            return int(regs.get(name, 0) or 0)

        return RailHomeOutcome(
            ok=ok,
            detail=detail,
            written=written,
            phase=self._phase.name,
            on_zero=reg("on_zero"),
            is_enabled=reg("is_enabled"),
            error=reg("error"),
            pos_m=self._pos_m if ok else None,
            sdk_codes=dict(codes) if codes else None,
            duration_s=time.monotonic() - t0,
        )

    # -- runtime ------------------------------------------------------------
    def set_target(self, pos_m: float) -> None:
        """Thread-safe latest-wins absolute target (m); clamped at send time.

        DROPPED (not queued) unless the track is ``READY`` and not homing: an
        unhomed / faulted track is never commanded, and a target parked while
        it was not commandable must not fire the moment it becomes so."""
        with self._lock:
            if self._phase is not RailPhase.READY or self._homing:
                return
            self._target_m = float(pos_m)

    def step(self) -> None:
        """5 Hz on the monitor thread: refresh pos, send sparse targets. A no-op
        while :attr:`homing` (the whole body sits under ``_io_lock`` so
        :meth:`home` can wait for an in-flight step before it writes)."""
        with self._io_lock:
            if self._homing:
                return
            if self._phase in (RailPhase.ABSENT, RailPhase.DETECTED, RailPhase.RAIL_ERROR):
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
