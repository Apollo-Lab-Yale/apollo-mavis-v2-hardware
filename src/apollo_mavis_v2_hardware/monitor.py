"""Read-only controller state monitor + explicit maintenance channel
(02-hardware §8.5; phase-09a / phase-09b).

``ArmStateMonitor`` polls ONE xArm7 control box for its joint angles, flange
pose, error/warn codes, controller state/mode, linear-track registers,
gripper opening and the controller-side safety read-backs (collision
sensitivity, TCP payload) while NEVER commanding it on its own: no
``motion_enable``, no ``set_mode``/``set_state``, no ``clean_*``, no ``set_*``.
The runtime uses it session-less (the Welcome page's real error codes and the
digital-twin overlay); when a hardware session wants the boxes it calls
``disconnect()`` — pause means RELEASE the connection, two SDK clients on one
control box are unevidenced — and ``start()`` again afterwards.

Zero-write contract (phase-09b wording): **zero writes unless an explicit
maintenance request**. ``READ_ONLY_SDK_METHODS`` / ``READ_ONLY_SDK_ATTRS`` are
the complete list of ``XArmAPI`` members the polling touches;
``MAINTENANCE_SDK_METHODS`` is the complete per-operation write set of
:meth:`ArmStateMonitor.maintenance` — ``clear_errors`` = ``clean_error`` +
``clean_warn`` (no enable: nothing moves, brakes stay engaged),
``apply_backstops`` = exactly the ``backstops.apply_backstops`` sequence,
``home_rail`` (phase-09c) = ``set_linear_track_back_origin`` +
``set_linear_track_enable`` + ``set_linear_track_speed`` — **the ONE
maintenance op that moves a mechanical part**: the carriage drives to the
track's zero end. It is operator-triggered, twin-gated by the runtime (a
full-travel sweep at the arm's CURRENT posture) and session-less; the request
carries the posture the sweep assumed and the op refuses, before any write,
when the arm has moved or a controller error is latched. ``motion_enable`` is
never part of it (the track is a separate RS-485 axis; the arm stays braked).
``set_collision_sensitivity`` (2026-09-11) = exactly one
``set_collision_sensitivity(level)`` with the operator's level 1..3
(``backstops.set_collision_sensitivity``; the request carries ``level``, anything
else is refused before the queue), judged from the rich-frame read-back like
``apply_backstops``; no motion, volatile - the next connect re-applies the
config value. ``recover`` (enable + servo mode) needs a session driver and is
refused here.
Maintenance requests are queued and executed ON THE POLL THREAD (one
``XArmAPI`` is never used from two threads); the caller only waits.
``tests/test_monitor.py`` runs all of it against a call-logging fake and
asserts nothing else is ever called.

``home_rail`` and SDK 1.18.5 (``x3/linear_motor.py:131-148``, verified):
``set_linear_motor_back_origin(wait, **kwargs)`` reads ``auto_enable``
(default True) and ``timeout`` (default 10 s) from kwargs; with ``wait`` it
polls the registers until ``on_zero == 1`` (0), track ``error != 0`` (80),
``sci == 0`` (81), 10 consecutive read failures (101) or the timeout (100) — and
the loop also exits with 100 as soon as ``api.connected`` drops. Then, if
``auto_enable``, it OVERWRITES that result with ``set_linear_motor_enable(True)``'s
return code, so a timed-out homing can come back as 0. The monitor therefore
passes ``auto_enable=False``, enables + sets the speed itself, and judges ONLY
from the after-sample registers (``on_zero == 1 and is_enabled == 1 and
error == 0``), never from the return codes (kept in ``sdk_codes`` for
diagnosis). While the op runs the poll thread is inside the SDK wait: no new
sample is published, the arm's status reads ``stale`` and ``maintenance_busy``
is true — expected, the UI shows "homing". ``stop()``/``disconnect()`` wait
for an in-flight ``home_rail`` (up to ``HOME_RAIL_TIMEOUT_S``) instead of
pulling the client out from under the wait loop.

SDK 1.18.5 side effects a read-only caller cannot avoid (verified in the
pinned source; relevant to "state/mode/error unchanged before/after"):

* ``XArmAPI.connect()`` calls ``clean_warn()`` when a controller WARNING
  (not error) is latched at connect time (``x3/base.py:519-521``).
* The first linear-track / gripper register read runs
  ``checkset_modbus_baud`` (``x3/base.py:2581-2620``): if the RS-485 baud the
  controller reports differs from the SDK default (2 000 000) the SDK WRITES
  the baud register, soft-reboots the end module (tool bus) and, should that
  raise C19/C28 (tool bus) or C111 (control-box bus), runs ``clean_error()``
  + ``set_state()``. At the factory baud nothing is written. Constructing
  ``XArmAPI(..., baud_checkset=False)`` would disable the whole path (a wrong
  baud then just fails the read); the monitor keeps the SDK default so it
  reads through exactly the same path as ``grippers.py`` / ``rail.py``.
* A controller in simulation mode answers track reads with ``(0, [])``
  without touching the bus (``@xarm_is_not_simulation_mode``) -> "no rail".
* While ``0 < error_code <= 17`` the SDK stops refreshing its cached joint
  angles from the report stream; ``get_servo_angle`` still asks the box.
* Reads succeed (code 0) while a controller error is latched: ``_check_code``
  maps ERR_CODE/WAR_CODE/STATE_NOT_READY to 0 for get-type calls — the
  Perception Arm's C19 does not stop this monitor.
* Safety read-backs: SDK 1.18.5 has NO ``get_tcp_load`` /
  ``get_collision_sensitivity``. ``XArmAPI.tcp_load`` (``[kg, [x, y, z] mm]``)
  and ``XArmAPI.collision_sensitivity`` (0..5) are PROPERTIES filled by the
  SDK report thread from the ``normal``/``rich`` report frame
  (``x3/base.py:1635-1636, 1784-1787``; bytes 115..132 of the 30001/30002
  frame) — never from the ``real`` 30003 stream the session driver uses. The
  monitor keeps the SDK default ``report_type='rich'`` (30002, ~10 Hz), so
  the properties are live; until the first frame arrives they read the SDK's
  initial ``[0, [0, 0, 0]]`` / ``0``, and a value written by ``set_*`` shows
  up one report period (~0.1 s) later — ``apply_backstops`` waits for that.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from . import units
from .backstops import (
    BACKSTOP_SDK_METHODS,
    COLLISION_SENSITIVITY_LEVELS,
    STATUS_ECHO_CODES,
    apply_backstops,
    set_collision_sensitivity,
)
from .config import XArmDriverConfig
from .driver import _default_api_factory
from .rail import HOME_RAIL_SDK_WAIT_S  # shared with XArmDriver.home_rail (phase-09d)

logger = logging.getLogger(__name__)

ArmMonitorStatus = Literal["off", "connecting", "running", "stale", "paused", "error"]
GripperKind = Literal["xarm", "xarm_g2", "none"]
MaintenanceOp = Literal[
    "clear_errors", "apply_backstops", "recover", "home_rail", "set_collision_sensitivity"
]
MAINTENANCE_OPS: tuple[str, ...] = (
    "clear_errors",
    "apply_backstops",
    "recover",
    "home_rail",
    "set_collision_sensitivity",
)

# STATUS_ECHO_CODES (1 ERR_CODE, 2 WAR_CODE, 9 STATE_NOT_READY - the raw ``UxbusState``
# echoes a box with something latched answers; NOT command failures) is defined in
# ``backstops.py`` (the leaf both this module and ``driver.py`` import) and imported
# above, so ``monitor.STATUS_ECHO_CODES`` keeps working; see :meth:`ArmStateMonitor._judge`.

# ---------------------------------------------------------------------------
# ZERO-WRITE ALLOWLIST — every XArmAPI member the POLLING may touch. Anything
# else (motion_enable, set_mode, set_state, clean_error, clean_warn, set_*,
# register_*, ...) is a contract violation unless it is the write set of an
# explicit maintenance request (MAINTENANCE_SDK_METHODS below);
# tests/test_monitor.py asserts both.
# ---------------------------------------------------------------------------
READ_ONLY_SDK_METHODS: frozenset[str] = frozenset(
    {
        "connect",
        "disconnect",
        "get_servo_angle",  # 7 joint angles, rad (is_radian=True)
        "get_position",  # FLANGE pose, mm + rad (tcp_offset is zero on both boxes); the RPY
        #   is extrinsic XYZ (Rz.Ry.Rx); the TCP is flange (+) (Rz(pi), +0.172 z) on a gripper
        #   arm - see units.sdk_to_tcp_pose. ArmMonitorSample.tcp_pose stays the raw flange.
        "get_err_warn_code",  # [error_code, warn_code]
        "get_linear_track_registers",  # {pos, status, error, is_enabled, on_zero, ...}
        "get_gripper_g2_position",  # G2: int mm — same call as G2Gripper.poll()
        "get_gripper_position",  # classic: pulses — same call as ClassicGripper.poll()
    }
)
READ_ONLY_SDK_ATTRS: frozenset[str] = frozenset(
    {
        "connected",
        "state",
        "mode",
        "collision_sensitivity",  # 0..5, from the rich report frame (no get_* in SDK 1.18.5)
        "tcp_load",  # [kg, [x, y, z] mm], from the rich report frame
    }
)
# Complete write set per maintenance operation (executed on the poll thread,
# only on an explicit request). "recover" is refused by the monitor. "home_rail"
# is the ONE op that moves something (the track carriage) — no motion_enable.
# "set_collision_sensitivity" (2026-09-11) is the operator's level override: one
# write, the same call apply_backstops issues at step (2), no motion.
MAINTENANCE_SDK_METHODS: Mapping[str, frozenset[str]] = {
    "clear_errors": frozenset({"clean_error", "clean_warn"}),
    "apply_backstops": frozenset(BACKSTOP_SDK_METHODS),
    "recover": frozenset(),
    "home_rail": frozenset(
        {"set_linear_track_back_origin", "set_linear_track_enable", "set_linear_track_speed"}
    ),
    "set_collision_sensitivity": frozenset({"set_collision_sensitivity"}),
}

MAX_RECONNECT_S = 10.0  # exponential backoff cap
RAIL_GRIPPER_HZ = 2.0  # modbus round-trips are slow; poll them at ~2 Hz
# start() after a stop()/disconnect() whose join timed out waits this long for the old
# thread to leave the SDK call it is blocked in (connect(): two sockets with their own
# timeouts + the version handshake; a register read: modbus timeouts) before reconnecting.
STALE_THREAD_JOIN_S = 15.0
# apply_backstops: the set_* replies arrive before the rich report frame that echoes
# the new values; wait at most this long for the read-back to reflect the config.
BACKSTOP_READBACK_SETTLE_S = 0.5
BACKSTOP_READBACK_POLL_S = 0.05
TCP_LOAD_MATCH_KG = 0.05  # |read-back - config| tolerance (runtime uses the same)
DEFAULT_MAINTENANCE_TIMEOUT_S = 10.0  # caller wait for clear_errors / apply_backstops
# home_rail (phase-09c): the SDK waits up to HOME_RAIL_SDK_WAIT_S for on_zero, the
# caller (and the REST handler) up to HOME_RAIL_TIMEOUT_S; stop()/disconnect() wait
# that long for an in-flight homing before releasing the box. Duration is
# unmeasured on the lab tracks ("~10 s" is folklore); 0.65 m at the track's own
# homing speed fits comfortably. HOME_RAIL_SDK_WAIT_S (30 s) lives in rail.py.
HOME_RAIL_TIMEOUT_S = 45.0
HOME_RAIL_Q_TOL_RAD = 0.02  # max |q_now - expected_q| per joint before homing is refused


def controller_error_title(code: int) -> str:
    """``'controller error 19: End Effector Communication Error'`` from the
    SDK's code table (``xarm/core/config/x_code.py``); '' for 0."""
    code = int(code)
    if code == 0:
        return ""
    try:
        from xarm.core.config.x_code import ControllerError  # data table only

        title = str(ControllerError(code, status=0).title["en"]).strip()
    except Exception:  # noqa: BLE001 — table lookup is best-effort
        title = ""
    return f"controller error {code}: {title}" if title else f"controller error {code}"


@dataclass(frozen=True)
class ArmMonitorSample:
    """One read-only reading; field-for-field the data part of core's
    ``ArmMonitorTelemetry`` (``status``/``detail`` live on the monitor,
    ``age_s = clock() - t_mono``)."""

    arm_id: str
    seq: int
    t_mono: float
    q: tuple[float, ...]  # 7 joint angles, rad, controller order (identity to the twin)
    tcp_pose: tuple[float, ...]  # FLANGE [x, y, z m, roll, pitch, yaw rad] base frame; () if unread
    error_code: int = 0
    warn_code: int = 0
    state: int | None = None  # controller state (4 = stopped / not enabled)
    mode: int | None = None
    rail_present: bool | None = None  # registers readable (None: not polled)
    rail_homed: bool | None = None  # on_zero == 1
    rail_enabled: bool | None = None  # is_enabled == 1
    rail_pos_m: float | None = None  # None unless homed AND enabled
    rail_raw_mm: float | None = None  # raw register, always when present
    rail_error: int | None = None  # track error register (0 = none); None when not present
    gripper_open_frac: float | None = None  # 0 closed .. 1 open; None for "none"
    gripper_raw: float | None = None  # raw SDK reading (G2 mm / classic pulses)
    # controller-side safety read-backs (phase-09b; slow rounds, rich report frame)
    collision_sensitivity: int | None = None  # 0..5; None until read
    tcp_load_kg: float | None = None  # set_tcp_load weight as the controller reports it
    tcp_load_cog_mm: tuple[float, ...] = ()  # (x, y, z) mm; () until read


# (present, homed, enabled, pos_m, raw_mm, error) as read by _read_rail
_RailRegs = tuple[bool | None, bool | None, bool | None, float | None, float | None, int | None]


@dataclass(frozen=True)
class MaintenanceOutcome:
    """Result of one :meth:`ArmStateMonitor.maintenance` request (the data part
    of core's ``ArmMaintenanceResult``; the runtime adds ``path``)."""

    arm_id: str
    op: str
    ok: bool
    detail: str = ""  # human-readable outcome
    sdk_codes: dict[str, int] = field(default_factory=dict)  # SDK call -> code, call order
    warnings: tuple[str, ...] = ()  # apply_backstops non-fatal codes
    before: ArmMonitorSample | None = None  # sample taken right before the op
    after: ArmMonitorSample | None = None  # sample taken right after (slow fields refreshed)


class _MaintenanceRequest:
    """One queued maintenance operation; completed by the poll thread.

    ``expected_q`` / ``q_tol_rad`` belong to ``home_rail``: the 7 joint angles the
    runtime's rail sweep was checked at; the op refuses before its first write when
    the before-sample deviates by more than ``q_tol_rad`` on any joint. ``level``
    belongs to ``set_collision_sensitivity``: the operator's level (1..3, validated
    in :meth:`ArmStateMonitor.maintenance`)."""

    __slots__ = (
        "op",
        "driver_cfg",
        "expected_q",
        "q_tol_rad",
        "level",
        "done",
        "outcome",
        "abandoned",
    )

    def __init__(
        self,
        op: str,
        driver_cfg: XArmDriverConfig | None,
        expected_q: tuple[float, ...] | None = None,
        q_tol_rad: float = HOME_RAIL_Q_TOL_RAD,
        level: int | None = None,
    ) -> None:
        self.op = op
        self.driver_cfg = driver_cfg
        self.expected_q = expected_q
        self.q_tol_rad = float(q_tol_rad)
        self.level = level
        self.done = threading.Event()
        self.outcome: MaintenanceOutcome | None = None
        self.abandoned = False  # the caller timed out; nobody reads the outcome

    def complete(self, outcome: MaintenanceOutcome) -> None:
        self.outcome = outcome
        self.done.set()


class ArmStateMonitor:
    """Read-only poller for one control box; one daemon thread, reconnects itself.

    ``snapshot()`` is lock-free for readers (frozen dataclass, reference swap);
    ``status`` derives ``"stale"`` from the last sample's age. ``stop()`` ends
    the thread and releases the box (status ``off``); ``disconnect()`` does the
    same but reports ``paused`` (hand-over to a session driver); ``start()``
    after either reconnects.

    Hand-over guarantee: the poll thread publishes its SDK client (``_api``,
    status ``running``, samples) only while it is the CURRENT generation and
    ``_running`` holds — both checked under the lock. So when ``stop()`` /
    ``disconnect()`` return before the thread left a blocking ``api.connect()``
    (the join timed out), the thread disconnects that client itself on return
    instead of adopting it, and the status stays ``paused`` / ``off``; ``start()``
    waits for such a thread before spawning the next one, so two clients never
    hold the same control box. Both shutdown calls return whether the box was
    already released (``join()`` waits for a late release).

    Maintenance channel (phase-09b): :meth:`maintenance` queues an explicit,
    operator-triggered operation; the poll thread executes it between polls
    (before/after samples, slow fields refreshed right after), the caller waits
    up to ``timeout_s``. Pending requests fail when the box is lost or the
    monitor is stopped / handed over; a request already executing re-checks the
    hand-over between its SDK calls and refuses BEFORE its first write, and
    ``stop()``/``disconnect()`` wait for an op that is mid-write (bounded by
    ``STALE_THREAD_JOIN_S``; ``HOME_RAIL_TIMEOUT_S`` for ``home_rail``) instead
    of disconnecting the client under it — so no maintenance write ever races a
    session driver that just took the box and no homing is ever cut mid-travel.
    """

    def __init__(
        self,
        arm_id: str,
        ip: str,
        *,
        gripper: GripperKind,
        expect_rail: bool,
        poll_hz: float = 10.0,
        stale_s: float = 0.5,
        reconnect_s: float = 2.0,
        api_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError("poll_hz must be > 0")
        if gripper not in ("xarm", "xarm_g2", "none"):
            raise ValueError(f"unknown gripper kind {gripper!r}")
        self.arm_id = arm_id
        self.ip = ip
        self.gripper: GripperKind = gripper
        self.expect_rail = bool(expect_rail)
        self.poll_hz = float(poll_hz)
        self.stale_s = float(stale_s)
        self.reconnect_s = float(reconnect_s)
        self._api_factory = api_factory or _default_api_factory
        self._clock = clock
        self._period = 1.0 / self.poll_hz
        self._slow_every = max(1, round(self.poll_hz / RAIL_GRIPPER_HZ))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._gen = 0  # bumped per start(); a thread only publishes while it is current
        self._api: Any = None
        self._status: ArmMonitorStatus = "off"
        self._detail = ""
        self._sample: ArmMonitorSample | None = None
        self._seq = 0
        self._tick = 0
        self._running_since: float | None = None
        self._backoff_s = self.reconnect_s
        self._connect_attempts = 0
        self._backoff_log: list[float] = []  # delays actually scheduled (tests)
        self._requests: deque[_MaintenanceRequest] = deque()  # serviced by the poll thread
        self._maintenance_active = False
        self._active_op: str | None = None  # op executing on the poll thread (shutdown budget)
        self._stop_detail = ""  # why requests are refused after stop()/disconnect()

    # -- public surface ----------------------------------------------------------
    def start(self) -> None:
        """Spawn the poll thread (idempotent); connects with backoff.

        If the previous thread is still finishing a ``stop()``/``disconnect()``
        (its join timed out inside a blocking SDK call) this waits for it —
        up to ``STALE_THREAD_JOIN_S`` — so the box is released before the new
        client connects."""
        with self._lock:
            stale = self._thread
            if stale is not None and stale.is_alive() and self._running:
                return  # already polling
        if stale is not None and stale.is_alive():
            stale.join(STALE_THREAD_JOIN_S)
            if stale.is_alive():
                logger.warning(
                    "%s: previous monitor thread still inside the SDK after %.0f s; "
                    "starting a new one (the old one releases its client on return)",
                    self.arm_id,
                    STALE_THREAD_JOIN_S,
                )
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._running:
                return  # started concurrently while we waited
            self._gen += 1
            self._running = True
            self._stop_detail = ""
            self._wake.clear()
            self._backoff_s = self.reconnect_s
            self._status, self._detail = "connecting", f"connecting to {self.ip}"
            self._thread = threading.Thread(
                target=self._run,
                args=(self._gen,),
                name=f"hw.{self.arm_id}.monitor-ro",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        """Stop polling and release the box; status -> ``off``. Idempotent.
        Returns True when the box is released on return (False: the thread is
        still inside an SDK call and releases it itself; see :meth:`join`)."""
        return self._shutdown("off", "stopped", timeout)

    def disconnect(self, timeout: float = 2.0) -> bool:
        """Release the box for a session driver (hand-over); status -> ``paused``.
        ``start()`` reconnects. Idempotent. Returns True when the box is
        released on return (see :meth:`stop`)."""
        return self._shutdown("paused", "released for hand-over", timeout)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the poll thread to exit (after ``stop()``/``disconnect()``);
        True when it has — the SDK client is then released for certain."""
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self) -> ArmMonitorSample | None:
        return self._sample  # frozen; single reference read

    def maintenance(
        self,
        op: MaintenanceOp,
        driver_cfg: XArmDriverConfig | None = None,
        timeout_s: float | None = None,
        *,
        expected_q: tuple[float, ...] | list[float] | None = None,
        q_tol_rad: float = HOME_RAIL_Q_TOL_RAD,
        level: int | None = None,
    ) -> MaintenanceOutcome:
        """Run one explicit maintenance operation ON THE POLL THREAD and wait for it.

        ``clear_errors``: ``clean_error()`` + ``clean_warn()`` — nothing else, no
        enable (measured 2026-09-04: no motion). ``apply_backstops``:
        ``backstops.apply_backstops(api, driver_cfg)`` (``driver_cfg`` required).
        ``home_rail`` (phase-09c, THE ONE MOTION OP — the carriage drives to the
        zero end): needs ``driver_cfg`` (``rail_speed_mm_s``) and ``expected_q``
        (the 7 joint angles the runtime's twin sweep was checked at); refused
        before any write unless the before-sample shows the arm within
        ``q_tol_rad`` of ``expected_q`` per joint, ``error_code == 0``, a track
        present and no track error. Writes exactly
        ``set_linear_track_back_origin(wait=True, timeout=HOME_RAIL_SDK_WAIT_S,
        auto_enable=False)`` -> ``set_linear_track_enable(True)`` ->
        ``set_linear_track_speed(cfg.rail_speed_mm_s)``; ``ok`` is judged from the
        after-sample registers only (``on_zero == 1``, ``is_enabled == 1``,
        ``error == 0``), never from the SDK return codes (module docstring).
        ``set_collision_sensitivity`` (2026-09-11): needs ``level`` in 1..3 (refused
        with ``ok=False`` otherwise, nothing queued); writes exactly
        ``set_collision_sensitivity(level, wait=False)``, waits up to
        ``BACKSTOP_READBACK_SETTLE_S`` for the rich frame to echo it, and is ``ok``
        iff the after-sample reads ``collision_sensitivity == level`` (the SDK code
        is diagnosis; a ``STATUS_ECHO_CODES`` echo is not a failure). ``driver_cfg``
        is optional here and only names the config value the next connect restores.
        ``recover`` needs a session driver -> ``ok=False``. Not connected / paused
        / stopped -> ``ok=False``. The outcome carries the SDK return codes in
        call order, the sample right before and the sample right after (slow
        fields — rail, gripper, sensitivity, payload — refreshed). ``timeout_s``
        defaults per op (``DEFAULT_MAINTENANCE_TIMEOUT_S``; ``HOME_RAIL_TIMEOUT_S``
        for ``home_rail``); on timeout the request is abandoned (its late result
        is dropped, the op itself still completes on the poll thread) and
        ``ok=False``.
        """
        if op not in MAINTENANCE_OPS:
            raise ValueError(f"unknown maintenance op {op!r}; expected one of {MAINTENANCE_OPS}")
        if op == "recover":
            return self._refuse(op, "recover needs a session")
        if op == "apply_backstops" and driver_cfg is None:
            return self._refuse(op, "apply_backstops needs the arm's driver config")
        if op == "set_collision_sensitivity":
            if (
                level is None
                or isinstance(level, bool)
                or int(level) != level
                or int(level) not in COLLISION_SENSITIVITY_LEVELS
            ):
                return self._refuse(
                    op,
                    "set_collision_sensitivity needs a level of 1, 2 or 3 "
                    f"(got {level!r}); 0 turns detection off and 4 / 5 false-trigger "
                    "under payload",
                )
            level = int(level)
        expected: tuple[float, ...] | None = None
        if op == "home_rail":
            if driver_cfg is None:
                return self._refuse(op, "home_rail needs the arm's driver config (rail speed)")
            if not self.expect_rail:
                return self._refuse(op, "home_rail: this monitor does not poll a linear track")
            if expected_q is None or len(expected_q) != 7:
                return self._refuse(
                    op,
                    "home_rail needs expected_q: the 7 joint angles the rail sweep was checked at",
                )
            expected = tuple(float(v) for v in expected_q)
            if q_tol_rad <= 0:
                return self._refuse(op, "home_rail: q_tol_rad must be > 0")
        if timeout_s is None:
            timeout_s = HOME_RAIL_TIMEOUT_S if op == "home_rail" else DEFAULT_MAINTENANCE_TIMEOUT_S
        with self._lock:
            if not self._running or self._api is None:
                status, detail = self._status, self._detail
                req = None
            else:
                req = _MaintenanceRequest(op, driver_cfg, expected, q_tol_rad, level=level)
                self._requests.append(req)
                self._wake.set()  # do not wait out the current poll period
        if req is None:
            what = f"monitor {status}" + (f": {detail}" if detail else "")
            return self._refuse(op, f"not connected to {self.ip} ({what})")
        if req.done.wait(max(0.0, float(timeout_s))):
            assert req.outcome is not None
            return req.outcome
        with self._lock:
            if req.outcome is not None:  # completed while we were timing out
                return req.outcome
            req.abandoned = True
        return self._refuse(
            op,
            f"{op} timed out after {float(timeout_s):g} s "
            "(the poll thread has not returned from the SDK); result dropped",
        )

    def _refuse(self, op: str, detail: str) -> MaintenanceOutcome:
        return MaintenanceOutcome(self.arm_id, op, ok=False, detail=detail)

    @property
    def maintenance_busy(self) -> bool:
        """A maintenance request is queued or executing on the poll thread."""
        with self._lock:
            return self._maintenance_active or any(not r.abandoned for r in self._requests)

    @property
    def status(self) -> ArmMonitorStatus:
        with self._lock:
            status, sample, since = self._status, self._sample, self._running_since
        if status == "running":
            now = self._clock()
            if sample is not None:
                if now - sample.t_mono > self.stale_s:
                    return "stale"
            elif since is not None and now - since > self.stale_s:
                return "stale"
        return status

    @property
    def detail(self) -> str:
        with self._lock:
            detail = self._detail
        if not detail and self.status == "stale":
            age = self.age_s
            return f"no fresh sample for {age:.1f} s" if age is not None else "no sample yet"
        return detail

    @property
    def age_s(self) -> float | None:
        sample = self._sample
        return None if sample is None else max(0.0, self._clock() - sample.t_mono)

    @property
    def connected(self) -> bool:
        return self._api is not None

    @property
    def connect_attempts(self) -> int:
        return self._connect_attempts

    @staticmethod
    def next_backoff(delay_s: float) -> float:
        """Exponential backoff: double, capped at ``MAX_RECONNECT_S``."""
        return min(float(delay_s) * 2.0, MAX_RECONNECT_S)

    # -- thread body -------------------------------------------------------------
    def _current(self, gen: int) -> bool:
        """This thread generation may still publish (not stopped, not superseded)."""
        return self._running and self._gen == gen

    def _run(self, gen: int) -> None:
        while self._current(gen):
            api = self._api
            if api is None:
                api = self._connect_once(gen)
                if api is None:
                    if self._current(gen):
                        self._backoff(gen)
                    continue
            req = self._next_request(gen)
            try:
                if req is not None:
                    self._service(gen, api, req)
                    continue  # back to polling without a sleep
                self._poll(gen, api, slow=(self._tick % self._slow_every == 0))
            except Exception as exc:  # noqa: BLE001 — the SDK raises bare Exception
                self._set(gen, "error", f"{type(exc).__name__}: {exc}".strip(": "))
                self._release_api(api)
                self._backoff(gen)
                continue
            self._tick += 1
            self._wait(gen, self._period)

    def _wait(self, gen: int, seconds: float) -> None:
        if self._wake.wait(max(0.0, seconds)):
            with self._lock:
                if self._current(gen):  # a superseded thread must not eat the new one's wake-up
                    self._wake.clear()

    def _backoff(self, gen: int) -> None:
        delay = self._backoff_s
        self._backoff_log.append(delay)
        self._backoff_s = self.next_backoff(delay)
        self._wait(gen, delay)

    def _set(self, gen: int, status: ArmMonitorStatus, detail: str) -> None:
        """Publish status/detail — only while this thread generation is current, so
        a thread finishing after stop()/disconnect() never overwrites ``off``/``paused``."""
        with self._lock:
            if self._current(gen):
                self._status, self._detail = status, detail

    def _connect_once(self, gen: int) -> Any:
        """Build + connect a fresh client; returns it once PUBLISHED as ``_api``,
        else None (connect failed, or stop()/disconnect() ran while ``connect()``
        blocked — then the client is disconnected right here and never adopted)."""
        self._connect_attempts += 1
        self._set(gen, "connecting", f"connecting to {self.ip} (attempt {self._connect_attempts})")
        api: Any = None
        try:
            # do_not_open + connect(): same socket set as the SDK default
            # (502 control + the 'rich' 30002 report stream, which keeps
            # .state/.mode/.tcp_load/.collision_sensitivity current)
            api = self._api_factory(self.ip, is_radian=True, do_not_open=True)
            api.connect()
            if getattr(api, "connected", True) is False:
                raise ConnectionError("SDK reports not connected after connect()")
        except Exception as exc:  # noqa: BLE001 — the SDK raises bare Exception
            _quiet_disconnect(api)
            self._set(gen, "error", f"connect failed: {exc}")
            return None
        with self._lock:
            current = self._current(gen)
            if current:
                self._api = api
                self._running_since = self._clock()
                self._status, self._detail = "running", ""
        if not current:
            _quiet_disconnect(api)  # hand-over happened meanwhile: release, do not adopt
            return None
        self._backoff_s = self.reconnect_s
        self._tick = 0
        return api

    def _release_api(self, api: Any = None) -> None:
        """Disconnect and forget the published client. With ``api`` given, only
        when it is still the published one (a thread finishing late must not
        release the client a newer start() published). Pending maintenance
        requests fail: their box is gone."""
        with self._lock:
            if api is not None and self._api is not api:
                return
            held, self._api = self._api, None
            self._running_since = None
        self._fail_pending("connection to the control box lost")
        _quiet_disconnect(held)

    def _shutdown(self, status: ArmMonitorStatus, detail: str, timeout: float) -> bool:
        stop_detail = f"monitor {status}: {detail}"
        with self._lock:
            self._running = False
            self._stop_detail = stop_detail  # _service() / _next_request() refuse with it
            thread = self._thread
            self._wake.set()
        self._fail_pending(stop_detail)
        released = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            active_op = self._maintenance_in_flight()
            if thread.is_alive() and active_op is not None:
                # An explicit maintenance op is executing on the poll thread. Never
                # pull the client from under its writes: the op re-checks the
                # hand-over between SDK calls (it refuses before its first write
                # and skips the read-back after the last), so this wait is at most
                # the SDK call in flight plus the writes already committed to. For
                # home_rail that call is the SDK's homing wait (the carriage is
                # moving): disconnecting would end the wait early with the track
                # still travelling, so the budget is HOME_RAIL_TIMEOUT_S.
                budget = HOME_RAIL_TIMEOUT_S if active_op == "home_rail" else STALE_THREAD_JOIN_S
                logger.warning(
                    "%s: maintenance op %s still executing %.1f s after %s; waiting for it "
                    "(up to %.0f s) before releasing the box",
                    self.arm_id,
                    active_op,
                    timeout,
                    detail,
                    budget,
                )
                thread.join(max(0.0, budget - timeout))
            if thread.is_alive():
                released = False
                logger.warning(
                    "%s: monitor thread still inside the SDK %.1f s after %s; "
                    "it releases its client itself when the call returns",
                    self.arm_id,
                    timeout,
                    detail,
                )
        self._release_api()
        with self._lock:
            if thread is self._thread and (thread is None or not thread.is_alive()):
                self._thread = None
            self._status, self._detail = status, detail
        return released

    # -- maintenance channel (poll thread) -------------------------------------------
    def _maintenance_in_flight(self) -> str | None:
        """Name of the op executing on the poll thread, or None."""
        with self._lock:
            return self._active_op if self._maintenance_active else None

    def _stopped_detail(self) -> str:
        """Refusal text once this generation may no longer publish."""
        with self._lock:
            return self._stop_detail or "monitor stopped before the request ran"

    def _next_request(self, gen: int) -> _MaintenanceRequest | None:
        """Pop the next live request (abandoned ones are dropped silently)."""
        with self._lock:
            while self._requests:
                req = self._requests.popleft()
                if req.abandoned:
                    continue
                if not self._current(gen):
                    detail = self._stop_detail or "monitor stopped before the request ran"
                    req.complete(self._refuse(req.op, detail))
                    continue
                self._maintenance_active = True
                self._active_op = req.op
                return req
        return None

    def _fail_pending(self, detail: str) -> None:
        with self._lock:
            pending = list(self._requests)
            self._requests.clear()
        for req in pending:
            if not req.abandoned:
                req.complete(self._refuse(req.op, detail))

    def _service(self, gen: int, api: Any, req: _MaintenanceRequest) -> None:
        """Execute one request on the poll thread: sample -> op -> sample (slow
        fields refreshed). SDK exceptions complete the request with ``ok=False``
        and propagate so the caller (`_run`) treats the box as lost.

        Hand-over safety: ``stop()``/``disconnect()`` may land while the
        before-sample is inside the SDK. The op re-checks :meth:`_current`
        right before its first write and refuses if the monitor was stopped /
        handed over meanwhile (nothing written); once the writes ran, a
        hand-over only skips the after-sample and the outcome still reports
        the writes (``after=None``, detail says why)."""
        codes: dict[str, int] = {}
        warnings: list[str] = []
        before: ArmMonitorSample | None = None
        rail_regs: _RailRegs | None = None  # home_rail: register read-back right after the writes
        try:
            with self._lock:  # home_rail: the before-sample must be NEWER than this
                prev_seq = self._sample.seq if self._sample is not None else -1
            before = self._sample_now(gen, api)
            if not self._current(gen):  # stopped / handed over during the before-sample
                req.complete(self._refuse(req.op, self._stopped_detail()))
                return
            if req.op == "clear_errors":
                codes["clean_error"] = int(api.clean_error())
                codes["clean_warn"] = int(api.clean_warn())
            elif req.op == "apply_backstops":
                assert req.driver_cfg is not None  # checked in maintenance()
                warnings = apply_backstops(api, req.driver_cfg, codes)
                if self._current(gen):  # a hand-over skips the read-back wait too
                    self._settle_backstops(api, req.driver_cfg)
            elif req.op == "set_collision_sensitivity":
                assert req.level is not None  # checked in maintenance()
                # ONE write (the apply_backstops step-(2) call with the operator's
                # level); the warnings list is the non-zero code, if any
                warnings = set_collision_sensitivity(api, req.level, codes)
                if self._current(gen):
                    self._settle_backstops(api, sensitivity=req.level)
            elif req.op == "home_rail":
                assert req.driver_cfg is not None  # checked in maintenance()
                refusal = self._home_rail_refusal(req, before, prev_seq)
                if refusal is not None:  # zero writes; the sample it was judged on rides along
                    req.complete(
                        MaintenanceOutcome(
                            self.arm_id, req.op, ok=False, detail=refusal, before=before
                        )
                    )
                    return
                # THE motion write: the carriage drives to the zero end; the SDK
                # blocks here until on_zero (or 30 s). Never auto_enable — its
                # return code would mask the wait result (module docstring).
                codes["set_linear_track_back_origin"] = int(
                    api.set_linear_track_back_origin(
                        wait=True, timeout=HOME_RAIL_SDK_WAIT_S, auto_enable=False
                    )
                )
                # non-motion follow-ups, always issued once homing was started
                codes["set_linear_track_enable"] = int(api.set_linear_track_enable(True))
                codes["set_linear_track_speed"] = int(
                    api.set_linear_track_speed(int(req.driver_cfg.rail_speed_mm_s))
                )
                # the judge needs the registers even when a hand-over landed meanwhile
                # (the client is still ours: _shutdown waits for this op); one modbus read
                rail_regs = self._read_rail(api, [])
            else:  # "recover" never reaches the queue; keep the refusal in one place
                req.complete(self._refuse(req.op, "recover needs a session"))
                return
            after = self._sample_now(gen, api) if self._current(gen) else None
            if req.op == "home_rail":
                ok, detail = self._judge_home_rail(req, codes, before, rail_regs)
            else:
                ok, detail = self._judge(req, codes, warnings, before, after)
            if after is None:  # the writes ran; the hand-over pre-empted the read-back
                detail += f" ({self._stopped_detail()} before the read-back)"
            req.complete(
                MaintenanceOutcome(
                    self.arm_id,
                    req.op,
                    ok=ok,
                    detail=detail,
                    sdk_codes=codes,
                    warnings=tuple(warnings),
                    before=before,
                    after=after,
                )
            )
        except Exception as exc:  # noqa: BLE001 — the SDK raises bare Exception
            req.complete(
                MaintenanceOutcome(
                    self.arm_id,
                    req.op,
                    ok=False,
                    detail=f"{req.op} failed: {type(exc).__name__}: {exc}".rstrip(": "),
                    sdk_codes=codes,
                    warnings=tuple(warnings),
                    before=before,
                )
            )
            raise
        finally:
            with self._lock:
                self._maintenance_active = False  # after complete(): busy covers the whole op
                self._active_op = None

    def _sample_now(self, gen: int, api: Any) -> ArmMonitorSample | None:
        """One full (slow) poll; returns the newest published sample."""
        self._poll(gen, api, slow=True)
        return self._sample

    def _settle_backstops(
        self,
        api: Any,
        cfg: XArmDriverConfig | None = None,
        *,
        sensitivity: int | None = None,
    ) -> None:
        """Wait (bounded by ``BACKSTOP_READBACK_SETTLE_S``) for the rich report frame
        to echo the values just set: with ``cfg`` the whole ``apply_backstops`` set
        (sensitivity AND payload), with a bare ``sensitivity`` (the
        ``set_collision_sensitivity`` op) that level alone."""
        want_sens = cfg.collision_sensitivity if cfg is not None else sensitivity
        want_kg = cfg.tcp_load_kg if cfg is not None else None
        deadline = time.monotonic() + BACKSTOP_READBACK_SETTLE_S
        while True:
            sens = _as_int(getattr(api, "collision_sensitivity", None))
            kg, _cog = _parse_tcp_load(getattr(api, "tcp_load", None))
            sens_ok = want_sens is None or sens == want_sens
            kg_ok = want_kg is None or (kg is not None and abs(kg - want_kg) <= TCP_LOAD_MATCH_KG)
            if sens_ok and kg_ok:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(BACKSTOP_READBACK_POLL_S)

    def _home_rail_refusal(
        self, req: _MaintenanceRequest, before: ArmMonitorSample | None, prev_seq: int = -1
    ) -> str | None:
        """Why ``home_rail`` must NOT start (None = go). Evaluated on the before-sample,
        before the first write: no FRESH sample (``_poll`` publishes nothing when
        ``get_servo_angle`` fails, so ``before`` would silently be the previous
        sample - equal to ``expected_q`` by construction - although the arm may have
        moved; ``prev_seq`` is the newest seq before the re-sample), a sample older
        than ``stale_s``, no track, controller error latched, track error latched, or
        the arm no longer in the posture the twin sweep assumed."""
        if before is None or before.seq <= prev_seq:
            return (
                "home_rail refused: could not take a fresh sample of the arm (get_servo_angle "
                "failed) - the posture cannot be verified; retry"
            )
        age = self._clock() - before.t_mono
        if age > self.stale_s:
            return (
                f"home_rail refused: the arm sample is {age:.1f} s old (stale after "
                f"{self.stale_s:g} s) - the posture cannot be verified; retry"
            )
        if before.rail_present is not True:
            return "home_rail refused: no linear track detected on this arm"
        if before.error_code != 0:
            return (
                f"home_rail refused: {controller_error_title(before.error_code)} is latched; "
                "clear errors first"
            )
        if before.rail_error:
            return (
                f"home_rail refused: linear track error {before.rail_error} is latched "
                "(power-cycle the track / check it in UFACTORY Studio)"
            )
        if req.expected_q is None or len(before.q) != 7:
            return "home_rail refused: no posture to compare against"
        worst = max(range(7), key=lambda i: abs(before.q[i] - req.expected_q[i]))
        dev = abs(before.q[worst] - req.expected_q[worst])
        if dev > req.q_tol_rad:
            return (
                f"home_rail refused: the arm moved since the sweep was checked "
                f"(joint {worst + 1} differs by {dev:.3f} rad, tolerance {req.q_tol_rad:g} rad); "
                "re-run the check at the current posture"
            )
        return None

    def _judge(
        self,
        req: _MaintenanceRequest,
        codes: dict[str, int],
        warnings: list[str],
        before: ArmMonitorSample | None,
        after: ArmMonitorSample | None,
    ) -> tuple[bool, str]:
        # SDK 1.18.5 returns clean_error() / clean_warn() RAW — they are the only
        # writes here that skip ``_check_code`` (``x3/base.py:2394-2413``). A box with
        # something latched answers with the STATUS ECHO, not a failure: ERR_CODE 1
        # ("an error is/was latched"), WAR_CODE 2 ("a warning is/was latched"),
        # STATE_NOT_READY 9 (stopped, brakes on) — precisely the three codes
        # ``_check_code`` maps to 0 for every non-move call. Judging on the raw code
        # made the operator's **Clear errors** click report "FAILED - clean_error
        # returned 2" on 2026-09-05 while the controller error was in fact cleared
        # (the after-sample read error_code 0). Like ``home_rail``, judge from the
        # READ-BACK and keep the codes in ``sdk_codes`` for diagnosis.
        failed = [
            f"{name} returned {code}"
            for name, code in codes.items()
            if code != 0 and code not in STATUS_ECHO_CODES
        ]
        if req.op == "clear_errors":
            err0 = before.error_code if before is not None else 0
            warn0 = before.warn_code if before is not None else 0
            if failed:
                return False, "; ".join(failed)
            if after is not None and after.error_code != 0:
                title = controller_error_title(after.error_code)
                return False, f"{title} re-latched right after clearing"
            if err0 or warn0:
                parts = []
                if err0:
                    parts.append(controller_error_title(err0))
                if warn0:
                    parts.append(f"controller warning {warn0}")
                return True, "cleared " + " and ".join(parts)
            return True, "no controller error or warning was latched; clean_error + clean_warn sent"
        if req.op == "set_collision_sensitivity":
            # set_collision_sensitivity returns the RAW uxbus reply too (x3/xarm.py:964,
            # no _check_code), so a latched box echoes 1 / 2 / 9 for a write that went
            # through: like clear_errors, the READ-BACK decides and the code is diagnosis.
            level = req.level
            assert level is not None  # checked in maintenance()
            if failed:
                return False, "; ".join(failed)
            echo = "; ".join(f"{n} returned {c}" for n, c in codes.items() if c != 0)
            if after is None:
                return False, (
                    f"collision sensitivity {level} written but not verified"
                    + (f" ({echo}: status echo)" if echo else "")
                )
            if after.collision_sensitivity != level:
                return False, (
                    f"collision sensitivity still reads {after.collision_sensitivity} "
                    f"after writing {level}" + (f" ({echo})" if echo else "")
                )
            notes: list[str] = []
            if before is not None and before.collision_sensitivity is not None:
                notes.append(f"was {before.collision_sensitivity}")
            if req.driver_cfg is not None:
                notes.append(
                    f"the config value {req.driver_cfg.collision_sensitivity} is re-applied "
                    "at the next connect"
                )
            else:
                notes.append("the config value is re-applied at the next connect")
            detail = f"collision sensitivity set to {level} ({'; '.join(notes)})"
            if echo:
                detail += f" ({echo}: status echo, value verified by read-back)"
            return True, detail
        # apply_backstops
        cfg = req.driver_cfg
        assert cfg is not None
        matched = (
            after is not None
            and after.collision_sensitivity == cfg.collision_sensitivity
            and after.tcp_load_kg is not None
            and abs(after.tcp_load_kg - cfg.tcp_load_kg) <= TCP_LOAD_MATCH_KG
        )
        # APIState 9 (STATE_NOT_READY) from set_tcp_load while the arm is stopped
        # (state 4/5) is tolerated when the read-back proves the controller stored the
        # value (live 2026-09-04); every other non-zero code is a failure.
        hard = [
            f"{name} returned {code}"
            for name, code in codes.items()
            if code != 0 and not (name == "set_tcp_load" and code == 9 and matched)
        ]
        if hard:
            return False, "; ".join(hard)
        cog = ", ".join(f"{v:g}" for v in cfg.tcp_load_cog_mm)
        detail = (
            f"safety settings applied: sensitivity {cfg.collision_sensitivity}, "
            f"payload {cfg.tcp_load_kg:.2f} kg at ({cog}) mm"
        )
        if cfg.reduced_tcp_boundary_mm is not None:
            detail += ", reduced-mode boundary on"
        if codes.get("set_tcp_load") == 9:
            detail += " (set_tcp_load returned 9: arm stopped, value verified by read-back)"
        if after is not None and not matched:
            detail += (
                f" (read-back not yet reflected: sensitivity {after.collision_sensitivity}, "
                f"payload {after.tcp_load_kg})"
            )
        return True, detail

    def _judge_home_rail(
        self,
        req: _MaintenanceRequest,
        codes: dict[str, int],
        before: ArmMonitorSample | None,
        regs: _RailRegs | None,
    ) -> tuple[bool, str]:
        """REGISTERS ONLY: ok iff the read-back taken right after the writes says
        ``on_zero == 1``, ``is_enabled == 1`` and track ``error == 0``. The SDK
        return codes are reported for diagnosis (100 wait timeout, 101 register
        reads failing, 80 track fault, 81 SCI low) but never decide — SDK 1.18.5
        can return 0 for a homing that did not finish and nonzero for one that did."""
        sdk_note = "; ".join(f"{name} returned {code}" for name, code in codes.items() if code != 0)
        if regs is None:
            return False, (
                "homing writes were sent but the register read-back was skipped; "
                "verify on_zero / is_enabled before using the track"
                + (f" ({sdk_note})" if sdk_note else "")
            )
        present, homed, enabled, pos_m, raw_mm, track_err = regs
        problems: list[str] = []
        if present is not True:
            problems.append("track registers unreadable after homing")
        if homed is not True:
            problems.append("on_zero still 0 (carriage did not reach the zero end)")
        if enabled is not True:
            problems.append("track not enabled")
        if track_err:
            problems.append(f"linear track error {track_err}")
        if problems:
            detail = "rail homing failed: " + ", ".join(problems)
            return False, detail + (f" ({sdk_note})" if sdk_note else "")
        assert req.driver_cfg is not None
        pos = pos_m if pos_m is not None else 0.0
        was_homed = before is not None and before.rail_homed is True
        detail = (
            f"rail {'re-' if was_homed else ''}homed: carriage at {pos:.3f} m (register "
            f"{raw_mm if raw_mm is not None else 0:g} mm), track enabled, positioning speed "
            f"{int(req.driver_cfg.rail_speed_mm_s)} mm/s"
        )
        if sdk_note:
            detail += f" (registers are authoritative; {sdk_note})"
        return True, detail

    # -- one poll ----------------------------------------------------------------
    def _poll(self, gen: int, api: Any, slow: bool) -> None:
        if getattr(api, "connected", True) is False:
            raise ConnectionError("SDK connection lost")
        prev = self._sample
        problems: list[str] = []

        code, angles = api.get_servo_angle(is_radian=True)
        if code != 0 or not angles or len(angles) < 7:
            # no sample this tick: the age-based stale detector reports it
            self._set(gen, "running", f"get_servo_angle returned code {code}")
            return
        q = tuple(float(v) for v in angles[:7])

        code_p, pose = api.get_position(is_radian=True)
        if code_p == 0 and pose is not None and len(pose) >= 6:
            tcp: tuple[float, ...] = (
                units.mm_to_m(pose[0]),
                units.mm_to_m(pose[1]),
                units.mm_to_m(pose[2]),
                float(pose[3]),
                float(pose[4]),
                float(pose[5]),
            )
        else:
            tcp = prev.tcp_pose if prev is not None else ()
            problems.append(f"get_position returned code {code_p}")

        code_e, ew = api.get_err_warn_code()
        if code_e == 0 and ew is not None and len(ew) >= 2:
            err, warn = int(ew[0]), int(ew[1])
        else:
            err, warn = (prev.error_code, prev.warn_code) if prev is not None else (0, 0)
            problems.append(f"get_err_warn_code returned code {code_e}")

        state = _as_int(getattr(api, "state", None))
        mode = _as_int(getattr(api, "mode", None))

        if slow or prev is None:
            rail = self._read_rail(api, problems)
            grip = self._read_gripper(api, prev, problems)
            sens = _as_int(getattr(api, "collision_sensitivity", None))
            load_kg, load_cog = _parse_tcp_load(getattr(api, "tcp_load", None))
        else:
            rail = (
                prev.rail_present,
                prev.rail_homed,
                prev.rail_enabled,
                prev.rail_pos_m,
                prev.rail_raw_mm,
                prev.rail_error,
            )
            grip = (prev.gripper_open_frac, prev.gripper_raw)
            sens, load_kg, load_cog = (
                prev.collision_sensitivity,
                prev.tcp_load_kg,
                prev.tcp_load_cog_mm,
            )

        self._seq += 1
        sample = ArmMonitorSample(
            arm_id=self.arm_id,
            seq=self._seq,
            t_mono=self._clock(),
            q=q,
            tcp_pose=tcp,
            error_code=err,
            warn_code=warn,
            state=state,
            mode=mode,
            rail_present=rail[0],
            rail_homed=rail[1],
            rail_enabled=rail[2],
            rail_pos_m=rail[3],
            rail_raw_mm=rail[4],
            rail_error=rail[5],
            gripper_open_frac=grip[0],
            gripper_raw=grip[1],
            collision_sensitivity=sens,
            tcp_load_kg=load_kg,
            tcp_load_cog_mm=load_cog,
        )
        parts: list[str] = []
        if err != 0:
            parts.append(controller_error_title(err))
        elif warn != 0:
            parts.append(f"controller warning {warn}")
        parts.extend(problems)
        with self._lock:
            if not self._current(gen):
                return  # stopped / handed over meanwhile: publish nothing
            self._sample = sample
            self._status, self._detail = "running", "; ".join(parts)

    def _read_rail(self, api: Any, problems: list[str]) -> _RailRegs:
        """(present, homed, enabled, pos_m, raw_mm, error); pos_m only when homed AND enabled."""
        if not self.expect_rail:
            return (None, None, None, None, None, None)
        code, regs = api.get_linear_track_registers()
        if code != 0:
            return (False, None, None, None, None, None)  # 3 timeout / 20 host id / 23 length
        if not isinstance(regs, Mapping) or "pos" not in regs:
            return (False, None, None, None, None, None)  # simulation-mode controller: (0, [])
        homed = int(regs.get("on_zero", 0) or 0) == 1
        enabled = int(regs.get("is_enabled", 0) or 0) == 1
        raw_mm = float(regs.get("pos", 0) or 0)
        track_err = int(regs.get("error", 0) or 0)
        if track_err:
            problems.append(f"linear track error {track_err}")
        pos_m = units.rail_mm_to_m(raw_mm) if (homed and enabled) else None
        return (True, homed, enabled, pos_m, raw_mm, track_err)

    def _read_gripper(
        self, api: Any, prev: ArmMonitorSample | None, problems: list[str]
    ) -> tuple[float | None, float | None]:
        """(open_frac, raw) via the SAME SDK call + conversion as grippers.py."""
        if self.gripper == "none":
            return (None, None)
        if self.gripper == "xarm_g2":
            code, mm = api.get_gripper_g2_position()  # == G2Gripper.poll()
            if code == 0 and mm is not None:
                return (units.g2_mm_to_frac(mm), float(mm))
            problems.append(f"get_gripper_g2_position returned code {code}")
        else:
            code, pulse = api.get_gripper_position()  # == ClassicGripper.poll()
            if code == 0 and pulse is not None:
                return (units.pulse_to_frac(pulse), float(pulse))
            problems.append(f"get_gripper_position returned code {code}")
        if prev is not None:
            return (prev.gripper_open_frac, prev.gripper_raw)
        return (None, None)


def _quiet_disconnect(api: Any) -> None:
    """``api.disconnect()`` that never raises (release paths); None is a no-op."""
    if api is None:
        return
    try:
        api.disconnect()
    except Exception:  # noqa: BLE001 — release never raises
        pass


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tcp_load(value: Any) -> tuple[float | None, tuple[float, ...]]:
    """SDK ``XArmAPI.tcp_load`` -> ``(kg, (x, y, z) mm)``; ``(None, ())`` when unreadable.

    The property is ``[weight, [x, y, z]]`` (``x3/base.py:1784``: mm for
    controller fw >= 0.2.1, which every lab box exceeds)."""
    try:
        weight, cog = value[0], value[1]
        kg = float(weight)
        cog_t = tuple(float(v) for v in cog)
    except (TypeError, ValueError, IndexError, KeyError):
        return (None, ())
    if len(cog_t) != 3:
        cog_t = ()
    return (kg, cog_t)
