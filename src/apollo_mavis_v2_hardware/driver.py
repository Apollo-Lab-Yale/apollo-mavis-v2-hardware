"""XArmDriver: core ArmInterface over the xArm SDK (02-hardware §3).

One driver per arm; one ``XArmAPI(ip, is_radian=True, report_type='real')``
per driver (never shared across arms/processes). ``XArmAPI`` is imported ONLY
here (injected everywhere else via ``api_factory``).

Phase-12 (02-hardware §4 additive; 14-dora §4.2 "Arm states without a
session"): ``connect(readonly=True)`` opens the same report stream but NEVER
writes to the control box — no enable, no mode/state change, no servo stream,
no gripper / rail writes, no error clearing; ``READONLY_ALLOWED_SDK_METHODS``
is the complete SDK surface that path may call. The runtime's ``IdleArmReader``
holds such a connection to each box between sessions.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from apollo_mavis_v2_core import (
    ArmConnectError,
    ArmIdentityError,
    ArmInterface,
    ArmState,
    BringupError,
    CommandError,
    GripperCommand,
    GripperState,
    Pose,
    RailExpectedError,
    RailUnavailableError,
)

from . import units
from .backstops import apply_backstops
from .config import ServoLimits, XArmDriverConfig
from .events import (
    DriverEvent,
    FaultEvent,
    GripperFaultEvent,
    RailEvent,
    RecoveredEvent,
    ReseedEvent,
    StaleEvent,
    StudioConflictWarning,
    TickStats,
)
from .grippers import GripperBackend, make_gripper, read_fw_tuple
from .rail import RailController, RailHomeOutcome

IS_REAL_READBACK_FW = (1, 9, 110)  # get_servo_angle(is_real=True) gate


class ArmFaultedError(CommandError):
    """Command sent while the driver is LATCHED (02-hardware §3.5)."""


class DriverPhase(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    READY = "ready"  # mode 0
    STREAMING = "streaming"  # mode 1
    FAULT = "fault"
    RECOVERING = "recovering"
    LATCHED = "latched"
    # phase-12: connected with connect(readonly=True) — report stream + read-only
    # polling only; the box is never enabled, never commanded, never cleared
    READONLY = "readonly"


class FaultKind(Enum):
    RECOVERABLE = "recoverable"
    UNRECOVERABLE = "unrecoverable"
    EXTERNAL = "external"
    RAIL_ONLY = "rail_only"


# Controller error classification (02-hardware §3.5)
RECOVERABLE_ERRORS = frozenset({22, 23, 24, 25, 31, 35})
UNRECOVERABLE_ERRORS = frozenset({1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 19, 28, 110})
RAIL_COMMS_ERROR = 111  # latches only the rail; the arm keeps streaming
RECOVERY_BUDGET = 3  # recoveries per rolling window, else LATCHED
RECOVERY_WINDOW_S = 30.0
C24_BACKOFF_S = 10.0  # halved vel/acc after a C24 recovery
C24_ERROR = 24

# Controller STATE codes (SDK ``XArmAPI.state``): 0 ready, 1 in motion, 2 standby
# ("sleeping"), 3 paused, 4 stopped, 5 stopped/collision-halt, 6 decelerating to a
# stop. The SDK's own readiness rule is ``ready = state not in (4, 5)``
# (``x3/base.py`` ``__handle_report_real`` and the normal/rich handlers all do
# ``if state in [4, 5]: self._is_ready = False else: ... = True``).
#
# **A healthy mode-1 arm that is not executing a servo step reports state 2.**
# Both lab boxes sit in ``mode 1 state 2`` for the whole session (2026-09-05) —
# the servo stream re-sends the held posture, which is not "motion". Treating
# anything outside {0, 1} as an external grab therefore fired the Studio-conflict
# detector ~0.5 s after every connect and every recovery and latched BOTH arms
# with "external mode/state conflict persisted" while no Studio was running.
SERVO_HEALTHY_STATES = frozenset({0, 1, 2})
# States that mean somebody else stopped/paused the arm under us (no error code):
# 3 paused, 4/5 stopped, 6 decelerating. Undocumented codes are NOT treated as a
# conflict — a detector that fires on an unrecognised state is what caused the
# 2026-09-05 false-positive latch; real trouble also surfaces as an error code or
# as a non-zero ``set_servo_angle_j`` return.
SERVO_CONFLICT_STATES = frozenset({3, 4, 5, 6})

STATE_NOT_READY_CODE = 9  # xarm ``APIState``/``UxbusState.STATE_NOT_READY``
# Entering servo mode is not instantaneous: after ``set_mode(1); set_state(0)`` the
# control box needs tens of ms before ``move_servoj`` is accepted (its TCP replies
# carry "not ready", bit 0x10 -> the SDK maps the move to APIState 9). A blind
# ``sleep(0.1)`` raced it on the real boxes (2026-09-05: the FIRST servo tick of the
# very first hardware session returned 9 and faulted the Perception Arm during
# bring-up), so the driver polls for a healthy state instead, and the streamer
# tolerates code 9 for a bounded grace right after it resumes.
SERVO_READY_TIMEOUT_S = 1.5
SERVO_READY_POLL_S = 0.02
SERVO_READY_MAX_POLLS = int(SERVO_READY_TIMEOUT_S / SERVO_READY_POLL_S) + 2
SERVO_NOT_READY_GRACE_S = 0.3

# ---------------------------------------------------------------------------
# READ-ONLY CONNECTION ALLOWLIST (phase-12): the COMPLETE set of XArmAPI methods
# ``connect(readonly=True)`` + its poll thread + get_state() / stop() /
# disconnect() may call. Everything a driving session writes — clean_*,
# motion_enable, set_mode / set_state, set_servo_angle_j, the backstop set_*,
# every gripper / linear-track set_* and save_conf — is absent by construction;
# tests/test_driver_readonly.py asserts the fake's call log stays inside this set.
# ``get_linear_track_registers`` / ``get_gripper_*_position`` are the same reads
# as the read-only monitor's (``monitor.READ_ONLY_SDK_METHODS``). The joint
# angles come from the 30003 report stream (``register_report_callback`` is a
# subscription on the SDK's own socket, not a controller write), so no
# ``get_servo_angle`` polling is needed.
# ---------------------------------------------------------------------------
READONLY_ALLOWED_SDK_METHODS: frozenset[str] = frozenset(
    {
        "disconnect",
        "register_report_callback",  # 30003 push subscription (joints, tcp, state)
        "get_err_warn_code",  # [error_code, warn_code] — the stream carries none
        "get_linear_track_registers",  # {pos, status, error, is_enabled, on_zero}
        "get_gripper_position",  # classic: pulses — same read as ClassicGripper.poll()
        "get_gripper_g2_position",  # G2: int mm — same read as G2Gripper.poll()
    }
)
# XArmAPI attributes the read-only path reads (SDK properties fed by its threads).
READONLY_ALLOWED_SDK_ATTRS: frozenset[str] = frozenset(
    {"connected", "sn", "version", "version_number", "mode", "joints_torque"}
)


def classify_error(err: int) -> FaultKind:
    """Map a controller error code to a recovery class. Unknown codes latch
    (conservative: only the explicitly recoverable list auto-resumes)."""
    if err == RAIL_COMMS_ERROR:
        return FaultKind.RAIL_ONLY
    if err in RECOVERABLE_ERRORS:
        return FaultKind.RECOVERABLE
    if err == 0:
        return FaultKind.EXTERNAL  # fault without a controller error = external actor
    return FaultKind.UNRECOVERABLE


@dataclass(frozen=True)
class _StateSnap:
    """Written only by the report callback; read by get_state() (ref swap)."""

    q: np.ndarray  # (7,) rad, actual_joint_angle
    dq: np.ndarray  # (7,) rad/s, finite-diff + EMA(alpha=0.5)
    ee_pose_sdk: tuple[float, ...]  # actual_tcp_pose[6], mm + rad, base frame
    tau: np.ndarray  # (7,) N*m
    mode: int
    state: int
    cmd_num: int
    mono_ts: float
    wallclock_ns: int


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of the last recovery sequence (auto or user-initiated); readable
    without consuming the event stream (``XArmDriver.recovery_result()``)."""

    seq: int  # increases with every completed sequence
    ok: bool  # True: streaming resumed (RecoveredEvent emitted); False: LATCHED
    error_code: int  # controller error captured at the start of the sequence
    detail: str = ""  # latch reason when not ok
    user_initiated: bool = False
    t_mono: float = 0.0


class _ServoStreamer:
    """Dedicated 100 Hz mode-1 streaming thread for one arm (§3.3).

    Mode 1 has no firmware smoothing and executes only the last instruction —
    this thread owns ALL velocity/accel limiting. After stalls it re-anchors
    and NEVER bursts catch-up ticks (catch-up = velocity spike = C24).
    """

    def __init__(
        self,
        api: Any,
        limits: ServoLimits,
        on_fault: Callable[[str, int], None],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "hw.arm.servo",
    ) -> None:
        self._api = api
        self._limits = limits
        self._on_fault = on_fault
        self._clock = clock
        self._sleep = sleep
        self._name = name
        self._dt = 1.0 / limits.rate_hz
        self._vel_step = np.asarray(limits.max_joint_vel, dtype=np.float64) * self._dt
        self._acc_step = np.asarray(limits.max_joint_acc, dtype=np.float64) * self._dt**2
        self._lever = np.asarray(limits.lever_arm_m, dtype=np.float64)
        lims = np.asarray(limits.joint_limits_rad, dtype=np.float64)
        self._q_lo = lims[:, 0] + limits.joint_limit_margin_rad
        self._q_hi = lims[:, 1] - limits.joint_limit_margin_rad
        self._lock = threading.Lock()
        self._target: np.ndarray | None = None  # latest-wins
        self._last_sent = np.zeros(7)
        self._prev_dq = np.zeros(7)
        self._scale = 1.0  # C24 backoff halves this for 10 s
        self._paused = True
        self._running = False
        self._thread: threading.Thread | None = None
        self._periods: deque[float] = deque(maxlen=2000)
        self._ticks = 0
        self._late_ticks = 0
        self._faults = 0
        # bounded STATE_NOT_READY tolerance right after resume() (see
        # SERVO_NOT_READY_GRACE_S): the control box may still be entering servo mode.
        # Bounded BOTH ways — wall time and tick count — so a stalled or frozen clock
        # cannot turn the window into "swallow code 9 forever".
        self._grace_until = 0.0
        self._grace_ticks = 0
        self._grace_ticks_max = max(1, int(SERVO_NOT_READY_GRACE_S * limits.rate_hz))
        self._not_ready_ticks = 0

    # -- control ------------------------------------------------------------
    def set_target(self, q7: np.ndarray) -> None:
        with self._lock:
            self._target = np.array(q7, dtype=np.float64)

    def reseed(self, q7: np.ndarray) -> None:
        """target := last_sent := q7, zero velocity (post-recovery re-anchor)."""
        with self._lock:
            q = np.array(q7, dtype=np.float64)
            self._target = q.copy()
            self._last_sent = q.copy()
            self._prev_dq = np.zeros(7)

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._prev_dq = np.zeros(7)

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            # the box may still be entering servo mode: tolerate APIState 9 briefly
            self._grace_until = self._clock() + SERVO_NOT_READY_GRACE_S
            self._grace_ticks = self._grace_ticks_max

    def set_scale(self, scale: float) -> None:
        with self._lock:
            self._scale = float(scale)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    @property
    def last_sent(self) -> np.ndarray:
        with self._lock:
            return self._last_sent.copy()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def not_ready_ticks(self) -> int:
        """Servo ticks swallowed as STATE_NOT_READY inside the post-resume grace
        window (diagnostics; a non-zero count is normal right after mode entry)."""
        return self._not_ready_ticks

    @property
    def stats(self) -> TickStats:
        periods = tuple(self._periods)
        ordered = sorted(periods)
        n = len(ordered)
        return TickStats(
            ticks=self._ticks,
            late_ticks=self._late_ticks,
            faults=self._faults,
            p50_s=ordered[n // 2] if n else 0.0,
            p99_s=ordered[min(n - 1, int(n * 0.99))] if n else 0.0,
            periods_s=periods,
        )

    # -- tick loop ------------------------------------------------------------
    def _run(self) -> None:
        dt = self._dt
        next_t = self._clock() + dt
        last_send_t: float | None = None
        while self._running:
            self._sleep(max(0.0, next_t - self._clock()))
            now = self._clock()
            if now - next_t > 2 * dt:
                # re-anchor after stalls: NEVER burst — catch-up ticks are a
                # velocity spike on the wire = C24
                next_t = now
                self._late_ticks += 1
            next_t += dt
            with self._lock:
                if self._paused:
                    last_send_t = None
                    continue
                target = self._target if self._target is not None else self._last_sent
                last_sent = self._last_sent
                prev_dq = self._prev_dq
                scale = self._scale
                in_grace = now < self._grace_until and self._grace_ticks > 0
            dq = np.clip(target - last_sent, -self._vel_step * scale, self._vel_step * scale)
            dq = np.clip(dq, prev_dq - self._acc_step * scale, prev_dq + self._acc_step * scale)
            cart_est = float(np.sum(np.abs(dq) * self._lever))  # conservative TCP bound
            if cart_est > self._limits.max_cart_step_m:
                dq = dq * (self._limits.max_cart_step_m / cart_est)
            q_cmd = np.clip(last_sent + dq, self._q_lo, self._q_hi)
            code = self._api.set_servo_angle_j(list(q_cmd), is_radian=True)
            if last_send_t is not None:
                self._periods.append(now - last_send_t)
            last_send_t = now
            if code == 0:
                self._ticks += 1
                with self._lock:
                    self._last_sent = q_cmd
                    self._prev_dq = q_cmd - last_sent
            elif int(code) == STATE_NOT_READY_CODE and in_grace:
                # the box has not finished entering servo mode: retry next tick,
                # do NOT advance _last_sent (nothing moved) and do NOT fault yet.
                # Past the grace window a 9 faults like any other bad return.
                self._not_ready_ticks += 1
                with self._lock:
                    self._grace_ticks -= 1
            else:
                self._faults += 1
                self.pause()  # §3.5 takes over on the monitor thread
                self._on_fault("servo", int(code))


def _default_api_factory(*args: Any, **kwargs: Any) -> Any:
    from xarm.wrapper import XArmAPI  # imported ONLY here (02-hardware §1)

    return XArmAPI(*args, **kwargs)


def _api_mode(api: Any) -> int:
    """Controller mode from the SDK's ``mode`` property (kept current by the
    SDK report thread from the 30003 ``state_mode`` byte).

    The report-callback PAYLOAD never carries ``mode`` (SDK 1.18.5
    ``x3/base.py:1284-1300``; only ``register_mode_changed_callback`` does).
    Reading ``data["mode"]`` yielded 0 and tripped the Studio-conflict detector
    ~1.2 s after every connect (fixed 2026-09-04).
    """
    try:
        return int(getattr(api, "mode", 0) or 0)
    except (TypeError, ValueError):
        return 0


class XArmDriver(ArmInterface):
    """core.ArmInterface over one xArm7 control box (02-hardware §3)."""

    def __init__(
        self,
        cfg: XArmDriverConfig,
        api_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self._api_factory = api_factory or _default_api_factory
        self._clock = clock
        self._sleep = sleep
        self._api: Any = None
        self._phase = DriverPhase.IDLE
        self._phase_lock = threading.Lock()
        self._streamer: _ServoStreamer | None = None
        self._monitor: _MonitorThread | None = None
        self._poller: _ReadonlyPoller | None = None  # connect(readonly=True) only
        self._readonly = False
        self._readonly_poll_ok = True  # False after a failed read-only poll -> stale
        self._rail: RailController | None = None
        self._gripper: GripperBackend | None = None
        self._has_rail = False
        self._fw: tuple[int, int, int] = (0, 0, 0)
        self._snap: _StateSnap | None = None  # ref-swapped by the report callback
        self._err_code = 0  # monitor cache (30003 carries no err/warn)
        self._warn_code = 0
        self._gripper_state = GripperState(open_frac=1.0)
        self._gripper_cmd_lock = threading.Lock()
        self._gripper_cmd: GripperCommand | None = None  # latest-wins queue
        self._events: deque[DriverEvent] = deque(maxlen=256)
        self._events_lock = threading.Lock()
        self._recovery_lock = threading.Lock()
        self._recovery_times: deque[float] = deque()
        self._c24_backoff_until: float | None = None
        self._fault_pending = False
        self._fault_code = 0
        self._fault_source = "user"
        self._user_recovery_pending = False  # request_recovery() -> monitor thread
        self._recovery_seq = 0
        self._recovery_result: RecoveryResult | None = None
        self._latch_reason = ""
        self._was_stale = True
        self._external_retry_at: float | None = None
        self._mode_set_at = -1e9  # grace window: reports may lag our set_mode(1)
        self._connect_warnings: list[str] = []
        lims = np.asarray(cfg.servo.joint_limits_rad, dtype=np.float64)
        self._q_lo = lims[:, 0] + cfg.servo.joint_limit_margin_rad
        self._q_hi = lims[:, 1] - cfg.servo.joint_limit_margin_rad

    # -- properties -----------------------------------------------------------
    @property
    def phase(self) -> DriverPhase:
        return self._phase

    @property
    def readonly(self) -> bool:
        """True when this driver was (last) opened with ``connect(readonly=True)``:
        it publishes state but never writes — every ``command_*`` raises
        ``CommandError`` and ``stop()`` / ``disconnect()`` skip the D6 writes."""
        return self._readonly

    @property
    def dof(self) -> int:
        return 8 if self._has_rail else 7

    @property
    def has_rail(self) -> bool:
        return self._has_rail

    @property
    def gripper_force_capable(self) -> bool:
        return self._gripper.force_capable if self._gripper else self.cfg.gripper == "xarm_g2"

    @property
    def tick_stats(self) -> TickStats:
        return self._streamer.stats if self._streamer else TickStats()

    @property
    def sn(self) -> str | None:
        return getattr(self._api, "sn", None) if self._api is not None else None

    @property
    def fw_version(self) -> str | None:
        return ".".join(str(v) for v in self._fw) if self._api is not None else None

    @property
    def gripper_kind(self) -> str:
        return self._gripper.kind if self._gripper is not None else self.cfg.gripper

    @property
    def rail_phase(self) -> str | None:
        return self._rail.phase.name if self._rail is not None else None

    @property
    def rail_position_known(self) -> bool:
        """True when the published rail slot is a MEASUREMENT: the track was seen
        homed + enabled (connect gate, a successful :meth:`home_rail`, or — on a
        read-only connection — the last register poll). False for an arm without
        a rail and for a track connected with ``rail_homing == "allow_unhomed"``
        (or read-only) before it was homed + enabled — then ``get_state().q[7]
        == rail_pos_m == 0.0`` is a placeholder (core's ``ArmState`` needs a
        finite ``q[7]``; consumers MUST check this flag)."""
        return self._rail is not None and self._rail.pos_known

    @property
    def connect_warnings(self) -> list[str]:
        return list(self._connect_warnings)

    def drain_events(self) -> list[DriverEvent]:
        """Bounded-deque pickup of every DriverEvent (FaultEvent / RecoveredEvent /
        ReseedEvent / StudioConflictWarning / RailEvent / GripperFaultEvent /
        StaleEvent) since the previous call; the runtime drains once per tick."""
        with self._events_lock:
            out = list(self._events)
            self._events.clear()
        return out

    def recovery_result(self) -> RecoveryResult | None:
        """Outcome of the most recent recovery sequence (None before the first);
        lets a waiter (REST) observe success/latch without draining the events
        the control loop consumes."""
        return self._recovery_result

    def _emit(self, event: DriverEvent) -> None:
        with self._events_lock:
            self._events.append(event)

    # -- bring-up (§3.2) -------------------------------------------------------
    def connect(self, readonly: bool = False) -> None:
        """Bring the arm up for a driving session (default) or, with
        ``readonly=True`` (phase-12), open a state-only connection — see
        :meth:`_connect_readonly`. The default path is unchanged."""
        cfg = self.cfg
        self._readonly = bool(readonly)
        if readonly:
            self._connect_readonly()
            return
        self._phase = DriverPhase.CONNECTING
        api = self._connect_api()
        self._api = api
        # 2. identity (cabling swaps) + fw gates
        sn = getattr(api, "sn", None)
        if cfg.expected_sn is not None and sn != cfg.expected_sn:
            self._phase = DriverPhase.IDLE
            raise ArmIdentityError(
                "identity", f"{cfg.arm_id}: sn={sn!r}, expected {cfg.expected_sn!r}"
            )
        self._fw = read_fw_tuple(api)  # api.version_number, NOT the raw api.version string
        # 3a. clear latched state + backstops (volatile settings, the same writes
        #     as the session-less apply_backstops maintenance op)
        api.clean_warn()
        api.clean_error()
        self._connect_warnings = apply_backstops(api, cfg)
        # 4. rail detection fixes dof; an unhomed / unverifiable track REFUSES the
        #    connect (phase-09c: the driver never homes — see rail.require_homed).
        #    Evaluated BEFORE motion_enable: register reads need no enable, so a
        #    refusal leaves the arm exactly as found (state 4, brakes engaged) and
        #    the contract's "zero writes to the arm on an unhomed refusal" holds.
        rail = RailController(api, cfg.rail_speed_mm_s, arm_id=cfg.arm_id)
        if cfg.expect_rail == "no":
            self._has_rail = False
        else:
            detected = rail.detect()
            if cfg.expect_rail == "yes" and not detected:
                self._phase = DriverPhase.IDLE
                raise RailExpectedError("rail", f"{cfg.arm_id}: expected rail not detected")
            self._has_rail = detected
        self._connect_warnings.extend(rail.warnings)  # e.g. SN unverifiable (SDK 1.18.5)
        if self._has_rail:
            n_warnings = len(rail.warnings)
            try:
                # RailNotHomedError (on_zero 0, unless cfg.rail_homing == "allow_unhomed":
                # phase-09d maintenance motion, the track stays DETECTED and the rail slot
                # is a 0.0 placeholder until home_rail()) or BringupError("rail", ...)
                # (register read / enable / speed failed); NEVER set_linear_track_back_origin
                rail.require_homed(allow_unhomed=cfg.rail_homing == "allow_unhomed")
            except BringupError:
                self._phase = DriverPhase.IDLE
                raise
            self._connect_warnings.extend(rail.warnings[n_warnings:])  # "position unknown"
            self._rail = rail
        # 3b. enable (required order: motion_enable -> set_mode -> set_state(0))
        api.motion_enable(True)
        api.set_mode(0)
        api.set_state(0)  # set_state(0) must follow every set_mode
        self._phase = DriverPhase.READY
        # 5. gripper backend + report callback
        self._gripper = make_gripper(cfg.gripper, self._clock)
        self._gripper.init(api, self._fw)  # raises GripperInitError
        # SDK 1.18.5 keywords ONLY (``xarm/wrapper/xarm_api.py:2222``): there is
        # no ``report_mode`` — passing it raised TypeError on the first real
        # connect (fixed 2026-09-04); mode comes from ``api.mode`` in _on_report.
        api.register_report_callback(
            self._on_report,
            report_cartesian=True,
            report_joints=True,
            report_state=True,
            report_error_code=False,  # 30003 carries none; the monitor polls them
            report_warn_code=False,
            report_mtable=False,
            report_mtbrake=False,
            report_cmd_num=True,
        )
        # 6. enter streaming; seed from the measured position
        self._enter_servo_mode()
        q_seed = self._read_measured_q()
        self._streamer = _ServoStreamer(
            api,
            cfg.servo,
            self._on_fault,
            self._clock,
            self._sleep,
            name=f"hw.{cfg.arm_id}.servo",
        )
        self._streamer.reseed(q_seed)
        self._streamer.resume()
        self._streamer.start()
        self._monitor = _MonitorThread(self)
        self._monitor.start()
        self._phase = DriverPhase.STREAMING

    def _connect_readonly(self) -> None:
        """``connect(readonly=True)`` (02-hardware §4 additive; 14-dora §4.2).

        Same ``XArmAPI(..., report_type='real', enable_report=True)`` client and
        identity / firmware reads as a normal connect, then ONLY:
        ``register_report_callback`` (joints, tcp pose, state at 100 Hz -> the
        same ``_StateSnap`` ``get_state()`` reads today), one read of the
        linear-track registers (presence fixes ``dof``; ``rail_position_known``
        iff ``on_zero == 1 and is_enabled == 1``, else the rail slot publishes the
        0.0 placeholder like the phase-09d ``allow_unhomed`` case — never
        ``require_homed``, never enable / speed / home), one read of the gripper
        opening (``get_gripper_position`` / ``get_gripper_g2_position``, mapped
        like the backends; the classic backend's ``init`` writes are skipped) and
        ``get_err_warn_code``; then a read-only poll thread at
        ``cfg.monitor_rate_hz`` refreshes those three. Phase ``READONLY``.

        NOT called, by construction: ``clean_warn`` / ``clean_error``,
        ``apply_backstops``, ``motion_enable``, ``set_mode`` / ``set_state``,
        ``set_servo_angle_j`` (no ``_ServoStreamer``), no ``_MonitorThread`` (its
        recovery path writes), no gripper / track ``set_*``. A controller error
        is REPORTED (``ArmState.error_code``), never recovered. The complete
        callable surface is :data:`READONLY_ALLOWED_SDK_METHODS`. ``expected_sn``
        and ``expect_rail == "yes"`` refuse exactly as the driving connect does.
        """
        cfg = self.cfg
        self._phase = DriverPhase.CONNECTING
        api = self._connect_api()
        self._api = api
        sn = getattr(api, "sn", None)
        if cfg.expected_sn is not None and sn != cfg.expected_sn:
            self._phase = DriverPhase.IDLE
            raise ArmIdentityError(
                "identity", f"{cfg.arm_id}: sn={sn!r}, expected {cfg.expected_sn!r}"
            )
        self._fw = read_fw_tuple(api)
        self._connect_warnings = []
        # SDK 1.18.5 keywords only (see connect()); a subscription on the SDK's own
        # report socket, not a controller write
        api.register_report_callback(
            self._on_report,
            report_cartesian=True,
            report_joints=True,
            report_state=True,
            report_error_code=False,
            report_warn_code=False,
            report_mtable=False,
            report_mtbrake=False,
            report_cmd_num=True,
        )
        # rail: ONE register read (detect keeps the dict) -> dof + position bookkeeping;
        # never require_homed (the gate enables + sets the speed on a homed track)
        rail = RailController(api, cfg.rail_speed_mm_s, arm_id=cfg.arm_id)
        if cfg.expect_rail == "no":
            self._has_rail = False
        else:
            detected = rail.detect()
            if cfg.expect_rail == "yes" and not detected:
                self._phase = DriverPhase.IDLE
                raise RailExpectedError("rail", f"{cfg.arm_id}: expected rail not detected")
            self._has_rail = detected
        self._connect_warnings.extend(rail.warnings)
        if self._has_rail:
            rail.observe_registers(rail.last_registers)
            if not rail.pos_known:
                self._connect_warnings.append(
                    f"{cfg.arm_id}: linear track not homed + enabled: carriage position "
                    "UNKNOWN - rail slot reads 0.000 m as a placeholder (read-only connection)"
                )
            self._rail = rail
        # gripper opening + error codes once, so the first get_state() is complete
        self._gripper = None  # no backend: its init()/poll() write (enable, clean)
        self._read_gripper_readonly()
        self._read_err_warn_readonly()
        self._readonly_poll_ok = True
        self._poller = _ReadonlyPoller(self)
        self._poller.start()
        self._phase = DriverPhase.READONLY

    def _read_err_warn_readonly(self) -> None:
        code, ew = self._api.get_err_warn_code()
        if code == 0 and ew is not None and len(ew) >= 2:
            self._err_code, self._warn_code = int(ew[0]), int(ew[1])

    def _read_gripper_readonly(self) -> None:
        """Gripper opening via the SAME read + conversion as the backends' poll()
        (and the read-only monitor), without the backends' writes."""
        kind = self.cfg.gripper
        if kind == "none":
            return
        if kind == "xarm_g2":
            code, mm = self._api.get_gripper_g2_position()
            if code == 0 and mm is not None:
                self._gripper_state = GripperState(open_frac=units.g2_mm_to_frac(mm))
        else:
            code, pulse = self._api.get_gripper_position()
            if code == 0 and pulse is not None:
                self._gripper_state = GripperState(open_frac=units.pulse_to_frac(pulse))

    def _read_rail_readonly(self) -> None:
        """Refresh the rail bookkeeping from the registers; a failed read = unknown."""
        rail = self._rail
        if rail is None:
            return
        code, regs = self._api.get_linear_track_registers()
        rail.observe_registers(regs if code == 0 else None)

    def _connect_api(self) -> Any:
        last_exc: Exception | None = None
        for attempt in range(3):  # retry 3x, 2 s apart
            if attempt:
                self._sleep(2.0)
            try:
                api = self._api_factory(
                    self.cfg.ip,
                    is_radian=True,
                    report_type="real",
                    enable_report=True,
                    check_joint_limit=True,
                )
            except Exception as exc:  # noqa: BLE001 — SDK raises bare Exceptions
                last_exc = exc
                continue
            if getattr(api, "connected", True):
                return api
        self._phase = DriverPhase.IDLE
        raise ArmConnectError(
            "connect", f"{self.cfg.arm_id} ({self.cfg.ip}): {last_exc or 'not connected'}"
        )

    def _await_servo_ready(self) -> int:
        """Block (≤ :data:`SERVO_READY_TIMEOUT_S`) until the control box reports a
        healthy servo state, then return that state.

        Read-only polling: ``get_state()`` is a plain 502 read whose reply also
        refreshes the SDK's "ready to move" flag, so the first
        ``set_servo_angle_j`` is not the thing that discovers the box was still
        entering servo mode (2026-09-05: a fixed 0.1 s sleep raced it and the
        first tick of the first hardware session faulted the arm with APIState 9).
        Returns the last state seen — the caller does NOT fault on a timeout, the
        streamer's bounded grace window and its normal fault path cover that.
        """
        api = self._api
        deadline = self._clock() + SERVO_READY_TIMEOUT_S
        state = -1
        # bounded by BOTH the deadline and the poll count: this runs inside connect()
        # and recovery, so it must never be able to hang on a clock that stands still
        for _ in range(SERVO_READY_MAX_POLLS):
            try:
                code, value = api.get_state()
                if code == 0:
                    state = int(value)
            except Exception:  # noqa: BLE001 — a read failure must not mask mode entry
                pass
            if state in SERVO_HEALTHY_STATES or self._clock() >= deadline:
                return state
            self._sleep(SERVO_READY_POLL_S)
        return state

    def _enter_servo_mode(self) -> int:
        """``set_mode(1)`` + ``set_state(0)`` + wait for readiness (§3.3).

        The single place the driver enters mode 1 (connect, recovery, external
        re-grab). Returns the controller state it settled on.
        """
        api = self._api
        api.set_mode(1)
        api.set_state(0)
        self._mode_set_at = self._clock()
        return self._await_servo_ready()

    def _read_measured_q(self) -> np.ndarray:
        """Measured joints; is_real needs fw >= 1.9.110, plain fallback otherwise."""
        api = self._api
        if self._fw >= IS_REAL_READBACK_FW:
            code, q = api.get_servo_angle(is_real=True)
            if code == 0:
                return np.asarray(q[:7], dtype=np.float64)
        code, q = api.get_servo_angle()
        if code != 0:
            raise ArmConnectError("seed", f"get_servo_angle returned {code}")
        return np.asarray(q[:7], dtype=np.float64)

    # -- state (§3.4) ----------------------------------------------------------
    def _on_report(self, data: dict[str, Any]) -> None:
        """SDK report-thread callback (100 Hz push on 30003); writes _StateSnap."""
        joints = data.get("joints")
        if not joints:
            return
        q = np.asarray(joints[:7], dtype=np.float64)
        now = self._clock()
        prev = self._snap
        if prev is not None and now > prev.mono_ts:
            raw_dq = (q - prev.q) / (now - prev.mono_ts)
            dq = 0.5 * raw_dq + 0.5 * prev.dq  # EMA alpha=0.5 (30003 has no velocities)
        else:
            dq = np.zeros(7)
        tau_raw = getattr(self._api, "joints_torque", None)
        tau = np.asarray(tau_raw[:7], dtype=np.float64) if tau_raw is not None else np.zeros(7)
        cart = data.get("cartesian") or [0.0] * 6
        self._snap = _StateSnap(  # single reference assignment = GIL-atomic swap
            q=q,
            dq=dq,
            ee_pose_sdk=tuple(float(v) for v in cart[:6]),
            tau=tau,
            mode=_api_mode(self._api),  # payload has no "mode" key (SDK 1.18.5)
            state=int(data.get("state", 0)),
            cmd_num=int(data.get("cmdnum", 0)),
            mono_ts=now,
            wallclock_ns=time.time_ns(),
        )

    def get_state(self) -> ArmState:
        """Lock-free latest snapshot; never blocks."""
        snap = self._snap
        now = self._clock()
        rail = self._rail if self._has_rail else None
        rail_pos = rail.pos_m if rail is not None else None
        if snap is None:  # nothing received yet: degraded-but-valid, stale=True
            q7, dq7 = np.zeros(7), np.zeros(7)
            ee = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
            mode, state, mono_ts, wall = 0, 4, 0.0, 0
            stale = True
        else:
            q7, dq7 = snap.q, snap.dq
            ee = units.sdk_to_pose(snap.ee_pose_sdk)
            mode, state = snap.mode, snap.state
            mono_ts, wall = snap.mono_ts, snap.wallclock_ns
            stale = (now - snap.mono_ts) > self.cfg.stale_after_s
        if not self._readonly_poll_ok:
            stale = True  # read-only poll failed / link lost (phase-12): never trust it
        if rail is not None:
            q = np.concatenate([q7, [rail_pos]])
            dq = np.concatenate([dq7, [0.0]])  # rail velocity unobservable
        else:
            q, dq = q7, dq7
        return ArmState(
            arm_id=self.cfg.arm_id,
            q=q,
            dq=dq,
            ee_pose=ee,
            gripper=self._gripper_state,
            rail_pos_m=rail_pos,
            error_code=self._err_code,
            warn_code=self._warn_code,
            mode=mode,
            state=state,
            stale=stale,
            t_mono=mono_ts,
            wallclock_ns=wall,
        )

    # -- commands --------------------------------------------------------------
    def _check_not_latched(self) -> None:
        if self._phase == DriverPhase.LATCHED:
            raise ArmFaultedError(
                f"{self.cfg.arm_id} is LATCHED ({self._latch_reason}); "
                "call clear_errors() to recover"
            )

    def _check_writable(self) -> None:
        """A read-only connection (phase-12) never commands: CommandError."""
        if self._readonly:
            raise CommandError(f"{self.cfg.arm_id}: read-only connection")

    def command_joints(self, q: np.ndarray) -> None:
        self._check_writable()
        self._check_not_latched()
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.dof,):
            raise CommandError(f"command_joints expects shape ({self.dof},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise CommandError("command_joints got non-finite values")
        if self._streamer is None:
            raise CommandError("driver not connected")
        q7 = np.clip(arr[:7], self._q_lo, self._q_hi)  # margin inside joint limits
        self._streamer.set_target(q7)
        if self._has_rail and self._rail is not None:
            self._rail.set_target(float(arr[7]))

    def command_gripper(self, cmd: GripperCommand) -> None:
        self._check_writable()
        self._check_not_latched()
        with self._gripper_cmd_lock:
            self._gripper_cmd = cmd  # latest-wins; drained by the monitor thread

    def command_rail(self, pos_m: float) -> None:
        self._check_writable()
        if not self._has_rail or self._rail is None:
            raise RailUnavailableError(f"{self.cfg.arm_id} has no rail")
        self._check_not_latched()
        if not np.isfinite(pos_m):
            raise CommandError("command_rail got non-finite value")
        if not self._rail.pos_known:
            # an explicit rail move on a track of unknown position is a programming
            # error (the rail slot of command_joints is dropped silently instead: the
            # control loop's hold target must keep flowing during the maintenance motion)
            raise CommandError(
                f"{self.cfg.arm_id}: linear track not homed (position unknown) - home_rail() first"
            )
        self._rail.set_target(float(pos_m))

    # -- rail homing on a connected driver (phase-09d) ---------------------------
    def home_rail(self) -> RailHomeOutcome:
        """Home the linear track while the servo stream HOLDS the joints — MOTION:
        the carriage drives to the track's zero end (the operator's LEFT, +X).

        The runtime's rail-homing maintenance job calls this on ITS OWN thread
        after it has (a) connected the driver with ``rail_homing ==
        "allow_unhomed"`` and (b) pre-positioned the arm along a twin-planned,
        rail-position-agnostic path; it blocks ≤ ``HOME_RAIL_SDK_WAIT_S`` (30 s)
        + a few register round-trips. Threads: the 100 Hz ``_ServoStreamer``
        keeps sending the hold posture (the SDK's per-instance command lock
        serialises its ``set_servo_angle_j`` with the homing's modbus calls; the
        SDK's homing wait is a 10 Hz register poll, not a held lock); the 5 Hz
        ``_MonitorThread`` keeps polling errors / the gripper, and its
        ``rail.step()`` is a no-op for the duration (homing latch, see
        :class:`RailController`). The rail slot of ``command_joints`` is dropped
        while homing and the target is cleared afterwards, so nothing queued
        can move the carriage once the track becomes commandable.

        Refused (``written=False``, nothing moved) unless the driver is
        ``STREAMING`` (a FAULT / LATCHED arm is not held by the stream) with no
        controller error latched; the track outcome itself is judged from the
        REGISTERS only (``on_zero == 1 and is_enabled == 1 and error == 0``) —
        success → rail ``READY``, position known (0.0 m), ``RailEvent``; failure
        → ``RAIL_ERROR`` + ``RailEvent``, position still unknown. Raises
        ``CommandError`` when not connected and ``RailUnavailableError`` without
        a track. Re-homing a homed track is allowed.
        """
        self._check_writable()
        if self._api is None or self._streamer is None or self._monitor is None:
            raise CommandError(f"{self.cfg.arm_id}: driver not connected")
        if not self._has_rail or self._rail is None:
            raise RailUnavailableError(f"{self.cfg.arm_id} has no rail")
        rail = self._rail
        if self._phase is not DriverPhase.STREAMING:
            return RailHomeOutcome(
                False,
                f"{self.cfg.arm_id}: rail homing refused: driver is {self._phase.value}, the "
                "servo stream is not holding the joints (recover first) - nothing written",
                phase=rail.phase.name,
            )
        if self._err_code != 0:
            return RailHomeOutcome(
                False,
                f"{self.cfg.arm_id}: rail homing refused: controller error {self._err_code} is "
                "latched - clear it first (nothing written)",
                phase=rail.phase.name,
            )
        outcome = rail.home()
        now = self._clock()
        # emit here (not only from the 5 Hz monitor tick) so the caller can see the
        # READY / RAIL_ERROR RailEvent right after home_rail() returns
        for phase_name, rcode, detail in rail.drain_events():
            self._emit(
                RailEvent(self.cfg.arm_id, phase=phase_name, code=rcode, detail=detail, t_mono=now)
            )
        return outcome

    # -- fault funnel & recovery (§3.5) -----------------------------------------
    def _on_fault(self, source: str, code: int) -> None:
        """All fault detectors funnel here; recovery runs on the monitor thread."""
        with self._phase_lock:
            if self._phase in (DriverPhase.FAULT, DriverPhase.RECOVERING, DriverPhase.LATCHED):
                return
            self._phase = DriverPhase.FAULT
            self._fault_pending = True
            self._fault_code = code
            self._fault_source = source
        if self._streamer is not None:
            self._streamer.pause()

    def _recover(self, user_initiated: bool = False) -> None:
        """clean_error -> motion_enable -> set_mode(1) -> set_state(0) -> re-seed
        from the MEASURED position (§3.5); emits ReseedEvent (runtime MUST
        re-anchor its IK target)."""
        with self._recovery_lock:
            api = self._api
            now = self._clock()
            self._phase = DriverPhase.RECOVERING
            self._fault_pending = False
            source = "user" if user_initiated else self._fault_source
            code, ew = api.get_err_warn_code()
            err, warn = (int(ew[0]), int(ew[1])) if code == 0 and ew else (0, 0)
            self._err_code, self._warn_code = err, warn
            self._emit(
                FaultEvent(
                    self.cfg.arm_id,
                    source=source,
                    code=self._fault_code,
                    error_code=err,
                    warn_code=warn,
                    t_mono=now,
                )
            )
            kind = classify_error(err)
            if kind == FaultKind.RAIL_ONLY and self._rail is not None:
                # 111 latches ONLY the rail; the arm re-enters streaming
                self._rail.latch_error(err)
            if not user_initiated:
                if kind == FaultKind.UNRECOVERABLE:
                    self._latch(f"controller error {err}", emit=False)
                    self._record_recovery(err, user_initiated)
                    return
                if kind == FaultKind.EXTERNAL:
                    self._handle_external(resume_from_fault=True)
                    self._record_recovery(err, user_initiated)
                    return
                if kind == FaultKind.RECOVERABLE and not self._budget_ok(now, err):
                    self._record_recovery(err, user_initiated)
                    return  # _budget_ok latched already
            # steps 3-6: the binding recovery sequence
            api.clean_error()
            api.clean_warn()
            if api.motion_enable(True) != 0:
                self._latch("motion_enable failed (release the physical e-stop?)")
                self._record_recovery(err, user_initiated)
                return
            if api.set_mode(1) != 0 or api.set_state(0) != 0:
                self._latch("set_mode/set_state failed during recovery")
                self._record_recovery(err, user_initiated)
                return
            self._mode_set_at = self._clock()
            self._await_servo_ready()
            try:
                q = self._read_measured_q()
            except ArmConnectError:
                self._latch("could not read measured position during recovery")
                self._record_recovery(err, user_initiated)
                return
            self._streamer.reseed(q)
            self._streamer.resume()
            self._emit(ReseedEvent(self.cfg.arm_id, tuple(float(v) for v in q), t_mono=now))
            if err == C24_ERROR:
                # halve vel/acc for 10 s after a speed-limit trip
                self._streamer.set_scale(0.5)
                self._c24_backoff_until = now + C24_BACKOFF_S
            if not user_initiated and kind == FaultKind.RECOVERABLE:
                self._recovery_times.append(now)
            self._err_code, self._warn_code = 0, warn
            self._phase = DriverPhase.STREAMING
            self._emit(RecoveredEvent(self.cfg.arm_id, err, t_mono=now))
            self._record_recovery(err, user_initiated)

    def _record_recovery(self, err: int, user_initiated: bool) -> None:
        """Publish the outcome of the sequence that just ended (ok = streaming again)."""
        self._recovery_seq += 1
        ok = self._phase == DriverPhase.STREAMING
        self._recovery_result = RecoveryResult(
            seq=self._recovery_seq,
            ok=ok,
            error_code=err,
            detail="" if ok else self._latch_reason,
            user_initiated=user_initiated,
            t_mono=self._clock(),
        )

    def _budget_ok(self, now: float, err: int) -> bool:
        """<= 3 recoveries per rolling 30 s; a second C24 inside the backoff
        window latches immediately."""
        if err == C24_ERROR and self._c24_backoff_until is not None:
            if now < self._c24_backoff_until:
                self._latch("second C24 inside the backoff window")
                return False
        while self._recovery_times and now - self._recovery_times[0] > RECOVERY_WINDOW_S:
            self._recovery_times.popleft()
        if len(self._recovery_times) >= RECOVERY_BUDGET:
            self._latch(
                f"recovery budget exhausted ({RECOVERY_BUDGET} in {RECOVERY_WINDOW_S:.0f} s)"
            )
            return False
        return True

    def _handle_external(self, resume_from_fault: bool = False) -> None:
        """The arm left servo mode with no error code — something outside this
        driver moved it (UFACTORY Studio "Live control" is the usual suspect, but
        who holds port 18333 is invisible to us, so the text only reports what was
        MEASURED). Pause, warn, retry mode 1 once; twice within 5 s -> LATCHED."""
        now = self._clock()
        snap = self._snap
        mode = snap.mode if snap else -1
        state = snap.state if snap else -1
        self._emit(
            StudioConflictWarning(
                self.cfg.arm_id,
                mode=mode,
                state=state,
                t_mono=now,
            )
        )
        if self._external_retry_at is not None and now - self._external_retry_at < 5.0:
            self._latch(
                f"arm left servo mode twice in 5 s (controller mode {mode} state {state}); "
                "close UFACTORY Studio live control if it is open"
            )
            return
        self._external_retry_at = now
        if self._streamer is not None:
            self._streamer.pause()
        self._enter_servo_mode()
        try:
            q = self._read_measured_q()
        except ArmConnectError:
            self._latch("could not re-seed after external conflict")
            return
        self._streamer.reseed(q)
        self._streamer.resume()
        self._emit(ReseedEvent(self.cfg.arm_id, tuple(float(v) for v in q), t_mono=now))
        self._phase = DriverPhase.STREAMING

    def _latch(self, reason: str, emit: bool = True) -> None:
        self._phase = DriverPhase.LATCHED
        self._latch_reason = reason
        if self._streamer is not None:
            self._streamer.pause()
        with self._gripper_cmd_lock:  # clear queues
            self._gripper_cmd = None
        if emit:
            self._emit(
                FaultEvent(
                    self.cfg.arm_id,
                    source="latch",
                    code=0,
                    error_code=self._err_code,
                    warn_code=self._warn_code,
                    detail=reason,
                    t_mono=self._clock(),
                )
            )

    # -- stop / clear_errors / disconnect (§3.6) --------------------------------
    def stop(self) -> None:
        """Software stop: set_state(4) + LATCHED. Does NOT clear errors and is
        NOT a hardware STO — the physical e-stop button is the real emergency
        path. Safe to call twice. On a read-only connection (phase-12) it is a
        no-op: nothing of ours is moving and ``set_state(4)`` is a write."""
        if self._readonly:
            return
        if self._streamer is not None:
            self._streamer.pause()
        if self._api is not None:
            try:
                self._api.set_state(4)
            except Exception:  # noqa: BLE001 — stop never raises
                pass
        if self._rail is not None:
            self._rail.set_target(self._rail.pos_m)
        self._latch("user_stop")

    def clear_errors(self) -> None:
        """User-initiated recovery on the CALLER's thread; only meaningful from
        LATCHED (§3.5). With the physical e-stop still engaged, motion_enable
        fails and the driver stays LATCHED. Prefer :meth:`request_recovery`
        from other threads (one XArmAPI must not be driven from two threads)."""
        if self._phase not in (DriverPhase.LATCHED, DriverPhase.FAULT):
            return
        self._reset_recovery_budget()
        self._recover(user_initiated=True)

    def request_recovery(self) -> None:
        """Operator-triggered recovery (phase-09b): flag it; the 5 Hz monitor
        thread runs ``_recover(user_initiated=True)`` — clean_error, clean_warn,
        motion_enable, set_mode(1), set_state(0), re-seed from the MEASURED
        position — bypassing classification and the budget. The outcome arrives
        as FaultEvent -> ReseedEvent + RecoveredEvent (or a latch FaultEvent) via
        :meth:`drain_events` and as :meth:`recovery_result`. Works from any
        connected phase (LATCHED / FAULT / STREAMING); never runs SDK calls on the
        caller's thread. Raises ``CommandError`` when the driver is not connected."""
        self._check_writable()
        if self._api is None or self._monitor is None:
            raise CommandError(f"{self.cfg.arm_id}: driver not connected")
        with self._phase_lock:
            self._user_recovery_pending = True

    def _reset_recovery_budget(self) -> None:
        """An explicit user action resets the auto-recovery budget and C24 backoff."""
        self._recovery_times.clear()
        self._c24_backoff_until = None
        if self._streamer is not None:
            self._streamer.set_scale(1.0)
        self._external_retry_at = None

    def _user_recover(self) -> None:
        """Monitor-thread half of :meth:`request_recovery`."""
        if self._api is None:
            return
        self._reset_recovery_budget()
        if self._streamer is not None:
            self._streamer.pause()  # the sequence re-seeds and resumes it
        try:
            self._recover(user_initiated=True)
        except Exception as exc:  # noqa: BLE001 — the SDK raises bare Exception
            self._latch(f"recovery failed: {type(exc).__name__}: {exc}".rstrip(": "))
            self._record_recovery(self._err_code, user_initiated=True)

    def disconnect(self) -> None:
        """Idempotent teardown; never raises (logs by returning silently).

        Hands the arm back in the power-on posture "stopped, brakes engaged"
        (phase-09c D6): after the threads stop, ``set_mode(0)`` ->
        ``set_state(4)`` -> ``motion_enable(False)``. The linear track is left
        alone — it keeps its homed flag (no ``set_linear_track_enable(False)``),
        so the next session needs no re-homing.

        A read-only connection (phase-12) only stops its poll thread and calls
        ``api.disconnect()``: the D6 hand-back writes (``set_mode`` /
        ``set_state`` / ``motion_enable``) belong to a driving session and are
        wrong for a client that never enabled the arm.
        """
        api, self._api = self._api, None
        if api is None:
            return
        if self._readonly:
            for closer in (
                lambda: self._poller.stop() if self._poller else None,
                lambda: api.disconnect(),
            ):
                try:
                    closer()
                except Exception:  # noqa: BLE001 — disconnect never raises
                    pass
            self._poller = None
            self._phase = DriverPhase.IDLE
            return
        for closer in (
            lambda: self._streamer.stop() if self._streamer else None,
            lambda: self._monitor.stop() if self._monitor else None,
            lambda: api.set_mode(0),
            lambda: api.set_state(4),  # stop (SDK: state 4 = stopped)
            lambda: api.motion_enable(False),  # brakes engaged, as found at power-on
            lambda: self._gripper.close() if self._gripper else None,
            lambda: api.disconnect(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 — disconnect never raises
                pass
        self._phase = DriverPhase.IDLE


class _ReadonlyPoller:
    """Read-only poll thread of ``connect(readonly=True)`` (phase-12) at
    ``cfg.monitor_rate_hz``: ``get_err_warn_code`` (the 30003 stream carries no
    codes), the linear-track registers and the gripper opening — three reads,
    zero writes. Deliberately NOT ``_MonitorThread``: that one recovers faults
    (``clean_error`` / ``motion_enable`` / ``set_mode`` ...), steps the rail and
    drains the gripper queue. A failed poll (SDK exception, or ``connected``
    dropping) only marks the published state stale until the next successful
    poll; nothing is ever retried with a write."""

    def __init__(self, driver: XArmDriver) -> None:
        self._d = driver
        self._running = False
        self._thread: threading.Thread | None = None
        self._period = 1.0 / driver.cfg.monitor_rate_hz

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"hw.{self._d.cfg.arm_id}.readonly", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while self._running:
            try:
                self.step()
            except Exception:  # noqa: BLE001 — a read-only poll never writes back
                self._d._readonly_poll_ok = False
            self._d._sleep(self._period)

    def step(self) -> None:
        d = self._d
        api = d._api
        if api is None:
            return
        if getattr(api, "connected", True) is False:
            d._readonly_poll_ok = False
            self._stale_edge()
            return
        d._read_err_warn_readonly()
        d._read_rail_readonly()
        d._read_gripper_readonly()
        d._readonly_poll_ok = True
        self._stale_edge()

    def _stale_edge(self) -> None:
        """StaleEvent on transitions, like the monitor's step 4 (get_state computes
        the flag itself; a failed poll counts as stale)."""
        d = self._d
        now = d._clock()
        snap = d._snap
        stale = (
            snap is None or (now - snap.mono_ts) > d.cfg.stale_after_s or not d._readonly_poll_ok
        )
        if stale != d._was_stale:
            age = (now - snap.mono_ts) if snap is not None else float("inf")
            d._emit(StaleEvent(d.cfg.arm_id, stale=stale, age_s=age, t_mono=now))
            d._was_stale = stale


class _MonitorThread:
    """5 Hz housekeeping: err/warn poll (30003 carries none), recovery
    execution, rail step, gripper queue + poll, staleness/backoff bookkeeping
    (02-hardware §10)."""

    def __init__(self, driver: XArmDriver) -> None:
        self._d = driver
        self._running = False
        self._thread: threading.Thread | None = None
        self._period = 1.0 / driver.cfg.monitor_rate_hz

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"hw.{self._d.cfg.arm_id}.monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run(self) -> None:
        while self._running:
            try:
                self.step()
            except Exception:  # noqa: BLE001 — monitor must keep ticking
                pass
            self._d._sleep(self._period)

    def step(self) -> None:  # noqa: C901 — linear checklist
        d = self._d
        api = d._api
        if api is None:
            return
        # 0. operator-requested recovery first (request_recovery), then a
        #    pending auto fault -> both run HERE (streamer already paused)
        with d._phase_lock:
            user_pending, d._user_recovery_pending = d._user_recovery_pending, False
        if user_pending:
            d._user_recover()
            return
        if d._fault_pending and d._phase == DriverPhase.FAULT:
            d._recover()
            return
        # 1. link loss
        if getattr(api, "connected", True) is False:
            if d._phase != DriverPhase.LATCHED:
                d._emit(FaultEvent(d.cfg.arm_id, source="report", code=-1, t_mono=d._clock()))
                d._latch("SDK connection lost")
            return
        # 2. err/warn poll (errors while holding)
        code, ew = api.get_err_warn_code()
        if code == 0 and ew:
            err, warn = int(ew[0]), int(ew[1])
            d._err_code, d._warn_code = err, warn
            if err == RAIL_COMMS_ERROR:
                # 111 latches ONLY the rail; arm keeps streaming
                if d._rail is not None and d._rail.phase.name != "RAIL_ERROR":
                    d._rail.latch_error(err)
                api.clean_error()
                d._err_code = 0
            elif err != 0 and d._phase == DriverPhase.STREAMING:
                d._on_fault("monitor", err)
                return
        # 3. the arm left servo mode with no error code (external actor, e.g. Studio
        #    "Live control"). Mode 0/2 = someone took position/teach control; state
        #    3/4/5/6 = someone paused or stopped it. **State 2 (standby) is HEALTHY**
        #    and is what a held mode-1 arm reports — see SERVO_HEALTHY_STATES.
        snap = d._snap
        if (
            d._phase == DriverPhase.STREAMING
            and snap is not None
            and d._err_code == 0
            and (snap.mode != 1 or snap.state in SERVO_CONFLICT_STATES)
            and (d._clock() - snap.mono_ts) <= d.cfg.stale_after_s
            and (d._clock() - d._mode_set_at) > 0.5  # reports lag our set_mode(1)
        ):
            d._handle_external()
        # 4. staleness transition events (get_state computes the flag itself)
        now = d._clock()
        stale = snap is None or (now - snap.mono_ts) > d.cfg.stale_after_s
        if stale != d._was_stale:
            age = (now - snap.mono_ts) if snap is not None else float("inf")
            d._emit(StaleEvent(d.cfg.arm_id, stale=stale, age_s=age, t_mono=now))
            d._was_stale = stale
        # 5. C24 backoff expiry
        if d._c24_backoff_until is not None and now >= d._c24_backoff_until:
            d._c24_backoff_until = None
            if d._streamer is not None:
                d._streamer.set_scale(1.0)
        # 6. rail slow axis + rail events
        if d._rail is not None:
            d._rail.step()
            for phase_name, rcode, detail in d._rail.drain_events():
                d._emit(
                    RailEvent(d.cfg.arm_id, phase=phase_name, code=rcode, detail=detail, t_mono=now)
                )
        # 7. gripper queue + poll
        if d._gripper is not None:
            with d._gripper_cmd_lock:
                cmd, d._gripper_cmd = d._gripper_cmd, None
            if cmd is not None and d._phase == DriverPhase.STREAMING:
                d._gripper.command(cmd)
            d._gripper_state = d._gripper.poll()
            for gcode in d._gripper.drain_faults():
                d._emit(GripperFaultEvent(d.cfg.arm_id, code=gcode, t_mono=now))
