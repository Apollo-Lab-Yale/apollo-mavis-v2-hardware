"""XArmDriver: core ArmInterface over the xArm SDK (02-hardware §3).

One driver per arm; one ``XArmAPI(ip, is_radian=True, report_type='real')``
per driver (never shared across arms/processes). ``XArmAPI`` is imported ONLY
here (injected everywhere else via ``api_factory``).
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
from apollo_xarm7_core import (
    ArmConnectError,
    ArmIdentityError,
    ArmInterface,
    ArmState,
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
from .grippers import GripperBackend, make_gripper, parse_fw
from .rail import RailController

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
            else:
                self._faults += 1
                self.pause()  # §3.5 takes over on the monitor thread
                self._on_fault("servo", int(code))


def _default_api_factory(*args: Any, **kwargs: Any) -> Any:
    from xarm.wrapper import XArmAPI  # imported ONLY here (02-hardware §1)

    return XArmAPI(*args, **kwargs)


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
    def connect_warnings(self) -> list[str]:
        return list(self._connect_warnings)

    def drain_events(self) -> list[DriverEvent]:
        with self._events_lock:
            out = list(self._events)
            self._events.clear()
        return out

    def _emit(self, event: DriverEvent) -> None:
        with self._events_lock:
            self._events.append(event)

    # -- bring-up (§3.2) -------------------------------------------------------
    def connect(self) -> None:
        cfg = self.cfg
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
        self._fw = parse_fw(str(getattr(api, "version", "") or "0.0.0"))
        # 3. clear latched state, backstops, enable (required order)
        api.clean_warn()
        api.clean_error()
        self._connect_warnings = apply_backstops(api, cfg)
        api.motion_enable(True)
        api.set_mode(0)
        api.set_state(0)  # set_state(0) must follow every set_mode
        self._phase = DriverPhase.READY
        # 4. rail detection fixes dof
        rail = RailController(api, cfg.rail_speed_mm_s)
        if cfg.expect_rail == "no":
            self._has_rail = False
        else:
            detected = rail.detect()
            if cfg.expect_rail == "yes" and not detected:
                self._phase = DriverPhase.IDLE
                raise RailExpectedError("rail", f"{cfg.arm_id}: expected rail not detected")
            self._has_rail = detected
        if self._has_rail:
            rail.ensure_homed()
            self._rail = rail
        # 5. gripper backend + report callback
        self._gripper = make_gripper(cfg.gripper, self._clock)
        self._gripper.init(api, self._fw)  # raises GripperInitError
        api.register_report_callback(
            self._on_report,
            report_cartesian=True,
            report_joints=True,
            report_state=True,
            report_mode=True,
            report_cmd_num=True,
        )
        # 6. enter streaming; seed from the measured position
        api.set_mode(1)
        api.set_state(0)
        self._mode_set_at = self._clock()
        self._sleep(0.1)
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
        tau = (
            np.asarray(tau_raw[:7], dtype=np.float64)
            if tau_raw is not None
            else np.zeros(7)
        )
        cart = data.get("cartesian") or [0.0] * 6
        self._snap = _StateSnap(  # single reference assignment = GIL-atomic swap
            q=q,
            dq=dq,
            ee_pose_sdk=tuple(float(v) for v in cart[:6]),
            tau=tau,
            mode=int(data.get("mode", 0)),
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

    def command_joints(self, q: np.ndarray) -> None:
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
        self._check_not_latched()
        with self._gripper_cmd_lock:
            self._gripper_cmd = cmd  # latest-wins; drained by the monitor thread

    def command_rail(self, pos_m: float) -> None:
        if not self._has_rail or self._rail is None:
            raise RailUnavailableError(f"{self.cfg.arm_id} has no rail")
        self._check_not_latched()
        if not np.isfinite(pos_m):
            raise CommandError("command_rail got non-finite value")
        self._rail.set_target(float(pos_m))

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
            source = getattr(self, "_fault_source", "user")
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
                    return
                if kind == FaultKind.EXTERNAL:
                    self._handle_external(resume_from_fault=True)
                    return
                if kind == FaultKind.RECOVERABLE and not self._budget_ok(now, err):
                    return  # _budget_ok latched already
            # steps 3-6: the binding recovery sequence
            api.clean_error()
            api.clean_warn()
            if api.motion_enable(True) != 0:
                self._latch("motion_enable failed (release the physical e-stop?)")
                return
            if api.set_mode(1) != 0 or api.set_state(0) != 0:
                self._latch("set_mode/set_state failed during recovery")
                return
            self._mode_set_at = self._clock()
            self._sleep(0.1)
            try:
                q = self._read_measured_q()
            except ArmConnectError:
                self._latch("could not read measured position during recovery")
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
        """Mode/state changed under us with no error code — UFACTORY Studio.
        Pause, warn, retry mode 1 once; twice within 5 s -> LATCHED."""
        now = self._clock()
        snap = self._snap
        self._emit(
            StudioConflictWarning(
                self.cfg.arm_id,
                mode=snap.mode if snap else -1,
                state=snap.state if snap else -1,
                t_mono=now,
            )
        )
        if self._external_retry_at is not None and now - self._external_retry_at < 5.0:
            self._latch("external mode/state conflict persisted (UFACTORY Studio?)")
            return
        self._external_retry_at = now
        api = self._api
        if self._streamer is not None:
            self._streamer.pause()
        api.set_mode(1)
        api.set_state(0)
        self._mode_set_at = self._clock()
        self._sleep(0.1)
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
        path. Safe to call twice."""
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
        """User-initiated recovery; only meaningful from LATCHED (§3.5). With
        the physical e-stop still engaged, motion_enable fails and the driver
        stays LATCHED."""
        if self._phase not in (DriverPhase.LATCHED, DriverPhase.FAULT):
            return
        self._recovery_times.clear()  # explicit user action resets the budget
        self._c24_backoff_until = None
        if self._streamer is not None:
            self._streamer.set_scale(1.0)
        self._external_retry_at = None
        self._recover(user_initiated=True)

    def disconnect(self) -> None:
        """Idempotent teardown; never raises (logs by returning silently)."""
        api, self._api = self._api, None
        if api is None:
            return
        for closer in (
            lambda: self._streamer.stop() if self._streamer else None,
            lambda: self._monitor.stop() if self._monitor else None,
            lambda: api.set_mode(0),
            lambda: api.set_state(0),
            lambda: self._gripper.close() if self._gripper else None,
            lambda: api.disconnect(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 — disconnect never raises
                pass
        self._phase = DriverPhase.IDLE


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
        # 0. pending fault -> run recovery here (streamer already paused)
        if d._fault_pending and d._phase == DriverPhase.FAULT:
            d._recover()
            return
        # 1. link loss
        if getattr(api, "connected", True) is False:
            if d._phase != DriverPhase.LATCHED:
                d._emit(
                    FaultEvent(d.cfg.arm_id, source="report", code=-1, t_mono=d._clock())
                )
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
        # 3. external mode/state grab (Studio) — no error code
        snap = d._snap
        if (
            d._phase == DriverPhase.STREAMING
            and snap is not None
            and d._err_code == 0
            and (snap.mode != 1 or snap.state not in (0, 1))
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
                    RailEvent(d.cfg.arm_id, phase=phase_name, code=rcode, detail=detail,
                              t_mono=now)
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
