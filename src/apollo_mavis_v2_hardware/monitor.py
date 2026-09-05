"""Read-only controller state monitor (02-hardware §8.5; phase-09a).

``ArmStateMonitor`` polls ONE xArm7 control box for its joint angles, flange
pose, error/warn codes, controller state/mode, linear-track registers and
gripper opening while NEVER commanding it: no ``motion_enable``, no
``set_mode``/``set_state``, no ``clean_*``, no ``set_*``. The runtime uses it
session-less (the Welcome page's real error codes and the digital-twin
overlay); when a hardware session wants the boxes it calls ``disconnect()``
— pause means RELEASE the connection, two SDK clients on one control box are
unevidenced — and ``start()`` again afterwards.

Zero-write contract: ``READ_ONLY_SDK_METHODS`` / ``READ_ONLY_SDK_ATTRS`` are
the complete list of ``XArmAPI`` members this module touches;
``tests/test_monitor.py`` runs it against a call-logging fake and asserts
nothing else is ever called.

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
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from . import units
from .driver import _default_api_factory

logger = logging.getLogger(__name__)

ArmMonitorStatus = Literal["off", "connecting", "running", "stale", "paused", "error"]
GripperKind = Literal["xarm", "xarm_g2", "none"]

# ---------------------------------------------------------------------------
# ZERO-WRITE ALLOWLIST — every XArmAPI member this module may touch. Anything
# else (motion_enable, set_mode, set_state, clean_error, clean_warn, set_*,
# register_*, ...) is a contract violation; tests/test_monitor.py asserts it.
# ---------------------------------------------------------------------------
READ_ONLY_SDK_METHODS: frozenset[str] = frozenset(
    {
        "connect",
        "disconnect",
        "get_servo_angle",  # 7 joint angles, rad (is_radian=True)
        "get_position",  # flange pose, mm + rad (tcp_offset is zero on both boxes)
        "get_err_warn_code",  # [error_code, warn_code]
        "get_linear_track_registers",  # {pos, status, error, is_enabled, on_zero, ...}
        "get_gripper_g2_position",  # G2: int mm — same call as G2Gripper.poll()
        "get_gripper_position",  # classic: pulses — same call as ClassicGripper.poll()
    }
)
READ_ONLY_SDK_ATTRS: frozenset[str] = frozenset({"connected", "state", "mode"})

MAX_RECONNECT_S = 10.0  # exponential backoff cap
RAIL_GRIPPER_HZ = 2.0  # modbus round-trips are slow; poll them at ~2 Hz
# start() after a stop()/disconnect() whose join timed out waits this long for the old
# thread to leave the SDK call it is blocked in (connect(): two sockets with their own
# timeouts + the version handshake; a register read: modbus timeouts) before reconnecting.
STALE_THREAD_JOIN_S = 15.0


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
    tcp_pose: tuple[float, ...]  # [x, y, z m, roll, pitch, yaw rad] base frame; () if unread
    error_code: int = 0
    warn_code: int = 0
    state: int | None = None  # controller state (4 = stopped / not enabled)
    mode: int | None = None
    rail_present: bool | None = None  # registers readable (None: not polled)
    rail_homed: bool | None = None  # on_zero == 1
    rail_enabled: bool | None = None  # is_enabled == 1
    rail_pos_m: float | None = None  # None unless homed AND enabled
    rail_raw_mm: float | None = None  # raw register, always when present
    gripper_open_frac: float | None = None  # 0 closed .. 1 open; None for "none"
    gripper_raw: float | None = None  # raw SDK reading (G2 mm / classic pulses)


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
            try:
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
            # (502 control + report stream, which keeps .state/.mode current)
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
        release the client a newer start() published)."""
        with self._lock:
            if api is not None and self._api is not api:
                return
            held, self._api = self._api, None
            self._running_since = None
        _quiet_disconnect(held)

    def _shutdown(self, status: ArmMonitorStatus, detail: str, timeout: float) -> bool:
        with self._lock:
            self._running = False
            thread = self._thread
            self._wake.set()
        released = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
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
        else:
            rail = (
                prev.rail_present,
                prev.rail_homed,
                prev.rail_enabled,
                prev.rail_pos_m,
                prev.rail_raw_mm,
            )
            grip = (prev.gripper_open_frac, prev.gripper_raw)

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
            gripper_open_frac=grip[0],
            gripper_raw=grip[1],
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

    def _read_rail(
        self, api: Any, problems: list[str]
    ) -> tuple[bool | None, bool | None, bool | None, float | None, float | None]:
        """(present, homed, enabled, pos_m, raw_mm); pos_m only when homed AND enabled."""
        if not self.expect_rail:
            return (None, None, None, None, None)
        code, regs = api.get_linear_track_registers()
        if code != 0:
            return (False, None, None, None, None)  # 3 timeout / 20 host id / 23 length
        if not isinstance(regs, Mapping) or "pos" not in regs:
            return (False, None, None, None, None)  # simulation-mode controller: (0, [])
        homed = int(regs.get("on_zero", 0) or 0) == 1
        enabled = int(regs.get("is_enabled", 0) or 0) == 1
        raw_mm = float(regs.get("pos", 0) or 0)
        track_err = int(regs.get("error", 0) or 0)
        if track_err:
            problems.append(f"linear track error {track_err}")
        pos_m = units.rail_mm_to_m(raw_mm) if (homed and enabled) else None
        return (True, homed, enabled, pos_m, raw_mm)

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
