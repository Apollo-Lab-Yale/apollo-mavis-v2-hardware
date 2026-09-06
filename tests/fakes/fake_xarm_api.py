"""Stateful double of the XArmAPI surface the driver uses (02-hardware §11).

Mirrors xarm-python-sdk 1.18.5 (the pinned rev), NOT a permissive stub:
``register_report_callback`` accepts exactly the SDK keywords (no
``report_mode``); the report payload carries the SDK keys only (no ``mode`` —
``mode``/``state`` are attributes kept current by the "report thread", as the
SDK does); ``version`` is the RAW controller string and ``version_number`` the
parsed tuple; there is NO ``get_linear_track_sn``; ``do_not_open=True`` leaves
the instance disconnected until ``connect()``.

Encodes the SDK gotchas as behavior: errors reset mode to 0; servo sends
return 1 (error latched) / 9 (not ready) / -8 (joint-limit reject);
``clean_error()`` alone is not readiness; unhomed rail returns 82; absent
rail returns code 3; a simulation-mode controller answers track reads with
``(0, [])`` without touching the bus. Records ``sent_joints`` + a ``calls``
log of every mutating call; ``emit_report`` (or the optional 30003 stream
reader) fires registered report callbacks.

Linear-track homing (phase-09c) mirrors ``x3/linear_motor.py:131-148`` +
``:242-261``: ``set_linear_track_back_origin(wait=True, **kwargs)`` blocks for
``homing_duration_s`` (honouring the ``timeout`` kwarg — SDK default 10 s — and
``self.connected``, exactly like ``__wait_linear_motor_back_origin``), returns
0 / 80 (track error) / 100 (timeout or link lost), then, when ``auto_enable``
(SDK default True!) OVERWRITES that code with the enable's return code —
``homing_result_code`` forces the wait result for tests that pin
"judge from registers, not from the code". ``inject_track_error(code)`` sets the
track error register (enable fails with 80, homing aborts with 80);
``homing_track_error`` (phase-09d) trips that register when the carriage would
have reached the zero end (wait returns 80, ``on_zero`` stays 0);
``rail_homed`` / ``rail_enabled`` are settable properties. The fake is used
from several threads at once (servo streamer, monitor, a homing caller):
every mutating call is a plain attribute write or list append under the GIL.
"""

from __future__ import annotations

import math
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .report_replayer import FrameSplitter, parse_real_frame

XARM7_LIMITS = [
    (-2 * math.pi, 2 * math.pi),
    (math.radians(-118), math.radians(120)),
    (-2 * math.pi, 2 * math.pi),
    (math.radians(-11), math.radians(225)),
    (-2 * math.pi, 2 * math.pi),
    (math.radians(-97), math.radians(180)),
    (-2 * math.pi, 2 * math.pi),
]

# what the two lab boxes answer to get_version() (fw v1.12.10, 2026-09-04)
REAL_VERSION_STRING = "7,7,XS1305,MC1303,v1.12.10"


@dataclass
class FaultScript:
    """Latch a controller error at the Nth set_servo_angle_j call (1-based)."""

    fault_at_tick: int | None = None
    error_code: int = 24
    servo_return: int = 1
    drop_mode_to_0: bool = True
    # Entering servo mode is not atomic on the real box: this many
    # set_servo_angle_j calls right after readiness answer APIState 9
    # (STATE_NOT_READY) before the first one lands. Reproduces 2026-09-05, when the
    # driver's blind sleep(0.1) after set_state(0) raced the controller and the very
    # first tick of the first hardware session faulted an arm during bring-up.
    not_ready_ticks: int = 0


class FakeXArmAPI:
    """Constructor mirrors XArmAPI(ip, is_radian=..., do_not_open=..., **kwargs)."""

    # exact keyword surface of XArmAPI.register_report_callback in SDK 1.18.5
    REPORT_CALLBACK_KEYWORDS = frozenset(
        {
            "report_cartesian",
            "report_joints",
            "report_state",
            "report_error_code",
            "report_warn_code",
            "report_mtable",
            "report_mtbrake",
            "report_cmd_num",
        }
    )

    def __init__(
        self,
        port: str = "192.168.1.235",
        is_radian: bool = False,
        do_not_open: bool = False,
        *,
        sn: str = "XA7-FAKE-0001",
        version: str = REAL_VERSION_STRING,
        initial_q: list[float] | None = None,
        initial_tcp: list[float] | None = None,
        has_rail: bool = False,
        rail_homed: bool = False,
        rail_enabled: bool = False,
        rail_sn: str = "AL1300FAKE1234",
        simulation_robot: bool = False,
        gripper_fw: str = "3.4.3",
        fault_script: FaultScript | None = None,
        auto_report_hz: float = 0.0,
        report_stream: tuple[str, int] | None = None,
        motion_enable_fails: bool = False,
        connect_fails: int = 0,
        connect_delay_s: float = 0.0,
        collision_sensitivity: int = 0,
        tcp_load: tuple[float, tuple[float, float, float]] = (0.0, (0.0, 0.0, 0.0)),
        homing_duration_s: float = 0.0,
        homing_result_code: int | None = None,
        homing_track_error: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        self.ctor_args = {
            "port": port,
            "is_radian": is_radian,
            "do_not_open": do_not_open,
            **kwargs,
        }
        self.ip = port
        self.is_radian = is_radian
        self.check_joint_limit = kwargs.get("check_joint_limit", True)
        self.report_type = kwargs.get("report_type", "rich")
        self.connected = not do_not_open  # SDK: do_not_open -> connect() later
        self.connect_failures_left = int(connect_fails)
        # connect() blocks this long (real SDK: two sockets with their own timeouts +
        # the version handshake -> seconds on a slow link); hand-over race tests
        self.connect_delay_s = float(connect_delay_s)
        self.sn = sn
        self.version = version  # RAW controller string, like the SDK property
        self.mode = 0  # SDK: updated by the report thread, never in the payload
        # Controller STATE, as the real boxes report it: 0 ready, 1 in motion,
        # 2 standby ("sleeping"), 3 paused, 4/5 stopped, 6 decelerating. A fresh box
        # is 2; a HELD mode-1 arm is ALSO 2 (measured on both lab boxes 2026-09-05)
        # — re-sending the same posture is not "motion". Readiness is a SEPARATE
        # flag on the wire (``ready_to_move``, the 0x10 bit of every TCP reply that
        # the SDK exposes as ``UxbusCmd.state_is_ready``), not a state value: that
        # split is why the driver must not infer "somebody grabbed the arm" from
        # ``state != 0``.
        self.state = 2
        self.ready_to_move = False  # set_state(0) after motion_enable + set_mode
        self.error_code = 0
        self.warn_code = 0
        self.motion_enabled = False
        self.motion_enable_fails = motion_enable_fails
        self.joints_torque = [0.0] * 7
        # SDK 1.18.5: read-only PROPERTIES refreshed by the report thread from the
        # normal/rich frame (no get_tcp_load / get_collision_sensitivity exist);
        # the lab boxes read 0 kg / sensitivity 3 (grip) and 1 (view) on 2026-09-04
        self.collision_sensitivity = int(collision_sensitivity)
        self.tcp_load: list[Any] = [float(tcp_load[0]), [float(v) for v in tcp_load[1]]]
        self._q = list(initial_q) if initial_q is not None else [0.0] * 7
        self._tcp = (
            list(initial_tcp) if initial_tcp is not None else [207.0, 0.0, 112.0, math.pi, 0.0, 0.0]
        )  # mm + rad, the SDK's get_position() units
        self._clock = clock
        self.fault_script = fault_script
        self.simulation_robot = simulation_robot
        # recording
        self.calls: list[tuple[str, tuple, dict]] = []
        self.sent_joints: list[tuple[float, list[float]]] = []
        self.servo_calls = 0
        # gripper
        self._gripper_enabled = False
        self._gripper_pulse = 850
        self._g2_mm = 84.0
        self._gripper_err = 0
        self._gripper_fw = gripper_fw
        # rail
        self._rail_present = has_rail
        self._rail_sn = rail_sn  # kept for subclasses that add get_linear_track_sn
        self._rail_on_zero = 1 if rail_homed else 0
        self._rail_pos_mm = 0
        self._rail_enabled = bool(rail_enabled)
        self._rail_speed = 0
        self._rail_error = 0
        self.rail_pos_commands: list[int] = []
        # homing (phase-09c): how long the carriage "travels" to the zero end; None
        # -> the wait result follows the SDK rules, else this code is returned
        # as the wait result (auto_enable may still overwrite it, like the SDK)
        self.homing_duration_s = float(homing_duration_s)
        self.homing_result_code = homing_result_code
        # phase-09d: the carriage trips this track error (e.g. 25/26 over-travel)
        # mid-travel: the wait returns 80, on_zero stays 0, the enable then fails
        self.homing_track_error = homing_track_error
        self.homing_started = 0  # set_linear_track_back_origin calls that reached the track
        self.homing_completed = 0  # times on_zero flipped to 1 through homing
        self._homing = False  # carriage travelling (status bit 0)
        # report plumbing
        self._callbacks: list[tuple[Callable[[dict], None], dict[str, bool]]] = []
        self._auto_report_hz = auto_report_hz
        self._report_stream = report_stream
        self._report_threads_running = False
        self._threads: list[threading.Thread] = []

    # -- recording helper ------------------------------------------------------
    def _rec(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def call_names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    # -- identity ----------------------------------------------------------------
    @property
    def version_number(self) -> tuple[int, int, int]:
        """SDK: (major, minor, revision) parsed from the raw version string."""
        m = re.search(r"v?(\d+)\.(\d+)\.(\d+)\s*$", self.version)
        if not m:
            return (0, 0, 0)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # -- error injection ---------------------------------------------------------
    def inject_error(self, code: int, warn: int = 0, drop_mode: bool = True) -> None:
        """Latch a controller error (e.g. 24 speed, 111 rail comms with
        drop_mode=False — a rail drop does not stop arm motion)."""
        self.error_code = code
        self.warn_code = warn
        if drop_mode:
            self.mode = 0  # errors silently reset the controller to mode 0
            self.state = 4
            self.ready_to_move = False

    def inject_track_error(self, code: int) -> None:
        """Latch a linear-track error register (e.g. 25/26 over-travel). The
        track drops its enable; ``set_linear_track_enable`` answers 80 until
        ``clean_linear_track_error()``; a homing in flight aborts with 80."""
        self._rail_error = int(code)
        if code:
            self._rail_enabled = False

    @property
    def rail_homed(self) -> bool:
        """``on_zero == 1``; settable (power-cycle -> False, Studio homing -> True)."""
        return self._rail_on_zero == 1

    @rail_homed.setter
    def rail_homed(self, value: bool) -> None:
        self._rail_on_zero = 1 if value else 0

    @property
    def rail_enabled(self) -> bool:
        return self._rail_enabled

    @rail_enabled.setter
    def rail_enabled(self, value: bool) -> None:
        self._rail_enabled = bool(value)

    @property
    def rail_error(self) -> int:
        return self._rail_error

    # -- lifecycle / mode / state -------------------------------------------------
    def connect(self, port: str | None = None, **kwargs: Any) -> None:
        self._rec("connect")
        if self.connected:
            return  # SDK: connect() on a connected instance is a no-op
        if self.connect_delay_s > 0:
            time.sleep(self.connect_delay_s)
        if self.connect_failures_left > 0:
            self.connect_failures_left -= 1
            raise Exception("connect socket failed")  # SDK raises bare Exception
        self.connected = True

    def disconnect(self) -> None:
        self._rec("disconnect")
        self.connected = False
        self.stop_fakes()

    def clean_warn(self) -> int:
        self._rec("clean_warn")
        self.warn_code = 0
        return 0

    def clean_error(self) -> int:
        # clearing the error does NOT restore readiness (docstring is explicit:
        # motion_enable + set_state are required afterwards)
        self._rec("clean_error")
        self.error_code = 0
        return 0

    def motion_enable(self, enable: bool = True, servo_id: Any = None) -> int:
        self._rec("motion_enable", enable)
        if self.motion_enable_fails:
            return 1  # e.g. physical e-stop still engaged
        self.motion_enabled = bool(enable)
        return 0

    def set_mode(self, mode: int = 0, detection_param: int = 0) -> int:
        self._rec("set_mode", mode)
        self.mode = mode
        self.state = 2  # standby
        self.ready_to_move = False  # every set_mode needs a set_state(0) after it
        return 0

    def get_state(self) -> tuple[int, int]:
        """SDK ``get_state()`` -> (code, state); a plain 502 read (the reply also
        refreshes the SDK's ready-to-move flag, which is why the driver polls it
        instead of sleeping a fixed time after entering servo mode). Recorded so
        tests can pin "readiness is polled, not slept for"."""
        self._rec("get_state")
        return 0, self.state

    def set_state(self, state: int = 0) -> int:
        self._rec("set_state", state)
        if state == 0:
            if self.motion_enabled and self.error_code == 0:
                # ready to accept motion, but the REPORTED state is standby until
                # something actually moves — the real box never idles in state 0
                self.state = 2
                self.ready_to_move = True
            # else: stays not-ready (latched error / not enabled)
        else:
            self.state = state
            self.ready_to_move = False
        return 0

    def get_err_warn_code(self, show: bool = False, **kwargs: Any) -> tuple[int, list[int]]:
        return 0, [self.error_code, self.warn_code]

    def get_servo_angle(
        self, servo_id: Any = None, is_radian: Any = None, is_real: bool = False
    ) -> tuple[int, list[float]]:
        self._rec("get_servo_angle", is_real=is_real)
        return 0, list(self._q)

    def get_position(self, is_radian: Any = None) -> tuple[int, list[float]]:
        """SDK: [x_mm, y_mm, z_mm, roll, pitch, yaw] (rad when is_radian)."""
        return 0, list(self._tcp)

    def emergency_stop(self) -> None:
        self._rec("emergency_stop")
        self.state = 4
        self.ready_to_move = False

    # -- mode-1 servo streaming -----------------------------------------------------
    def set_servo_angle_j(
        self,
        angles: list[float],
        speed: Any = None,
        mvacc: Any = None,
        mvtime: Any = None,
        is_radian: Any = None,
        **kwargs: Any,
    ) -> int:
        self.servo_calls += 1
        script = self.fault_script
        if script is not None and script.fault_at_tick == self.servo_calls:
            self.inject_error(script.error_code, drop_mode=script.drop_mode_to_0)
            script.fault_at_tick = None  # fire once
            return script.servo_return
        if self.error_code != 0:
            return 1  # HAS_ERROR until cleaned
        if self.mode != 1 or not self.ready_to_move or not self.motion_enabled:
            return 9  # STATE_NOT_READY (the reply's 0x10 bit)
        if script is not None and script.not_ready_ticks > 0:
            script.not_ready_ticks -= 1
            return 9  # still entering servo mode (see FaultScript.not_ready_ticks)
        if self.check_joint_limit:
            for value, (lo, hi) in zip(angles[:7], XARM7_LIMITS, strict=False):
                if not lo <= float(value) <= hi:
                    return -8  # SDK-side OUT_OF_RANGE
        self._q = [float(v) for v in angles[:7]]
        self.sent_joints.append((self._clock(), list(self._q)))
        return 0

    # -- backstops (§6) ---------------------------------------------------------------
    def set_tcp_load(self, weight: float, cog: list[float], wait: bool = False, **kw: Any) -> int:
        self._rec("set_tcp_load", weight, tuple(cog))
        # the controller echoes the new load in the next report frame
        self.tcp_load = [float(weight), [float(v) for v in cog]]
        return 0

    def set_gravity_direction(self, direction: list[float], wait: bool = True) -> int:
        self._rec("set_gravity_direction", tuple(direction))
        return 0

    def set_collision_sensitivity(self, value: int, wait: bool = True) -> int:
        self._rec("set_collision_sensitivity", value)
        self.collision_sensitivity = int(value)
        return 0

    def set_self_collision_detection(self, on: bool) -> int:
        self._rec("set_self_collision_detection", on)
        return 0

    def set_collision_tool_model(self, tool_type: int, **params: Any) -> int:
        self._rec("set_collision_tool_model", tool_type, **params)
        return 0

    def set_reduced_tcp_boundary(self, boundary: list[int]) -> int:
        self._rec("set_reduced_tcp_boundary", tuple(boundary))
        return 0

    def set_reduced_mode(self, on: bool) -> int:
        self._rec("set_reduced_mode", on)
        return 0

    def set_collision_rebound(self, on: bool) -> list[int]:
        self._rec("set_collision_rebound", on)
        return [0, 0]  # SDK 1.18.5 returns the raw reply list here, not ret[0]

    def save_conf(self) -> int:  # the driver must NEVER call this
        self._rec("save_conf")
        return 0

    # -- classic gripper ------------------------------------------------------------
    def set_gripper_enable(self, enable: bool) -> int:
        self._rec("set_gripper_enable", enable)
        self._gripper_enabled = bool(enable)
        return 0

    def set_gripper_mode(self, mode: int) -> int:
        self._rec("set_gripper_mode", mode)
        return 0

    def set_gripper_speed(self, speed: float) -> int:
        self._rec("set_gripper_speed", speed)
        return 0

    def set_gripper_position(
        self,
        pos: int,
        wait: bool = False,
        speed: Any = None,
        auto_enable: bool = False,
        timeout: Any = None,
        **kwargs: Any,
    ) -> int:
        # SDK: wait_motion (default True) runs wait_move() before the modbus write
        self._rec(
            "set_gripper_position", pos, wait=wait, wait_motion=kwargs.get("wait_motion", True)
        )
        self._gripper_pulse = int(pos)
        return 0

    def get_gripper_position(self, **kwargs: Any) -> tuple[int, int]:
        return 0, self._gripper_pulse

    def get_gripper_status(self) -> tuple[int, int]:
        return 0, 0

    def get_gripper_err_code(self) -> tuple[int, int]:
        return 0, self._gripper_err

    def clean_gripper_error(self) -> int:
        self._rec("clean_gripper_error")
        self._gripper_err = 0
        return 0

    def get_gripper_version(self) -> tuple[int, str]:
        return 0, self._gripper_fw

    def set_external_device_monitor_params(self, dev_type: int = 1, frequency: int = 10) -> int:
        self._rec("set_external_device_monitor_params", dev_type=dev_type, frequency=frequency)
        return 0

    # -- G2 gripper --------------------------------------------------------------------
    def set_gripper_g2_position(
        self,
        pos: float,
        speed: int = 100,
        force: int = 50,
        wait: bool = False,
        timeout: Any = None,
        **kwargs: Any,
    ) -> int:
        # SDK: wait_motion (default True) runs wait_move() before the modbus write
        self._rec(
            "set_gripper_g2_position",
            pos,
            speed=speed,
            force=force,
            wait=wait,
            wait_motion=kwargs.get("wait_motion", True),
        )
        self._g2_mm = float(pos)
        return 0

    def get_gripper_g2_position(self, **kwargs: Any) -> tuple[int, int]:
        return 0, int(self._g2_mm)  # SDK returns int mm

    # -- linear track (rail) --------------------------------------------------------------
    # NOTE: no get_linear_track_sn / get_linear_track_version — SDK 1.18.5 has none.
    def get_linear_track_registers(self, **kwargs: Any) -> tuple[int, Any]:
        self._rec("get_linear_track_registers")
        if self.simulation_robot:
            return 0, []  # @xarm_is_not_simulation_mode(ret=(0, [])): bus untouched
        if not self._rail_present:
            return 3, {}  # response timeout: nothing on the RS-485 bus
        return 0, {
            "pos": self._rail_pos_mm,
            "status": (0 if self._rail_on_zero else 2) | (1 if self._homing else 0),
            "error": self._rail_error,
            "is_enabled": int(self._rail_enabled),
            "on_zero": self._rail_on_zero,
            "sci": 1,
            "sco": [0, 0],
        }

    def get_linear_track_on_zero(self) -> tuple[int, int]:
        if not self._rail_present:
            return 3, 0
        return 0, self._rail_on_zero

    def set_linear_track_back_origin(self, wait: bool = True, **kwargs: Any) -> int:
        """SDK 1.18.5 ``set_linear_motor_back_origin``: kwargs ``auto_enable``
        (default True) and ``timeout`` (default 10). The homing write goes out,
        then (``wait``) the register poll loop; then ``auto_enable`` OVERWRITES the
        code with the enable's; finally 80 if the track error register is set."""
        auto_enable = bool(kwargs.get("auto_enable", True))
        timeout = kwargs.get("timeout", 10)
        self._rec("set_linear_track_back_origin", wait=wait, **kwargs)
        if self.simulation_robot:
            return 0  # @xarm_is_not_simulation_mode: bus untouched
        if not self._rail_present:
            return 3  # modbus timeout
        if not timeout or not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 10  # SDK default
        self.homing_started += 1
        code = 0
        if self._rail_error:
            code = 80  # the wait loop returns LINEAR_MOTOR_HAS_FAULT right away
        elif wait:
            code = self._wait_back_origin(float(timeout))
        else:
            self._finish_homing()  # no dynamics on the non-blocking path
        if self.homing_result_code is not None:
            code = int(self.homing_result_code)
        if auto_enable:
            code = self._enable_track(True)  # SDK: ret[0] = set_linear_motor_enable(True)
        return code if self._rail_error == 0 else 80

    def _wait_back_origin(self, timeout: float) -> int:
        """``__wait_linear_motor_back_origin``: 0 on on_zero, 80 on a track error,
        100 on timeout — and 100 when ``connected`` drops (the loop condition)."""
        self._homing = True
        try:
            deadline = time.monotonic() + timeout
            done_at = time.monotonic() + self.homing_duration_s
            while self.connected and time.monotonic() < deadline:
                if self._rail_error:
                    return 80
                if time.monotonic() >= done_at:
                    if self.homing_track_error:
                        self.inject_track_error(int(self.homing_track_error))
                        return 80  # LINEAR_MOTOR_HAS_FAULT: carriage stopped short
                    self._finish_homing()
                    return 0
                time.sleep(0.002)  # the SDK polls the registers every 0.1 s
            return 100  # WAIT_FINISH_TIMEOUT (also the exit code when the link drops)
        finally:
            self._homing = False

    def _finish_homing(self) -> None:
        self._rail_on_zero = 1
        self._rail_pos_mm = 0
        self.homing_completed += 1

    def _enable_track(self, enable: bool) -> int:
        """``set_linear_motor_enable`` semantics: with a track error latched the
        enable does not take and the code is 80."""
        if self._rail_error:
            self._rail_enabled = False
            return 80
        self._rail_enabled = bool(enable)
        return 0

    def set_linear_track_enable(self, enable: bool) -> int:
        self._rec("set_linear_track_enable", enable)
        if not self._rail_present:
            return 3
        return self._enable_track(enable)

    def set_linear_track_speed(self, speed: int) -> int:
        self._rec("set_linear_track_speed", speed)
        if not self._rail_present:
            return 3
        self._rail_speed = int(speed)
        return 0

    @property
    def rail_speed(self) -> int:
        """Positioning speed last written (mm/s); 0 until set."""
        return self._rail_speed

    def get_linear_track_pos(self) -> tuple[int, int]:
        if not self._rail_present:
            return 3, 0
        return 0, self._rail_pos_mm

    def set_linear_track_pos(self, pos: int, wait: bool = True, **kwargs: Any) -> int:
        self._rec("set_linear_track_pos", pos, wait=wait)
        if self._rail_on_zero != 1:
            return 82  # LINEAR_MOTOR_NOT_INIT: homing is mandatory per power-on
        self.rail_pos_commands.append(int(pos))
        self._rail_pos_mm = int(pos)  # instant move (fake has no dynamics)
        return 0

    def clean_linear_track_error(self) -> int:
        self._rec("clean_linear_track_error")
        self._rail_error = 0
        return 0

    # -- report stream -----------------------------------------------------------------------
    def register_report_callback(
        self,
        callback: Callable[[dict], None] | None = None,
        report_cartesian: bool = True,
        report_joints: bool = True,
        report_state: bool = True,
        report_error_code: bool = True,
        report_warn_code: bool = True,
        report_mtable: bool = True,
        report_mtbrake: bool = True,
        report_cmd_num: bool = True,
    ) -> bool:
        """Exact SDK 1.18.5 signature: an unknown keyword (e.g. ``report_mode``)
        is a TypeError here just as on the real ``XArmAPI``."""
        flags = {
            "report_cartesian": report_cartesian,
            "report_joints": report_joints,
            "report_state": report_state,
            "report_error_code": report_error_code,
            "report_warn_code": report_warn_code,
            "report_mtable": report_mtable,
            "report_mtbrake": report_mtbrake,
            "report_cmd_num": report_cmd_num,
        }
        self._rec("register_report_callback", **flags)
        if callback is not None:
            self._callbacks.append((callback, flags))
        if not self._report_threads_running:
            self._report_threads_running = True
            if self._report_stream is not None:
                t = threading.Thread(
                    target=self._stream_reader, daemon=True, name="fake-report-stream"
                )
                t.start()
                self._threads.append(t)
            elif self._auto_report_hz > 0:
                t = threading.Thread(target=self._auto_report, daemon=True, name="fake-report-auto")
                t.start()
                self._threads.append(t)
            else:
                self.emit_report()  # SDK's report thread pushes immediately
        return True

    def _fire_report(self, joints: list[float], cartesian: list[float], cmdnum: int) -> None:
        """Build the payload exactly like SDK ``_report_callback`` (no ``mode`` key)."""
        for cb, flags in list(self._callbacks):
            data: dict[str, Any] = {}
            if flags["report_cartesian"]:
                data["cartesian"] = list(cartesian)
            if flags["report_joints"]:
                data["joints"] = list(joints)
            if flags["report_error_code"]:
                data["error_code"] = self.error_code
            if flags["report_warn_code"]:
                data["warn_code"] = self.warn_code
            if flags["report_state"]:
                data["state"] = self.state
            if flags["report_mtable"]:
                data["mtable"] = [True] * 7
            if flags["report_mtbrake"]:
                data["mtbrake"] = [True] * 7
            if flags["report_cmd_num"]:
                data["cmdnum"] = cmdnum
            cb(data)

    def emit_report(
        self,
        q: list[float] | None = None,
        tcp: list[float] | None = None,
        tau: list[float] | None = None,
        mode: int | None = None,
        state: int | None = None,
        cmdnum: int = 0,
    ) -> None:
        """Push one report; ``mode``/``state`` update the attributes (as the SDK
        report thread does) — they are NOT part of the payload."""
        if tau is not None:
            self.joints_torque = list(tau)
        if mode is not None:
            self.mode = mode
        if state is not None:
            self.state = state
        self._fire_report(
            list(q) if q is not None else list(self._q),
            list(tcp) if tcp is not None else [0.0] * 6,
            cmdnum,
        )

    def _auto_report(self) -> None:
        period = 1.0 / self._auto_report_hz
        while self._report_threads_running and self.connected:
            self.emit_report()
            time.sleep(period)

    def _stream_reader(self) -> None:
        """Faithful stand-in for the SDK report thread: reads the 30003 byte
        stream (here: the test replayer), parses frames, updates the cached
        mode/state/torque like ``__handle_report_real``, fires callbacks."""
        try:
            sock = socket.create_connection(self._report_stream, timeout=2.0)
        except OSError:
            return
        sock.settimeout(0.1)
        splitter = FrameSplitter()
        while self._report_threads_running and self.connected:
            try:
                data = sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            for frame in splitter.feed(data):
                parsed = parse_real_frame(frame)
                self.joints_torque = parsed["torques"]
                self.mode = parsed["mode"]
                self.state = parsed["state"]
                self._fire_report(parsed["joints"], parsed["cartesian"], parsed["cmdnum"])
        try:
            sock.close()
        except OSError:
            pass

    def stop_fakes(self) -> None:
        self._report_threads_running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
