"""Stateful double of the XArmAPI surface the driver uses (02-hardware §11).

Encodes the SDK gotchas as behavior: errors reset mode to 0; servo sends
return 1 (error latched) / 9 (not ready) / -8 (joint-limit reject);
``clean_error()`` alone is not readiness; unhomed rail returns 82; absent
rail returns code 3; sim-mode controllers answer track calls with a bogus SN.
Records ``sent_joints`` + a full ``calls`` log; ``emit_report`` (or the
optional 30003 stream reader) fires registered report callbacks.
"""

from __future__ import annotations

import math
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


@dataclass
class FaultScript:
    """Latch a controller error at the Nth set_servo_angle_j call (1-based)."""

    fault_at_tick: int | None = None
    error_code: int = 24
    servo_return: int = 1
    drop_mode_to_0: bool = True


class FakeXArmAPI:
    """Constructor mirrors XArmAPI(ip, is_radian=..., report_type=..., ...)."""

    def __init__(
        self,
        port: str = "192.168.1.235",
        is_radian: bool = False,
        do_not_open: bool = False,
        *,
        sn: str = "XA7-FAKE-0001",
        version: str = "v2.6.107",
        initial_q: list[float] | None = None,
        has_rail: bool = False,
        rail_homed: bool = False,
        rail_sn: str = "AL1300FAKE1234",
        gripper_fw: str = "3.4.3",
        fault_script: FaultScript | None = None,
        auto_report_hz: float = 0.0,
        report_stream: tuple[str, int] | None = None,
        motion_enable_fails: bool = False,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        self.ctor_args = {"port": port, "is_radian": is_radian, **kwargs}
        self.ip = port
        self.is_radian = is_radian
        self.check_joint_limit = kwargs.get("check_joint_limit", True)
        self.report_type = kwargs.get("report_type", "rich")
        self.connected = True
        self.sn = sn
        self.version = version
        self.mode = 0
        self.state = 2  # sleeping/standby after a fresh connect
        self.error_code = 0
        self.warn_code = 0
        self.motion_enabled = False
        self.motion_enable_fails = motion_enable_fails
        self.joints_torque = [0.0] * 7
        self._q = list(initial_q) if initial_q is not None else [0.0] * 7
        self._clock = clock
        self.fault_script = fault_script
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
        self._rail_sn = rail_sn
        self._rail_on_zero = 1 if rail_homed else 0
        self._rail_pos_mm = 0
        self._rail_enabled = False
        self._rail_speed = 0
        self.rail_pos_commands: list[int] = []
        # report plumbing
        self._callbacks: list[Callable[[dict], None]] = []
        self._auto_report_hz = auto_report_hz
        self._report_stream = report_stream
        self._report_threads_running = False
        self._threads: list[threading.Thread] = []

    # -- recording helper ------------------------------------------------------
    def _rec(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def call_names(self) -> list[str]:
        return [name for name, _, _ in self.calls]

    # -- error injection ---------------------------------------------------------
    def inject_error(self, code: int, warn: int = 0, drop_mode: bool = True) -> None:
        """Latch a controller error (e.g. 24 speed, 111 rail comms with
        drop_mode=False — a rail drop does not stop arm motion)."""
        self.error_code = code
        self.warn_code = warn
        if drop_mode:
            self.mode = 0  # errors silently reset the controller to mode 0
            self.state = 4

    # -- lifecycle / mode / state -------------------------------------------------
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
        self.state = 2  # mode change moves state out of ready
        return 0

    def set_state(self, state: int = 0) -> int:
        self._rec("set_state", state)
        if state == 0:
            if self.motion_enabled and self.error_code == 0:
                self.state = 0
            # else: stays not-ready (latched error / not enabled)
        else:
            self.state = state
        return 0

    def get_err_warn_code(self, show: bool = False, **kwargs: Any) -> tuple[int, list[int]]:
        return 0, [self.error_code, self.warn_code]

    def get_servo_angle(
        self, servo_id: Any = None, is_radian: Any = None, is_real: bool = False
    ) -> tuple[int, list[float]]:
        self._rec("get_servo_angle", is_real=is_real)
        return 0, list(self._q)

    def emergency_stop(self) -> None:
        self._rec("emergency_stop")
        self.state = 4

    # -- mode-1 servo streaming -----------------------------------------------------
    def set_servo_angle_j(
        self, angles: list[float], speed: Any = None, mvacc: Any = None,
        mvtime: Any = None, is_radian: Any = None, **kwargs: Any,
    ) -> int:
        self.servo_calls += 1
        script = self.fault_script
        if script is not None and script.fault_at_tick == self.servo_calls:
            self.inject_error(script.error_code, drop_mode=script.drop_mode_to_0)
            script.fault_at_tick = None  # fire once
            return script.servo_return
        if self.error_code != 0:
            return 1  # HAS_ERROR until cleaned
        if self.mode != 1 or self.state != 0 or not self.motion_enabled:
            return 9  # state not ready
        if self.check_joint_limit:
            for value, (lo, hi) in zip(angles[:7], XARM7_LIMITS, strict=False):
                if not lo <= float(value) <= hi:
                    return -8  # SDK-side OUT_OF_RANGE
        self._q = [float(v) for v in angles[:7]]
        self.sent_joints.append((self._clock(), list(self._q)))
        return 0

    # -- backstops (§6) ---------------------------------------------------------------
    def set_tcp_load(self, weight: float, cog: list[float]) -> int:
        self._rec("set_tcp_load", weight, tuple(cog))
        return 0

    def set_gravity_direction(self, direction: list[float]) -> int:
        self._rec("set_gravity_direction", tuple(direction))
        return 0

    def set_collision_sensitivity(self, value: int, wait: bool = True) -> int:
        self._rec("set_collision_sensitivity", value)
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

    def set_collision_rebound(self, on: bool) -> int:
        self._rec("set_collision_rebound", on)
        return 0

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

    def set_gripper_position(self, pos: int, wait: bool = False, **kwargs: Any) -> int:
        self._rec("set_gripper_position", pos, wait=wait)
        self._gripper_pulse = int(pos)
        return 0

    def get_gripper_position(self) -> tuple[int, int]:
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

    def set_external_device_monitor_params(self, dev_type: int = 1,
                                           frequency: int = 10) -> int:
        self._rec("set_external_device_monitor_params", dev_type=dev_type,
                  frequency=frequency)
        return 0

    # -- G2 gripper --------------------------------------------------------------------
    def set_gripper_g2_position(self, pos: float, speed: int = 100, force: int = 50,
                                wait: bool = False, **kwargs: Any) -> int:
        self._rec("set_gripper_g2_position", pos, speed=speed, force=force, wait=wait)
        self._g2_mm = float(pos)
        return 0

    def get_gripper_g2_position(self) -> tuple[int, float]:
        return 0, self._g2_mm

    # -- linear track (rail) --------------------------------------------------------------
    def get_linear_track_registers(self, **kwargs: Any) -> tuple[int, dict]:
        self._rec("get_linear_track_registers")
        if not self._rail_present:
            return 3, {}  # response timeout: nothing on the RS-485 bus
        return 0, {
            "pos": self._rail_pos_mm, "status": 0, "error": 0,
            "is_enabled": int(self._rail_enabled), "on_zero": self._rail_on_zero,
        }

    def get_linear_track_sn(self) -> tuple[int, str]:
        self._rec("get_linear_track_sn")
        if not self._rail_present:
            return 3, ""
        return 0, self._rail_sn

    def get_linear_track_on_zero(self) -> tuple[int, int]:
        return 0, self._rail_on_zero

    def set_linear_track_back_origin(self, wait: bool = True, **kwargs: Any) -> int:
        self._rec("set_linear_track_back_origin", wait=wait)
        self._rail_on_zero = 1
        self._rail_pos_mm = 0
        return 0

    def set_linear_track_enable(self, enable: bool) -> int:
        self._rec("set_linear_track_enable", enable)
        self._rail_enabled = bool(enable)
        return 0

    def set_linear_track_speed(self, speed: int) -> int:
        self._rec("set_linear_track_speed", speed)
        self._rail_speed = int(speed)
        return 0

    def get_linear_track_pos(self) -> tuple[int, int]:
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
        return 0

    # -- report stream -----------------------------------------------------------------------
    def register_report_callback(self, callback: Callable[[dict], None] = None,
                                 **kwargs: Any) -> bool:
        self._rec("register_report_callback")
        if callback is not None:
            self._callbacks.append(callback)
        if not self._report_threads_running:
            self._report_threads_running = True
            if self._report_stream is not None:
                t = threading.Thread(target=self._stream_reader, daemon=True,
                                     name="fake-report-stream")
                t.start()
                self._threads.append(t)
            elif self._auto_report_hz > 0:
                t = threading.Thread(target=self._auto_report, daemon=True,
                                     name="fake-report-auto")
                t.start()
                self._threads.append(t)
            else:
                self.emit_report()  # SDK's report thread pushes immediately
        return True

    def emit_report(self, q: list[float] | None = None, tcp: list[float] | None = None,
                    tau: list[float] | None = None, mode: int | None = None,
                    state: int | None = None, cmdnum: int = 0) -> None:
        if tau is not None:
            self.joints_torque = list(tau)
        data = {
            "joints": list(q) if q is not None else list(self._q),
            "cartesian": list(tcp) if tcp is not None else [0.0] * 6,
            "mode": self.mode if mode is None else mode,
            "state": self.state if state is None else state,
            "cmdnum": cmdnum,
            "error_code": self.error_code,
            "warn_code": self.warn_code,
        }
        for cb in list(self._callbacks):
            cb(data)

    def _auto_report(self) -> None:
        period = 1.0 / self._auto_report_hz
        while self._report_threads_running and self.connected:
            self.emit_report()
            time.sleep(period)

    def _stream_reader(self) -> None:
        """Faithful stand-in for the SDK report thread: reads the 30003 byte
        stream (here: the test replayer), parses frames, fires callbacks."""
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
                report = {
                    "joints": parsed["joints"],
                    "cartesian": parsed["cartesian"],
                    "mode": parsed["mode"],
                    "state": parsed["state"],
                    "cmdnum": parsed["cmdnum"],
                    "error_code": self.error_code,
                    "warn_code": self.warn_code,
                }
                for cb in list(self._callbacks):
                    cb(report)
        try:
            sock.close()
        except OSError:
            pass

    def stop_fakes(self) -> None:
        self._report_threads_running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
