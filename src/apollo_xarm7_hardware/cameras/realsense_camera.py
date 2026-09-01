"""Intel RealSense camera backend (02-hardware §8), import-guarded.

pyrealsense2 is an OPTIONAL extra (``pip install apollo-xarm7-hardware[realsense]``);
this module imports without it and raises CameraInitError on use. Tests inject
a fake ``rs`` module via ``rs_mod``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from apollo_xarm7_core import CameraFrame, CameraInitError
from apollo_xarm7_core.interfaces import CameraInterface
from apollo_xarm7_core.schemas import CameraConfig

try:  # optional [realsense] extra
    import pyrealsense2 as _rs
except ImportError:  # pragma: no cover - exercised via rs_mod injection
    _rs = None

WARMUP_S = 1.0  # sensor needs >= 1 s before frames are usable (LeRobot)
STALE_AFTER_S = 0.5


class RealSenseCamera(CameraInterface):
    """One RealSense, addressed by serial (never by index)."""

    def __init__(self, cfg: CameraConfig, rs_mod: Any = None, warmup_s: float = WARMUP_S):
        if cfg.kind != "realsense":
            raise CameraInitError("camera", f"RealSenseCamera got kind={cfg.kind!r}")
        if not cfg.serial:
            raise CameraInitError("camera", f"{cfg.id}: realsense camera needs serial")
        self.cfg = cfg
        self._rs = rs_mod if rs_mod is not None else _rs
        self._warmup_s = warmup_s
        self._pipeline: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._seq = 0
        self._failed = False

    @property
    def camera_id(self) -> str:
        return self.cfg.id

    @property
    def resolution(self) -> tuple[int, int]:
        return self.cfg.resolution

    @property
    def fps(self) -> float:
        return float(self.cfg.fps)

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        if self._running:
            return
        rs = self._rs
        if rs is None:
            raise CameraInitError(
                "camera",
                f"{self.cfg.id}: pyrealsense2 not installed "
                "(pip install apollo-xarm7-hardware[realsense])",
            )
        try:
            self._pipeline = self._start_pipeline()
        except Exception:  # noqa: BLE001 — wedged after unclean shutdown
            self._hardware_reset()
            try:
                self._pipeline = self._start_pipeline()
            except Exception as exc:  # noqa: BLE001
                raise CameraInitError(
                    "camera", f"{self.cfg.id}: pipeline start failed twice: {exc}"
                ) from exc
        if self._warmup_s > 0:
            time.sleep(self._warmup_s)
        self._failed = False
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"cam.{self.cfg.id}", daemon=True
        )
        self._thread.start()

    def _start_pipeline(self) -> Any:
        rs = self._rs
        config = rs.config()
        config.enable_device(str(self.cfg.serial))
        w, h = self.cfg.resolution
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, self.cfg.fps)
        pipeline = rs.pipeline()
        pipeline.start(config)
        return pipeline

    def _hardware_reset(self) -> None:
        rs = self._rs
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == str(self.cfg.serial):
                    dev.hardware_reset()
                    time.sleep(2.0)
                    return
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None

    def latest(self) -> CameraFrame | None:
        with self._lock:
            frame = self._frame
        if frame is None or time.monotonic() - frame.t_mono > STALE_AFTER_S:
            return None
        return frame

    def _capture_loop(self) -> None:
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                rgb = np.asanyarray(color.get_data())  # already RGB (rs.format.rgb8)
            except Exception:  # noqa: BLE001
                self._failed = True
                self._running = False
                return
            self._seq += 1
            frame = CameraFrame(
                camera_id=self.cfg.id,
                rgb=np.ascontiguousarray(rgb),
                t_mono=time.monotonic(),
                wallclock_ns=time.time_ns(),
                seq=self._seq,
            )
            with self._lock:
                self._frame = frame

    @staticmethod
    def find_cameras(rs_mod: Any = None) -> list[dict[str, Any]]:
        """Enumerate RealSense devices by serial (empty without pyrealsense2)."""
        rs = rs_mod if rs_mod is not None else _rs
        if rs is None:
            return []
        found: list[dict[str, Any]] = []
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                found.append(
                    {
                        "kind": "realsense",
                        "serial": dev.get_info(rs.camera_info.serial_number),
                        "label": dev.get_info(rs.camera_info.name),
                    }
                )
        except Exception:  # noqa: BLE001
            return []
        return found
