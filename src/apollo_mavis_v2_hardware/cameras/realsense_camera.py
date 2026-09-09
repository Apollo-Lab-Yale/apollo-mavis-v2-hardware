"""Intel RealSense camera backend (02-hardware §8), import-guarded.

pyrealsense2 is an OPTIONAL extra (``pip install apollo-mavis-v2-hardware[realsense]``);
this module imports without it and raises CameraInitError on use. Tests inject
a fake ``rs`` module via ``rs_mod``.

Depth (phase-12; 14-dora §4.2 "Depth"): with ``CameraConfig.depth`` the z16
depth stream is enabled at the colour resolution / fps and published as
``CameraFrame.depth`` ((H, W) uint16 in units of ``CameraFrame.depth_scale_m``,
read ONCE from the depth sensor after the pipeline starts; 0.001 = mm when the
SDK does not answer). With ``align_depth_to_color`` (default) the capture
thread runs ``rs.align(rs.stream.color)`` on every frameset (~2 ms/frame budget
on the lab machine) so depth pixels index the colour image. ``depth: false``
(the default) leaves the colour-only behaviour untouched: no depth stream, no
align object, ``depth is None``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from apollo_mavis_v2_core import CameraFrame, CameraInitError
from apollo_mavis_v2_core.interfaces import CameraInterface
from apollo_mavis_v2_core.schemas import CameraConfig

try:  # optional [realsense] extra
    import pyrealsense2 as _rs
except ImportError:  # pragma: no cover - exercised via rs_mod injection
    _rs = None

WARMUP_S = 1.0  # sensor needs >= 1 s before frames are usable (LeRobot)
STALE_AFTER_S = 0.5
DEFAULT_DEPTH_SCALE_M = 0.001  # z16 in mm — the D4xx default, used when the SDK read fails


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
        self._align: Any = None  # rs.align(rs.stream.color), created once per start()
        self._depth_scale_m = DEFAULT_DEPTH_SCALE_M

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

    @property
    def depth_scale_m(self) -> float:
        """Metres per depth unit (read once after the pipeline starts; 0.001 when
        depth is off or the sensor did not answer)."""
        return self._depth_scale_m

    def start(self) -> None:
        if self._running:
            return
        rs = self._rs
        if rs is None:
            raise CameraInitError(
                "camera",
                f"{self.cfg.id}: pyrealsense2 not installed "
                "(pip install apollo-mavis-v2-hardware[realsense])",
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
        if self.cfg.depth:
            self._depth_scale_m = self._read_depth_scale()
            self._align = rs.align(rs.stream.color) if self.cfg.align_depth_to_color else None
        else:
            self._depth_scale_m = DEFAULT_DEPTH_SCALE_M
            self._align = None
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
        if self.cfg.depth:
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.cfg.fps)
        pipeline = rs.pipeline()
        pipeline.start(config)
        return pipeline

    def _read_depth_scale(self) -> float:
        """``first_depth_sensor().get_depth_scale()`` of the active device, once;
        ``DEFAULT_DEPTH_SCALE_M`` when anything in that chain raises."""
        try:
            profile = self._pipeline.get_active_profile()
            scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception:  # noqa: BLE001 — optional metadata, never fatal
            return DEFAULT_DEPTH_SCALE_M
        if not np.isfinite(scale) or scale <= 0.0:
            return DEFAULT_DEPTH_SCALE_M
        return scale

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
        self._align = None

    def latest(self) -> CameraFrame | None:
        with self._lock:
            frame = self._frame
        if frame is None or time.monotonic() - frame.t_mono > STALE_AFTER_S:
            return None
        return frame

    def _capture_loop(self) -> None:
        while self._running:
            depth_img: np.ndarray | None = None
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if self._align is not None:
                    frames = self._align.process(frames)  # depth pixels -> colour pixels
                color = frames.get_color_frame()
                if not color:
                    continue
                rgb = np.ascontiguousarray(np.asanyarray(color.get_data()))  # RGB (rs.format.rgb8)
                if self.cfg.depth:
                    depth = frames.get_depth_frame()
                    if depth:  # a frameset without depth keeps the colour frame
                        depth_img = np.ascontiguousarray(
                            np.asanyarray(depth.get_data()), dtype=np.uint16
                        )
                        if depth_img.shape != rgb.shape[:2]:
                            depth_img = None  # never publish depth that does not index rgb
            except Exception:  # noqa: BLE001
                self._failed = True
                self._running = False
                return
            self._seq += 1
            frame = CameraFrame(
                camera_id=self.cfg.id,
                rgb=rgb,
                t_mono=time.monotonic(),
                wallclock_ns=time.time_ns(),
                seq=self._seq,
                depth=depth_img,
                depth_scale_m=self._depth_scale_m,
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
