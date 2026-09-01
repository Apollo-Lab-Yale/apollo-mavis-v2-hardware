"""V4L2 camera backend over OpenCV (02-hardware §8), LeRobot-style.

cv2 is an OPTIONAL import: the module loads (and enumeration returns [])
without it; opening a camera then raises CameraInitError. Tests inject a
fake cv2 module via ``cv2_mod``.
"""

from __future__ import annotations

import glob as _glob
import threading
import time
from typing import Any

from apollo_xarm7_core import CameraFrame, CameraInitError
from apollo_xarm7_core.interfaces import CameraInterface
from apollo_xarm7_core.schemas import CameraConfig

try:  # optional dependency (heavy wheel; absent in slim test envs)
    import cv2 as _cv2
except ImportError:  # pragma: no cover - exercised via cv2_mod injection
    _cv2 = None

BY_ID_GLOB = "/dev/v4l/by-id/*-video-index0"
BY_PATH_GLOB = "/dev/v4l/by-path/*-video-index0"
DEV_GLOB = "/dev/video*"
MAX_CONSECUTIVE_FAILURES = 5
STALE_AFTER_S = 0.5


class OpenCVCamera(CameraInterface):
    """One UVC camera opened by stable device PATH (never index)."""

    def __init__(self, cfg: CameraConfig, cv2_mod: Any = None) -> None:
        if cfg.kind != "v4l2":
            raise CameraInitError("camera", f"OpenCVCamera got kind={cfg.kind!r}")
        if not cfg.device_path:
            raise CameraInitError("camera", f"{cfg.id}: v4l2 camera needs device_path")
        self.cfg = cfg
        self._cv2 = cv2_mod if cv2_mod is not None else _cv2
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None
        self._seq = 0
        self._failed = False
        self._reopened_once = False

    # -- properties -----------------------------------------------------------
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

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        if self._cv2 is None:
            raise CameraInitError("camera", f"{self.cfg.id}: cv2 not installed")
        self._cv2.setNumThreads(1)  # BEFORE any capture work
        self._cap = self._open_and_configure()
        self._failed = False
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"cam.{self.cfg.id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

    def latest(self) -> CameraFrame | None:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if time.monotonic() - frame.t_mono > STALE_AFTER_S:
            return None
        return frame

    # -- internals ---------------------------------------------------------------
    def _open_and_configure(self) -> Any:
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.cfg.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraInitError("camera", f"{self.cfg.id}: cannot open {self.cfg.device_path}")
        w, h = self.cfg.resolution
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # order matters: FOURCC first (UVC reaches 30 fps at 640x480+ only in
        # MJPG), then FPS, then W/H — verify every set() by read-back (V4L2
        # silently clamps)
        for prop, value, label in (
            (cv2.CAP_PROP_FOURCC, fourcc, "FOURCC=MJPG"),
            (cv2.CAP_PROP_FPS, float(self.cfg.fps), f"FPS={self.cfg.fps}"),
            (cv2.CAP_PROP_FRAME_WIDTH, float(w), f"WIDTH={w}"),
            (cv2.CAP_PROP_FRAME_HEIGHT, float(h), f"HEIGHT={h}"),
        ):
            cap.set(prop, value)
            actual = cap.get(prop)
            if int(actual) != int(value):
                cap.release()
                raise CameraInitError(
                    "camera",
                    f"{self.cfg.id}: {label} not honored (got {actual})",
                )
        return cap

    def _capture_loop(self) -> None:
        cv2 = self._cv2
        failures = 0
        while self._running:
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    if not self._reopened_once:  # release + reopen ONCE
                        self._reopened_once = True
                        failures = 0
                        try:
                            self._cap.release()
                            self._cap = self._open_and_configure()
                            continue
                        except Exception:  # noqa: BLE001 — fall through to failed
                            pass
                    self._failed = True  # UI greys the tile; never crash the workcell
                    self._running = False
                    return
                continue
            failures = 0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR->RGB once, here
            self._seq += 1
            frame = CameraFrame(
                camera_id=self.cfg.id,
                rgb=rgb,
                t_mono=time.monotonic(),
                wallclock_ns=time.time_ns(),
                seq=self._seq,
            )
            with self._lock:
                self._frame = frame

    # -- enumeration ----------------------------------------------------------------
    @staticmethod
    def find_cameras(
        cv2_mod: Any = None,
        glob_fn: Any = _glob.glob,
        test_read: bool = True,
    ) -> list[dict[str, Any]]:
        """Enumerate V4L2 capture nodes by stable path.

        Prefer /dev/v4l/by-id (stable across reboots), fall back to
        /dev/v4l/by-path (port identity, serial-less cameras), then raw
        /dev/video*. One test read() per node — the odd per-UVC metadata node
        opens but yields no frames.
        """
        cv2 = cv2_mod if cv2_mod is not None else _cv2
        paths = sorted(glob_fn(BY_ID_GLOB)) or sorted(glob_fn(BY_PATH_GLOB)) or sorted(
            glob_fn(DEV_GLOB)
        )
        found: list[dict[str, Any]] = []
        for path in paths:
            entry = {"kind": "v4l2", "device_path": path, "label": path.rsplit("/", 1)[-1]}
            if test_read:
                if cv2 is None:
                    continue
                cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
                ok = cap.isOpened()
                if ok:
                    ok, frame = cap.read()
                    ok = bool(ok) and frame is not None
                cap.release()
                if not ok:
                    continue
            found.append(entry)
        return found