"""Camera backends: fake cv2 (FOURCC order, read-back, reopen) + RS dedup (§8)."""

import time

import numpy as np
import pytest
from apollo_mavis_v2_core import CameraInitError
from apollo_mavis_v2_core.schemas import CameraConfig

from apollo_mavis_v2_hardware.cameras import find_all_cameras, make_camera
from apollo_mavis_v2_hardware.cameras.opencv_camera import OpenCVCamera
from apollo_mavis_v2_hardware.cameras.realsense_camera import RealSenseCamera

BGR_FRAME = np.zeros((4, 6, 3), dtype=np.uint8)
BGR_FRAME[..., 0] = 10  # B
BGR_FRAME[..., 1] = 20  # G
BGR_FRAME[..., 2] = 30  # R


class FakeCapture:
    def __init__(self, path, backend, honor=True, fps_clamp=None, fail_reads=False):
        self.path = path
        self.backend = backend
        self.honor = honor
        self.fps_clamp = fps_clamp
        self.fail_reads = fail_reads
        self.props: dict[int, float] = {}
        self.set_order: list[int] = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_order.append(prop)
        if prop == FakeCv2.CAP_PROP_FPS and self.fps_clamp is not None:
            self.props[prop] = self.fps_clamp  # V4L2 silently clamps
        else:
            self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        if self.fail_reads:
            return False, None
        return True, BGR_FRAME.copy()

    def release(self):
        self.released = True


class FakeCv2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 6
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    COLOR_BGR2RGB = 4

    def __init__(self, **capture_kwargs):
        self.capture_kwargs = capture_kwargs
        self.captures: list[FakeCapture] = []
        self.num_threads: int | None = None

    def setNumThreads(self, n):
        self.num_threads = n

    def VideoWriter_fourcc(self, *chars):
        code = 0
        for i, ch in enumerate(chars):
            code |= ord(ch) << (8 * i)
        return code

    def VideoCapture(self, path, backend):
        cap = FakeCapture(path, backend, **self.capture_kwargs)
        self.captures.append(cap)
        return cap

    def cvtColor(self, img, code):
        assert code == self.COLOR_BGR2RGB
        return img[..., ::-1].copy()


def _cfg(cam_id="cam0", **kw) -> CameraConfig:
    return CameraConfig(
        id=cam_id, kind="v4l2",
        device_path="/dev/v4l/by-id/usb-FAKE_Cam_1234-video-index0",
        resolution=(6, 4), fps=30, **kw,
    )


def test_configure_order_fourcc_then_fps_then_size():
    cv2 = FakeCv2()
    cam = OpenCVCamera(_cfg(), cv2_mod=cv2)
    cam.start()
    try:
        cap = cv2.captures[0]
        assert cap.set_order == [
            FakeCv2.CAP_PROP_FOURCC,  # MJPG FIRST (UVC 30 fps needs it)
            FakeCv2.CAP_PROP_FPS,
            FakeCv2.CAP_PROP_FRAME_WIDTH,
            FakeCv2.CAP_PROP_FRAME_HEIGHT,
        ]
        assert cv2.num_threads == 1
        assert cap.backend == FakeCv2.CAP_V4L2
    finally:
        cam.stop()


def test_readback_mismatch_raises_camera_init_error():
    cv2 = FakeCv2(fps_clamp=15.0)  # driver silently clamps 30 -> 15
    cam = OpenCVCamera(_cfg(), cv2_mod=cv2)
    with pytest.raises(CameraInitError, match="FPS"):
        cam.start()
    assert cv2.captures[0].released


def test_bgr_to_rgb_once_in_capture_thread():
    cv2 = FakeCv2()
    cam = OpenCVCamera(_cfg(), cv2_mod=cv2)
    cam.start()
    try:
        deadline = time.monotonic() + 2.0
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = cam.latest()
            time.sleep(0.005)
        assert frame is not None
        assert frame.rgb[0, 0].tolist() == [30, 20, 10]  # R,G,B (was B,G,R)
        assert frame.camera_id == "cam0"
    finally:
        cam.stop()


def test_five_read_failures_reopen_once_then_mark_failed():
    cv2 = FakeCv2(fail_reads=True)
    cam = OpenCVCamera(_cfg(), cv2_mod=cv2)
    cam.start()
    deadline = time.monotonic() + 2.0
    while not cam.failed and time.monotonic() < deadline:
        time.sleep(0.005)
    cam.stop()
    assert cam.failed  # UI greys the tile; the workcell never crashes
    assert len(cv2.captures) == 2  # reopened exactly once
    assert cam.latest() is None


def test_find_cameras_prefers_by_id_and_test_reads():
    cv2 = FakeCv2()
    by_id = ["/dev/v4l/by-id/usb-A-video-index0", "/dev/v4l/by-id/usb-B-video-index0"]

    def glob_fn(pattern):
        return by_id if "by-id" in pattern else ["/dev/video0"]

    found = OpenCVCamera.find_cameras(cv2_mod=cv2, glob_fn=glob_fn)
    assert [f["device_path"] for f in found] == by_id  # by-id wins over /dev/video*


class _FakeRsDevice:
    def __init__(self, serial, name):
        self._info = {"serial_number": serial, "name": name}

    def get_info(self, key):
        return self._info[key]


class _FakeRsContext:
    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


class FakeRsModule:
    class camera_info:
        serial_number = "serial_number"
        name = "name"

    def __init__(self, serials):
        self._devices = [_FakeRsDevice(s, f"Intel RealSense D435 ({s})") for s in serials]

    def context(self):
        return _FakeRsContext(self._devices)


def test_find_all_cameras_dedups_realsense_v4l2_ghosts():
    rs = FakeRsModule(["823112061234"])
    cv2 = FakeCv2()

    def glob_fn(pattern):
        if "by-id" in pattern:
            return [
                "/dev/v4l/by-id/usb-Intel_RealSense_D435_823112061234-video-index0",
                "/dev/v4l/by-id/usb-Logitech_C920_5678-video-index0",
            ]
        return []

    merged = find_all_cameras(cv2_mod=cv2, rs_mod=rs, glob_fn=glob_fn, test_read=False)
    kinds = [(m["kind"], m.get("serial") or m.get("device_path")) for m in merged]
    assert kinds[0] == ("realsense", "823112061234")  # RS first
    assert len(merged) == 2  # the RS ghost /dev/v4l node is gone
    assert "Logitech" in merged[1]["device_path"]


def test_realsense_requires_module_and_serial():
    cfg = CameraConfig(id="rs0", kind="realsense", serial="823112061234")
    cam = RealSenseCamera(cfg, rs_mod=None, warmup_s=0.0)
    with pytest.raises(CameraInitError, match="pyrealsense2"):
        cam.start()
    with pytest.raises(Exception):  # noqa: B017 — core validator or our guard
        make_camera(CameraConfig(id="rs1", kind="realsense"))


def test_make_camera_dispatch():
    cam = make_camera(_cfg(), cv2_mod=FakeCv2())
    assert isinstance(cam, OpenCVCamera)
    with pytest.raises(CameraInitError):
        make_camera(CameraConfig(id="s", kind="sim"))
