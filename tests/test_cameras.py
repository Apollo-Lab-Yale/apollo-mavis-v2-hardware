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
    def __init__(self, path, backend, honor=True, fps_clamp=None, fail_reads=False, fixed=None):
        self.path = path
        self.backend = backend
        self.honor = honor
        self.fps_clamp = fps_clamp
        self.fail_reads = fail_reads
        self.fixed = fixed or {}  # props the driver refuses to change (e.g. a depth node's Z16)
        self.props: dict[int, float] = {}
        self.set_order: list[int] = []
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.set_order.append(prop)
        if prop in self.fixed:
            self.props[prop] = self.fixed[prop]
        elif prop == FakeCv2.CAP_PROP_FPS and self.fps_clamp is not None:
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

    def __init__(self, per_path=None, **capture_kwargs):
        self.capture_kwargs = capture_kwargs
        self.per_path = per_path or {}  # path -> FakeCapture kwargs override
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
        cap = FakeCapture(path, backend, **{**self.capture_kwargs, **self.per_path.get(path, {})})
        self.captures.append(cap)
        return cap

    def cvtColor(self, img, code):
        assert code == self.COLOR_BGR2RGB
        return img[..., ::-1].copy()


def _cfg(cam_id="cam0", **kw) -> CameraConfig:
    return CameraConfig(
        id=cam_id,
        kind="v4l2",
        device_path="/dev/v4l/by-id/usb-FAKE_Cam_1234-video-index0",
        resolution=(6, 4),
        fps=30,
        **kw,
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


# -- USB-serial addressing (RealSense D435i colour over UVC) -----------------------------------
D435I_SERIAL = "322143060792"


def _fake_sysfs(tmp_path, serial=D435I_SERIAL):
    """videoN/device -> <usb>:<iface> dir; <usb>/serial holds the USB serial.

    Mirrors the lab machine: video0 = depth (interface 0), video1 = its metadata
    node (index 1), video4 = colour (interface 3) of an Intel (8086) RealSense;
    video7 = another vendor's camera.
    """
    root = tmp_path / "video4linux"
    root.mkdir()
    usb = tmp_path / "devices" / "4-2"
    usb.mkdir(parents=True)
    (usb / "serial").write_text(serial + "\n")
    (usb / "idVendor").write_text("8086\n")
    other = tmp_path / "devices" / "6-2"
    other.mkdir()
    (other / "serial").write_text("999999999999\n")
    (other / "idVendor").write_text("046d\n")
    for n, dev, iface, index in (
        (4, usb, "03", 0),
        (0, usb, "00", 0),
        (1, usb, "00", 1),
        (7, other, "00", 0),
    ):
        iface_dir = dev / f"{dev.name}:1.{int(iface)}"
        iface_dir.mkdir(exist_ok=True)
        (iface_dir / "bInterfaceNumber").write_text(iface + "\n")
        node = root / f"video{n}"
        node.mkdir()
        (node / "index").write_text(f"{index}\n")
        (node / "device").symlink_to(iface_dir)
    return root


def test_v4l2_nodes_by_usb_serial_orders_interfaces_and_skips_metadata(tmp_path):
    from apollo_mavis_v2_hardware.cameras.opencv_camera import v4l2_nodes_by_usb_serial

    root = _fake_sysfs(tmp_path)
    assert v4l2_nodes_by_usb_serial(D435I_SERIAL, root) == ["/dev/video0", "/dev/video4"]
    assert v4l2_nodes_by_usb_serial("999999999999", root) == ["/dev/video7"]
    assert v4l2_nodes_by_usb_serial("nope", root) == []
    assert v4l2_nodes_by_usb_serial(D435I_SERIAL, tmp_path / "missing") == []


def _serial_cfg(**kw) -> CameraConfig:
    base: dict = {
        "id": "view_wrist",
        "kind": "v4l2",
        "serial": D435I_SERIAL,
        "fourcc": "YUYV",
        "resolution": (6, 4),
        "fps": 30,
    }
    base.update(kw)
    return CameraConfig(**base)


def test_serial_config_tries_interfaces_until_fourcc_honored(tmp_path):
    """The depth node keeps Z16 whatever we set -> skipped; the colour node wins."""
    root = _fake_sysfs(tmp_path)
    z16 = FakeCv2().VideoWriter_fourcc(*"Z16 ")
    yuyv = FakeCv2().VideoWriter_fourcc(*"YUYV")
    cv2 = FakeCv2(per_path={"/dev/video0": {"fixed": {FakeCv2.CAP_PROP_FOURCC: z16}}})
    cam = OpenCVCamera(_serial_cfg(), cv2_mod=cv2, sysfs_root=root)
    cam.start()
    try:
        assert cam.device_path == "/dev/video4"
        assert [c.path for c in cv2.captures] == ["/dev/video0", "/dev/video4"]
        assert cv2.captures[0].released  # the rejected depth node is released
        assert cv2.captures[1].props[FakeCv2.CAP_PROP_FOURCC] == yuyv  # cfg.fourcc, not MJPG
        time.sleep(0.05)
        assert cam.latest() is not None
    finally:
        cam.stop()


def test_serial_config_without_matching_node_raises(tmp_path):
    root = _fake_sysfs(tmp_path)
    cam = OpenCVCamera(_serial_cfg(serial="000000000000"), cv2_mod=FakeCv2(), sysfs_root=root)
    with pytest.raises(CameraInitError, match="USB serial"):
        cam.start()
    z16 = FakeCv2().VideoWriter_fourcc(*"Z16 ")
    cv2 = FakeCv2(fixed={FakeCv2.CAP_PROP_FOURCC: z16})  # every node refuses YUYV
    with pytest.raises(CameraInitError, match="video0.*not honored.*video4.*not honored"):
        OpenCVCamera(_serial_cfg(), cv2_mod=cv2, sysfs_root=root).start()
    with pytest.raises(CameraInitError, match="device_path or serial"):
        OpenCVCamera(
            CameraConfig(id="c", kind="v4l2", device_path="/dev/video0").model_copy(
                update={"device_path": None}
            )
        )


# -- RealSense cold-boot wake ----------------------------------------------------------------
def test_usb_vendor_of_node_reads_sysfs(tmp_path):
    from apollo_mavis_v2_hardware.cameras.opencv_camera import usb_vendor_of_node

    root = _fake_sysfs(tmp_path)
    assert usb_vendor_of_node("/dev/video4", root) == "8086"
    assert usb_vendor_of_node("/dev/video7", root) == "046d"
    assert usb_vendor_of_node("/dev/video99", root) is None


def test_realsense_node_triggers_wake_once_non_realsense_does_not(tmp_path):
    root = _fake_sysfs(tmp_path)
    calls = []
    z16 = FakeCv2().VideoWriter_fourcc(*"Z16 ")
    cv2 = FakeCv2(per_path={"/dev/video0": {"fixed": {FakeCv2.CAP_PROP_FOURCC: z16}}})
    # Two RealSense opens (depth rejected, colour accepted) -> the wake hook runs per
    # open attempt here because the test hook has no once-per-process memory; the
    # real hook (wake_realsense) keeps that state itself.
    cam = OpenCVCamera(_serial_cfg(), cv2_mod=cv2, sysfs_root=root, rs_wake=lambda: calls.append(1))
    cam.start()
    try:
        assert cam.device_path == "/dev/video4" and len(calls) == 2
    finally:
        cam.stop()
    other = CameraConfig(id="c", kind="v4l2", serial="999999999999", resolution=(6, 4), fps=30)
    calls.clear()
    cam2 = OpenCVCamera(other, cv2_mod=FakeCv2(), sysfs_root=root, rs_wake=lambda: calls.append(1))
    cam2.start()
    try:
        assert cam2.device_path == "/dev/video7" and calls == []  # Logitech: no wake
    finally:
        cam2.stop()


def test_wake_realsense_runs_tool_once_and_tolerates_absence(monkeypatch):
    from apollo_mavis_v2_hardware.cameras import opencv_camera as mod

    monkeypatch.setattr(mod, "_rs_wake_done", False)
    runs = []

    class Proc:
        returncode = 0
        stdout = "RealSense D435I 243522071002 5.15.1"
        stderr = ""

    def runner(cmd, **kw):
        runs.append((tuple(cmd), kw["timeout"]))
        return Proc()

    assert mod.wake_realsense(runner=runner) is True
    assert mod.wake_realsense(runner=runner) is True  # cached: no second run
    assert runs == [(("rs-enumerate-devices", "-s"), mod.RS_WAKE_TIMEOUT_S)]
    assert mod.wake_realsense(runner=runner, force=True) is True and len(runs) == 2

    monkeypatch.setattr(mod, "_rs_wake_done", False)

    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    assert mod.wake_realsense(runner=missing) is False
    assert mod._rs_wake_done is True  # a missing tool is not retried for every camera

    monkeypatch.setattr(mod, "_rs_wake_done", False)

    class Bad(Proc):
        returncode = 1
        stderr = "No device detected"

    assert mod.wake_realsense(runner=lambda cmd, **kw: Bad()) is False
