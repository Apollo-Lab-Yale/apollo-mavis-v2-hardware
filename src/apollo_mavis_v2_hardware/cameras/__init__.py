"""Camera backends (02-hardware §8): dispatch + merged enumeration with
RealSense-first ghost dedup (RS color sensors also appear as /dev/video*)."""

from __future__ import annotations

from typing import Any

from apollo_mavis_v2_core import CameraInitError
from apollo_mavis_v2_core.interfaces import CameraInterface
from apollo_mavis_v2_core.schemas import CameraConfig

from .opencv_camera import OpenCVCamera
from .realsense_camera import RealSenseCamera

__all__ = ["OpenCVCamera", "RealSenseCamera", "find_all_cameras", "make_camera"]


def make_camera(cfg: CameraConfig, **backend_kwargs: Any) -> CameraInterface:
    """Dispatch on the tagged-union CameraConfig.kind."""
    if cfg.kind == "v4l2":
        return OpenCVCamera(cfg, **backend_kwargs)
    if cfg.kind == "realsense":
        return RealSenseCamera(cfg, **backend_kwargs)
    raise CameraInitError("camera", f"{cfg.id}: unsupported kind {cfg.kind!r} in hardware")


def _is_realsense_ghost(entry: dict[str, Any]) -> bool:
    """V4L2 node that is actually a RealSense color sensor (by-id symlinks
    contain the vendor string)."""
    path = str(entry.get("device_path", ""))
    return "Intel_RealSense" in path or "RealSense" in path


def find_all_cameras(
    cv2_mod: Any = None,
    rs_mod: Any = None,
    glob_fn: Any = None,
    test_read: bool = True,
) -> list[dict[str, Any]]:
    """Merge both backends; RealSense entries win over their V4L2 ghosts.

    Feeds ``GET /api/cameras`` (runtime). Ordering: RealSense (by serial)
    first, then remaining genuine V4L2 nodes.
    """
    rs_entries = RealSenseCamera.find_cameras(rs_mod)
    kwargs: dict[str, Any] = {"cv2_mod": cv2_mod, "test_read": test_read}
    if glob_fn is not None:
        kwargs["glob_fn"] = glob_fn
    v4l2_entries = OpenCVCamera.find_cameras(**kwargs)
    have_rs = bool(rs_entries)
    merged = list(rs_entries)
    for entry in v4l2_entries:
        if have_rs and _is_realsense_ghost(entry):
            continue  # RS-first dedup
        merged.append(entry)
    return merged
