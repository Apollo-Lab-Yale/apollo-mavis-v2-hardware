"""30003 report-stream replayer + frame builder/parser (02-hardware §11).

Frame layout (87 bytes, 'real' report):
  [0:4]   u32 big-endian total size (87)
  [4]     state_mode byte: (mode << 4) | state
  [5:7]   u16 big-endian cmd_num
  [7:35]  7 x f32 little-endian actual_joint_angle (rad)
  [35:59] 6 x f32 little-endian actual_tcp_pose (mm + rad)
  [59:87] 7 x f32 little-endian estimated_joint_torque (N*m)
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

REAL_FRAME_SIZE = 87
MAX_FRAME_SIZE = 4096  # resync guard: no sane frame is bigger


def build_real_frame(
    q: Sequence[float],
    tcp_pose: Sequence[float] | None = None,
    tau: Sequence[float] | None = None,
    state: int = 0,
    mode: int = 1,
    cmd_num: int = 0,
) -> bytes:
    tcp_pose = list(tcp_pose) if tcp_pose is not None else [0.0] * 6
    tau = list(tau) if tau is not None else [0.0] * 7
    assert len(q) == 7 and len(tcp_pose) == 6 and len(tau) == 7
    payload = (
        struct.pack("B", ((mode & 0x0F) << 4) | (state & 0x0F))
        + struct.pack(">H", cmd_num & 0xFFFF)
        + struct.pack("<7f", *[float(v) for v in q])
        + struct.pack("<6f", *[float(v) for v in tcp_pose])
        + struct.pack("<7f", *[float(v) for v in tau])
    )
    frame = struct.pack(">I", REAL_FRAME_SIZE) + payload
    assert len(frame) == REAL_FRAME_SIZE
    return frame


def parse_real_frame(frame: bytes) -> dict:
    """Parse one complete 87-byte frame (size prefix included)."""
    assert len(frame) == REAL_FRAME_SIZE
    state_mode = frame[4]
    (cmd_num,) = struct.unpack(">H", frame[5:7])
    joints = list(struct.unpack("<7f", frame[7:35]))
    tcp = list(struct.unpack("<6f", frame[35:59]))
    tau = list(struct.unpack("<7f", frame[59:87]))
    return {
        "mode": state_mode >> 4,
        "state": state_mode & 0x0F,
        "cmdnum": cmd_num,
        "joints": joints,
        "cartesian": tcp,
        "torques": tau,
    }


class FrameSplitter:
    """Incremental stream splitter: yields complete frames, resyncs on garbage
    (truncated/torn frames or a bogus size prefix), never raises."""

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, data: bytes) -> list[bytes]:
        self._buf += data
        frames: list[bytes] = []
        while len(self._buf) >= 4:
            (size,) = struct.unpack(">I", self._buf[:4])
            if size < 5 or size > MAX_FRAME_SIZE:
                # garbage: shift one byte and hunt for the next sane prefix
                self._buf = self._buf[1:]
                continue
            if len(self._buf) < size:
                break  # torn frame: wait for more bytes
            frame, self._buf = self._buf[:size], self._buf[size:]
            if size == REAL_FRAME_SIZE:
                frames.append(frame)
            # non-real sizes are skipped whole (other report flavors)
        return frames


def write_fixture(path: str | Path, frames: Iterable[bytes]) -> None:
    Path(path).write_bytes(b"".join(frames))


def load_fixture(path: str | Path) -> list[bytes]:
    splitter = FrameSplitter()
    return splitter.feed(Path(path).read_bytes())


class ReportReplayer:
    """TCP server on 127.0.0.1:<ephemeral> pushing frames at a fixed rate.

    ``pause()``/``resume()`` model report silence (staleness tests); frames
    cycle until stop(). One client at a time (like the controller port)."""

    def __init__(self, frames: Sequence[bytes], rate_hz: float = 100.0) -> None:
        if not frames:
            raise ValueError("need at least one frame")
        self._frames = list(frames)
        self._period = 1.0 / rate_hz
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.address: tuple[str, int] = self._server.getsockname()
        self._paused = threading.Event()  # set = paused
        self._running = False
        self._thread: threading.Thread | None = None
        self.frames_sent = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._serve, name="report-replayer",
                                        daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._running = False
        try:
            self._server.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _serve(self) -> None:
        self._server.settimeout(0.2)
        conn: socket.socket | None = None
        idx = 0
        while self._running:
            if conn is None:
                try:
                    conn, _ = self._server.accept()
                except (TimeoutError, OSError):
                    continue
            if self._paused.is_set():
                time.sleep(0.005)
                continue
            try:
                conn.sendall(self._frames[idx % len(self._frames)])
            except OSError:
                conn = None
                continue
            self.frames_sent += 1
            idx += 1
            time.sleep(self._period)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
