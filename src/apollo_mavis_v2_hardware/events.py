"""DriverEvent union (02-hardware §1) — bounded-deque events drained by runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FaultEvent:
    """A fault was detected (controller error, bad servo return, link loss)."""

    arm_id: str
    # "servo" | "monitor" | "report" | "external" | "user_stop" | "user" (start of an
    # operator-requested recovery) | "latch" (recovery gave up: budget / unrecoverable)
    source: str
    code: int  # API return code or controller error code
    error_code: int = 0  # controller error at capture time
    warn_code: int = 0
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class RecoveredEvent:
    """Recovery sequence completed; streaming resumed."""

    arm_id: str
    error_code: int
    t_mono: float = 0.0


@dataclass(frozen=True)
class ReseedEvent:
    """Streamer re-seeded from the measured position — runtime MUST re-anchor
    its IK target pose (overview §6 watchdog invariant)."""

    arm_id: str
    q: tuple[float, ...]  # measured joints, rad (7)
    t_mono: float = 0.0


@dataclass(frozen=True)
class StudioConflictWarning:
    """Mode/state changed under us with no error code — UFACTORY Studio
    'Live control' is fighting the SDK stream (02-hardware §9)."""

    arm_id: str
    mode: int
    state: int
    detail: str = "close UFACTORY Studio live control"
    t_mono: float = 0.0


@dataclass(frozen=True)
class RailEvent:
    """Rail phase change or rail fault (error 111 latches only the rail)."""

    arm_id: str
    phase: str  # RailPhase.name
    code: int = 0
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class GripperFaultEvent:
    """Gripper error that survived one clean+re-enable cycle (arm unaffected)."""

    arm_id: str
    code: int
    detail: str = ""
    t_mono: float = 0.0


@dataclass(frozen=True)
class StaleEvent:
    """30003 report silence crossed ``stale_after_s`` (True) or recovered (False).

    Addition to the 02-hardware §1 union: §3.4 requires 'stale=True + a
    DriverEvent'; this is that event.
    """

    arm_id: str
    stale: bool
    age_s: float = 0.0
    t_mono: float = 0.0


@dataclass(frozen=True)
class JitterWarning:
    """Sustained streamer tick p99 above threshold (GIL-pressure cue, §3.3)."""

    arm_id: str
    p99_s: float
    t_mono: float = 0.0


DriverEvent = (
    FaultEvent
    | RecoveredEvent
    | ReseedEvent
    | StudioConflictWarning
    | RailEvent
    | GripperFaultEvent
    | StaleEvent
    | JitterWarning
)


@dataclass
class TickStats:
    """Servo-streamer tick statistics (02-hardware §3.1)."""

    ticks: int = 0
    late_ticks: int = 0  # tick period > 2 * dt
    faults: int = 0
    p50_s: float = 0.0
    p99_s: float = 0.0
    periods_s: tuple[float, ...] = field(default=(), repr=False)
