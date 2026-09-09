"""apollo-mavis-v2-hardware: real xArm7 drivers for the apollo-mavis-v2 stack.

Implements core's ArmInterface/CameraInterface/WorkcellInterface over the
xArm-Python-SDK (pinned 1.18.5), plus NetworkManager auto-matching (netsetup),
V4L2/RealSense camera backends and a read-only controller state monitor
(``ArmStateMonitor``, phase-09a) with an explicit operator maintenance channel
(``maintenance()``, phase-09b; ``home_rail`` — the one motion op — phase-09c).
The driver never homes the rail at connect (``RailNotHomedError`` by default,
phase-09c); phase-09d adds ``XArmDriver.home_rail()`` for the runtime's
operator-confirmed, twin-planned rail-homing maintenance motion on a driver
connected with ``rail_homing: "allow_unhomed"``. Phase-12 adds
``XArmDriver.connect(readonly=True)`` (state-only connection for the runtime's
idle arm reader; ``READONLY_ALLOWED_SDK_METHODS`` is its complete SDK surface)
and depth on ``RealSenseCamera`` (``CameraConfig.depth``). Depends only on
apollo-mavis-v2-core.
"""

from . import units
from .backstops import BACKSTOP_SDK_METHODS, apply_backstops, expected_backstop_sequence
from .config import ServoLimits, XArmDriverConfig
from .driver import (
    READONLY_ALLOWED_SDK_ATTRS,
    READONLY_ALLOWED_SDK_METHODS,
    ArmFaultedError,
    DriverPhase,
    RecoveryResult,
    XArmDriver,
)
from .events import (
    DriverEvent,
    FaultEvent,
    GripperFaultEvent,
    JitterWarning,
    RailEvent,
    RecoveredEvent,
    ReseedEvent,
    StaleEvent,
    StudioConflictWarning,
    TickStats,
)
from .grippers import ClassicGripper, G2Gripper, GripperBackend, NoGripper, make_gripper
from .monitor import (
    HOME_RAIL_Q_TOL_RAD,
    HOME_RAIL_SDK_WAIT_S,
    HOME_RAIL_TIMEOUT_S,
    MAINTENANCE_OPS,
    MAINTENANCE_SDK_METHODS,
    READ_ONLY_SDK_ATTRS,
    READ_ONLY_SDK_METHODS,
    ArmMonitorSample,
    ArmMonitorStatus,
    ArmStateMonitor,
    MaintenanceOp,
    MaintenanceOutcome,
    controller_error_title,
)
from .netsetup import ArmNet, NetSetup
from .rail import (
    UNKNOWN_RAIL_POS_M,
    RailController,
    RailHomeOutcome,
    RailNotHomedError,
    RailPhase,
)
from .workcell import ArmBringupStatus, HardwareWorkcell

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "units",
    # driver
    "XArmDriver",
    "XArmDriverConfig",
    "ServoLimits",
    "DriverPhase",
    "ArmFaultedError",
    "RecoveryResult",
    "READONLY_ALLOWED_SDK_METHODS",  # connect(readonly=True) call surface (phase-12)
    "READONLY_ALLOWED_SDK_ATTRS",
    "apply_backstops",
    "expected_backstop_sequence",
    "BACKSTOP_SDK_METHODS",
    # events
    "DriverEvent",
    "FaultEvent",
    "RecoveredEvent",
    "ReseedEvent",
    "StudioConflictWarning",
    "RailEvent",
    "GripperFaultEvent",
    "StaleEvent",
    "JitterWarning",
    "TickStats",
    # grippers / rail
    "GripperBackend",
    "ClassicGripper",
    "G2Gripper",
    "NoGripper",
    "make_gripper",
    "RailController",
    "RailPhase",
    "RailHomeOutcome",  # XArmDriver.home_rail() result (phase-09d)
    "UNKNOWN_RAIL_POS_M",
    "RailNotHomedError",  # core's class (re-exported for convenience)
    # read-only monitor (phase-09a) + maintenance channel (phase-09b)
    "ArmStateMonitor",
    "ArmMonitorSample",
    "ArmMonitorStatus",
    "MaintenanceOp",
    "MaintenanceOutcome",
    "MAINTENANCE_OPS",
    "MAINTENANCE_SDK_METHODS",
    "HOME_RAIL_TIMEOUT_S",
    "HOME_RAIL_SDK_WAIT_S",
    "HOME_RAIL_Q_TOL_RAD",
    "READ_ONLY_SDK_METHODS",
    "READ_ONLY_SDK_ATTRS",
    "controller_error_title",
    # netsetup / workcell
    "NetSetup",
    "ArmNet",
    "HardwareWorkcell",
    "ArmBringupStatus",
]
