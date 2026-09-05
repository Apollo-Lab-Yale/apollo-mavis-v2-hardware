"""apollo-mavis-v2-hardware: real xArm7 drivers for the apollo-mavis-v2 stack.

Implements core's ArmInterface/CameraInterface/WorkcellInterface over the
xArm-Python-SDK (pinned 1.18.5), plus NetworkManager auto-matching (netsetup),
V4L2/RealSense camera backends and a read-only controller state monitor
(``ArmStateMonitor``, phase-09a). Depends only on apollo-mavis-v2-core.
"""

from . import units
from .backstops import apply_backstops
from .config import ServoLimits, XArmDriverConfig
from .driver import ArmFaultedError, DriverPhase, XArmDriver
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
    READ_ONLY_SDK_ATTRS,
    READ_ONLY_SDK_METHODS,
    ArmMonitorSample,
    ArmMonitorStatus,
    ArmStateMonitor,
    controller_error_title,
)
from .netsetup import ArmNet, NetSetup
from .rail import RailController, RailPhase
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
    "apply_backstops",
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
    # read-only monitor (phase-09a)
    "ArmStateMonitor",
    "ArmMonitorSample",
    "ArmMonitorStatus",
    "READ_ONLY_SDK_METHODS",
    "READ_ONLY_SDK_ATTRS",
    "controller_error_title",
    # netsetup / workcell
    "NetSetup",
    "ArmNet",
    "HardwareWorkcell",
    "ArmBringupStatus",
]
