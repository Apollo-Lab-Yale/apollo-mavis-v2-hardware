"""apollo-xarm7-hardware: real xArm7 drivers for the apollo-xarm7 stack.

Implements core's ArmInterface/CameraInterface/WorkcellInterface over the
xArm-Python-SDK (pinned 1.18.5), plus NetworkManager auto-matching (netsetup)
and V4L2/RealSense camera backends. Depends only on apollo-xarm7-core.
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
    # netsetup / workcell
    "NetSetup",
    "ArmNet",
    "HardwareWorkcell",
    "ArmBringupStatus",
]
