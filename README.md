# apollo-xarm7-hardware

Real xArm7 hardware backends for the apollo-xarm7 stack: XArmDriver (mode-1
servo streaming, recovery state machine, rail, grippers), NetworkManager
auto-matching (netsetup), V4L2/RealSense camera backends, and the
HardwareWorkcell assembly. Depends only on apollo-xarm7-core and the pinned
xArm-Python-SDK. See docs/design/02-hardware.md in the workspace.
