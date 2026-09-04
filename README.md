# apollo-mavis-v2-hardware

Real hardware backends for the MAVIS v2 cell (two UFACTORY xArm7 arms on linear
tracks): XArmDriver (mode-1
servo streaming, recovery state machine, rail, grippers), NetworkManager
auto-matching (netsetup), V4L2/RealSense camera backends, and the
HardwareWorkcell assembly. Depends only on apollo-mavis-v2-core and the pinned
xArm-Python-SDK. See docs/design/02-hardware.md in the workspace.
