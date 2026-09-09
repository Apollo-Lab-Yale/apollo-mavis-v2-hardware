"""Import guard: hardware depends only on core (pyproject ``banned-api`` + a subprocess
check that importing the package pulls in none of the banned modules — phase-12 adds
``dora`` and ``pyarrow``, the runtime's dora_bridge is the only place they may live)."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BANNED = (
    "dora",
    "pyarrow",
    "mujoco",
    "fastapi",
    "torch",
    "lerobot",
    "mink",
    "apollo_mavis_v2_sim",
    "apollo_mavis_v2_runtime",
)
PROBE = r"""
import json, sys
import apollo_mavis_v2_hardware
import apollo_mavis_v2_hardware.cameras.realsense_camera
import apollo_mavis_v2_hardware.cameras.opencv_camera
import apollo_mavis_v2_hardware.monitor
import apollo_mavis_v2_hardware.workcell
import apollo_mavis_v2_hardware.netsetup
banned = json.loads(sys.argv[1])
loaded = sorted(m for m in sys.modules if m.split(".")[0] in banned)
print(json.dumps(loaded))
"""


def test_pyproject_bans_dora_and_pyarrow() -> None:
    text = (REPO / "pyproject.toml").read_text()
    assert "[tool.ruff.lint.flake8-tidy-imports.banned-api]" in text
    for name in BANNED:
        assert f'"{name}".msg = ' in text, f"{name} missing from banned-api"
    assert "TID251" in text  # the rule that enforces the list


def test_importing_the_package_loads_no_banned_module() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(list(BANNED))],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], f"banned modules imported: {loaded}"
