"""Persisted arm -> NIC mapping (02-hardware §7.1): nic_map.json.

Two locations exist:

* ``USER_STATE_PATH``   ``~/.config/apollo-mavis-v2/nic_map.json`` — written by a
  user-run ``match`` (runtime session start, manual CLI).
* ``SYSTEM_STATE_PATH`` ``/etc/apollo-mavis-v2/nic_map.json`` — written by root:
  ``netsetup install`` and the NetworkManager dispatcher hook (§7.4). Root-owned,
  world-readable, so every account's ``verify`` can consume it.

Resolution (``resolve_state_path``): explicit ``--state`` > user path if it exists
> system path if it exists > user path (fresh default for writing).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .types import NicMapEntry

USER_STATE_PATH = Path("~/.config/apollo-mavis-v2/nic_map.json")
SYSTEM_STATE_PATH = Path("/etc/apollo-mavis-v2/nic_map.json")
DEFAULT_STATE_PATH = USER_STATE_PATH  # backwards-compatible alias (write default)


def user_state_path() -> Path:
    return USER_STATE_PATH.expanduser()


def resolve_state_path(
    explicit: Path | None = None,
    user_path: Path | None = None,
    system_path: Path | None = None,
) -> Path:
    """Pick the nic_map.json to use: explicit > existing user > existing system > user.

    ``user_path``/``system_path`` default to the module constants *at call time*
    (tests monkeypatch them)."""
    if explicit is not None:
        return Path(explicit).expanduser()
    user = Path(user_path or USER_STATE_PATH).expanduser()
    if user.is_file():
        return user
    system = Path(system_path or SYSTEM_STATE_PATH).expanduser()
    if system.is_file():
        return system
    return user


def load_nic_map(path: Path = DEFAULT_STATE_PATH) -> dict[str, NicMapEntry]:
    path = path.expanduser()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, NicMapEntry] = {}
    for arm, entry in raw.items():
        try:
            out[arm] = NicMapEntry(
                mac=entry["mac"],
                ifname=entry.get("ifname", ""),
                profile_uuid=entry["profile_uuid"],
                arm_ip=entry.get("arm_ip", ""),
                ts=float(entry.get("ts", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            continue  # tolerate foreign/stale entries
    return out


def save_nic_map(entries: dict[str, NicMapEntry], path: Path = DEFAULT_STATE_PATH) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        arm: {
            "mac": e.mac,
            "ifname": e.ifname,
            "profile_uuid": e.profile_uuid,
            "arm_ip": e.arm_ip,
            "ts": e.ts,
        }
        for arm, e in entries.items()
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)  # atomic: a crashed save never corrupts


def holdoff_path(state_path: Path) -> Path:
    """Sibling stamp of nic_map.json: wall-clock time of the last repair() round
    that re-probed arms which were merely *unreachable* on their pinned NIC."""
    return state_path.expanduser().with_suffix(".holdoff")


def read_holdoff(state_path: Path) -> float | None:
    """Wall-clock time of the last unreachable-arm re-probe; None = never."""
    try:
        text = holdoff_path(state_path).read_text().strip()
        return float(text) if text else None
    except (OSError, ValueError):
        return None


def write_holdoff(state_path: Path, ts: float) -> None:
    path = holdoff_path(state_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{ts:.3f}\n")
    except OSError:
        pass  # best effort: a missing stamp only means one extra probe round
