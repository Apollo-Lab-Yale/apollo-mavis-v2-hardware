"""Persisted arm -> NIC mapping (02-hardware §7.1): nic_map.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .types import NicMapEntry

DEFAULT_STATE_PATH = Path("~/.config/apollo-xarm7/nic_map.json")


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
