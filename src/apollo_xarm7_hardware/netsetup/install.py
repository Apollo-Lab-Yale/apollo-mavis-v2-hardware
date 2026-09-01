"""One-time polkit grant install (02-hardware §7.3).

Ubuntu 22.04 ships polkitd 0.105 = LocalAuthority (.pkla) backend ONLY —
JavaScript ``.rules`` files are silently ignored (no JS engine in the
binary). Headless/SSH operation therefore needs this ``.pkla`` grant plus
membership in the ``netdev`` group.
"""

from __future__ import annotations

import os
from pathlib import Path

from .probe import SysRunner, sys_run

PKLA_PATH = Path("/etc/polkit-1/localauthority/50-local.d/46-apollo-networkmanager.pkla")

PKLA_CONTENT = """\
[apollo: let netdev group manage NetworkManager]
Identity=unix-group:netdev
Action=org.freedesktop.NetworkManager.network-control;org.freedesktop.NetworkManager.settings.modify.system
ResultAny=yes
ResultInactive=yes
ResultActive=yes
"""

NETDEV_GROUP = "netdev"


def install_commands(user: str | None = None) -> list[str]:
    """The exact sudo commands (printed for the operator to confirm/run)."""
    user = user or os.environ.get("USER", "$USER")
    return [
        f"sudo tee {PKLA_PATH} <<'EOF'\n{PKLA_CONTENT}EOF",
        f"sudo usermod -aG {NETDEV_GROUP} {user}   # re-login required",
    ]


def check(
    run_sys: SysRunner = sys_run,
    pkla_path: Path = PKLA_PATH,
    user: str | None = None,
) -> list[str]:
    """Verify grant + group; returns problems (empty = healthy). verify() runs
    this so a missing grant becomes an actionable landing-page warning, not a
    mid-bring-up failure."""
    problems: list[str] = []
    try:
        content = pkla_path.read_text()
        if content.strip() != PKLA_CONTENT.strip():
            problems.append(f"{pkla_path} exists but differs from the expected grant")
    except FileNotFoundError:
        problems.append(f"polkit grant missing: {pkla_path} (run: netsetup install)")
    except PermissionError:
        problems.append(f"cannot read {pkla_path} (permissions)")
    proc = run_sys(["id", "-nG"] + ([user] if user else []))
    groups = proc.stdout.split() if proc.returncode == 0 else []
    if NETDEV_GROUP not in groups:
        problems.append(
            f"user not in {NETDEV_GROUP!r} group (run: sudo usermod -aG {NETDEV_GROUP} "
            f"{user or os.environ.get('USER', '$USER')}; then re-login)"
        )
    return problems


def install(
    run_sys: SysRunner = sys_run,
    assume_yes: bool = False,
    pkla_path: Path = PKLA_PATH,
    user: str | None = None,
) -> int:
    """Print the sudo commands; execute only after explicit confirmation.

    polkitd watches the directory — no restart needed; the group change needs
    a re-login.
    """
    problems = check(run_sys, pkla_path, user)
    if not problems:
        print("netsetup install: already healthy (grant + group present)")
        return 0
    print("netsetup install needs the following one-time sudo commands:\n")
    for cmd in install_commands(user):
        print(cmd, end="\n\n")
    if not assume_yes:
        answer = input("Run them now via sudo? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted; run the commands above manually")
            return 1
    user = user or os.environ.get("USER", "")
    proc = run_sys(
        ["sudo", "sh", "-c", f"cat > {pkla_path} <<'EOF'\n{PKLA_CONTENT}EOF"]
    )
    if proc.returncode != 0:
        print(f"writing {pkla_path} failed (rc={proc.returncode})")
        return proc.returncode
    proc = run_sys(["sudo", "usermod", "-aG", NETDEV_GROUP, user])
    if proc.returncode != 0:
        print(f"usermod failed (rc={proc.returncode})")
        return proc.returncode
    print("installed; re-login for the group change to take effect")
    return 0
