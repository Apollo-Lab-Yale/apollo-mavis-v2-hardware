"""One-time system install (02-hardware §7.3 / §7.4).

Four root-owned artefacts make netsetup work for every account, headless:

1. polkit ``.pkla`` grant — Ubuntu 22.04 ships polkitd 0.105 = LocalAuthority
   backend ONLY; JavaScript ``.rules`` are silently ignored. Lets ``netdev``
   members run ``nmcli`` unprivileged from SSH / services.
2. ``netdev`` group membership for the operator (re-login required).
3. the NetworkManager dispatcher hook (``dispatcher.py``) — root re-runs
   ``match --repair`` on ethernet link events, no polkit involved.
4. ``/etc/apollo-mavis-v2/`` (root, world-readable) holding the system
   ``nic_map.json`` the dispatcher writes and every account's ``verify`` reads;
   seeded by one ``match`` through sudo when the arms are reachable.

``install --check`` is read-only; ``install`` prints the exact sudo commands and
executes them only after confirmation (``--yes`` skips the prompt).
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import dispatcher as dispatcher_mod
from .dispatcher import DISPATCHER_PATH
from .probe import SysRunner, TcpConnect, sys_run, tcp_probe
from .state import SYSTEM_STATE_PATH
from .types import ArmNet

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
SYSTEM_STATE_DIR = SYSTEM_STATE_PATH.parent


def invoking_user(default: str = "$USER") -> str:
    """The operator, even under ``sudo python -m ... install`` (SUDO_USER)."""
    return os.environ.get("SUDO_USER") or os.environ.get("USER") or default


@dataclass(frozen=True)
class InstallStep:
    """One sudo action: the shell line shown to the operator + the argv run."""

    shell: str
    argv: tuple[str, ...]
    note: str = ""


def _shell_join(argv: Sequence[str]) -> str:
    import shlex

    return shlex.join(list(argv))


def plan_install(
    user: str | None = None,
    python: Path | None = None,
    arms: Sequence[ArmNet] = (),
    *,
    dispatcher_only: bool = False,
    initial_match: bool = True,
    script_src: Path | None = None,
    pkla_path: Path = PKLA_PATH,
    dispatcher_path: Path = DISPATCHER_PATH,
    state_path: Path = SYSTEM_STATE_PATH,
) -> list[InstallStep]:
    """The exact sudo steps, in order. ``script_src`` is where the rendered
    dispatcher script sits on disk (``install`` writes a temp file); when None a
    placeholder is shown (plan / print only)."""
    user = user or invoking_user()
    python = python or dispatcher_mod.default_python()
    arms = list(arms)
    steps: list[InstallStep] = []
    if not dispatcher_only:
        steps.append(InstallStep(
            f"sudo tee {pkla_path} <<'EOF'\n{PKLA_CONTENT}EOF",
            ("sudo", "sh", "-c", f"cat > {pkla_path} <<'EOF'\n{PKLA_CONTENT}EOF"),
            "polkit grant (LocalAuthority .pkla; polkitd watches the dir)",
        ))
        steps.append(InstallStep(
            f"sudo usermod -aG {NETDEV_GROUP} {user}   # re-login required",
            ("sudo", "usermod", "-aG", NETDEV_GROUP, user),
            "netdev group for unprivileged nmcli",
        ))
    state_dir = state_path.parent
    steps.append(InstallStep(
        f"sudo install -d -m 755 -o root -g root {state_dir}",
        ("sudo", "install", "-d", "-m", "755", "-o", "root", "-g", "root", str(state_dir)),
        "system state dir (root-owned, world-readable nic_map.json)",
    ))
    if arms:
        src = str(script_src) if script_src is not None else "<rendered 90-mavis-netsetup>"
        steps.append(InstallStep(
            f"sudo install -D -m 755 -o root -g root {src} {dispatcher_path}",
            ("sudo", "install", "-D", "-m", "755", "-o", "root", "-g", "root", src,
             str(dispatcher_path)),
            "NM dispatcher hook: root re-runs `match --repair` on ethernet up/down",
        ))
        if initial_match and not dispatcher_only:
            argv = ("sudo", *dispatcher_mod.match_argv(python, arms, state_path))
            steps.append(InstallStep(
                _shell_join(argv),
                argv,
                "seed the system nic_map + pin both profiles (root: no polkit needed yet)",
            ))
    return steps


def install_commands(user: str | None = None, **kwargs) -> list[str]:
    """The exact sudo commands (printed for the operator to confirm/run)."""
    return [s.shell for s in plan_install(user, **kwargs)]


def check(
    run_sys: SysRunner = sys_run,
    pkla_path: Path = PKLA_PATH,
    user: str | None = None,
    dispatcher_path: Path = DISPATCHER_PATH,
    expected_script: str | None = None,
) -> list[str]:
    """Verify grant + group + dispatcher hook; returns problems (empty = healthy).
    verify() runs this so a missing piece becomes an actionable landing-page
    warning, not a mid-bring-up failure."""
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
    if NETDEV_GROUP not in groups and not (user is None and os.geteuid() == 0):
        # root (the dispatcher) needs no polkit grant / group
        problems.append(
            f"user not in {NETDEV_GROUP!r} group (run: sudo usermod -aG {NETDEV_GROUP} "
            f"{user or invoking_user()}; then re-login)"
        )
    try:
        script = dispatcher_path.read_text()
        if not os.access(dispatcher_path, os.X_OK):
            problems.append(f"NM dispatcher hook {dispatcher_path} is not executable")
        elif expected_script is not None and script != expected_script:
            problems.append(
                f"NM dispatcher hook {dispatcher_path} differs from the rendered script "
                f"(run: netsetup install --dispatcher-only)"
            )
    except FileNotFoundError:
        problems.append(
            f"NM dispatcher hook missing: {dispatcher_path} (run: netsetup install) — "
            f"arm NICs will not re-match at boot / hot-plug without it"
        )
    except PermissionError:
        problems.append(f"cannot read {dispatcher_path} (permissions)")
    return problems


def arms_reachable(arms: Sequence[ArmNet], tcp_connect: TcpConnect = tcp_probe) -> list[str]:
    """Names of arms whose control box does NOT answer (open/refused) right now."""
    return [a.name for a in arms if tcp_connect(a.ip, 502, 1.0) == "unreachable"]


def install(
    run_sys: SysRunner = sys_run,
    assume_yes: bool = False,
    pkla_path: Path = PKLA_PATH,
    user: str | None = None,
    *,
    python: Path | None = None,
    arms: Sequence[ArmNet] = (),
    dispatcher_only: bool = False,
    dispatcher_path: Path = DISPATCHER_PATH,
    state_path: Path = SYSTEM_STATE_PATH,
    tcp_connect: TcpConnect = tcp_probe,
    out=None,
) -> int:
    """Print the sudo commands; execute only after explicit confirmation.

    polkitd watches its directory — no restart needed; the group change needs
    a re-login; NetworkManager-dispatcher picks up new scripts immediately.
    """
    out = out or sys.stdout
    python = python or dispatcher_mod.default_python()
    arms = list(arms)
    user = user or invoking_user("")
    expected = dispatcher_mod.render_dispatcher(python, arms, state_path) if arms else None
    problems = check(run_sys, pkla_path, user or None, dispatcher_path, expected)
    if dispatcher_only:
        problems = [p for p in problems if "dispatcher" in p]
    state_exists = state_path.expanduser().is_file()
    if not problems and (dispatcher_only or state_exists or not arms):
        print("netsetup install: already healthy (grant + group + dispatcher present)",
              file=out)
        return 0
    if arms:
        unreachable = arms_reachable(arms, tcp_connect)
    else:
        unreachable = []
        print("note: no --arm/--config given -> dispatcher hook and initial match skipped",
              file=out)
    initial_match = bool(arms) and not unreachable and not dispatcher_only
    if arms and unreachable and not dispatcher_only:
        print(f"note: arm(s) {', '.join(unreachable)} do not answer TCP 502 now -> initial "
              f"match skipped; the dispatcher will match on the next link event, or run:\n"
              f"  {_shell_join(('sudo', *dispatcher_mod.match_argv(python, arms, state_path)))}",
              file=out)
    script_tmp: Path | None = None
    if arms and expected is not None:
        fd, name = tempfile.mkstemp(prefix="90-mavis-netsetup.", suffix=".sh")
        with os.fdopen(fd, "w") as fh:
            fh.write(expected)
        script_tmp = Path(name)
    steps = plan_install(
        user or None, python, arms, dispatcher_only=dispatcher_only,
        initial_match=initial_match, script_src=script_tmp, pkla_path=pkla_path,
        dispatcher_path=dispatcher_path, state_path=state_path,
    )
    if not problems:  # healthy artefacts, only the initial match is missing
        steps = [s for s in steps if "match" in s.argv]
    print("netsetup install needs the following one-time sudo commands:\n", file=out)
    for step in steps:
        print(f"# {step.note}\n{step.shell}\n", file=out)
    if not assume_yes:
        answer = input("Run them now via sudo? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted; run the commands above manually", file=out)
            return 1
    try:
        for step in steps:
            proc = run_sys(list(step.argv))
            if proc.returncode != 0:
                print(f"step failed (rc={proc.returncode}): {step.shell.splitlines()[0]}",
                      file=out)
                return proc.returncode
    finally:
        if script_tmp is not None:
            try:
                script_tmp.unlink()
            except OSError:
                pass
    print("installed; re-login for the group change to take effect", file=out)
    return 0
