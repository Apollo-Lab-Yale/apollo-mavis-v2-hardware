"""netsetup CLI: python -m apollo_mavis_v2_hardware.netsetup <command>.

Commands: verify | match [--repair] | reconcile [--apply] | install | status.
``reconcile`` prints the plan only; pass ``--apply`` to execute it.
``match --repair`` is the NM dispatcher's entry point (02-hardware §7.4):
verify first, re-probe only what is broken, never disturb a healthy arm.
``--fixtures FILE...`` replays recorded nmcli transcripts instead of touching
the real NetworkManager (dry-run / test harness); the real runner is wired
only when no fixtures are given.

State file: ``--state`` > ``~/.config/apollo-mavis-v2/nic_map.json`` if it
exists > ``/etc/apollo-mavis-v2/nic_map.json`` if it exists > the user path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import NetSetup, TranscriptRunner
from . import install as install_mod
from .state import SYSTEM_STATE_PATH, USER_STATE_PATH
from .types import ArmNet


def _parse_arm(spec: str) -> ArmNet:
    """--arm name=ip[/prefix][,host=IP]"""
    name, _, rest = spec.partition("=")
    if not rest:
        raise argparse.ArgumentTypeError(f"--arm needs name=ip[/prefix], got {spec!r}")
    host_ip = None
    if "," in rest:
        rest, _, hostpart = rest.partition(",")
        if hostpart.startswith("host="):
            host_ip = hostpart[5:]
    ip, _, prefix = rest.partition("/")
    return ArmNet(name=name, ip=ip, prefix=int(prefix or 24), host_ip=host_ip)


def _arms_from_config(path: str) -> list[ArmNet]:
    from apollo_mavis_v2_core import load_workcell_config

    cfg = load_workcell_config(path)
    return [ArmNet(name=a.id, ip=a.ip) for a in cfg.arms if a.ip]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m apollo_mavis_v2_hardware.netsetup")
    p.add_argument("command", choices=["verify", "match", "reconcile", "install", "status"])
    p.add_argument("--arm", action="append", type=_parse_arm, default=[],
                   help="name=ip[/prefix][,host=IP]; repeatable")
    p.add_argument("--config", help="workcell YAML (arms read from it)")
    p.add_argument("--state", type=Path, default=None,
                   help=f"nic_map.json path (default: {USER_STATE_PATH} if present, else "
                        f"{SYSTEM_STATE_PATH} if present, else the former)")
    p.add_argument("--fixtures", nargs="+", default=None, metavar="FILE",
                   help="replay nmcli transcripts (never touches NetworkManager)")
    p.add_argument("--apply", action="store_true",
                   help="reconcile: execute the plan (default: plan only)")
    p.add_argument("--repair", action="store_true",
                   help="match: verify first and re-probe only broken arms (dispatcher mode)")
    p.add_argument("--deadline", type=float, default=None,
                   help="match --repair: seconds to wait for an unreachable arm to boot")
    p.add_argument("--holdoff", type=float, default=None,
                   help="match --repair: min seconds between re-probes of unreachable arms")
    p.add_argument("--check", action="store_true", help="install: verify only")
    p.add_argument("--yes", action="store_true", help="install: skip confirmation")
    p.add_argument("--dispatcher-only", action="store_true",
                   help="install: only the NM dispatcher hook + system state dir")
    p.add_argument("--python", type=Path, default=None,
                   help="install: interpreter baked into the dispatcher hook "
                        "(default: the running venv python)")
    p.add_argument("--user", default=None,
                   help="install: account added to the netdev group / checked by --check "
                        "(default: $SUDO_USER, else $USER; e.g. the operations account)")
    return p


def _print_results(results, install_problems, warnings, notes, fail_mark: str) -> int:
    for prob in install_problems:
        print(f"WARNING: {prob}")
    for warn in warnings:
        print(f"WARNING: {warn}")
    for note in notes:
        print(f"note: {note}")
    bad = 0
    for name, res in results.items():
        mark = "OK " if res.ok else fail_mark
        bad += 0 if res.ok else 1
        print(f"[{mark}] {name}: probe={res.probe} nic={res.ifname or '-'} "
              f"mac={res.mac or '-'} uuid={res.profile_uuid or '-'} {res.detail}")
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms: list[ArmNet] = list(args.arm)
    if args.config:
        arms.extend(_arms_from_config(args.config))

    if args.command == "install":
        if args.check:
            expected = None
            if arms:
                from .dispatcher import default_python, render_dispatcher

                expected = render_dispatcher(args.python or default_python(), arms)
            problems = install_mod.check(user=args.user, expected_script=expected)
            for prob in problems:
                print(f"PROBLEM: {prob}")
            print("ok" if not problems else f"{len(problems)} problem(s)")
            return 0 if not problems else 1
        return install_mod.install(
            assume_yes=args.yes, user=args.user, python=args.python, arms=arms,
            dispatcher_only=args.dispatcher_only,
        )

    if args.fixtures:
        # dry-run harness: EVERYTHING stubbed — transcripts for nmcli, inert
        # seams for ping/ip/sockets/carrier. Never touches real network state.
        import subprocess

        ns = NetSetup(
            arms,
            state_path=args.state,
            run=TranscriptRunner.from_files(*args.fixtures),
            run_sys=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
            tcp_connect=lambda ip, port, timeout: "unreachable",
            carrier_of=lambda dev: False,
        )
    else:
        ns = NetSetup(arms, state_path=args.state)  # the ONLY real-runner wiring

    if args.command == "status":
        for nic in ns.status():
            carrier = "carrier" if nic.carrier else "no-carrier"
            print(f"{nic.dev:18s} {nic.type:10s} {nic.state:12s} {carrier:10s} "
                  f"{nic.mac:17s} {nic.connection}")
        return 0
    if args.command == "verify":
        results = ns.verify()
        return _print_results(results, ns.install_problems, ns.warnings, [], "MISS")
    if args.command == "match":
        if args.repair:
            kwargs = {}
            if args.deadline is not None:
                kwargs["deadline_s"] = args.deadline
            if args.holdoff is not None:
                kwargs["holdoff_s"] = args.holdoff
            if args.fixtures:
                kwargs.update(deadline_s=0.0, sleep=lambda s: None)
            results = ns.repair(**kwargs)
            print(f"state: {ns.state_path}")
            return _print_results(results, ns.install_problems, ns.warnings, ns.notes, "FAIL")
        results = ns.match()
        print(f"state: {ns.state_path}")
        return _print_results(results, [], ns.warnings, [], "FAIL")
    if args.command == "reconcile":
        plan = ns.reconcile(apply=args.apply)
        print(plan.render())
        print(f"\n{len(plan.steps)} step(s); "
              + ("APPLIED" if plan.applied else "plan only — pass --apply to execute"))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
