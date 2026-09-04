"""netsetup CLI: python -m apollo_mavis_v2_hardware.netsetup <command>.

Commands: verify | match | reconcile | install | status.
``reconcile`` prints the plan only; pass ``--apply`` to execute it.
``--fixtures FILE...`` replays recorded nmcli transcripts instead of touching
the real NetworkManager (dry-run / test harness); the real runner is wired
only when no fixtures are given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import NetSetup, TranscriptRunner
from . import install as install_mod
from .state import DEFAULT_STATE_PATH
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
    p.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH,
                   help="nic_map.json path")
    p.add_argument("--fixtures", nargs="+", default=None, metavar="FILE",
                   help="replay nmcli transcripts (never touches NetworkManager)")
    p.add_argument("--apply", action="store_true",
                   help="reconcile: execute the plan (default: plan only)")
    p.add_argument("--check", action="store_true", help="install: verify only")
    p.add_argument("--yes", action="store_true", help="install: skip confirmation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        if args.check:
            problems = install_mod.check()
            for prob in problems:
                print(f"PROBLEM: {prob}")
            print("ok" if not problems else f"{len(problems)} problem(s)")
            return 0 if not problems else 1
        return install_mod.install(assume_yes=args.yes)

    arms: list[ArmNet] = list(args.arm)
    if args.config:
        arms.extend(_arms_from_config(args.config))
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
        for prob in ns.install_problems:
            print(f"WARNING: {prob}")
        bad = 0
        for name, res in results.items():
            mark = "OK " if res.ok else "MISS"
            bad += 0 if res.ok else 1
            print(f"[{mark}] {name}: probe={res.probe} nic={res.ifname or '-'} "
                  f"uuid={res.profile_uuid or '-'} {res.detail}")
        return 0 if bad == 0 else 1
    if args.command == "match":
        results = ns.match()
        bad = 0
        for name, res in results.items():
            mark = "OK " if res.ok else "FAIL"
            bad += 0 if res.ok else 1
            print(f"[{mark}] {name}: probe={res.probe} nic={res.ifname or '-'} "
                  f"mac={res.mac or '-'} uuid={res.profile_uuid or '-'} {res.detail}")
        return 0 if bad == 0 else 1
    if args.command == "reconcile":
        plan = ns.reconcile(apply=args.apply)
        print(plan.render())
        print(f"\n{len(plan.steps)} step(s); "
              + ("APPLIED" if plan.applied else "plan only — pass --apply to execute"))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
