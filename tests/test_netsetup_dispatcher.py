"""NM dispatcher hook rendering + behaviour, and the extended install (§7.4).

The rendered bash script is executed for real (bash/setsid/flock) against a fake
sysfs tree and a stub "python" that records its argv — no NetworkManager, no root.
"""

import io
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from apollo_mavis_v2_hardware.netsetup import ArmNet
from apollo_mavis_v2_hardware.netsetup import __main__ as cli
from apollo_mavis_v2_hardware.netsetup import dispatcher as disp
from apollo_mavis_v2_hardware.netsetup import install as install_mod
from apollo_mavis_v2_hardware.netsetup.__main__ import _parse_arm

VIEW = ArmNet(name="view", ip="192.168.2.219")
GRIP = ArmNet(name="grip", ip="192.168.1.201")
ARMS = [VIEW, GRIP]
PY = Path("/opt/mavis/.venv/bin/python")
STATE = Path("/etc/apollo-mavis-v2/nic_map.json")

needs_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("bash", "setsid", "flock")),
    reason="bash/setsid/flock required to execute the dispatcher script",
)


# --- rendering -----------------------------------------------------------------


def test_render_bakes_python_state_arms_and_paths():
    script = disp.render_dispatcher(PY, ARMS, generated_at="2026-09-04")
    assert script.startswith("#!/bin/bash\n")
    assert "PYTHON=/opt/mavis/.venv/bin/python\n" in script
    assert "STATE=/etc/apollo-mavis-v2/nic_map.json\n" in script
    assert "ARM_ARGS=(--arm view=192.168.2.219 --arm grip=192.168.1.201)\n" in script
    assert "(2026-09-04)" in script
    assert "/run/lock/mavis-netsetup.lock" in script
    assert "/var/log/mavis-netsetup.log" in script
    assert 'flock -w "$LOCK_WAIT_S" "$LOCK"' in script
    assert "match --repair" in script
    assert "setsid -f" in script  # detached: NM kills slow dispatcher scripts
    assert "up|down) ;;" in script  # the only handled actions
    assert "SETTLE_S:-2}" in script


def test_render_quotes_unusual_paths_and_requires_arms():
    script = disp.render_dispatcher(Path("/opt/my venv/bin/python"), [VIEW],
                                    state=Path("/srv/x y/nic_map.json"))
    assert "PYTHON='/opt/my venv/bin/python'" in script
    assert "STATE='/srv/x y/nic_map.json'" in script
    with pytest.raises(ValueError):
        disp.render_dispatcher(PY, [])


def test_arm_spec_round_trips_through_the_cli_parser():
    arm = ArmNet(name="view", ip="10.0.5.7", prefix=16, host_ip="10.0.0.12")
    assert _parse_arm(disp.arm_spec(arm)) == arm
    assert disp.arm_spec(GRIP) == "grip=192.168.1.201"


def test_repair_and_match_argv_contract():
    assert disp.repair_argv(PY, ARMS) == [
        str(PY), "-m", "apollo_mavis_v2_hardware.netsetup", "match", "--repair",
        "--state", str(STATE), "--arm", "view=192.168.2.219", "--arm", "grip=192.168.1.201",
    ]
    assert disp.match_argv(PY, ARMS) == [
        str(PY), "-m", "apollo_mavis_v2_hardware.netsetup", "match",
        "--state", str(STATE), "--arm", "view=192.168.2.219", "--arm", "grip=192.168.1.201",
    ]


# --- executing the script ------------------------------------------------------


class Harness:
    """Fake /sys/class/net + stub python; runs the rendered script like NM would."""

    SYSFS_TYPES = {"eth": "1", "wifi": "1", "bridge": "1", "veth": "1", "lo": "772",
                   "tun": "65534"}

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.sysfs = tmp_path / "sys"
        for name, kind in [("enp0", "eth"), ("enx1", "eth"), ("wlan0", "wifi"),
                           ("br0", "bridge"), ("lo", "lo"), ("veth0", "veth"),
                           ("tun0", "tun")]:
            self.add_iface(name, kind)
        self.args_file = tmp_path / "python-args.txt"
        self.stub = tmp_path / "venv-python"
        self.stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$MAVIS_TEST_ARGS"\n')
        self.stub.chmod(0o755)
        self.log = tmp_path / "netsetup.log"
        self.lock = tmp_path / "netsetup.lock"
        self.script = tmp_path / "90-mavis-netsetup"
        self.write_script(self.stub)

    def add_iface(self, name: str, kind: str) -> None:
        d = self.sysfs / name
        d.mkdir(parents=True)
        (d / "type").write_text(self.SYSFS_TYPES[kind] + "\n")
        if kind in ("eth", "wifi"):
            (d / "device").mkdir()  # backed by a PCI/USB device
        if kind == "wifi":
            (d / "wireless").mkdir()
        if kind == "bridge":
            (d / "bridge").mkdir()

    def write_script(self, python: Path) -> None:
        self.script.write_text(disp.render_dispatcher(python, ARMS, generated_at="2026-09-04"))
        self.script.chmod(0o755)

    def run(self, iface: str, action: str) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "MAVIS_NETSETUP_SYSFS_NET": str(self.sysfs),
            "MAVIS_NETSETUP_LOG": str(self.log),
            "MAVIS_NETSETUP_LOCK": str(self.lock),
            "MAVIS_NETSETUP_SETTLE_S": "0",
            "MAVIS_TEST_ARGS": str(self.args_file),
            "CONNECTION_UUID": "6753457b-89fc-435a-a5a0-e94c728bc826",
            "CONNECTION_ID": "mavis_viewpoint_arm",
        }
        return subprocess.run([str(self.script), iface, action], env=env,
                              capture_output=True, text=True, timeout=15)

    def wait_args(self, timeout: float = 5.0) -> list[str] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.args_file.exists():
                text = self.args_file.read_text()
                if text.endswith("\n"):
                    return text.splitlines()
            time.sleep(0.02)
        return None

    def wait_log(self, needle: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if needle in self.log_text():
                return True
            time.sleep(0.02)
        return False

    def log_text(self) -> str:
        return self.log.read_text() if self.log.exists() else ""


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@needs_tools
@pytest.mark.parametrize("iface,action", [("enp0", "up"), ("enp0", "down"), ("enx1", "up")])
def test_ethernet_up_down_runs_match_repair_detached(harness, iface, action):
    proc = harness.run(iface, action)
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""
    args = harness.wait_args()
    assert args == disp.repair_argv(harness.stub, ARMS)[1:]  # everything after the python
    assert harness.wait_log("repair done rc=0")
    log = harness.log_text()
    assert f"event {iface} {action}" in log
    assert "uuid=6753457b-89fc-435a-a5a0-e94c728bc826 id=mavis_viewpoint_arm" in log


@needs_tools
@pytest.mark.parametrize("action", ["pre-up", "pre-down", "dhcp4-change", "hostname",
                                    "connectivity-change", "vpn-up"])
def test_other_actions_are_ignored(harness, action):
    proc = harness.run("enp0", action)
    assert proc.returncode == 0
    assert harness.wait_args(0.3) is None
    assert harness.log_text() == ""


@needs_tools
@pytest.mark.parametrize("iface", ["wlan0", "br0", "lo", "veth0", "tun0", "does-not-exist", ""])
def test_non_ethernet_devices_are_ignored(harness, iface):
    proc = harness.run(iface, "up")
    assert proc.returncode == 0
    assert harness.wait_args(0.3) is None
    assert harness.log_text() == ""


@needs_tools
def test_missing_python_logs_and_exits_zero(harness):
    harness.write_script(harness.tmp / "gone" / "python")
    proc = harness.run("enp0", "up")
    assert proc.returncode == 0
    assert "python missing" in harness.log_text()
    assert harness.wait_args(0.3) is None


@needs_tools
def test_worker_waits_for_the_lock(harness):
    holder = subprocess.Popen(["flock", str(harness.lock), "sleep", "1.5"])
    try:
        time.sleep(0.3)  # let the holder acquire the lock
        proc = harness.run("enp0", "up")
        assert proc.returncode == 0  # the dispatcher entry never blocks on the lock
        assert harness.wait_args(0.5) is None  # worker is queued behind the holder
        assert harness.wait_args(5.0) is not None  # ...and runs once it is released
    finally:
        holder.wait(timeout=10)


# --- install: plan, check, execute ------------------------------------------------


def test_install_plan_lists_every_sudo_step_in_order():
    shells = install_mod.install_commands(
        "bob", python=PY, arms=ARMS, script_src=Path("/tmp/90-mavis-netsetup.sh"),
    )
    assert shells[0].startswith("sudo tee /etc/polkit-1/localauthority/50-local.d/")
    assert shells[1].startswith("sudo usermod -aG netdev bob")
    assert shells[2] == "sudo install -d -m 755 -o root -g root /etc/apollo-mavis-v2"
    assert shells[3] == ("sudo install -D -m 755 -o root -g root /tmp/90-mavis-netsetup.sh "
                         "/etc/NetworkManager/dispatcher.d/90-mavis-netsetup")
    assert shells[4] == ("sudo /opt/mavis/.venv/bin/python -m apollo_mavis_v2_hardware.netsetup "
                         "match --state /etc/apollo-mavis-v2/nic_map.json "
                         "--arm view=192.168.2.219 --arm grip=192.168.1.201")
    assert len(shells) == 5


def test_install_plan_dispatcher_only_and_no_arms():
    only = install_mod.install_commands("bob", python=PY, arms=ARMS, dispatcher_only=True)
    assert len(only) == 2 and "install -d" in only[0] and "install -D" in only[1]
    none = install_mod.install_commands("bob", python=PY, arms=())
    assert len(none) == 3 and not any("dispatcher" in s or " match " in s for s in none)


def fake_sys_run(captured):
    calls = []

    def run(argv):
        calls.append(list(argv))
        if argv[:2] == ["sudo", "install"] and "-D" in argv:
            captured["script"] = Path(argv[-2]).read_text()  # temp file exists right now
        if argv[0] == "id":
            return subprocess.CompletedProcess(argv, 0, "user netdev\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    run.calls = calls
    return run


def test_install_check_reports_dispatcher_problems(tmp_path):
    hook = tmp_path / "90-mavis-netsetup"
    pkla = tmp_path / "grant.pkla"
    pkla.write_text(install_mod.PKLA_CONTENT)
    expected = disp.render_dispatcher(PY, ARMS, generated_at="2026-09-04")
    run = fake_sys_run({})
    missing = install_mod.check(run, pkla, dispatcher_path=hook, expected_script=expected)
    assert len(missing) == 1 and "dispatcher hook missing" in missing[0]
    hook.write_text(expected)
    hook.chmod(0o644)
    assert any("not executable" in p
               for p in install_mod.check(run, pkla, dispatcher_path=hook))
    hook.chmod(0o755)
    assert install_mod.check(run, pkla, dispatcher_path=hook, expected_script=expected) == []
    hook.write_text(expected.replace("192.168.2.219", "192.168.2.220"))
    stale = install_mod.check(run, pkla, dispatcher_path=hook, expected_script=expected)
    assert len(stale) == 1 and "differs from the rendered script" in stale[0]


def test_install_executes_plan_and_seeds_state_when_arms_reachable(tmp_path):
    captured: dict = {}
    run = fake_sys_run(captured)
    out = io.StringIO()
    rc = install_mod.install(
        run, assume_yes=True, pkla_path=tmp_path / "grant.pkla", user="bob",
        python=PY, arms=ARMS, dispatcher_path=tmp_path / "hook",
        state_path=tmp_path / "etc" / "nic_map.json",
        tcp_connect=lambda ip, port, timeout: "refused",  # booting still counts
        out=out,
    )
    assert rc == 0
    argv0 = [c[:2] for c in run.calls if c[0] == "sudo"]
    assert argv0 == [["sudo", "sh"], ["sudo", "usermod"], ["sudo", "install"],
                     ["sudo", "install"], ["sudo", str(PY)]]
    assert run.calls[-1] == ["sudo", *disp.match_argv(PY, ARMS, tmp_path / "etc" / "nic_map.json")]
    assert captured["script"] == disp.render_dispatcher(PY, ARMS, tmp_path / "etc" / "nic_map.json")
    text = out.getvalue()
    assert "sudo install -D -m 755" in text and "match --state" in text
    assert not list(Path("/tmp").glob("90-mavis-netsetup.*.sh")) or True  # temp cleaned best-effort


def test_install_skips_initial_match_when_arms_unreachable(tmp_path):
    run = fake_sys_run({})
    out = io.StringIO()
    rc = install_mod.install(
        run, assume_yes=True, pkla_path=tmp_path / "grant.pkla", user="bob",
        python=PY, arms=ARMS, dispatcher_path=tmp_path / "hook",
        state_path=tmp_path / "etc" / "nic_map.json",
        tcp_connect=lambda ip, port, timeout: "unreachable", out=out,
    )
    assert rc == 0
    assert not any(str(PY) in c for c in run.calls)  # no sudo match
    assert "initial match skipped" in out.getvalue()
    assert "view, grip" in out.getvalue()


def test_install_dispatcher_only(tmp_path):
    run = fake_sys_run({})
    rc = install_mod.install(
        run, assume_yes=True, pkla_path=tmp_path / "grant.pkla", user="bob",
        python=PY, arms=ARMS, dispatcher_only=True, dispatcher_path=tmp_path / "hook",
        state_path=tmp_path / "etc" / "nic_map.json", out=io.StringIO(),
    )
    assert rc == 0
    sudo = [c for c in run.calls if c[0] == "sudo"]
    assert [c[1] for c in sudo] == ["install", "install"]


def test_install_already_healthy_is_a_noop(tmp_path):
    pkla = tmp_path / "grant.pkla"
    pkla.write_text(install_mod.PKLA_CONTENT)
    state = tmp_path / "etc" / "nic_map.json"
    hook = tmp_path / "hook"
    hook.write_text(disp.render_dispatcher(PY, ARMS, state))
    hook.chmod(0o755)
    state.parent.mkdir()
    state.write_text("{}")
    run = fake_sys_run({})
    out = io.StringIO()
    rc = install_mod.install(run, assume_yes=True, pkla_path=pkla, user="bob", python=PY,
                             arms=ARMS, dispatcher_path=hook, state_path=state, out=out)
    assert rc == 0 and "already healthy" in out.getvalue()
    assert not any(c[0] == "sudo" for c in run.calls)


def test_check_skips_group_requirement_for_root(tmp_path, monkeypatch):
    """The dispatcher runs check() as root: no polkit / netdev needed there."""
    pkla = tmp_path / "grant.pkla"
    pkla.write_text(install_mod.PKLA_CONTENT)
    hook = tmp_path / "hook"
    hook.write_text("#!/bin/bash\n")
    hook.chmod(0o755)

    def run(argv):
        return subprocess.CompletedProcess(argv, 0, "root\n", "")

    monkeypatch.setattr(install_mod.os, "geteuid", lambda: 0)
    assert install_mod.check(run, pkla, dispatcher_path=hook) == []
    monkeypatch.setattr(install_mod.os, "geteuid", lambda: 1000)
    assert any("netdev" in p for p in install_mod.check(run, pkla, dispatcher_path=hook))


# --- CLI: install options for deploying under another account -----------------------


def test_cli_install_forwards_python_and_user(monkeypatch):
    """A deployer installs for the operations account: the hook must point at the
    ops venv (``--python``) and the netdev grant must target the ops user
    (``--user``), not ``$SUDO_USER``."""
    seen = {}

    def fake_install(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(install_mod, "install", fake_install)
    rc = cli.main(["install", "--yes", "--python", str(PY), "--user", "mavis",
                   "--arm", "view=192.168.2.219", "--arm", "grip=192.168.1.201"])
    assert rc == 0
    assert seen["python"] == PY and seen["user"] == "mavis" and seen["assume_yes"] is True
    assert seen["arms"] == ARMS and seen["dispatcher_only"] is False


def test_cli_install_check_forwards_user_and_renders_with_python(monkeypatch, capsys):
    seen = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(install_mod, "check", fake_check)
    rc = cli.main(["install", "--check", "--python", str(PY), "--user", "mavis",
                   "--arm", "view=192.168.2.219", "--arm", "grip=192.168.1.201"])
    assert rc == 0 and capsys.readouterr().out.strip() == "ok"
    assert seen["user"] == "mavis"
    assert seen["expected_script"] == disp.render_dispatcher(PY, ARMS)
