"""nmcli subprocess wrapper + terse-output parsing (02-hardware §7.1).

Everything runs with ``LC_ALL=C``; profiles are addressed **by UUID only**
(the target machine has two profiles both named ``xarm7_1``); activation
always passes ``-w 10..15`` (nmcli's default wait is 90 s).

``NmcliRunner`` is the test seam: tests inject a transcript-backed runner;
the real one is wired only by the CLI / default construction.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

# signature: runner(*args, timeout=...) -> CompletedProcess
NmcliRunner = Callable[..., subprocess.CompletedProcess]


def nmcli(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    """The real runner: ``nmcli <args>`` with stable C-locale output."""
    return subprocess.run(
        ["nmcli", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        timeout=timeout,
        check=False,
    )


def split_terse(line: str) -> list[str]:
    r"""Split one line of ``nmcli -t`` output on unescaped colons.

    Literal colons arrive escaped as ``\:`` (MAC addresses!); ``\\`` is a
    literal backslash.
    """
    fields: list[str] = []
    cur: list[str] = []
    it = iter(line)
    for ch in it:
        if ch == "\\":
            cur.append(next(it, ""))  # unescape \: \\ etc.
        elif ch == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def unescape_value(value: str) -> str:
    r"""Unescape a single ``-g`` detail value (one value per line).

    All literal colons come escaped, so :func:`split_terse` yields one field;
    re-joining tolerates malformed input without dropping anything.
    """
    return ":".join(split_terse(value))


MUTATING_VERBS = frozenset({"add", "modify", "delete", "up", "down", "reload", "clone"})


def is_mutating(args: tuple[str, ...] | list[str]) -> bool:
    """True when an nmcli argv would change NetworkManager state."""
    positional = [a for a in args if not a.startswith("-")]
    # drop option values that follow -w/-f/-g/-m (they are not verbs)
    verbs = set(positional) & MUTATING_VERBS
    return bool(verbs) and "show" not in positional and "status" not in positional


class TranscriptRunner:
    """Replays recorded nmcli transcripts; refuses anything unscripted.

    Transcript format (``tests/fixtures/nmcli/*.txt`` and the CLI
    ``--fixtures`` dry-run path)::

        $ nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status
        enp36s0f0:ethernet:connected:xarm7_1
        ...
        $ rc=10 nmcli connection show uuid missing

    A ``rc=N`` prefix sets the exit code (default 0); stdout is every line
    until the next ``$``. Unexpected commands raise (tests fail loudly) —
    this runner never touches the real NetworkManager.
    """

    def __init__(self, transcripts: dict[tuple[str, ...], tuple[int, str]]) -> None:
        self._transcripts = dict(transcripts)
        self.calls: list[tuple[str, ...]] = []

    @classmethod
    def from_files(cls, *paths: str | Path) -> TranscriptRunner:
        table: dict[tuple[str, ...], tuple[int, str]] = {}
        for path in paths:
            text = Path(path).read_text()
            key: tuple[str, ...] | None = None
            rc = 0
            out: list[str] = []
            for line in text.splitlines():
                if line.startswith("$ "):
                    if key is not None:
                        table[key] = (rc, "\n".join(out) + ("\n" if out else ""))
                    rest = line[2:].strip()
                    rc = 0
                    if rest.startswith("rc="):
                        rc_str, rest = rest.split(None, 1)
                        rc = int(rc_str[3:])
                    argv = shlex.split(rest)
                    if argv and argv[0] == "nmcli":
                        argv = argv[1:]
                    key = tuple(argv)
                    out = []
                elif key is not None and not line.startswith("#"):
                    out.append(line)
            if key is not None:
                table[key] = (rc, "\n".join(out) + ("\n" if out else ""))
        return cls(table)

    def __call__(self, *args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
        key = tuple(args)
        self.calls.append(key)
        if key not in self._transcripts:
            raise AssertionError(f"unscripted nmcli command: nmcli {' '.join(args)}")
        rc, stdout = self._transcripts[key]
        return subprocess.CompletedProcess(
            args=["nmcli", *args], returncode=rc, stdout=stdout, stderr=""
        )
