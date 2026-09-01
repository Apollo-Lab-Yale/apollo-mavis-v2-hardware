"""Shared test plumbing: fake clock + path helpers. No network, no hardware."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))  # make `fakes` importable


class FakeClock:
    """Deterministic monotonic clock: ``sleep`` advances fake time.

    ``inject_stall(after_sleeps, extra)`` adds ``extra`` seconds during the
    Nth subsequent sleep — models a scheduler stall for the no-burst test.
    """

    def __init__(self, start: float = 100.0) -> None:
        self._now = start
        self._lock = threading.Lock()
        self._sleep_count = 0
        self._stall_at: int | None = None
        self._stall_extra = 0.0

    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, dt: float) -> None:
        with self._lock:
            self._sleep_count += 1
            self._now += max(dt, 0.0)
            if self._stall_at is not None and self._sleep_count >= self._stall_at:
                self._now += self._stall_extra
                self._stall_at = None
        time.sleep(0)  # yield the GIL so other threads make progress

    def inject_stall(self, after_sleeps: int, extra: float) -> None:
        with self._lock:
            self._stall_at = self._sleep_count + after_sleeps
            self._stall_extra = extra

    @property
    def stall_pending(self) -> bool:
        with self._lock:
            return self._stall_at is not None

    def jump(self, dt: float) -> None:
        with self._lock:
            self._now += dt


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def nmcli_fixtures() -> Path:
    return FIXTURES / "nmcli"


@pytest.fixture
def report_fixtures() -> Path:
    return FIXTURES / "report"
