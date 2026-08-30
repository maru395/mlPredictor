"""One lightweight background collector per running Streamlit server."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading

from .collector import load_settings, read_json, utc_now
from .match_schedule import next_check, run_scheduled_collection, schedule_path, timestamp

LOGGER = logging.getLogger(__name__)
OPEN_WAIT_SECONDS = 15
OPEN_CACHE_SECONDS = 300


@dataclass
class _RefreshAttempt:
    """An immutable result after done is set, shared by concurrent visitors."""

    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)


class CollectionService:
    """Files are shared state; no Streamlit functions run on this daemon thread."""

    def __init__(self, root: Path, collector=run_scheduled_collection):
        self.root = root
        self.collector = collector
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._guard = threading.Lock()
        self._requested = False
        self._running = False
        self._attempt: _RefreshAttempt | None = None
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="mpl-ph-collector", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    @property
    def running(self) -> bool:
        with self._guard:
            return self._running

    @property
    def pending(self) -> bool:
        with self._guard:
            return self._requested

    def request_refresh(self) -> bool:
        with self._guard:
            if self._stop.is_set() or self._running or self._requested:
                return False
            self._attempt = _RefreshAttempt()
            self._requested = True
        self._wake.set()
        return True

    def refresh_on_open(self, *, timeout: float = OPEN_WAIT_SECONDS,
                        max_age: float = OPEN_CACHE_SECONDS) -> dict:
        """Join/request one scheduled check, without bypassing completion timers.

        Nearby openings reuse a recent successful check unless a job is due.
        Pausing and source retry backoff are honored. The bounded wait does not
        cancel a slow collector; atomic publication and the app's revision
        watcher will pick up its eventual result.
        """
        try:
            with self._guard:
                if self._stop.is_set():
                    return {"state": "stopped"}
                settings = load_settings(self.root)
                if not settings["enabled"]:
                    return {"state": "paused"}
                if not self._running and not self._requested:
                    schedule = read_json(schedule_path(self.root), {})
                    now = utc_now()
                    due = next_check(settings, schedule, now)
                    retry = timestamp(schedule.get("retry_at"))
                    if schedule.get("state") == "error" and retry and retry > now:
                        return {**schedule, "cached": True}
                    checked = timestamp(schedule.get("last_checked_at"))
                    if (schedule.get("state") == "success" and checked and
                            0 <= (now - checked).total_seconds() < max_age and
                            due is not None and due > now):
                        return {**schedule, "cached": True}
                    self._attempt = _RefreshAttempt()
                    self._requested = True
                attempt = self._attempt
            self._wake.set()
            if attempt is None:
                return {"state": "error", "error": "No refresh attempt was created"}
            if not attempt.done.wait(max(0.0, min(float(timeout), OPEN_WAIT_SECONDS))):
                return {"state": "timeout"}
            return dict(attempt.result)
        except Exception as error:
            return {"state": "error", "error": f"{type(error).__name__}: {error}"}

    def settings_changed(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._guard:
            if self._attempt and not self._attempt.done.is_set() and not self._running:
                self._requested = False
                self._attempt.result = {"state": "stopped"}
                self._attempt.done.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                settings = load_settings(self.root)
                status = read_json(schedule_path(self.root), {})
                now = utc_now()
                due = next_check(settings, status, now)
                with self._guard:
                    should_run = not self._stop.is_set() and (self._requested or (due is not None and due <= now))
                    if should_run:
                        if not self._requested:
                            self._attempt = _RefreshAttempt()
                        attempt = self._attempt
                        self._requested = False
                        self._running = True
                if should_run:
                    result = {"state": "error", "error": "Check did not finish"}
                    try:
                        report = self.collector(self.root)
                        result = report if isinstance(report, dict) else {"state": "success"}
                        self.last_error = None
                    except Exception as error:
                        self.last_error = f"{type(error).__name__}: {error}"
                        result = {"state": "error", "error": self.last_error}
                        LOGGER.exception("Opening-time collection failed; retaining saved predictions")
                    finally:
                        with self._guard:
                            self._running = False
                            attempt.result = result
                            attempt.done.set()
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                with self._guard:
                    if self._requested and self._attempt:
                        self._requested = False
                        self._attempt.result = {"state": "error", "error": self.last_error}
                        self._attempt.done.set()
                LOGGER.exception("Background data collection failed; retaining saved predictions")
            # Re-read persisted settings regularly; a button wakes the worker immediately.
            if self._stop.is_set():
                break
            self._wake.wait(15)
