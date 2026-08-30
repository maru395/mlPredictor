"""One lightweight background collector per running Streamlit server."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
import threading

from .collector import load_settings, read_json, utc_now
from .match_schedule import next_check, run_scheduled_collection, schedule_path

LOGGER = logging.getLogger(__name__)


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
            if self._running or self._requested:
                return False
            self._requested = True
        self._wake.set()
        return True

    def settings_changed(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                settings = load_settings(self.root)
                status = read_json(schedule_path(self.root), {})
                now = utc_now()
                due = next_check(settings, status, now)
                with self._guard:
                    should_run = self._requested or (due is not None and due <= now)
                    if should_run:
                        self._requested = False
                        self._running = True
                if should_run:
                    try:
                        self.collector(self.root)
                        self.last_error = None
                    finally:
                        with self._guard:
                            self._running = False
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                LOGGER.exception("Background data collection failed; retaining saved predictions")
            # Re-read persisted settings regularly; a button wakes the worker immediately.
            self._wake.wait(15)
