"""Opening-time checks remain bounded, shared, and match-schedule aware."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from mlbb_predictor.collection_service import CollectionService
from mlbb_predictor.collector import atomic_json, read_json, save_settings
from mlbb_predictor.match_schedule import run_scheduled_collection, schedule_path
from mlbb_predictor.startup import startup_notice


NOW = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)


def successful_schedule(age=0):
    return {"state": "success", "last_checked_at": (NOW - timedelta(seconds=age)).isoformat(),
            "next_discovery_at": (NOW + timedelta(days=1)).isoformat(), "matches": {}}


class OpeningServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.services = []
        self.gates = []
        self.clock = patch("mlbb_predictor.collection_service.utc_now", return_value=NOW)
        self.clock.start()

    def tearDown(self):
        for service in self.services:
            service.stop()
        for gate in self.gates:
            gate.set()
        for service in self.services:
            service._thread.join(3)
            self.assertFalse(service._thread.is_alive())
        self.clock.stop()
        self.directory.cleanup()

    def check(self, root):
        result = successful_schedule()
        atomic_json(schedule_path(root), result)
        return result

    def service(self, *, age=600, enabled=True, collector=None, schedule=None):
        save_settings(self.root, False)
        atomic_json(schedule_path(self.root), schedule if schedule is not None else successful_schedule(age))
        collector = collector if collector is not None else Mock(side_effect=self.check)
        service = CollectionService(self.root, collector=collector)
        self.services.append(service)
        save_settings(self.root, enabled)
        return service, collector

    def test_open_requests_fresh_check_even_before_daily_discovery_is_due(self):
        service, collector = self.service()
        result = service.refresh_on_open(timeout=2)
        self.assertEqual(result["state"], "success")
        collector.assert_called_once_with(self.root)
        self.assertEqual(result["last_checked_at"], NOW.isoformat())

    def test_recent_visits_reuse_success_without_extra_collection(self):
        service, collector = self.service(age=60)
        self.assertTrue(service.refresh_on_open(timeout=2)["cached"])
        collector.assert_not_called()

    def test_pausing_prevents_an_open_from_forcing_collection(self):
        service, collector = self.service(enabled=False)
        self.assertEqual(service.refresh_on_open(), {"state": "paused"})
        collector.assert_not_called()

    def test_due_match_overrides_the_five_minute_cache(self):
        schedule = successful_schedule(age=30)
        schedule["matches"]["due"] = {"status": "completed", "present": True,
                                       "collect_after": (NOW - timedelta(seconds=1)).isoformat()}
        service, collector = self.service(schedule=schedule)
        self.assertEqual(service.refresh_on_open(timeout=2)["state"], "success")
        collector.assert_called_once()

    def test_future_clock_is_not_considered_a_recent_check(self):
        service, collector = self.service(age=-10)
        self.assertEqual(service.refresh_on_open(timeout=2)["state"], "success")
        collector.assert_called_once()

    def test_source_failure_backoff_is_honored_on_reopen(self):
        schedule = {**successful_schedule(), "state": "error", "error": "Source unavailable",
                    "retry_at": (NOW + timedelta(hours=1)).isoformat()}
        service, collector = self.service(schedule=schedule)
        result = service.refresh_on_open(timeout=2)
        self.assertEqual(result["state"], "error")
        self.assertTrue(result["cached"])
        collector.assert_not_called()

    def test_two_visitors_join_one_inflight_check(self):
        started, finish = threading.Event(), threading.Event()
        self.gates.append(finish)

        def check(root):
            started.set()
            self.assertTrue(finish.wait(3))
            return self.check(root)

        service, collector = self.service(collector=Mock(side_effect=check))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(service.refresh_on_open, timeout=2)
            self.assertTrue(started.wait(2))
            second = pool.submit(service.refresh_on_open, timeout=2)
            self.assertFalse(service.request_refresh())
            finish.set()
            self.assertEqual(first.result(3)["state"], "success")
            self.assertEqual(second.result(3)["state"], "success")
        collector.assert_called_once()

    def test_slow_check_returns_saved_data_control_without_cancelling_worker(self):
        started, finish = threading.Event(), threading.Event()
        self.gates.append(finish)

        def check(root):
            started.set()
            self.assertTrue(finish.wait(3))
            return self.check(root)

        service, collector = self.service(collector=Mock(side_effect=check))
        before = time.monotonic()
        self.assertEqual(service.refresh_on_open(timeout=.03)["state"], "timeout")
        self.assertLess(time.monotonic() - before, 1)
        self.assertTrue(started.wait(2))
        self.assertTrue(service.running)
        finish.set()
        self.assertEqual(service.refresh_on_open(timeout=2)["state"], "success")
        collector.assert_called_once()

    def test_collector_exception_unblocks_open_with_error(self):
        service, collector = self.service(collector=Mock(side_effect=OSError("offline")))
        with self.assertLogs("mlbb_predictor.collection_service", level="ERROR"):
            result = service.refresh_on_open(timeout=2)
        self.assertEqual(result["state"], "error")
        self.assertIn("offline", result["error"])
        self.assertFalse(service.running)
        collector.assert_called_once()

    def test_stopped_worker_does_not_accept_opening_requests(self):
        service, collector = self.service()
        service.stop()
        self.assertEqual(service.refresh_on_open(), {"state": "stopped"})
        self.assertFalse(service.request_refresh())
        collector.assert_not_called()

    def test_open_does_not_bypass_a_newly_observed_completion_delay(self):
        url = "https://mldb.gg/match/MPL_Philippines_Season_18-TLPHvsRORA-test"
        card = (f'<a class="matches-item" href="{url}"><div data-utc="2026-08-29T11:30:00+00:00"></div>'
                '<span>BO3</span><span class="team-info-short-name">TLPH</span>'
                '<span class="team-info-short-name">RORA</span><span class="team-info-score">2</span>'
                '<span class="team-info-score">0</span><span class="match-status completed">Completed</span></a>')
        full_collector = Mock()
        check = lambda root: run_scheduled_collection(root, fetcher=lambda _: json.dumps({"html": card}),
                                                       collector=full_collector, now=NOW)
        service, _ = self.service(collector=check)
        result = service.refresh_on_open(timeout=2)
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["matches"][url]["collect_after"], (NOW + timedelta(hours=24)).isoformat())
        full_collector.assert_not_called()
        self.assertEqual(read_json(schedule_path(self.root))["matches"][url]["first_seen_completed_at"], NOW.isoformat())


class StartupNoticeTests(unittest.TestCase):
    def test_timeout_and_error_never_claim_fresh_data(self):
        for state in ("timeout", "error", "busy", "stopped", "paused"):
            kind, message = startup_notice({"state": state})
            self.assertIn(kind, ("info", "warning"))
            self.assertIn("saved", message.lower())

    def test_partial_collection_is_not_reported_as_success(self):
        result = {**successful_schedule(), "last_collection_report": {"at": NOW.isoformat(), "state": "partial"}}
        self.assertEqual(startup_notice(result)[0], "warning")

    def test_old_reports_do_not_claim_new_games_were_just_downloaded(self):
        result = {**successful_schedule(), "last_collection_report": {
            "at": (NOW - timedelta(days=1)).isoformat(), "state": "success", "new_games": 2}}
        self.assertNotIn("Collected 2", startup_notice(result)[1])
        result["last_collection_report"]["at"] = NOW.isoformat()
        self.assertIn("Collected 2", startup_notice(result)[1])
        result["cached"] = True
        self.assertNotIn("Collected 2", startup_notice(result)[1])


if __name__ == "__main__":
    unittest.main()
