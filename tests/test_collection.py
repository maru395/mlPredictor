"""Offline collector tests: validation, safe merges, atomic publication and cadence."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from mlbb_predictor.collector import (
    atomic_json, collect_series, collection_lock, discover_completed, discover_schedule, merge_series,
    load_settings, read_json, run_collection, safe_url, save_settings, status_path,
)
from mlbb_predictor.collection_service import CollectionService, next_check
from mlbb_predictor.history import valid_lineups, valid_picks
from mlbb_predictor.live_data import history_fingerprint, load_prediction_state, snapshot_path
from mlbb_predictor.match_schedule import (
    DAY, collection_due, next_collection, observe_schedule, run_scheduled_collection, schedule_path,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 30, 7, tzinfo=timezone.utc)
URL = "https://mldb.gg/match/MPL_Philippines_Season_18-TLPHvsRORA-test"


def fixture_card(*, state="completed", codes=("TLPH", "RORA"), scores=(2, 0),
                 timestamp="2026-08-29T11:30:00+00:00", url=URL):
    return (f'<div data-stage="Regular Season"><a class="matches-item" href="{url}">'
            f'<div data-utc="{timestamp}"></div><span class="bo">BO3</span>'
            + "".join(f'<span class="team-info-short-name">{code}</span>' for code in codes)
            + "".join(f'<span class="team-info-score">{score}</span>' for score in scores)
            + f'<span class="match-status {state}">{state}</span></a></div>')


def game(number=1):
    return {
        "date": "2026-08-29", "series_started_at": "2026-08-29T11:30:00+00:00",
        "season": "MPL Philippines Season 18", "team_a": "TLPH", "team_b": "RORA",
        "winner": "TLPH", "game": number, "series_url": URL,
        "players_a": [f"Player A{i}" for i in range(5)], "players_b": [f"Player B{i}" for i in range(5)],
        "heroes_a": [f"Hero A{i}" for i in range(5)], "heroes_b": [f"Hero B{i}" for i in range(5)],
        "lineups_verified": True, "picks_verified": True,
        "bans_a": [f"Ban A{i}" for i in range(5)], "bans_b": [f"Ban B{i}" for i in range(5)], "bans_verified": True,
    }


def incomplete(row):
    result = deepcopy(row)
    for side in ("a", "b"):
        result[f"players_{side}"] = result[f"players_{side}"][:4]
        result[f"reported_heroes_{side}"] = result.pop(f"heroes_{side}")[:4]
    result.update(lineups_verified=False, picks_verified=False, bans_verified=False,
                  bans_a=[], bans_b=[], lineup_data_issue="missing", pick_data_issue="missing")
    return result


def series_html(rows):
    content = '<div class="summary-info" data-utc="2026-08-29T11:30:00+00:00">'
    for row in rows:
        content += f'<div class="tab-pane" id="game-{row["game"]}">'
        content += '<a href="https://mldb.gg/team/team-liquid-ph" class="team-name">Liquid</a>'
        content += '<a href="https://mldb.gg/team/aurora-gaming-ph" class="team-name">Aurora</a>'
        content += '<div class="mdc-header-team win team-one"></div><div class="mdc-header-team lose team-two"></div>'
        for index in range(len(row["players_a"])):
            for side in ("a", "b"):
                content += f'<a href="#" class="mdc-pf-player-name">{row[f"players_{side}"][index]}</a>'
                content += f'<a href="#" class="mdc-pf-hero-name">{row.get(f"heroes_{side}", row.get(f"reported_heroes_{side}"))[index]}</a>'
    return content


class DiscoveryTests(unittest.TestCase):
    def test_only_completed_past_configured_league_teams(self):
        page = (fixture_card() + fixture_card(state="live", codes=("FNOP", "OMG"))
                + fixture_card(state="upcoming", codes=("FLCN", "TNC"))
                + fixture_card(timestamp="2027-08-29T11:30:00+00:00")
                + fixture_card(codes=("TBD", "TLPH")))
        fixtures = discover_completed(page, NOW)
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0]["stage"], "Regular Season")
        self.assertEqual(fixtures[0]["date"], "2026-08-29")

    def test_source_alias_and_philippine_date(self):
        item = discover_completed(fixture_card(codes=("ONIC", "FLCP"), timestamp="2026-08-28T19:30:00+00:00"), NOW)[0]
        self.assertEqual((item["team_a"], item["team_b"]), ("FNOP", "FLCN"))
        self.assertEqual(item["date"], "2026-08-29")

    def test_missing_empty_and_bad_finished_scores_fail_closed(self):
        for page in ("", fixture_card(scores=(1, 0)), fixture_card(scores=(2, 2)),
                     fixture_card(timestamp="2026-08-29T11:30:00")):
            with self.subTest(page=page), self.assertRaises(ValueError):
                discover_completed(page, NOW)

    def test_duplicate_cards_deduplicate_and_conflicts_are_rejected(self):
        self.assertEqual(len(discover_completed(fixture_card() * 2, NOW)), 1)
        with self.assertRaises(ValueError):
            discover_completed(fixture_card() + fixture_card(scores=(0, 2)), NOW)

    def test_source_url_is_restricted(self):
        for url in ("https://example.com/", "http://mldb.gg/match/x", "https://mldb.gg/event/other-league",
                    "https://mldb.gg/match/MPL_Philippines_Season_17-x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                safe_url(url)
        self.assertEqual(safe_url(URL + "?load_draft=false&game_seq=1"), URL + "?load_draft=false&game_seq=1")


class SeriesTests(unittest.TestCase):
    def test_parser_collects_actual_players_without_guessing_bans(self):
        rows = [game(1), game(2)]
        fixture = discover_completed(fixture_card(), NOW)[0]
        fetcher = lambda url, ajax=False: json.dumps({"html": ""}) if ajax else series_html(rows)
        collected = collect_series(fixture, {"teams": []}, fetcher)
        self.assertEqual(len(collected), 2)
        self.assertTrue(all(valid_lineups(row) and valid_picks(row) for row in collected))
        self.assertTrue(all(not row["bans_verified"] and not row["bans_a"] for row in collected))
        self.assertEqual(collected[0]["players_a"], rows[0]["players_a"])
        self.assertEqual(collected[0]["heroes_a"], rows[0]["heroes_a"])

    def test_incomplete_source_keeps_result_but_excludes_lineups_and_picks(self):
        rows = [incomplete(game(1)), incomplete(game(2))]
        fixture = discover_completed(fixture_card(), NOW)[0]
        fetcher = lambda url, ajax=False: json.dumps({"html": ""}) if ajax else series_html(rows)
        collected = collect_series(fixture, {"teams": []}, fetcher)
        self.assertFalse(valid_lineups(collected[0]))
        self.assertFalse(valid_picks(collected[0]))
        self.assertNotIn("heroes_a", collected[0])
        self.assertEqual(collected[0]["winner"], "TLPH")

    def test_score_conflicts_cannot_enter_history(self):
        fixture = discover_completed(fixture_card(scores=(0, 2)), NOW)[0]
        with self.assertRaises(ValueError):
            collect_series(fixture, {"teams": []}, lambda *args, **kwargs: series_html([game(1), game(2)]))

    def test_incomplete_records_can_be_repaired(self):
        repaired = merge_series([incomplete(game())], [game()])[0]
        self.assertTrue(valid_lineups(repaired) and valid_picks(repaired))
        self.assertNotIn("lineup_data_issue", repaired)
        self.assertNotIn("pick_data_issue", repaired)
        self.assertNotIn("reported_heroes_a", repaired)

    def test_verified_records_never_downgrade(self):
        old = game()
        merged = merge_series([old], [incomplete(old)])[0]
        for field in ("players_a", "players_b", "heroes_a", "heroes_b", "bans_a", "bans_b"):
            self.assertEqual(merged[field], old[field])
        self.assertTrue(valid_lineups(merged) and valid_picks(merged) and merged["bans_verified"])

    def test_side_or_player_order_does_not_change_identity_mapping(self):
        old, new = game(), game()
        for field in ("team", "players", "heroes", "bans"):
            new[f"{field}_a"], new[f"{field}_b"] = new[f"{field}_b"], new[f"{field}_a"]
        for field in ("players_a", "players_b", "heroes_a", "heroes_b"):
            new[field].reverse()
        merged = merge_series([old], [new])[0]
        self.assertEqual(merged["team_a"], old["team_a"])
        self.assertEqual(dict(zip(merged["players_a"], merged["heroes_a"])), dict(zip(old["players_a"], old["heroes_a"])))

    def test_result_player_hero_and_ban_conflicts_are_quarantined(self):
        for field, value in (("winner", "RORA"), ("players_a", ["Changed"] + game()["players_a"][1:]),
                             ("heroes_a", list(reversed(game()["heroes_a"]))),
                             ("bans_a", ["Different ban"] + game()["bans_a"][1:])):
            new = game()
            new[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                merge_series([game()], [new])

    def test_separate_incomplete_sources_cannot_create_a_false_player_hero_mapping(self):
        old, new = game(), game()
        old["lineups_verified"] = False
        old["players_a"] = old["players_a"][:4]
        new["picks_verified"] = False
        new.pop("heroes_a")
        with self.assertRaisesRegex(ValueError, "independently ordered"):
            merge_series([old], [new])


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for name in ("config/mpl_ph_teams.json", "config/meta_tiers.json", "models/player_elo_model.json",
                     "data/processed/mpl_ph_s17_player_games.json", "data/processed/mpl_ph_s18_player_games.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)

    def collect(self, rows=None):
        with patch("mlbb_predictor.collector.collect_series", return_value=rows or [game(1), game(2)]):
            return run_collection(self.root, fetcher=lambda url: json.dumps({"html": fixture_card()}), now=NOW)

    def test_publish_retrains_once_and_repeated_imports_are_idempotent(self):
        first = self.collect()
        self.assertEqual(first["state"], "success")
        self.assertEqual(first["new_games"], 2)
        initial = snapshot_path(self.root).read_bytes()
        model, metrics, metadata, rows = load_prediction_state(self.root)
        self.assertEqual(metrics["match_count"], 190)
        self.assertEqual(metrics["selected_k_factor"], 16)
        self.assertEqual(metrics["history_fingerprint"], history_fingerprint(rows))
        self.assertEqual(metadata["games"], 192)
        with patch("mlbb_predictor.collector.train_collected_model", side_effect=AssertionError("Must not train unchanged data")):
            second = self.collect()
        self.assertEqual(second["state"], "success")
        self.assertEqual(second["new_games"], 0)
        self.assertEqual(snapshot_path(self.root).read_bytes(), initial)
        self.assertEqual(load_prediction_state(self.root)[0].ratings, model.ratings)

    def test_original_imports_rosters_tiers_and_seed_model_are_untouched(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        self.collect()
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, path)

    def test_network_failure_keeps_last_good_snapshot(self):
        self.collect()
        initial = snapshot_path(self.root).read_bytes()
        def fail(url):
            raise OSError("offline")
        result = run_collection(self.root, fetcher=fail, now=NOW)
        self.assertEqual(result["state"], "error")
        self.assertEqual(snapshot_path(self.root).read_bytes(), initial)
        self.assertEqual(result["last_success_at"], NOW.isoformat())

    def test_training_failure_cannot_publish_half_an_update(self):
        self.collect()
        initial = snapshot_path(self.root).read_bytes()
        updated = [game(1), game(2)]
        updated[0]["stage"] = "Added metadata"
        with patch("mlbb_predictor.collector.train_collected_model", side_effect=ValueError("training failed")):
            result = self.collect(updated)
        self.assertEqual(result["state"], "error")
        self.assertEqual(snapshot_path(self.root).read_bytes(), initial)

    def test_source_conflict_reports_partial_without_overwriting_good_game(self):
        self.collect()
        initial = snapshot_path(self.root).read_bytes()
        bad = [game(1), game(2)]
        bad[0]["winner"] = "RORA"
        result = self.collect(bad)
        self.assertEqual(result["state"], "partial")
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(snapshot_path(self.root).read_bytes(), initial)

    def test_os_lock_prevents_concurrent_collection(self):
        with collection_lock(self.root) as acquired:
            self.assertTrue(acquired)
            result = run_collection(self.root, fetcher=lambda url: self.fail("Must not fetch when locked"), now=NOW)
        self.assertEqual(result["state"], "busy")
        with collection_lock(self.root) as acquired:
            self.assertTrue(acquired)

    def test_snapshot_fingerprint_rejects_mismatched_model_and_history(self):
        self.collect()
        snapshot = read_json(snapshot_path(self.root))
        snapshot["history_rows"][0]["winner"] = "Changed"
        atomic_json(snapshot_path(self.root), snapshot)
        with self.assertRaisesRegex(ValueError, "same snapshot"):
            load_prediction_state(self.root)
        # Offline tests can still explicitly load immutable seed data.
        self.assertEqual(load_prediction_state(self.root, offline=True)[1]["match_count"], 188)

    def test_due_batch_does_not_sweep_in_a_newly_finished_unrelated_match(self):
        fixture = discover_schedule(fixture_card(), NOW)[0]
        unrelated = discover_schedule(fixture_card(codes=("FNOP", "OMG"), url=URL + "-other"), NOW)[0]
        with patch("mlbb_predictor.collector.collect_series", return_value=[game(1), game(2)]) as collect:
            result = run_collection(self.root, fixture_feed=[fixture, unrelated], selected_urls={URL}, now=NOW)
        self.assertEqual(result["state"], "success")
        collect.assert_called_once()
        self.assertEqual(collect.call_args.args[0]["url"], URL)
        self.assertEqual(result["new_games"], 2)


class MatchScheduleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.fixture = discover_schedule(fixture_card(), NOW)[0]
        self.fetcher = Mock(return_value=json.dumps({"html": fixture_card()}))
        self.collector = Mock(return_value={"state": "success", "new_games": 2,
                                           "series_outcomes": [{"url": URL, "state": "complete"}]})

    def run_check(self, when=NOW):
        return run_scheduled_collection(self.root, fetcher=self.fetcher, collector=self.collector, now=when)

    def test_upcoming_fixture_is_discovered_without_a_score_or_finish_time(self):
        page = fixture_card(state="upcoming", scores=("", ""), timestamp="2026-08-30T11:30:00+00:00")
        fixture = discover_schedule(page, NOW)[0]
        self.assertIsNone(fixture["score_a"])
        schedule = observe_schedule({}, [fixture], NOW)
        self.assertIsNone(next_collection(schedule))
        self.assertEqual(next_check({"enabled": True}, schedule, NOW), datetime(2026, 8, 30, 14, 30, tzinfo=timezone.utc))
        self.assertNotIn("collect_after", schedule["matches"][URL])

    def test_newly_observed_completion_waits_24_hours_not_start_plus_24(self):
        state = self.run_check()
        self.collector.assert_not_called()
        self.assertEqual(next_collection(state), NOW + DAY)
        # A very old scheduled start does not prove when a delayed match ended.
        old = {**self.fixture, "date": "2026-08-01", "series_started_at": "2026-08-01T11:30:00+00:00"}
        self.assertEqual(next_collection(observe_schedule({}, [old], NOW)), NOW + DAY)

    def test_delay_survives_polling_and_restart_and_runs_once_at_boundary(self):
        self.run_check()
        self.run_check(NOW + DAY - timedelta(seconds=1))
        self.collector.assert_not_called()
        state = self.run_check(NOW + DAY)
        self.collector.assert_called_once()
        self.assertEqual(self.collector.call_args.kwargs["selected_urls"], {URL})
        self.assertEqual(state["matches"][URL]["first_seen_completed_at"], NOW.isoformat())
        self.assertIsNone(next_collection(state))
        self.run_check(NOW + 2 * DAY)
        self.collector.assert_called_once()

    def test_sleeping_pc_catches_up_without_restarting_the_wait(self):
        self.run_check()
        state = self.run_check(NOW + 3 * DAY)
        self.collector.assert_called_once()
        self.assertEqual(state["matches"][URL]["collect_after"], (NOW + DAY).isoformat())

    def test_only_due_matches_collect_when_another_just_finished(self):
        self.run_check()
        self.fetcher.return_value = json.dumps({"html": fixture_card() + fixture_card(codes=("FNOP", "OMG"), url=URL + "-new")})
        state = self.run_check(NOW + DAY)
        self.assertEqual(self.collector.call_args.kwargs["selected_urls"], {URL})
        self.assertEqual(collection_due(state["matches"][URL + "-new"]), NOW + 2 * DAY)

    def test_postponement_withdraws_completion_and_restarts_timer(self):
        self.run_check()
        self.fetcher.return_value = json.dumps({"html": fixture_card(state="upcoming", timestamp="2026-09-04T11:30:00+00:00")})
        postponed = self.run_check(NOW + DAY)
        self.collector.assert_not_called()
        self.assertIsNone(next_collection(postponed))
        self.assertNotIn("first_seen_completed_at", postponed["matches"][URL])
        later = NOW + 6 * DAY
        self.fetcher.return_value = json.dumps({"html": fixture_card(timestamp="2026-09-04T11:30:00+00:00")})
        self.assertEqual(next_collection(self.run_check(later)), later + DAY)

    def test_live_at_collection_time_does_not_import_cached_complete_result(self):
        self.run_check()
        self.fetcher.return_value = json.dumps({"html": fixture_card(state="live")})
        state = self.run_check(NOW + DAY)
        self.collector.assert_not_called()
        self.assertIsNone(next_collection(state))

    def test_disappearing_fixture_is_not_collected_from_stale_cache(self):
        self.run_check()
        self.fetcher.return_value = json.dumps({"html": fixture_card(state="upcoming", codes=("FNOP", "OMG"), url=URL + "-other")})
        state = self.run_check(NOW + DAY)
        self.collector.assert_not_called()
        self.assertFalse(state["matches"][URL]["present"])
        self.assertIsNone(next_collection(state))

    def test_network_failure_preserves_timer_and_retries_in_one_hour(self):
        self.run_check()
        self.fetcher.side_effect = OSError("offline")
        state = self.run_check(NOW + DAY)
        self.assertEqual(state["state"], "error")
        self.assertEqual(state["matches"][URL]["first_seen_completed_at"], NOW.isoformat())
        self.assertEqual(next_check({"enabled": True}, state, NOW + DAY), NOW + DAY + timedelta(hours=1))
        self.collector.assert_not_called()

    def test_incomplete_or_conflicting_details_retry_next_day(self):
        for outcome in ("incomplete", "error"):
            with self.subTest(outcome=outcome):
                atomic_json(schedule_path(self.root), {})
                self.collector.reset_mock()
                self.collector.return_value = {"state": "partial", "series_outcomes": [{"url": URL, "state": outcome}]}
                self.run_check()
                state = self.run_check(NOW + DAY)
                self.assertEqual(next_collection(state), NOW + 2 * DAY)
                self.run_check(NOW + DAY + timedelta(minutes=30))
                self.collector.assert_called_once()

    def test_successful_game_data_is_not_redownloaded_every_day(self):
        self.run_check()
        self.run_check(NOW + DAY)
        for day in range(2, 7):
            self.run_check(NOW + day * DAY)
        self.collector.assert_called_once()

    def test_existing_complete_imports_are_preserved_without_a_new_download(self):
        atomic_json(snapshot_path(self.root), {"season_rows": [game(1), game(2)]})
        state = self.run_check()
        self.assertEqual(state["matches"][URL]["collection_state"], "already_saved")
        self.assertIsNone(next_collection(state))
        self.run_check(NOW + DAY)
        self.collector.assert_not_called()

    def test_failed_model_publish_does_not_mark_queue_as_collected(self):
        self.run_check()
        self.collector.return_value = {"state": "error", "series_outcomes": [{"url": URL, "state": "complete"}]}
        state = self.run_check(NOW + DAY)
        self.assertIsNone(state["matches"][URL].get("last_collected_at"))
        self.assertEqual(next_collection(state), NOW + 2 * DAY)

    def test_legacy_interval_settings_migrate_and_keep_pause(self):
        atomic_json(self.root / "config/data_collection.json", {"enabled": False, "interval_minutes": 15})
        settings = load_settings(self.root)
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["completion_delay_hours"], 24)
        self.assertEqual(settings["mode"], "day_after_completion")
        self.assertNotIn("interval_minutes", settings)

    def test_empty_or_invalid_feed_cannot_trigger_due_downloads(self):
        self.run_check()
        self.fetcher.return_value = json.dumps({"html": "changed website"})
        state = self.run_check(NOW + DAY)
        self.assertEqual(state["state"], "error")
        self.collector.assert_not_called()

    def test_schedule_check_lock_prevents_parallel_queue_writes(self):
        with collection_lock(self.root, "schedule.lock"):
            state = self.run_check()
        self.assertEqual(state["state"], "busy")
        self.fetcher.assert_not_called()


class SchedulerTests(unittest.TestCase):
    def test_daily_discovery_pause_overdue_and_future_clock_handling(self):
        settings = {"enabled": True}
        self.assertEqual(next_check(settings, {}, NOW), NOW)
        schedule = {"last_checked_at": NOW.isoformat(), "next_discovery_at": (NOW + timedelta(days=1)).isoformat()}
        self.assertEqual(next_check(settings, schedule, NOW), NOW + timedelta(days=1))
        self.assertLess(next_check(settings, schedule, NOW + timedelta(days=2)), NOW + timedelta(days=2))
        self.assertEqual(next_check(settings, {"last_checked_at": (NOW + timedelta(days=1)).isoformat()}, NOW), NOW)
        self.assertIsNone(next_check({**settings, "enabled": False}, {}, NOW))

    def test_paused_worker_allows_one_manual_refresh_and_coalesces_clicks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_settings(root, False)
            started, finish = threading.Event(), threading.Event()
            calls = []
            def collect(path):
                calls.append(path)
                started.set()
                finish.wait(3)
            service = CollectionService(root, collector=collect)
            try:
                self.assertTrue(service.request_refresh())
                self.assertTrue(started.wait(3))
                self.assertTrue(service.running)
                self.assertFalse(service.request_refresh())
            finally:
                service.stop()
                finish.set()
                service._thread.join(3)
            self.assertEqual(calls, [root])
            self.assertFalse(service._thread.is_alive())

    def test_enabled_worker_collects_without_user_click(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_settings(root, True)
            started = threading.Event()
            def collect(path):
                atomic_json(status_path(path), {"last_attempt_at": datetime.now(timezone.utc).isoformat()})
                started.set()
            service = CollectionService(root, collector=collect)
            try:
                self.assertTrue(started.wait(3))
            finally:
                service.stop()
                service._thread.join(3)
            self.assertFalse(service._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
