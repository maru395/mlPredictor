"""Automatic profiles and cross-feature projections from one collected history."""

from copy import deepcopy
from datetime import timedelta
import html
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlbb_predictor.automatic_collection import next_check, run_automatic_collection
from mlbb_predictor.collector import atomic_json, discover_schedule, read_json, run_collection
from mlbb_predictor.history import recent_player_picks, recent_team_picks
from mlbb_predictor.live_data import data_revision, load_prediction_state
from mlbb_predictor.profile_collection import (
    parse_profile, profile_snapshot_path, profile_status_path, refresh_profiles,
)
from mlbb_predictor.teams import current_team_profiles, load_team_profiles, rating_lineup
from test_collection import NOW, fixture_card, game, series_html

ROOT = Path(__file__).resolve().parents[1]


def official_page(team, season="Season 18"):
    members = [(member["player"], member["role"]) for member in team["roster"]]
    members.append(("Coach " + team["code"], "Coach"))
    return ('<div class="team-description"><p>Updated official summary</p></div>'
            '<div class="achievements"><h3>Achievements</h3>'
            '<div class="d-flex justify-content-between"><div>Season 17</div><div>1st</div></div></div>'
            f'<span>Roster for {season}</span>' + ''.join(
                f'<input name="player-{i}-ign" value="{html.escape(name)}">'
                f'<input name="player-{i}-role" value="{html.escape(role)}">'
                for i, (name, role) in enumerate(members)))


class ProfileUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "config").mkdir()
        shutil.copy2(ROOT / "config/mpl_ph_teams.json", self.root / "config/mpl_ph_teams.json")
        self.profiles = load_team_profiles(self.root / "config/mpl_ph_teams.json")
        self.pages = {team["official_url"]: official_page(team) for team in self.profiles["teams"]}

    def refresh(self, now=NOW, **kwargs):
        return refresh_profiles(self.root, now=now, fetcher=lambda url: self.pages[url], **kwargs)

    def test_daily_refresh_aliases_staff_summary_and_seed_preservation(self):
        seed = (self.root / "config/mpl_ph_teams.json").read_bytes()
        result = self.refresh()
        self.assertEqual(result["state"], "success")
        saved = read_json(profile_snapshot_path(self.root))
        aurora = saved["teams"][1]
        self.assertEqual(aurora["summary"], "Updated official summary")
        self.assertEqual(aurora["achievements"], ["Season 17 — 1st"])
        self.assertEqual(aurora["staff"][0]["role"], "Coach")
        self.assertEqual(next(m for m in aurora["roster"] if m["player"] == "Domengkite")["history_alias"], "Domeng")
        self.assertEqual((self.root / "config/mpl_ph_teams.json").read_bytes(), seed)
        fetcher = Mock(side_effect=AssertionError("Not due"))
        refresh_profiles(self.root, now=NOW + timedelta(hours=23), fetcher=fetcher)
        fetcher.assert_not_called()
        self.assertEqual(self.refresh(NOW + timedelta(days=1))["last_success_at"], (NOW + timedelta(days=1)).isoformat())

    def test_partial_outage_preserves_failed_team_and_retries_hourly(self):
        self.refresh()
        old = read_json(profile_snapshot_path(self.root))
        team = self.profiles["teams"][1]
        self.pages[team["official_url"]] = "Source unavailable"
        result = self.refresh(NOW + timedelta(days=1))
        self.assertEqual(result["state"], "partial")
        self.assertEqual(result["next_check_at"], (NOW + timedelta(hours=25)).isoformat())
        self.assertEqual(read_json(profile_snapshot_path(self.root))["teams"][1], old["teams"][1])

    def test_season_rollover_missing_roles_and_duplicate_players_are_rejected(self):
        team = deepcopy(self.profiles["teams"][0])
        with self.assertRaises(ValueError):
            parse_profile(official_page(team, "Season 19"), team, {})
        team["roster"][0]["role"] = "Unknown"
        with self.assertRaises(ValueError):
            parse_profile(official_page(team), team, {})
        team = deepcopy(self.profiles["teams"][0])
        team["roster"][-1]["player"] = team["roster"][0]["player"]
        with self.assertRaises(ValueError):
            parse_profile(official_page(team), team, {})

    def test_role_spelling_variants_and_single_cell_achievement(self):
        team = self.profiles["teams"][0]
        page = official_page(team).replace('value="Gold Lane"', 'value="Goldlane"')
        page = page.replace('<div>Season 17</div><div>1st</div>', '<div>Season 17 - 1st</div>')
        result = parse_profile(page, team, {})
        self.assertIn("Gold Lane", [member["role"] for member in result["roster"]])
        self.assertEqual(result["achievements"], ["Season 17 - 1st"])

    def test_profile_due_overrides_recent_schedule_and_pause_stops_both(self):
        schedule = {"state": "success", "last_checked_at": NOW.isoformat(),
                    "next_discovery_at": (NOW + timedelta(days=1)).isoformat()}
        self.assertEqual(next_check({"enabled": True}, schedule, NOW, self.root), NOW)
        self.assertIsNone(next_check({"enabled": False}, schedule, NOW, self.root))
        self.refresh()
        self.assertEqual(next_check({"enabled": True}, schedule, NOW, self.root), NOW + timedelta(days=1))

    def test_match_source_backoff_does_not_block_profile_refresh(self):
        atomic_json(self.root / "data/processed/match_schedule.json", {
            "state": "error", "retry_at": (NOW + timedelta(hours=1)).isoformat()})
        with patch("mlbb_predictor.automatic_collection.utc_now", return_value=NOW), \
                patch("mlbb_predictor.automatic_collection.run_scheduled_collection") as matches, \
                patch("mlbb_predictor.automatic_collection.refresh_profiles", return_value={"state": "success"}) as profiles:
            result = run_automatic_collection(self.root)
        matches.assert_not_called()
        profiles.assert_called_once_with(self.root)
        self.assertEqual(result["profile_report"]["state"], "success")

    def test_revision_watches_profiles_tiers_and_collected_matches(self):
        initial = data_revision(self.root)
        self.refresh()
        profiles = data_revision(self.root)
        self.assertNotEqual(initial, profiles)
        atomic_json(self.root / "config/meta_tiers.json", {"tiers": {"New hero": "S"}})
        tiers = data_revision(self.root)
        self.assertNotEqual(tiers, profiles)
        atomic_json(self.root / "data/processed/live_snapshot.json", {})
        self.assertNotEqual(tiers, data_revision(self.root))

    def test_collected_substitute_updates_prediction_lineup_and_profile_records(self):
        self.refresh()
        rows = [game(1), game(2)]
        team = self.profiles["teams"][0]
        opponent = self.profiles["teams"][1]
        names = rating_lineup(team)
        names[0] = "Allen"
        for row in rows:
            row.update(team_a=team["code"], team_b=opponent["code"], winner=team["code"],
                       players_a=names, players_b=rating_lineup(opponent))
        projected = current_team_profiles(self.root, rows)
        updated = projected["teams"][0]
        self.assertIn("Allen", rating_lineup(updated))
        self.assertNotIn("JIMPINKMAN", rating_lineup(updated))
        self.assertIn("Latest verified", updated["lineup_basis"])
        self.assertEqual(updated["standing"]["matches_won"], 1)
        self.assertEqual(updated["standing"]["games_won"], 2)
        # The same incoming heroes feed profile favorites and player meta pools.
        favorites = recent_team_picks(rows, {team["code"]})
        self.assertEqual(favorites["teams"][team["code"]][0]["picks"], 2)
        self.assertTrue(recent_player_picks(rows, names)["allen"]["heroes"])

    def test_unregistered_player_is_not_silently_added_to_prediction(self):
        self.refresh()
        row = game()
        row.update(team_a="APBR", players_a=["Unregistered"] + rating_lineup(self.profiles["teams"][0])[1:])
        projected = current_team_profiles(self.root, [row])["teams"][0]
        self.assertNotIn("Unregistered", rating_lineup(projected))
        self.assertIn("lineup_warning", projected)

    def test_standings_exclude_playoffs_incomplete_series_and_duplicate_imports(self):
        rows = [game(1), game(2)]
        playoff = [dict(row, date="2026-08-30", stage="Playoffs") for row in rows]
        incomplete = [dict(game(), date="2026-08-31")]
        projected = current_team_profiles(self.root, rows + rows + playoff + incomplete)
        liquid = next(t for t in projected["teams"] if t["code"] == "TLPH")
        self.assertEqual(liquid["standing"]["games_won"], 2)
        self.assertEqual(projected["standings_as_of"], "2026-08-29")

    def test_stage_does_not_leak_from_upcoming_playoffs_into_completed_results(self):
        card = fixture_card().replace('<div data-stage="Regular Season">', '<div>')
        url = game()["series_url"]
        page = ('<div data-stage="Playoffs"></div>' + card +
                '<span class="td-dr-week-title">Week 2</span>' +
                f'<tr class="td-dr-row" data-href="{url}"></tr>')
        self.assertEqual(discover_schedule(page, NOW)[0]["stage"], "Regular Season")
        self.assertEqual(discover_schedule('<div data-stage="Playoffs"></div>' + card, NOW)[0]["stage"], "Unknown")

    def test_published_collection_reaches_app_history_profiles_meta_and_hero_editor(self):
        import json
        import os
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit unavailable")
        for name in ("models/player_elo_model.json", "data/processed/mpl_ph_s17_player_games.json",
                     "data/processed/mpl_ph_s18_player_games.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        before = load_prediction_state(self.root)
        teams = {team["code"]: team for team in self.profiles["teams"]}
        rows = [game(1), game(2)]
        for row in rows:
            row.update(players_a=rating_lineup(teams["TLPH"]), players_b=rating_lineup(teams["RORA"]))
        self.refresh()
        report = run_collection(self.root, now=NOW, fetcher=lambda url, ajax=False:
                                json.dumps({"html": ""}) if ajax else series_html(rows),
                                fixture_feed=discover_schedule(fixture_card(), NOW))
        self.assertEqual(report["state"], "success")
        after = load_prediction_state(self.root)
        self.assertEqual(after[1]["match_count"], before[1]["match_count"] + 2)
        self.assertNotEqual(after[0].ratings, before[0].ratings)
        projected = current_team_profiles(self.root, after[3])
        revision = ("cross-feature-test", str(self.root), *data_revision(self.root))
        with patch.dict(os.environ, {"MLBB_OFFLINE_TEST_MODE": "1"}), \
                patch("mlbb_predictor.live_data.load_prediction_state", return_value=after), \
                patch("mlbb_predictor.live_data.data_revision", return_value=revision), \
                patch("mlbb_predictor.teams.current_team_profiles", return_value=projected):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            app.selectbox(key="predict_team_b").set_value("RORA")
            app.selectbox(key="inspector_team").set_value("TLPH").run()
        self.assertFalse(app.exception)
        standings = next(frame.value for frame in app.dataframe if "Match points" in frame.value.columns)
        liquid = standings[standings["Team"] == "Team Liquid PH"].iloc[0]
        self.assertEqual(liquid["Match W-L"], "3-1")
        self.assertEqual(liquid["Match points"], 3)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Series record"), "3-1")
        imported = [frame.value for frame in app.dataframe if "Used for Elo" in frame.value.columns]
        self.assertEqual(sum(len(table) for table in imported), 23)
        self.assertIn("Hero A0", " ".join(value for table in imported for value in table["Team A picks"]))
        self.assertTrue(any("Hero A0" in value for value in app.multiselect(key="tier_editor_S").options))
        roster = next(frame.value for frame in app.dataframe if "Most-used picks this season" in frame.value.columns)
        self.assertTrue(all(value != "No usable Season 18 picks" for value in roster["Most-used picks this season"][:5]))
        pools = recent_player_picks(after[3], rating_lineup(teams["TLPH"]))
        self.assertIn({"hero": "Hero A0", "picks": 2}, pools["sanford"]["heroes"])
        evidence = next(frame.value for frame in app.dataframe if "Meta Elo share" in frame.value.columns)
        self.assertTrue(any("2026-08-29" in str(value) for value in evidence.values.flatten()))


if __name__ == "__main__":
    unittest.main()
