"""Season totals keep every appearance and exclude older seasons and duplicates."""

from copy import deepcopy
from collections import Counter
from datetime import date, timedelta
import unittest

from mlbb_predictor.history import season_player_picks, season_player_pick_records, season_team_picks, matches_by_week
from mlbb_predictor.meta import roster_meta_profile
from mlbb_predictor.collector import discover_schedule
from test_collection import game, fixture_card, NOW


class SeasonStatsTests(unittest.TestCase):
    def rows(self, count):
        rows = []
        for i in range(count):
            row = game()
            row.update(date=(date(2026, 8, 21) + timedelta(days=i)).isoformat())
            row["players_a"][0] = "KarlTzy"
            row["heroes_a"][0] = "Paquito"
            rows.append(row)
        return rows

    def test_tenth_pick_becomes_eleventh_and_repeat_import_does_not_add_twelfth(self):
        rows = self.rows(11)
        old_season = dict(rows[0], season="MPL Philippines Season 17", date="2026-01-01")
        initial = season_player_picks([old_season] + rows[:10], ["KarlTzy"])
        self.assertEqual(initial["karltzy"]["heroes"], [{"hero": "Paquito", "picks": 10}])
        updated = season_player_picks([old_season] + rows + [deepcopy(rows[-1])], ["KarlTzy"])
        self.assertEqual(updated["karltzy"]["heroes"], [{"hero": "Paquito", "picks": 11}])
        self.assertEqual(updated["karltzy"]["wins"], 11)
        team = season_team_picks(rows + [rows[-1]], {"TLPH"})
        self.assertIn({"hero": "Paquito", "picks": 11}, team["teams"]["TLPH"])
        lineup = ["KarlTzy", "Sanford", "Sanji", "Teddy", "Jaypee"]
        before_meta = roster_meta_profile(lineup, initial, {"Paquito": "S"})
        after_meta = roster_meta_profile(lineup, updated, {"Paquito": "S"})
        self.assertGreater(after_meta["tier_score"], before_meta["tier_score"])

    def test_incomplete_mappings_and_bans_do_not_become_player_picks(self):
        rows = self.rows(3)
        rows[1]["lineups_verified"] = False
        rows[2]["heroes_a"][0] = "Nolan"
        rows[2]["bans_a"] = ["Paquito"]
        result = season_player_picks(rows, ["KarlTzy"])["karltzy"]
        self.assertEqual(result["heroes"], [{"hero": "Nolan", "picks": 1}, {"hero": "Paquito", "picks": 1}])
        self.assertEqual(result["missing_pick_games"], 1)

    def test_weeks_follow_extracted_week_and_calendar_for_seed_games(self):
        rows = self.rows(1)
        rows[0].update(date="2026-08-28", week=2, stage="Regular Season")
        rows.append(dict(rows[0], date="2026-09-06"))
        rows[1].pop("week")
        groups = matches_by_week(rows)
        self.assertEqual([group["label"] for group in groups], ["Week 3", "Week 2"])

    def test_event_week_is_extracted_from_result_section(self):
        url = game()["series_url"]
        page = fixture_card() + f'<span class="td-dr-week-title">Week 2</span><tr class="td-dr-row" data-href="{url}"></tr>'
        fixture = discover_schedule(page, NOW)[0]
        self.assertEqual(fixture["week"], 2)
        self.assertEqual(fixture["stage"], "Regular Season")

    def test_previous_season_yve_cannot_enter_sanji_meta_or_audit_records(self):
        rows = self.rows(13)
        heroes = ["Novaria"] * 5 + ["Eudora"] * 4 + ["Selena"] * 3 + ["Valentina"]
        for row, hero in zip(rows, heroes):
            row["players_a"][0] = "Sanji"
            row["heroes_a"][0] = hero
        old = deepcopy(rows[0])
        old.update(date="2026-05-01", season="MPL Philippines Season 17")
        old["heroes_a"][0] = "Yve"
        combined = [old] + rows + [deepcopy(rows[-1])]
        pools = season_player_picks(combined, ["Sanji"])
        expected = Counter({"Novaria": 5, "Eudora": 4, "Selena": 3, "Valentina": 1})
        self.assertEqual({p["hero"]: p["picks"] for p in pools["sanji"]["heroes"]}, expected)
        evidence = season_player_pick_records(combined, "sAnJi")
        self.assertEqual(Counter(row["Hero"] for row in evidence), expected)
        self.assertTrue(all(row["Season"] == "MPL Philippines Season 18" and row["Source"] for row in evidence))
        lineup = ["Sanford", "KarlTzy", "Sanji", "Teddy", "Jaypee"]
        with_old = roster_meta_profile(lineup, pools, {"Yve": "S", "Eudora": "A"})
        only_current = roster_meta_profile(lineup, season_player_picks(rows, ["Sanji"]), {"Yve": "S", "Eudora": "A"})
        self.assertEqual(with_old, only_current)
        old_only = season_player_picks([old], ["Sanji"])
        self.assertEqual(old_only["sanji"]["heroes"], [])
        self.assertEqual(season_player_pick_records([old], "Sanji"), [])


if __name__ == "__main__":
    unittest.main()
