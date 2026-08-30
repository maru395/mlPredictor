from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from download_recent_data import parse_draft, verify_series
from mlbb_predictor.history import (
    load_game_history, merge_game_history, recent_team_picks,
    series_key, valid_lineups, valid_picks,
)
from mlbb_predictor.player_model import PlayerEloPredictor, train_player_elo
from mlbb_predictor.teams import load_team_profiles, rating_lineup

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / f"data/processed/mpl_ph_s{season}_player_games.json" for season in (17, 18)]


def synthetic_game(series: int, game: int = 1) -> dict:
    return {
        "date": f"2026-01-{series:02d}", "team_a": "A", "team_b": "B",
        "game": game, "winner": "A", "series_url": f"https://example.test/{series}",
        "players_a": [f"A{i}" for i in range(5)], "players_b": [f"B{i}" for i in range(5)],
        "heroes_a": [f"HeroA{i}" for i in range(5)], "heroes_b": [f"HeroB{i}" for i in range(5)],
        "bans_a": ["NeverPicked"], "bans_b": ["AnotherBan"],
    }


class RecentHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.rows = load_game_history(PATHS)
        cls.new = [row for row in cls.rows if row.get("season") == "MPL Philippines Season 18"]

    def test_all_supplied_results_are_preserved(self):
        self.assertEqual((self.payload["series"], self.payload["games"]), (73, 190))
        self.assertEqual(len(self.new), 21)
        manual = json.loads((ROOT / "data/manual/mpl_ph_s18_aug21_28.json").read_text())
        self.assertEqual(len(manual["series"]), 9)
        for expected in manual["series"]:
            key = (expected["date"], *sorted((expected["team_a"], expected["team_b"])))
            verify_series(expected, [row for row in self.new if series_key(row) == key])

    def test_bans_are_separate_from_played_heroes(self):
        self.assertEqual(sum(row["bans_verified"] for row in self.new), 17)
        row = next(row for row in self.new if row["team_a"] == "TLPH" and row["team_b"] == "FLCN" and row["game"] == 1)
        self.assertIn("Fanny", row["bans_a"])
        self.assertNotIn("Fanny", row["heroes_a"])
        self.assertEqual(set(row["heroes_a"]), {"Rafaela", "Paquito", "Novaria", "Clint", "Uranus"})
        self.assertEqual(row["winner"], "TLPH")

    def test_game_winners_not_guessed_from_series_winner(self):
        outcomes = [row["winner"] for row in self.new if {row["team_a"], row["team_b"]} == {"APBR", "TWIS"}]
        self.assertEqual(outcomes, ["TWIS", "APBR", "APBR"])

    def test_missing_lineups_are_not_fabricated(self):
        self.assertEqual(sum(valid_lineups(row) for row in self.new), 19)
        self.assertEqual(sum(valid_picks(row) for row in self.new), 19)
        missing = [row for row in self.new if not valid_lineups(row)]
        self.assertEqual(len(missing), 2)
        for row in missing:
            self.assertEqual({row["team_a"], row["team_b"]}, {"TWIS", "FNOP"})
            self.assertEqual(len(row["players_a"]), 4)
            self.assertNotIn("heroes_a", row)

    def test_source_duplicate_heroes_are_excluded(self):
        old = [row for row in self.rows if row.get("season") == "MPL Philippines Season 17"]
        self.assertEqual(sum(not valid_picks(row) for row in old), 9)
        self.assertEqual(sum(valid_lineups(row) for row in old), 169)

    def test_last_ten_means_series_and_includes_every_game(self):
        rows = [synthetic_game(i, game) for i in range(1, 12) for game in (1, 2, 3)]
        recent = recent_team_picks(list(reversed(rows)), {"A", "B"})
        coverage = recent["coverage"]["A"]
        self.assertEqual(coverage["series"], 10)
        self.assertEqual(coverage["games"], 30)
        self.assertEqual(coverage["from"], "2026-01-02")
        self.assertEqual(recent["teams"]["A"][0]["picks"], 30)
        self.assertNotIn("NeverPicked", [row["hero"] for row in recent["teams"]["A"]])

    def test_missing_draft_does_not_extend_window(self):
        rows = [synthetic_game(i) for i in range(1, 12)]
        rows[-1]["picks_verified"] = False
        recent = recent_team_picks(rows, {"A"})
        self.assertEqual(recent["coverage"]["A"]["from"], "2026-01-02")
        self.assertEqual(recent["coverage"]["A"]["games_with_picks"], 9)
        self.assertEqual(recent["teams"]["A"][0]["picks"], 9)

    def test_duplicate_import_does_not_double_count(self):
        rows = [synthetic_game(1), synthetic_game(2)]
        duplicate = deepcopy(rows[0])
        duplicate["series_url"] += "?ref=alternate"
        self.assertEqual(len(merge_game_history(rows, [duplicate])), 2)
        recent = recent_team_picks(rows + [duplicate], {"A"})
        self.assertEqual(recent["teams"]["A"][0]["picks"], 2)

    def test_conflicting_duplicate_is_rejected(self):
        row = synthetic_game(1)
        conflicting = {**row, "winner": "B"}
        with self.assertRaises(ValueError):
            merge_game_history([row, conflicting])

    def test_missing_game_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_game_history([synthetic_game(1, 1), synthetic_game(1, 3)])

    def test_fewer_than_ten_series_uses_available_history(self):
        recent = recent_team_picks([synthetic_game(1)], {"A", "C"})
        self.assertEqual(recent["coverage"]["A"]["series"], 1)
        self.assertEqual(recent["coverage"]["C"]["games_with_picks"], 0)
        self.assertEqual(recent["teams"]["C"], [])

    def test_all_teams_have_ten_series_with_correct_pick_denominators(self):
        codes = {row["team_a"] for row in self.rows} | {row["team_b"] for row in self.rows}
        recent = recent_team_picks(self.rows, codes)
        for team, counts in recent["teams"].items():
            coverage = recent["coverage"][team]
            self.assertEqual(coverage["series"], 10)
            self.assertEqual(sum(row["picks"] for row in counts), coverage["games_with_picks"] * 5)
            self.assertEqual(coverage["games"], coverage["games_with_picks"] + len(coverage["missing_pick_games"]))
            self.assertEqual(counts, sorted(counts, key=lambda row: (-row["picks"], row["hero"].casefold())))

    def test_saved_elo_uses_only_complete_lineups_and_keeps_aliases(self):
        model, metrics = PlayerEloPredictor.load(ROOT / "models/player_elo_model.json")
        old = [row for row in self.rows if row.get("season") == "MPL Philippines Season 17"]
        rebuilt, _ = train_player_elo(old)
        rebuilt.fit([row for row in self.new if valid_lineups(row)])
        self.assertEqual(model.ratings, rebuilt.ratings)
        self.assertEqual(sum(model.games.values()), 1880)
        self.assertEqual(metrics["update_game_count"], 19)
        self.assertEqual(metrics["excluded_lineup_games"], 2)
        self.assertEqual(model.player_summary("Nathzz")["games"], 5)
        self.assertEqual(model.player_summary("Nathzz")["wins"], 1)
        self.assertEqual(model.player_summary("JIMPINKMAN")["games"], 7)
        self.assertEqual(model.player_summary("Vin")["games"], 5)
        self.assertNotIn("teddyqt", model.ratings)
        for profile in load_team_profiles(ROOT / "config/mpl_ph_teams.json")["teams"]:
            self.assertEqual(model.lineup_summary(rating_lineup(profile))["known_players"], 5)

    def test_desktop_draft_parser_ignores_repeated_mobile_carousel(self):
        parts = []
        for side in ("A", "B"):
            for kind in ("ban", "pick"):
                for i in range(5):
                    extra = " banned-hero" if kind == "ban" else ""
                    parts.append(f'<div class="hero-portrait{extra}"><img class="hero-img" alt="{side}-{kind}-{i}"></div>')
        desktop = "".join(parts)
        bans_a, bans_b, picks_a, picks_b = parse_draft(desktop + '<div id="mobileDraftSlider-1">' + desktop)
        self.assertEqual(len(bans_a + bans_b + picks_a + picks_b), 20)
        self.assertEqual(picks_a[0], "A-pick-0")
        self.assertEqual(bans_b[-1], "B-ban-4")

    def test_incomplete_draft_parser_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_draft("")


if __name__ == "__main__":
    unittest.main()
