from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlbb_predictor.context import build_context_data
from mlbb_predictor.data import canonical_team, load_all_matches
from mlbb_predictor.meta import (
    analyze_matchup,
    analyze_rating_matchup,
    draft_tier_score,
    load_meta_config,
    save_meta_config,
)
from mlbb_predictor.model import EloPredictor, train_and_evaluate
from mlbb_predictor.player_model import (
    PlayerEloPredictor,
    load_player_games,
    train_player_elo,
)
from mlbb_predictor.teams import (
    load_team_hero_picks,
    load_team_profiles,
    profiles_by_code,
    rating_lineup,
    starting_roster,
)


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_team_aliases_connect_adjacent_seasons(self) -> None:
        self.assertEqual(canonical_team(" AP.Bren "), "FCAP")
        self.assertEqual(canonical_team("ECHO"), "TLPH")
        self.assertEqual(canonical_team("TNC Pro Team"), "TNC")

    def test_real_datasets_normalize_to_expected_counts(self) -> None:
        matches = load_all_matches(ROOT / "data" / "raw")
        self.assertEqual(len(matches), 543)
        self.assertEqual(sum(row["source"] == "Kaggle" for row in matches), 160)
        self.assertEqual(sum(row["source"] == "Hugging Face" for row in matches), 383)
        for row in matches:
            self.assertIn(row["winner"], {row["team_a"], row["team_b"]})

    def test_context_includes_players_and_team_hero_history(self) -> None:
        context = build_context_data(ROOT / "data" / "raw")
        self.assertGreaterEqual(len(context["hero_pool"]), 100)
        self.assertIn("TLPH", context["players"])
        self.assertIn("TLPH", context["team_heroes"])
        echo_players = {row["player"] for row in context["players"]["TLPH"]}
        self.assertIn("KarlTzy", echo_players)
        self.assertIn("KarlTzy", context["player_history"])
        self.assertGreater(context["player_history"]["KarlTzy"]["games"], 0)
        self.assertTrue(context["player_history"]["KarlTzy"]["top_picks"])

    def test_current_team_profiles_are_limited_to_the_eight_mpl_ph_teams(self) -> None:
        payload = load_team_profiles(ROOT / "config" / "mpl_ph_teams.json")
        profiles = profiles_by_code(payload)
        expected_names = [
            "AP BREN",
            "Aurora Gaming PH",
            "Fnatic ONIC PH",
            "Smart Omega",
            "Team Falcons PH",
            "Team Liquid PH",
            "TNC Pro Team",
            "Twisted Minds PH",
        ]
        self.assertEqual([team["name"] for team in payload["teams"]], expected_names)
        self.assertEqual(set(profiles), {"APBR", "RORA", "FNOP", "OMG", "FLCN", "TLPH", "TNC", "TWIS"})
        for team in payload["teams"]:
            self.assertGreaterEqual(len(team["roster"]), 6)
            self.assertIn("standing", team)
            self.assertTrue(team["official_url"].startswith("https://ph-mpl.com/team/"))

        self.assertTrue(profiles["APBR"]["cold_start"])
        self.assertTrue(profiles["TWIS"]["cold_start"])
        self.assertEqual(profiles["FLCN"]["model_code"], "FCAP")

    def test_last_season_hero_picks_cover_every_current_team(self) -> None:
        profiles = profiles_by_code(
            load_team_profiles(ROOT / "config" / "mpl_ph_teams.json")
        )
        payload = load_team_hero_picks(
            ROOT / "config" / "mpl_ph_s17_hero_picks.json", set(profiles)
        )
        self.assertEqual(payload["games"], 169)
        self.assertEqual(payload["total_picks"], 1690)
        self.assertEqual(payload["teams"]["TLPH"][0], {"hero": "Claude", "picks": 19})
        self.assertEqual(payload["teams"]["RORA"][0], {"hero": "Lapu-Lapu", "picks": 14})
        self.assertEqual(payload["teams"]["FLCN"][:2], [
            {"hero": "Claude", "picks": 21},
            {"hero": "Suyou", "picks": 21},
        ])

    def test_player_games_cover_every_s17_game_and_use_real_lineups(self) -> None:
        payload, rows = load_player_games(
            ROOT / "data" / "processed" / "mpl_ph_s17_player_games.json"
        )
        self.assertEqual(payload["series"], 64)
        self.assertEqual(payload["games"], 169)
        self.assertEqual(len(rows), 169)
        self.assertEqual(rows[0]["players_a"], ["Sanford", "KarlTzy", "Sanji", "Jaypee", "Daiki"])
        self.assertEqual(rows[0]["players_b"], ["Edward", "Demonkite", "Yue", "Light", "Domeng"])

    def test_every_team_profile_has_exactly_five_starters(self) -> None:
        payload = load_team_profiles(ROOT / "config" / "mpl_ph_teams.json")
        for profile in payload["teams"]:
            self.assertEqual(len(starting_roster(profile)), 5)
            self.assertEqual(len(rating_lineup(profile)), 5)


class ModelTests(unittest.TestCase):
    def test_probabilities_are_symmetric(self) -> None:
        model = EloPredictor(k_factor=24)
        model.ratings = {"A": 1620, "B": 1480}
        self.assertAlmostEqual(model.predict("A", "B") + model.predict("B", "A"), 1.0)

    def test_winner_gains_rating_and_model_round_trips(self) -> None:
        model = EloPredictor(k_factor=24)
        model.update("A", "B", "A", "2024-01-01")
        self.assertGreater(model.ratings["A"], model.ratings["B"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path, {"match_count": 1})
            loaded, metrics = EloPredictor.load(path)
        self.assertEqual(loaded.games["A"], 1)
        self.assertEqual(metrics["match_count"], 1)

    def test_training_produces_validation_metrics(self) -> None:
        matches = load_all_matches(ROOT / "data" / "raw")
        model, metrics = train_and_evaluate(matches)
        self.assertEqual(metrics["match_count"], 543)
        self.assertGreater(metrics["validation_count"], 0)
        self.assertEqual(len(model.ratings), metrics["team_count"])

    def test_player_elo_updates_only_the_players_in_the_game(self) -> None:
        model = PlayerEloPredictor(k_factor=16)
        row = {
            "team_a": "A",
            "team_b": "B",
            "players_a": ["A1", "A2", "A3", "A4", "A5"],
            "players_b": ["B1", "B2", "B3", "B4", "B5"],
            "winner": "A",
            "date": "2026-01-01",
        }
        model.update(row)
        self.assertGreater(model.player_summary("A1")["rating"], 1500)
        self.assertLess(model.player_summary("B1")["rating"], 1500)
        self.assertEqual(model.player_summary("New Player")["rating"], 1500)

    def test_current_roster_model_favors_tlph_over_rebuilt_aurora(self) -> None:
        _, rows = load_player_games(
            ROOT / "data" / "processed" / "mpl_ph_s17_player_games.json"
        )
        model, metrics = train_player_elo(rows)
        profiles = profiles_by_code(
            load_team_profiles(ROOT / "config" / "mpl_ph_teams.json")
        )
        tlph = rating_lineup(profiles["TLPH"])
        aurora = rating_lineup(profiles["RORA"])
        probability = model.predict_lineups(tlph, aurora)
        self.assertGreater(probability, 0.65)
        self.assertEqual(model.lineup_summary(tlph)["known_players"], 5)
        self.assertEqual(model.lineup_summary(aurora)["known_players"], 3)
        self.assertEqual(metrics["match_count"], 169)

    def test_rating_based_draft_adjustment_keeps_player_elo_as_base(self) -> None:
        result = analyze_rating_matchup(
            1600,
            1500,
            [],
            [],
            {},
            {},
            context_team_a="A",
            context_team_b="B",
        )
        self.assertFalse(result["draft_applied"])
        self.assertGreater(result["base_probability_a"], 0.5)
        self.assertEqual(result["base_probability_a"], result["final_probability_a"])

    def test_complete_draft_applies_meta_and_comfort(self) -> None:
        model = EloPredictor(k_factor=24)
        model.ratings = {"A": 1500, "B": 1500}
        context = {
            "team_heroes": {
                "A": [{"hero": hero, "picks": 10} for hero in ["H1", "H2", "H3", "H4", "H5"]],
                "B": [{"hero": hero, "picks": 10} for hero in ["H6", "H7", "H8", "H9", "H10"]],
            }
        }
        draft_a = ["H1", "H2", "H3", "H4", "H5"]
        draft_b = ["H6", "H7", "H8", "H9", "H10"]
        tiers = {**{hero: "S" for hero in draft_a}, **{hero: "F" for hero in draft_b}}
        result = analyze_matchup(model, "A", "B", draft_a, draft_b, tiers, context)
        self.assertTrue(result["draft_applied"])
        self.assertGreater(result["final_probability_a"], 0.5)
        self.assertEqual(draft_tier_score(draft_a, tiers), 5.0)

    def test_partial_or_duplicate_draft_does_not_adjust_elo(self) -> None:
        model = EloPredictor(k_factor=24)
        model.ratings = {"A": 1550, "B": 1500}
        result = analyze_matchup(model, "A", "B", ["H1"], ["H2"], {}, {})
        self.assertFalse(result["draft_applied"])
        self.assertEqual(result["base_probability_a"], result["final_probability_a"])

    def test_meta_config_round_trips_in_a_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            save_meta_config(
                path,
                {
                    "season": "Test Season",
                    "tiers": {"Joy": "S", "Nana": "F"},
                    "custom_heroes": ["New Hero"],
                },
            )
            loaded = load_meta_config(path)
        self.assertEqual(loaded["season"], "Test Season")
        self.assertEqual(loaded["tiers"]["Joy"], "S")
        self.assertEqual(loaded["custom_heroes"], ["New Hero"])

    def test_supplied_current_meta_config_is_loaded(self) -> None:
        loaded = load_meta_config(ROOT / "config" / "meta_tiers.json")
        self.assertEqual(loaded["season"], "September 2026 Meta (Season 41)")
        self.assertGreaterEqual(len(loaded["tiers"]), 120)
        self.assertEqual(loaded["tiers"]["Paquito"], "S")
        self.assertEqual(loaded["tiers"]["Miya"], "S")
        self.assertEqual(loaded["tiers"]["Marcel"], "S")
        self.assertEqual(loaded["tiers"]["Yve"], "S")
        self.assertEqual(loaded["tiers"]["Zetian"], "S")
        self.assertEqual(loaded["tiers"]["Gatotkaca"], "A")
        self.assertEqual(loaded["tiers"]["Kalea"], "A")
        self.assertEqual(loaded["tiers"]["Zhuxin"], "A")
        self.assertEqual(loaded["role_tiers"]["EXP Lane"]["Atlas"], "A")
        self.assertEqual(loaded["role_tiers"]["EXP Lane"]["Paquito"], "S")
        self.assertEqual(loaded["role_tiers"]["Roam"]["Atlas"], "S")
        self.assertEqual(loaded["role_tiers"]["Roam"]["Marcel"], "S")
        self.assertEqual(loaded["role_tiers"]["Mid Lane"]["Yve"], "S")
        self.assertEqual(loaded["role_tiers"]["Mid Lane"]["Zhuxin"], "A")
        self.assertEqual(loaded["tiers"]["Zilong"], "D")
        self.assertIn("Marcel", loaded["custom_heroes"])
        self.assertIn("tier_conversion", loaded["import"])


if __name__ == "__main__":
    unittest.main()
