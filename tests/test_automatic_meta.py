from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from mlbb_predictor.history import load_game_history, recent_player_picks
from mlbb_predictor.meta import (
    analyze_rating_matchup, load_meta_config, roster_meta_profile, save_meta_config,
)
from mlbb_predictor.player_model import player_key

ROOT = Path(__file__).resolve().parents[1]


def pools(prefix: str, count: int = 10, hero: str = "Strong") -> tuple[list[str], dict]:
    players = [f"{prefix}{index}" for index in range(5)]
    history = {player_key(player): {"heroes": [{"hero": hero, "picks": count}]} for player in players}
    return players, history


def matchup(profile_a, profile_b, **kwargs):
    return analyze_rating_matchup(
        1500, 1500, kwargs.pop("draft_a", []), kwargs.pop("draft_b", []),
        kwargs.pop("tiers", {}), {}, context_team_a="A", context_team_b="B",
        automatic_profile_a=profile_a, automatic_profile_b=profile_b, **kwargs,
    )


class AutomaticMetaTests(unittest.TestCase):
    def test_more_s_tier_usage_increases_probability_without_a_draft(self):
        players_a, history_a = pools("A", hero="Strong")
        players_b, history_b = pools("B", hero="Weak")
        tiers = {"Strong": "S", "Weak": "F"}
        a = roster_meta_profile(players_a, history_a, tiers)
        b = roster_meta_profile(players_b, history_b, tiers)
        result = matchup(a, b)
        self.assertEqual(result["adjustment_mode"], "automatic")
        self.assertTrue(result["automatic_meta_applied"])
        self.assertFalse(result["draft_applied"])
        self.assertGreater(result["final_probability_a"], 0.5)
        self.assertLess(result["final_probability_a"], 1)
        self.assertEqual(result["comfort_adjustment_a"], 0)

    def test_pick_frequency_not_just_hero_presence_weights_fit(self):
        players, history = pools("A", count=9)
        for row in history.values():
            row["heroes"].append({"hero": "Weak", "picks": 1})
        frequent_s = roster_meta_profile(players, history, {"Strong": "S", "Weak": "F"})
        for row in history.values():
            row["heroes"][0]["picks"] = 1
            row["heroes"][1]["picks"] = 9
        frequent_f = roster_meta_profile(players, history, {"Strong": "S", "Weak": "F"})
        self.assertGreater(frequent_s["tier_score"], frequent_f["tier_score"])
        self.assertAlmostEqual(frequent_s["players"][0]["raw_tier_score"], 4.5)
        self.assertAlmostEqual(frequent_s["players"][0]["sa_pick_share"], .9)

    def test_sparse_history_is_shrunk_toward_neutral(self):
        players, few = pools("A", count=1)
        _, many = pools("A", count=30)
        small = roster_meta_profile(players, few, {"Strong": "S"})
        large = roster_meta_profile(players, many, {"Strong": "S"})
        self.assertGreater(small["tier_score"], 2.5)
        self.assertLess(small["tier_score"], large["tier_score"])
        self.assertLess(large["tier_score"], 5)

    def test_missing_and_unrated_history_remain_neutral(self):
        players, history = pools("A")
        missing = roster_meta_profile(players, {}, {"Strong": "S"})
        unrated = roster_meta_profile(players, history, {})
        self.assertEqual(missing["tier_score"], 2.5)
        self.assertEqual(unrated["tier_score"], 2.5)
        self.assertEqual(unrated["players_with_rated_picks"], 0)
        self.assertEqual(matchup(missing, unrated)["final_probability_a"], .5)

    def test_only_current_starters_count_and_each_has_equal_weight(self):
        players, history = pools("A", count=10)
        history["departed player"] = {"heroes": [{"hero": "Weak", "picks": 1000}]}
        history[player_key(players[0])] = {"heroes": [{"hero": "Weak", "picks": 100}]}
        result = roster_meta_profile(players, history, {"Strong": "S", "Weak": "F"})
        expected = ((12.5 / 105) + 4 * (62.5 / 15)) / 5
        self.assertAlmostEqual(result["tier_score"], expected)
        self.assertNotIn("departed player", [row["player"] for row in result["players"]])

    def test_equal_meta_fit_does_not_change_relative_chance(self):
        players, history = pools("A")
        profile = roster_meta_profile(players, history, {"Strong": "S"})
        result = analyze_rating_matchup(1650, 1500, [], [], {}, {}, context_team_a="A", context_team_b="B",
                                        automatic_profile_a=profile, automatic_profile_b=profile)
        self.assertAlmostEqual(result["base_probability_a"], result["final_probability_a"])

    def test_swap_symmetry_and_disable_or_zero_weight(self):
        a, b = {"tier_score": 4.1}, {"tier_score": 2.1}
        self.assertAlmostEqual(matchup(a, b)["final_probability_a"] + matchup(b, a)["final_probability_a"], 1)
        disabled = matchup(a, b, automatic_meta_enabled=False)
        self.assertEqual(disabled["adjustment_mode"], "base")
        self.assertEqual(disabled["final_probability_a"], .5)
        self.assertEqual(matchup(a, b, tier_elo_per_step=0)["final_probability_a"], .5)

    def test_complete_draft_replaces_auto_without_double_counting(self):
        draft_a = [f"H{i}" for i in range(5)]
        draft_b = [f"J{i}" for i in range(5)]
        tiers = {hero: "S" for hero in draft_a} | {hero: "F" for hero in draft_b}
        with_auto = matchup({"tier_score": 0}, {"tier_score": 5}, draft_a=draft_a, draft_b=draft_b, tiers=tiers)
        without_auto = matchup(None, None, draft_a=draft_a, draft_b=draft_b, tiers=tiers)
        self.assertEqual(with_auto, without_auto)
        self.assertEqual(with_auto["adjustment_mode"], "draft")
        self.assertFalse(with_auto["automatic_meta_applied"])

    def test_partial_or_overlapping_draft_keeps_auto_active(self):
        profiles = ({"tier_score": 4}, {"tier_score": 2})
        base = matchup(*profiles)
        self.assertEqual(matchup(*profiles, draft_a=["H1"]), base)
        same = [f"H{i}" for i in range(5)]
        self.assertEqual(matchup(*profiles, draft_a=same, draft_b=same), base)

    def test_saved_tier_changes_recalculate_without_retraining(self):
        players, history = pools("A")
        original = deepcopy(history)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiers.json"
            save_meta_config(path, {"tiers": {"Strong": "F"}})
            low = roster_meta_profile(players, history, load_meta_config(path)["tiers"])
            save_meta_config(path, {"tiers": {"Strong": "S"}})
            high = roster_meta_profile(players, history, load_meta_config(path)["tiers"])
        self.assertGreater(high["tier_score"], low["tier_score"])
        self.assertEqual(history, original)

    def test_real_player_pools_are_limited_to_ten_series(self):
        _, rows = load_game_history([ROOT / f"data/processed/mpl_ph_s{s}_player_games.json" for s in (17, 18)])
        result = recent_player_picks(rows, ["KarlTzy", "Nathzz", "Unknown"])
        self.assertEqual(result["karltzy"]["series"], 10)
        self.assertEqual(result["nathzz"]["series"], 2)
        self.assertEqual(result["nathzz"]["games_with_picks"], 5)
        self.assertEqual(result["unknown"]["heroes"], [])
        for row in result.values():
            self.assertEqual(sum(p["picks"] for p in row["heroes"]), row["games_with_picks"])

    def test_player_window_follows_transfers_and_preserves_gaps(self):
        rows = []
        for i in range(1, 12):
            rows.append({
                "date": f"2026-01-{i:02d}", "team_a": "Old" if i < 6 else "New", "team_b": "Other",
                "game": 1, "winner": "Other", "series_url": f"https://example.test/{i}",
                "players_a": ["Transfer", "A2", "A3", "A4", "A5"],
                "players_b": ["B1", "B2", "B3", "B4", "B5"],
                "heroes_a": ["Used", "H2", "H3", "H4", "H5"],
                "heroes_b": ["I1", "I2", "I3", "I4", "I5"], "bans_a": ["Banned"],
            })
        rows[-1]["picks_verified"] = False
        result = recent_player_picks(list(reversed(rows)), ["  TRANSFER  "])["transfer"]
        self.assertEqual(result["series"], 10)
        self.assertEqual(result["from"], "2026-01-02")
        self.assertEqual(result["games_with_picks"], 9)
        self.assertEqual(result["missing_pick_games"], 1)
        self.assertEqual(result["heroes"], [{"hero": "Used", "picks": 9}])
        rows[-1]["picks_verified"] = True
        rows[-1]["lineups_verified"] = False
        result = recent_player_picks(rows, ["Transfer"])["transfer"]
        self.assertEqual(result["games_with_picks"], 9)


if __name__ == "__main__":
    unittest.main()
