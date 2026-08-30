"""Local Streamlit integration checks; run with the project's .venv Python."""

from pathlib import Path
import os
import unittest
from unittest.mock import patch
import re

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(AppTest is None, "Streamlit is installed in the project .venv")
class AppTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"MLBB_OFFLINE_TEST_MODE": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        self.app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(self.app.exception)

    def test_import_table_and_recent_profile_counts(self):
        tables = [frame.value for frame in self.app.dataframe]
        imported = next(table for table in tables if "Used for Elo" in table.columns)
        self.assertEqual(len(imported), 21)
        self.assertEqual(int(imported["Used for Elo"].sum()), 19)
        self.app.selectbox(key="inspector_team").set_value("TWIS").run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("window is not extended" in warning.value for warning in self.app.warning))
        favorites = next(frame.value for frame in self.app.dataframe if "Recent picks" in frame.value.columns)
        self.assertEqual(favorites.iloc[0]["Hero"], "Claude")
        self.assertEqual(favorites.iloc[0]["Recent picks"], 9)

    def test_collection_ui_explains_match_aware_timing_without_interval_control(self):
        self.assertNotIn("collection_interval", [element.key for element in self.app.selectbox])
        self.assertTrue(any("first observed completion" in item.value for item in self.app.warning))
        self.assertEqual(self.app.button(key="collect_now").label, "Check schedule now")
        self.assertTrue(self.app.toggle(key="collection_enabled").value)

    def test_complete_drafts_apply_to_current_player_ratings(self):
        self.app.selectbox(key="predict_team_b").set_value("RORA").run()
        self.app.multiselect(key="draft_a").set_value(["Claude", "Paquito", "Yve", "Harley", "Phoveus"])
        self.app.multiselect(key="draft_b").set_value(["Valentina", "Gatotkaca", "Suyou", "Hylos", "Belerick"])
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Meta-adjusted favorite" in item.value for item in self.app.markdown))
        adjustment = next(frame.value for frame in self.app.dataframe if "Last-10-series comfort" in frame.value.columns)
        self.assertEqual(len(adjustment), 2)

    def test_same_team_selection_and_incomplete_draft_are_safe(self):
        self.app.selectbox(key="predict_team_b").set_value("TLPH").run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("two different teams" in item.value for item in self.app.warning))
        self.app.selectbox(key="predict_team_b").set_value("RORA").run()
        self.app.multiselect(key="draft_a").set_value(["Claude"]).run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any("Meta-adjusted favorite" in item.value for item in self.app.markdown))
        self.assertTrue(any("Automatic meta favorite" in item.value for item in self.app.markdown))

    def test_automatic_meta_is_on_without_draft_input_and_can_be_disabled(self):
        self.assertTrue(self.app.toggle(key="automatic_meta_enabled").value)
        self.assertEqual(self.app.multiselect(key="draft_a").value, [])
        self.assertTrue(any("Automatic meta favorite" in item.value for item in self.app.markdown))
        table = next(frame.value for frame in self.app.dataframe if "Automatic meta Elo" in frame.value.columns)
        self.assertEqual(len(table), 2)
        self.app.toggle(key="automatic_meta_enabled").set_value(False).run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Current-roster favorite" in item.value for item in self.app.markdown))
        self.assertFalse(any("Automatic meta favorite" in item.value for item in self.app.markdown))

    def test_automatic_evidence_uses_current_aurora_players(self):
        self.app.selectbox(key="predict_team_b").set_value("RORA").run()
        evidence = next(frame.value for frame in self.app.dataframe if "Meta Elo share" in frame.value.columns)
        aurora = evidence[evidence["Team"] == "Aurora Gaming PH"]
        self.assertEqual(set(aurora["Player"]), {"Nathzz", "Koyl", "Yue", "Domengkite", "Light"})
        self.assertNotIn("Demonkite", set(evidence["Player"]))
        self.assertNotIn("Edward", set(evidence["Player"]))
        self.app.slider(key="tier_elo_weight").set_value(0).run()
        table = next(frame.value for frame in self.app.dataframe if "Automatic meta Elo" in frame.value.columns)
        self.assertTrue(all(float(value) == 0 for value in table["Automatic meta Elo"]))

    def test_redesigned_navigation_and_forecast_precede_optional_settings(self):
        self.assertEqual([tab.label for tab in self.app.tabs], [
            "Predict", "MPL PH team profiles", "Season meta tiers", "Collected matches", "Data collection",
        ])
        # Check actual element-tree positions, not just the order of calls.
        output = self.app.main.get("markdown")
        card = next(item for item in output if 'aria-label="Match forecast"' in item.value)
        self.assertIn("Player-Elo baseline", card.value)
        self.assertIn("Current players", next(item.value for item in output if "<h1>" in item.value))
        draft_section = next(item for item in self.app.expander if item.label == "Automatic meta and optional drafts")
        positions = {id(node): index for index, node in enumerate(self.app.main)}
        self.assertLess(positions[id(card)], positions[id(draft_section)])
        self.assertFalse(draft_section.proto.expanded)
        self.assertIn("one game", card.value)

    def test_swapping_teams_swaps_probabilities(self):
        def probabilities():
            markup = next(item.value for item in self.app.markdown if 'aria-label="Match forecast"' in item.value)
            return re.findall(r'class="matchdesk-chance">([\d.]+)', markup)
        before = probabilities()
        self.app.selectbox(key="predict_team_a").set_value("FNOP")
        self.app.selectbox(key="predict_team_b").set_value("TLPH").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(probabilities(), before[::-1])

    def test_duplicate_draft_still_uses_automatic_mode(self):
        self.app.multiselect(key="draft_a").set_value(["Claude", "Paquito", "Yve", "Harley", "Phoveus"])
        self.app.multiselect(key="draft_b").set_value(["Claude", "Gatotkaca", "Suyou", "Hylos", "Belerick"])
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("same hero cannot be on both sides" in item.value for item in self.app.warning))
        self.assertTrue(any("Automatic meta favorite" in item.value for item in self.app.markdown))

    def test_all_eight_team_profiles_and_tier_editor_remain_available(self):
        for team in ("APBR", "RORA", "FNOP", "OMG", "FLCN", "TLPH", "TNC", "TWIS"):
            self.app.selectbox(key="inspector_team").set_value(team).run()
            self.assertFalse(self.app.exception, team)
            visible = " ".join(item.value for item in self.app.markdown)
            self.assertNotRegex(visible, "[\u2013\u2014]")
        for tier in "SABCDF":
            self.assertEqual(self.app.multiselect(key=f"tier_editor_{tier}").label, f"{tier} tier")

    def test_tier_changes_can_be_saved_without_changing_the_real_config(self):
        hero = "Akai"
        for tier in "SABCDF":
            widget = self.app.multiselect(key=f"tier_editor_{tier}")
            values = [item for item in widget.value if item != hero]
            widget.set_value(values + [hero] if tier == "S" else values)
        save_button = next(button for button in self.app.button if button.label == "Save season tiers")
        with patch("mlbb_predictor.meta.save_meta_config") as save:
            save_button.click().run()
        self.assertFalse(self.app.exception)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[1]["tiers"][hero], "S")

    def test_current_live_snapshot_renders_without_starting_collection(self):
        from mlbb_predictor.live_data import load_prediction_state, snapshot_path
        path = snapshot_path(ROOT)
        if not path.exists():
            self.skipTest("No optional collected snapshot on this machine")
        snapshot = load_prediction_state(ROOT, offline=False)
        revision = ("presentation-live-snapshot", path.stat().st_mtime_ns)
        with patch("mlbb_predictor.live_data.load_prediction_state", return_value=snapshot), \
             patch("mlbb_predictor.live_data.data_revision", return_value=revision):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        strip = next(item.value for item in app.markdown if 'class="matchdesk-data-strip"' in item.value)
        self.assertIn(str(snapshot[1]["match_count"]), strip)
        self.assertIn(snapshot[2]["cutoff"], strip)
        self.assertTrue(app.button(key="collect_now").disabled)


if __name__ == "__main__":
    unittest.main()
