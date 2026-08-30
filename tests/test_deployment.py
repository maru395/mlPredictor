"""Exercise a portable checkout without raw datasets or machine-local state."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from mlbb_predictor.context import read_context_data
from mlbb_predictor.history import valid_lineups
from mlbb_predictor.live_data import load_prediction_state


ROOT = Path(__file__).resolve().parents[1]
SAVED_FILES = (
    "models/player_elo_model.json",
    "data/processed/context.json",
    "data/processed/mpl_ph_s17_player_games.json",
    "data/processed/mpl_ph_s18_player_games.json",
)
SUPPORT_FILES = (
    "app.py", "requirements.txt", "download_recent_data.py", "download_player_data.py",
    "config/mpl_ph_teams.json", "config/meta_tiers.json", "config/data_collection.json",
    "ui/matchdesk.css", ".streamlit/config.toml",
)


def copy_checkout(destination, *, omit=(), live=False):
    names = list(SAVED_FILES + SUPPORT_FILES)
    names.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / "mlbb_predictor").glob("*.py"))
    if live:
        names.append("data/processed/live_snapshot.json")
    for name in names:
        if name in omit:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


class DeploymentTests(unittest.TestCase):
    def test_saved_predictor_loads_without_raw_datasets_or_collection_state(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            copy_checkout(checkout)
            _, metrics, _, rows = load_prediction_state(checkout, offline=True)
            self.assertEqual(sum(valid_lineups(row) for row in rows), metrics["match_count"])
            self.assertTrue(read_context_data(checkout / "data/processed/context.json")["hero_pool"])
            self.assertFalse((checkout / "data/raw").exists())
            self.assertFalse((checkout / "data/processed/match_schedule.json").exists())

    def test_optional_snapshot_preserves_current_collected_ratings(self):
        if not (ROOT / "data/processed/live_snapshot.json").exists():
            self.skipTest("No optional collected snapshot")
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            copy_checkout(checkout, live=True)
            expected = load_prediction_state(ROOT)
            actual = load_prediction_state(checkout)
            self.assertEqual(actual[0].ratings, expected[0].ratings)
            self.assertEqual(actual[1]["history_fingerprint"], expected[1]["history_fingerprint"])
            self.assertEqual(actual[2]["cutoff"], expected[2]["cutoff"])

    def test_git_allows_saved_artifacts_but_ignores_local_runtime_files(self):
        if not shutil.which("git"):
            self.skipTest("Git is not installed")
        repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True)
        if repo.returncode:
            self.skipTest("Not a Git checkout")
        files = (*SAVED_FILES, "data/processed/live_snapshot.json")
        for name in files:
            with self.subTest(name=name):
                result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", name], cwd=ROOT)
                self.assertEqual(result.returncode, 1, "A required saved file is still ignored by Git")
        for name in ("data/processed/collection.lock", "data/processed/schedule.lock",
                     "data/processed/match_schedule.json", "data/processed/collection_status.json",
                     "data/processed/live_snapshot.previous.json", "models/elo_model.json",
                     ".streamlit/secrets.toml"):
            with self.subTest(name=name):
                result = subprocess.run(["git", "check-ignore", "--no-index", "--quiet", name], cwd=ROOT)
                self.assertEqual(result.returncode, 0)

    def run_isolated_app(self, *, omit=(), missing=None, opening_outcome=None):
        try:
            import streamlit  # noqa: F401
        except ImportError:
            self.skipTest("Streamlit is installed in the project .venv")
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            copy_checkout(checkout, omit=omit)
            # A separate interpreter prevents cached imports or cached model
            # resources from accidentally satisfying an incomplete checkout.
            script = """
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import mlbb_predictor
assert Path(mlbb_predictor.__file__).resolve().parent == Path.cwd() / 'mlbb_predictor'
with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected startup download')), patch('mlbb_predictor.live_data.train_player_elo', side_effect=AssertionError('Unexpected startup training')):
    app = AppTest.from_file('app.py').run(timeout=30)
assert not app.exception, str(app.exception)
"""
            if opening_outcome is not None:
                script = """
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from mlbb_predictor.live_data import load_prediction_state
class FakeService:
    running = pending = False
    last_error = None
    calls = 0
    def refresh_on_open(self):
        self.calls += 1
        return OUTCOME
    def stop(self):
        pass
service = FakeService()
def load_after_check(*args, **kwargs):
    assert service.calls >= 1, 'Prediction was read before the opening check'
    return load_prediction_state(*args, **kwargs)
with patch('mlbb_predictor.collection_service.CollectionService', return_value=service), patch('mlbb_predictor.live_data.load_prediction_state', side_effect=load_after_check), patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected real network')):
    app = AppTest.from_file('app.py').run(timeout=30)
    assert not app.exception, str(app.exception)
    assert service.calls == 1
    assert app.session_state['opening_data_checked']
    if OUTCOME['state'] == 'timeout':
        assert any('still running' in item.value for item in app.info)
    if OUTCOME['state'] == 'error':
        assert any('opening data check could not finish' in item.value for item in app.warning)
    app.selectbox(key='predict_team_b').set_value('RORA').run()
    assert not app.exception, str(app.exception)
    assert service.calls == 1, 'Widget rerun requested another opening check'
    another = AppTest.from_file('app.py').run(timeout=30)
    assert not another.exception, str(another.exception)
    assert service.calls == 2, 'New browser session did not request its data check'
""".replace("OUTCOME", repr(opening_outcome))
            if missing:
                script += f"""
assert any('missing required' in item.value for item in app.error)
assert any({missing!r} in item.value for item in app.code)
assert not app.tabs
"""
            else:
                script += """
assert not app.error, str(app.error)
assert len(app.tabs) == 5
assert any('Match forecast' in item.value for item in app.markdown)
assert not (Path.cwd() / 'data/raw').exists()
"""
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=checkout,
                env={**os.environ, "MLBB_OFFLINE_TEST_MODE": "0" if opening_outcome is not None else "1"},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_clean_checkout_starts_without_downloading_or_retraining(self):
        self.run_isolated_app()

    def test_missing_model_lists_the_exact_path(self):
        name = "models/player_elo_model.json"
        self.run_isolated_app(omit=(name,), missing=name)

    def test_missing_stylesheet_is_reported_before_loading_it(self):
        name = "ui/matchdesk.css"
        self.run_isolated_app(omit=(name,), missing=name)

    def test_missing_tiers_does_not_silently_use_neutral_defaults(self):
        name = "config/meta_tiers.json"
        self.run_isolated_app(omit=(name,), missing=name)

    def test_opening_check_precedes_prediction_and_runs_once_per_session(self):
        self.run_isolated_app(opening_outcome={"state": "success"})

    def test_opening_timeout_still_renders_saved_predictor(self):
        self.run_isolated_app(opening_outcome={"state": "timeout"})

    def test_opening_source_failure_still_renders_saved_predictor(self):
        self.run_isolated_app(opening_outcome={"state": "error", "error": "offline"})


if __name__ == "__main__":
    unittest.main()
