"""Read a consistent collected-data/model snapshot, or the original seed files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .history import load_game_history, merge_game_history, series_key, valid_lineups
from .player_model import PlayerEloPredictor, train_player_elo


def snapshot_path(root: Path) -> Path:
    return root / "data/processed/live_snapshot.json"


def data_revision(root: Path, offline: bool = False) -> tuple:
    names = ["models/player_elo_model.json", "data/processed/mpl_ph_s17_player_games.json",
             "data/processed/mpl_ph_s18_player_games.json", "config/mpl_ph_teams.json",
             "config/meta_tiers.json", "data/processed/context.json"]
    if not offline:
        names.append("data/processed/live_snapshot.json")
        names.append("data/processed/team_profiles.json")
        names.append("data/processed/match_schedule.json")
    return tuple((root / name).stat().st_mtime_ns if (root / name).exists() else 0 for name in names)


def history_fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def train_collected_model(old_rows: list[dict], season_rows: list[dict]) -> tuple[PlayerEloPredictor, dict]:
    combined = merge_game_history(old_rows, season_rows)
    model, metrics = train_player_elo(old_rows)
    new_games = [row for row in combined if row.get("season") == "MPL Philippines Season 18" and valid_lineups(row)]
    model.fit(new_games)
    metrics.update(
        match_count=len(old_rows) + len(new_games), player_count=len(model.ratings),
        imported_game_count=len(combined), excluded_lineup_games=sum(not valid_lineups(row) for row in combined),
        update_game_count=len(new_games), rating_cutoff=max(row["date"] for row in combined),
        season="MPL Philippines Seasons 17–18", validation_season="MPL Philippines Season 17",
        history_fingerprint=history_fingerprint(combined),
    )
    return model, metrics


def load_prediction_state(root: Path, offline: bool = False) -> tuple:
    paths = [root / f"data/processed/mpl_ph_s{s}_player_games.json" for s in (17, 18)]
    if not offline and snapshot_path(root).exists():
        snapshot = json.loads(snapshot_path(root).read_text(encoding="utf-8"))
        rows = snapshot["history_rows"]
        model, metrics = PlayerEloPredictor.from_dict(snapshot["model"])
        if metrics.get("history_fingerprint") != history_fingerprint(rows):
            raise ValueError("Collected data and model do not belong to the same snapshot")
        return model, metrics, snapshot["history_metadata"], rows
    metadata, rows = load_game_history(paths)
    model, metrics = PlayerEloPredictor.load(root / "models/player_elo_model.json")
    return model, metrics, metadata, rows
