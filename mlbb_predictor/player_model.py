"""Player-level Elo ratings for five-player MLBB lineups."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .model import ELO_SCALE, START_RATING, _clip_probability, elo_probability


PLAYER_K_CANDIDATES = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)


def player_key(player: str) -> str:
    return " ".join(str(player).split()).casefold()


def load_player_games(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    if int(payload.get("games", -1)) != len(rows):
        raise ValueError("Player-game metadata does not match its rows.")
    for row in rows:
        teams = {str(row["team_a"]), str(row["team_b"])}
        if len(teams) != 2 or str(row["winner"]) not in teams:
            raise ValueError("A player-game winner is not part of the matchup.")
        for field_name in ("players_a", "players_b"):
            lineup = [player_key(name) for name in row[field_name]]
            if len(lineup) != 5 or len(set(lineup)) != 5 or not all(lineup):
                raise ValueError("Every player-game row must have two unique five-player lineups.")
        if {player_key(name) for name in row["players_a"]} & {player_key(name) for name in row["players_b"]}:
            raise ValueError("A player cannot appear on both sides of one game.")
    return payload, rows


@dataclass
class PlayerEloPredictor:
    k_factor: float
    scale: float = ELO_SCALE
    start_rating: float = START_RATING
    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    recent: dict[str, deque[int]] = field(default_factory=dict)
    last_match_date: dict[str, str] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)

    def _ensure_player(self, player: str) -> str:
        key = player_key(player)
        self.ratings.setdefault(key, self.start_rating)
        self.games.setdefault(key, 0)
        self.wins.setdefault(key, 0)
        self.recent.setdefault(key, deque(maxlen=5))
        self.display_names.setdefault(key, " ".join(str(player).split()))
        return key

    def lineup_rating(self, players: list[str]) -> float:
        keys = [self._ensure_player(player) for player in players]
        if len(keys) != 5 or len(set(keys)) != 5:
            raise ValueError("A lineup must contain five unique players.")
        return mean(self.ratings[key] for key in keys)

    def predict_lineups(self, players_a: list[str], players_b: list[str]) -> float:
        return elo_probability(
            self.lineup_rating(players_a), self.lineup_rating(players_b), self.scale
        )

    def update(self, row: dict) -> float:
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        winner = str(row["winner"])
        if winner not in {team_a, team_b}:
            raise ValueError("Winner is not part of the player-Elo matchup.")
        players_a = [str(player) for player in row["players_a"]]
        players_b = [str(player) for player in row["players_b"]]
        probability_a = self.predict_lineups(players_a, players_b)
        actual_a = 1.0 if winner == team_a else 0.0
        change = self.k_factor * (actual_a - probability_a)
        date = str(row.get("date", ""))

        for player, won, adjustment in (
            *((player, int(actual_a), change) for player in players_a),
            *((player, int(not actual_a), -change) for player in players_b),
        ):
            key = self._ensure_player(player)
            self.ratings[key] += adjustment
            self.games[key] += 1
            self.wins[key] += won
            self.recent[key].append(won)
            if date:
                self.last_match_date[key] = date
        return probability_a

    def fit(self, rows: list[dict]) -> "PlayerEloPredictor":
        for row in rows:
            self.update(row)
        return self

    def player_summary(self, player: str) -> dict[str, float | int | str | bool]:
        key = self._ensure_player(player)
        recent_values = list(self.recent[key])
        return {
            "key": key,
            "name": self.display_names.get(key, player),
            "rating": round(self.ratings[key], 1),
            "games": self.games[key],
            "wins": self.wins[key],
            "losses": self.games[key] - self.wins[key],
            "known": self.games[key] > 0,
            "recent_wins": sum(recent_values),
            "recent_games": len(recent_values),
            "last_match_date": self.last_match_date.get(key, "N/A"),
        }

    def lineup_summary(self, players: list[str]) -> dict:
        summaries = [self.player_summary(player) for player in players]
        return {
            "rating": round(mean(float(row["rating"]) for row in summaries), 1),
            "known_players": sum(bool(row["known"]) for row in summaries),
            "unknown_players": [players[i] for i, row in enumerate(summaries) if not row["known"]],
            "players": summaries,
        }

    def to_dict(self, metrics: dict | None = None) -> dict:
        return {
            "model_type": "Player lineup Elo",
            "k_factor": self.k_factor,
            "scale": self.scale,
            "start_rating": self.start_rating,
            "ratings": self.ratings,
            "games": self.games,
            "wins": self.wins,
            "recent": {key: list(values) for key, values in self.recent.items()},
            "last_match_date": self.last_match_date,
            "display_names": self.display_names,
            "metrics": metrics or {},
        }

    def save(self, path: Path, metrics: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(metrics), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> tuple["PlayerEloPredictor", dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> tuple["PlayerEloPredictor", dict]:
        """Load a model from an atomic data/model snapshot."""
        model = cls(
            k_factor=float(payload["k_factor"]),
            scale=float(payload["scale"]),
            start_rating=float(payload["start_rating"]),
        )
        model.ratings = {key: float(value) for key, value in payload["ratings"].items()}
        model.games = {key: int(value) for key, value in payload["games"].items()}
        model.wins = {key: int(value) for key, value in payload["wins"].items()}
        model.recent = {
            key: deque((int(item) for item in values), maxlen=5)
            for key, values in payload["recent"].items()
        }
        model.last_match_date = dict(payload.get("last_match_date", {}))
        model.display_names = dict(payload.get("display_names", {}))
        return model, dict(payload.get("metrics", {}))


def _evaluate_k(training: list[dict], validation: list[dict], k_factor: float) -> dict:
    model = PlayerEloPredictor(k_factor=k_factor).fit(training)
    losses: list[float] = []
    brier_scores: list[float] = []
    correct = 0
    for row in validation:
        actual = 1.0 if row["winner"] == row["team_a"] else 0.0
        probability = _clip_probability(
            model.predict_lineups(row["players_a"], row["players_b"])
        )
        losses.append(
            -(actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability))
        )
        brier_scores.append((probability - actual) ** 2)
        correct += int((probability >= 0.5) == bool(actual))
        model.update(row)
    count = len(validation)
    return {
        "log_loss": sum(losses) / count,
        "brier_score": sum(brier_scores) / count,
        "accuracy": correct / count,
    }


def train_player_elo(
    rows: list[dict], validation_fraction: float = 0.25
) -> tuple[PlayerEloPredictor, dict]:
    if len(rows) < 20:
        raise ValueError("At least 20 player-game rows are required.")
    split_index = max(1, int(len(rows) * (1.0 - validation_fraction)))
    training, validation = rows[:split_index], rows[split_index:]
    results = {k: _evaluate_k(training, validation, k) for k in PLAYER_K_CANDIDATES}
    best_k = min(results, key=lambda key: results[key]["log_loss"])
    best = results[best_k]
    model = PlayerEloPredictor(k_factor=best_k).fit(rows)
    metrics = {
        "match_count": len(rows),
        "player_count": len(model.ratings),
        "training_count": len(training),
        "validation_count": len(validation),
        "validation_accuracy": round(best["accuracy"], 4),
        "validation_log_loss": round(best["log_loss"], 4),
        "validation_brier_score": round(best["brier_score"], 4),
        "baseline_log_loss": round(math.log(2.0), 4),
        "selected_k_factor": best_k,
        "season": "MPL Philippines Season 17",
    }
    return model, metrics
