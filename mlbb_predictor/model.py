"""A small, transparent Elo probability model for MLBB team matchups."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .data import DISPLAY_NAMES, display_name


START_RATING = 1500.0
ELO_SCALE = 400.0
K_CANDIDATES = (8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 64.0)


def elo_probability(rating_a: float, rating_b: float, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def _clip_probability(value: float) -> float:
    return min(max(value, 1e-12), 1.0 - 1e-12)


@dataclass
class EloPredictor:
    k_factor: float
    scale: float = ELO_SCALE
    start_rating: float = START_RATING
    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    recent: dict[str, deque[int]] = field(default_factory=dict)
    last_match_date: dict[str, str] = field(default_factory=dict)

    def _ensure_team(self, team: str) -> None:
        self.ratings.setdefault(team, self.start_rating)
        self.games.setdefault(team, 0)
        self.wins.setdefault(team, 0)
        self.recent.setdefault(team, deque(maxlen=5))

    def predict(self, team_a: str, team_b: str) -> float:
        self._ensure_team(team_a)
        self._ensure_team(team_b)
        return elo_probability(self.ratings[team_a], self.ratings[team_b], self.scale)

    def update(self, team_a: str, team_b: str, winner: str, date: str = "") -> float:
        if winner not in {team_a, team_b}:
            raise ValueError(f"Winner {winner!r} is not in matchup {team_a!r} vs {team_b!r}")
        probability_a = self.predict(team_a, team_b)
        actual_a = 1.0 if winner == team_a else 0.0
        change = self.k_factor * (actual_a - probability_a)
        self.ratings[team_a] += change
        self.ratings[team_b] -= change
        self.games[team_a] += 1
        self.games[team_b] += 1
        self.wins[team_a] += int(actual_a)
        self.wins[team_b] += int(not actual_a)
        self.recent[team_a].append(int(actual_a))
        self.recent[team_b].append(int(not actual_a))
        if date:
            self.last_match_date[team_a] = date
            self.last_match_date[team_b] = date
        return probability_a

    def fit(self, matches: Iterable[dict[str, str | int]]) -> "EloPredictor":
        for match in matches:
            self.update(
                str(match["team_a"]),
                str(match["team_b"]),
                str(match["winner"]),
                str(match["date"]),
            )
        return self

    def team_summary(self, team: str) -> dict[str, float | int | str]:
        self._ensure_team(team)
        recent_values = list(self.recent[team])
        return {
            "code": team,
            "name": display_name(team),
            "rating": round(self.ratings[team], 1),
            "games": self.games[team],
            "wins": self.wins[team],
            "losses": self.games[team] - self.wins[team],
            "win_rate": self.wins[team] / self.games[team] if self.games[team] else 0.5,
            "recent_wins": sum(recent_values),
            "recent_games": len(recent_values),
            "last_match_date": self.last_match_date.get(team, "N/A"),
        }

    def to_dict(self, metrics: dict[str, float | int] | None = None) -> dict:
        return {
            "model_type": "Elo",
            "k_factor": self.k_factor,
            "scale": self.scale,
            "start_rating": self.start_rating,
            "ratings": self.ratings,
            "games": self.games,
            "wins": self.wins,
            "recent": {team: list(values) for team, values in self.recent.items()},
            "last_match_date": self.last_match_date,
            "display_names": {team: DISPLAY_NAMES.get(team, team) for team in self.ratings},
            "metrics": metrics or {},
        }

    def save(self, path: Path, metrics: dict[str, float | int] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(metrics), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> tuple["EloPredictor", dict[str, float | int]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        return model, dict(payload.get("metrics", {}))


def _evaluate_k(
    training: list[dict[str, str | int]],
    validation: list[dict[str, str | int]],
    k_factor: float,
) -> dict[str, float]:
    model = EloPredictor(k_factor=k_factor).fit(training)
    losses: list[float] = []
    brier_scores: list[float] = []
    correct = 0

    for match in validation:
        team_a = str(match["team_a"])
        team_b = str(match["team_b"])
        actual = 1.0 if str(match["winner"]) == team_a else 0.0
        probability = _clip_probability(model.predict(team_a, team_b))
        losses.append(-(actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability)))
        brier_scores.append((probability - actual) ** 2)
        correct += int((probability >= 0.5) == bool(actual))
        model.update(team_a, team_b, str(match["winner"]), str(match["date"]))

    count = len(validation)
    return {
        "log_loss": sum(losses) / count,
        "brier_score": sum(brier_scores) / count,
        "accuracy": correct / count,
    }


def train_and_evaluate(
    matches: list[dict[str, str | int]], validation_fraction: float = 0.25
) -> tuple[EloPredictor, dict[str, float | int]]:
    if len(matches) < 20:
        raise ValueError("At least 20 matches are required to train and validate the model.")
    split_index = max(1, int(len(matches) * (1.0 - validation_fraction)))
    training = matches[:split_index]
    validation = matches[split_index:]
    if not validation:
        raise ValueError("Validation split is empty.")

    results = {k: _evaluate_k(training, validation, k) for k in K_CANDIDATES}
    best_k = min(results, key=lambda key: results[key]["log_loss"])
    best = results[best_k]
    model = EloPredictor(k_factor=best_k).fit(matches)

    sources: defaultdict[str, int] = defaultdict(int)
    for match in matches:
        sources[str(match["source"])] += 1

    metrics: dict[str, float | int] = {
        "match_count": len(matches),
        "team_count": len(model.ratings),
        "training_count": len(training),
        "validation_count": len(validation),
        "validation_accuracy": round(best["accuracy"], 4),
        "validation_log_loss": round(best["log_loss"], 4),
        "validation_brier_score": round(best["brier_score"], 4),
        "baseline_log_loss": round(math.log(2.0), 4),
        "selected_k_factor": best_k,
        "kaggle_match_count": sources["Kaggle"],
        "huggingface_match_count": sources["Hugging Face"],
    }
    return model, metrics

