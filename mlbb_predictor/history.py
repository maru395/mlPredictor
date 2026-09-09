"""Chronological game history and rolling, series-based hero pick counts."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
import json
from pathlib import Path

from .player_model import player_key


def series_key(row: dict) -> tuple[str, str, str]:
    # MPL PH has at most one series for a given pairing on a calendar day.
    # A semantic key also deduplicates alternative URLs for the same series.
    return (row["date"], *sorted((row["team_a"], row["team_b"])))


def merge_game_history(*datasets: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    order: dict[tuple, int] = {}
    for dataset in datasets:
        for row in dataset:
            series = series_key(row)
            order.setdefault(series, len(order))
            key = (*series, int(row["game"]))
            if key in unique:
                previous = unique[key]
                fields = ("team_a", "team_b", "winner", "players_a", "players_b", "heroes_a", "heroes_b")
                if any(previous.get(field) != row.get(field) for field in fields):
                    raise ValueError(f"Conflicting duplicate game: {key}")
                continue
            unique[key] = row
    rows = sorted(unique.values(), key=lambda row: (
        row["date"], row.get("series_started_at", row["date"]),
        order[series_key(row)], int(row["game"]),
    ))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[series_key(row)].append(row)
    for key, games in groups.items():
        if [row["game"] for row in games] != list(range(1, len(games) + 1)):
            raise ValueError(f"Incomplete game sequence: {key}")
    return rows


def load_game_history(paths: list[Path]) -> tuple[dict, list[dict]]:
    sources, datasets = [], []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        if len(rows) != payload["games"]:
            raise ValueError(f"Game metadata does not match its rows: {path}")
        for row in rows:
            if row["team_a"] == row["team_b"] or row["winner"] not in {row["team_a"], row["team_b"]}:
                raise ValueError(f"Invalid game result: {path}")
            if not valid_lineups(row) and row.get("lineups_verified") is not False:
                raise ValueError(f"Incomplete lineup without an explicit missing-data flag: {path}")
        sources.append({key: value for key, value in payload.items() if key != "rows"})
        datasets.append(rows)
    rows = merge_game_history(*datasets)
    return {
        "sources": sources, "games": len(rows),
        "series": len({series_key(row) for row in rows}),
        "cutoff": max(row["date"] for row in rows),
    }, rows


def valid_lineups(row: dict) -> bool:
    if row.get("lineups_verified") is False:
        return False
    names_a = [player_key(name) for name in row.get("players_a", [])]
    names_b = [player_key(name) for name in row.get("players_b", [])]
    return len(names_a) == len(names_b) == 5 and all(names_a + names_b) and len(set(names_a + names_b)) == 10


def valid_picks(row: dict) -> bool:
    if row.get("picks_verified") is False:
        return False
    heroes_a, heroes_b = row.get("heroes_a", []), row.get("heroes_b", [])
    heroes = heroes_a + heroes_b
    return len(heroes_a) == len(heroes_b) == 5 and all(heroes) and len(set(heroes)) == 10


def recent_team_picks(rows: list[dict], teams: set[str], limit: int | None = 10) -> dict:
    """Count played heroes in the last N series, not the last N individual games.

    The series window includes games with bad pick data. They are excluded from
    the pick-rate denominator and reported as gaps, never replaced by older data.
    Bans (verified or user-reported) are never counted as picks.
    """
    if limit is not None and limit < 1:
        raise ValueError("The recent-series window must be positive")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in merge_game_history(rows):
        groups[series_key(row)].append(row)
    result = {"window": limit, "unit": "series", "teams": {}, "coverage": {}}
    for team in sorted(teams):
        selected = [games for key, games in groups.items() if team in key[1:]]
        if limit is not None:
            selected = selected[-limit:]
        counts: Counter = Counter()
        valid_games, missing, series_rows = 0, [], []
        for games in selected:
            first = games[0]
            opponent = first["team_b"] if first["team_a"] == team else first["team_a"]
            wins = sum(row["winner"] == team for row in games)
            series_rows.append({
                "date": first["date"], "opponent": opponent,
                "score": f"{wins}–{len(games) - wins}", "games": len(games),
                "season": first.get("season", "Unknown"), "source": first["series_url"],
            })
            for row in games:
                if not valid_picks(row):
                    missing.append({"date": row["date"], "game": row["game"], "source": row["series_url"]})
                    continue
                side = "a" if row["team_a"] == team else "b"
                counts.update(row[f"heroes_{side}"])
                valid_games += 1
        result["teams"][team] = [
            {"hero": hero, "picks": count}
            for hero, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
        ]
        result["coverage"][team] = {
            "series": len(selected), "games": sum(len(games) for games in selected),
            "games_with_picks": valid_games, "missing_pick_games": missing,
            "from": series_rows[0]["date"] if series_rows else None,
            "through": series_rows[-1]["date"] if series_rows else None,
            "series_rows": list(reversed(series_rows)),
        }
    return result


def recent_player_picks(rows: list[dict], players: list[str], limit: int | None = 10) -> dict:
    """Recent hero pools follow the player, including appearances for other teams.

    Select each player's last N recorded series before filtering invalid drafts.
    Never map heroes onto an incomplete lineup or infer absent player identities.
    """
    if limit is not None and limit < 1:
        raise ValueError("The recent-series window must be positive")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in merge_game_history(rows):
        groups[series_key(row)].append(row)
    result = {}
    for player in players:
        key = player_key(player)
        selected = [games for games in groups.values() if any(
            key in {player_key(name) for name in row.get("players_a", []) + row.get("players_b", [])}
            for row in games
        )]
        if limit is not None:
            selected = selected[-limit:]
        counts: Counter = Counter()
        appearances = wins = losses = 0
        for games in selected:
            for row in games:
                for side in ("a", "b"):
                    names = [player_key(name) for name in row.get(f"players_{side}", [])]
                    if key not in names:
                        continue
                    appearances += 1
                    if valid_lineups(row):
                        wins += int(row["winner"] == row[f"team_{side}"])
                        losses += int(row["winner"] != row[f"team_{side}"])
                    if valid_lineups(row) and valid_picks(row):
                        counts[row[f"heroes_{side}"][names.index(key)]] += 1
        result[key] = {
            "player": player, "series": len(selected), "games": appearances, "wins": wins, "losses": losses,
            "games_with_picks": sum(counts.values()),
            "missing_pick_games": appearances - sum(counts.values()),
            "from": selected[0][0]["date"] if selected else None,
            "through": selected[-1][-1]["date"] if selected else None,
            "heroes": [{"hero": hero, "picks": count} for hero, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0].casefold())
            )],
        }
    return result


CURRENT_SEASON = "MPL Philippines Season 18"


def season_games(rows: list[dict], season: str = CURRENT_SEASON) -> list[dict]:
    return merge_game_history([row for row in rows if row.get("season") == season])


def season_player_picks(rows: list[dict], players: list[str], season: str = CURRENT_SEASON) -> dict:
    """Cumulative, deduplicated player/hero appearances in one season only."""
    result = recent_player_picks(season_games(rows, season), players, limit=None)
    for stats in result.values():
        stats["season"] = season
    return result


def season_player_pick_records(rows: list[dict], player: str, season: str = CURRENT_SEASON) -> list[dict]:
    """Auditable played-hero appearances, with the same eligibility as meta pools."""
    records = []
    key = player_key(player)
    for row in season_games(rows, season):
        if not valid_lineups(row) or not valid_picks(row):
            continue
        for side, other in (("a", "b"), ("b", "a")):
            names = list(map(player_key, row[f"players_{side}"]))
            if key in names:
                records.append({"Date": row["date"], "Team": row[f"team_{side}"],
                                "Opponent": row[f"team_{other}"], "Game": row["game"],
                                "Hero": row[f"heroes_{side}"][names.index(key)], "Season": season,
                                "Source": row["series_url"]})
    return records


def season_team_picks(rows: list[dict], teams: set[str], season: str = CURRENT_SEASON) -> dict:
    return recent_team_picks(season_games(rows, season), teams, limit=None)


def matches_by_week(rows: list[dict]) -> list[dict]:
    """Prefer extracted event week numbers; infer missing seed weeks by calendar."""
    rows = merge_game_history(rows)
    if not rows:
        return []
    first = date.fromisoformat(min(row["date"] for row in rows))
    start = first - timedelta(days=first.weekday())
    anchored = next((row for row in rows if isinstance(row.get("week"), int) and row["week"] > 0
                     and row.get("stage", "Regular Season") == "Regular Season"), None)
    if anchored:
        anchor_date = date.fromisoformat(anchored["date"])
        start = anchor_date - timedelta(days=anchor_date.weekday() + 7 * (anchored["week"] - 1))
    groups = {}
    for row in rows:
        extracted = row.get("week")
        week = (int(extracted) if isinstance(extracted, int) and extracted > 0 else
                (date.fromisoformat(row["date"]) - start).days // 7 + 1)
        stage = row.get("stage", "Regular Season")
        label = f"Week {week}" if stage == "Regular Season" else f"{stage} · Week {week}"
        group = groups.setdefault(label, {"label": label, "rows": [], "from": row["date"], "through": row["date"]})
        group["rows"].append(row)
        group["through"] = max(group["through"], row["date"])
    return sorted(groups.values(), key=lambda group: group["from"], reverse=True)
