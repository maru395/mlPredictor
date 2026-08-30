"""Load the curated MPL Philippines team profiles used by the app."""

from __future__ import annotations

import json
from pathlib import Path


def load_team_profiles(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    teams = payload.get("teams", [])
    codes = [team["code"] for team in teams]
    if len(teams) != 8 or len(set(codes)) != 8:
        raise ValueError("The MPL PH profile file must contain exactly eight unique teams.")
    return payload


def profiles_by_code(payload: dict) -> dict[str, dict]:
    return {team["code"]: team for team in payload["teams"]}


def starting_roster(profile: dict) -> list[dict]:
    starters = [
        member
        for member in profile["roster"]
        if str(member.get("status", "")).casefold() == "main"
    ]
    if len(starters) != 5:
        raise ValueError(f"{profile['name']} must have exactly five Main players.")
    return starters


def rating_lineup(profile: dict) -> list[str]:
    return [member.get("history_alias", member["player"]) for member in starting_roster(profile)]


def load_team_hero_picks(path: Path, expected_codes: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    teams = payload.get("teams", {})
    team_games = payload.get("team_games", {})
    if set(teams) != expected_codes or set(team_games) != expected_codes:
        raise ValueError("Last-season hero picks must cover every current MPL PH team.")
    for code, rows in teams.items():
        if not rows or any(not row.get("hero") or int(row.get("picks", 0)) <= 0 for row in rows):
            raise ValueError(f"Invalid last-season hero picks for {code}.")
        if int(team_games[code]) <= 0:
            raise ValueError(f"Invalid last-season game count for {code}.")
    return payload
