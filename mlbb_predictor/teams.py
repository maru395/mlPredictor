"""Load the curated MPL Philippines team profiles used by the app."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
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
    if "prediction_roster" in profile:
        return profile["prediction_roster"]
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


def current_team_profiles(root: Path, rows: list[dict], *, offline=False) -> dict:
    """Project profiles, prediction lineups and standings from the loaded history."""
    from .history import merge_game_history, series_key, valid_lineups
    from .player_model import player_key

    payload = load_team_profiles(root / "config/mpl_ph_teams.json")
    saved = root / "data/processed/team_profiles.json"
    if not offline and saved.exists():
        payload = load_team_profiles(saved)
    payload = deepcopy(payload)
    season_rows = [row for row in merge_game_history(rows)
                   if row.get("season") == f"MPL Philippines {payload['season']}"]
    for team in payload["teams"]:
        # Official pages list registered members, not starters. A complete
        # collected lineup is evidence for prediction, not a permanent transfer.
        if not offline:
            members = {player_key(member.get("history_alias", member["player"])): member for member in team["roster"]}
            latest = next((row for row in reversed(season_rows)
                           if team["code"] in (row["team_a"], row["team_b"]) and valid_lineups(row)), None)
            selected = None
            if latest:
                side = "a" if latest["team_a"] == team["code"] else "b"
                keys = list(map(player_key, latest[f"players_{side}"]))
                if all(key in members for key in keys):
                    selected = [deepcopy(members[key]) for key in keys]
                    team["lineup_basis"] = f"Latest verified collected lineup · {latest['date']}"
            if selected is None:
                by_role = defaultdict(list)
                for member in team["roster"]:
                    by_role[member["role"]].append(member)
                if len(by_role) == 5 and all(len(group) == 1 for group in by_role.values()):
                    selected = deepcopy(team["roster"])
                    team["lineup_basis"] = "Official roster · one registered player per role"
                else:
                    selected = deepcopy(starting_roster(team))
                    team["lineup_basis"] = "Saved prediction lineup · official starters are not identified"
                    team["lineup_warning"] = "No complete recent lineup could be matched to the official roster; the saved prediction five is retained."
            order = {role: i for i, role in enumerate(("EXP Lane", "Jungle", "Mid Lane", "Gold Lane", "Roam"))}
            team["prediction_roster"] = sorted(selected, key=lambda member: order.get(member["role"], 99))
            selected_keys = {player_key(member.get("history_alias", member["player"])) for member in selected}
            for member in team["roster"]:
                member["status"] = "Main" if player_key(member.get("history_alias", member["player"])) in selected_keys else "Registered reserve"

    # The same rows shown in Collected matches power records; incomplete drafts
    # still have usable results. Exclude playoffs and incomplete BO3 series.
    groups = defaultdict(list)
    for row in season_rows:
        if row.get("stage", "Regular Season").casefold() == "regular season":
            groups[series_key(row)].append(row)
    standings = {team["code"]: dict(matches_won=0, matches_lost=0, games_won=0, games_lost=0, game_diff=0)
                 for team in payload["teams"]}
    dates = []
    for key, games in groups.items():
        a, b = key[1:]
        if a not in standings or b not in standings:
            continue
        wins = sum(row["winner"] == a for row in games)
        losses = sum(row["winner"] == b for row in games)
        if (wins + losses != len(games) or max(wins, losses) != 2 or min(wins, losses) >= 2 or
                sorted(int(row["game"]) for row in games) != list(range(1, len(games) + 1))):
            continue
        dates.append(key[0])
        for code, won, lost in ((a, wins, losses), (b, losses, wins)):
            record = standings[code]
            record["matches_won"] += int(won > lost)
            record["matches_lost"] += int(won < lost)
            record["games_won"] += won
            record["games_lost"] += lost
            record["game_diff"] += won - lost
    # One match point per series win, as used by this season's standings.
    for record in standings.values():
        record["match_points"] = record["matches_won"]
    def score(code):
        record = standings[code]
        return record["match_points"], record["game_diff"], record["games_won"]
    for team in payload["teams"]:
        code = team["code"]
        standings[code]["position"] = 1 + sum(score(other) > score(code) for other in standings)
        team["standing"] = standings[code]
    payload["standings_as_of"] = max(dates, default="No collected results")
    payload["standings_note"] = "Collected regular-season results; ordered by series wins, game difference, then game wins. Equal records share rank; official tiebreakers are not applied."
    return payload
