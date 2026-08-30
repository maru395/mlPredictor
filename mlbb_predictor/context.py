"""Build player, hero-pick, and team-comfort summaries from the raw data."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .data import canonical_team, find_kaggle_boxmatch


HF_ROLE_FIELDS = ("explaner", "jungler", "midlaner", "goldlaner", "roamer")


def _hero_name(raw_name: str) -> str:
    return " ".join(raw_name.strip().split())


def _new_stat() -> dict[str, int]:
    return {"picks": 0, "wins": 0}


def build_context_data(raw_dir: Path) -> dict:
    """Create JSON-ready player and hero-pick summaries.

    Player names exist only in the Kaggle MPL PH S13 file. Team-level hero
    comfort uses Kaggle S13 plus both Hugging Face S14 leagues.
    """
    player_totals: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0}
    )
    player_heroes: defaultdict[tuple[str, str], defaultdict[str, dict[str, int]]] = (
        defaultdict(lambda: defaultdict(_new_stat))
    )
    team_heroes: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_new_stat)
    )
    team_games: defaultdict[str, int] = defaultdict(int)
    global_player_totals: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0}
    )
    global_player_heroes: defaultdict[str, defaultdict[str, dict[str, int]]] = (
        defaultdict(lambda: defaultdict(_new_stat))
    )
    global_player_teams: defaultdict[str, set[str]] = defaultdict(set)
    hero_pool: set[str] = set()

    kaggle_path = find_kaggle_boxmatch(raw_dir)
    with kaggle_path.open("r", encoding="utf-8-sig", newline="") as handle:
        kaggle_rows = list(csv.DictReader(handle))

    for offset in range(0, len(kaggle_rows), 10):
        chunk = kaggle_rows[offset : offset + 10]
        for team in {canonical_team(row["Team"]) for row in chunk}:
            team_games[team] += 1
        for row in chunk:
            team = canonical_team(row["Team"])
            player = " ".join(row["Players"].strip().split())
            hero = _hero_name(row["Pick"])
            won = int(row["Win"].strip().lower() == "win")
            if not player or not hero:
                continue
            hero_pool.add(hero)
            player_totals[(team, player)]["games"] += 1
            player_totals[(team, player)]["wins"] += won
            player_heroes[(team, player)][hero]["picks"] += 1
            player_heroes[(team, player)][hero]["wins"] += won
            global_player_totals[player]["games"] += 1
            global_player_totals[player]["wins"] += won
            global_player_heroes[player][hero]["picks"] += 1
            global_player_heroes[player][hero]["wins"] += won
            global_player_teams[player].add(team)
            team_heroes[team][hero]["picks"] += 1
            team_heroes[team][hero]["wins"] += won

    hf_sources = (
        raw_dir / "hf_mpl_id_s14.csv",
        raw_dir / "hf_mpl_ph_s14.csv",
    )
    for path in hf_sources:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            blue_team = canonical_team(row["blue_team"])
            red_team = canonical_team(row["red_team"])
            blue_won = int(row["result"].strip().upper() == "BLUE")
            for team in (blue_team, red_team):
                team_games[team] += 1
            for side, team, won in (
                ("blue", blue_team, blue_won),
                ("red", red_team, 1 - blue_won),
            ):
                for role in HF_ROLE_FIELDS:
                    hero = _hero_name(row[f"{side}_{role}"])
                    if not hero:
                        continue
                    hero_pool.add(hero)
                    team_heroes[team][hero]["picks"] += 1
                    team_heroes[team][hero]["wins"] += won

    players_by_team: defaultdict[str, list[dict]] = defaultdict(list)
    for (team, player), totals in player_totals.items():
        top_picks = []
        for hero, values in sorted(
            player_heroes[(team, player)].items(),
            key=lambda item: (-item[1]["picks"], item[0].lower()),
        ):
            top_picks.append(
                {
                    "hero": hero,
                    "picks": values["picks"],
                    "wins": values["wins"],
                    "win_rate": round(values["wins"] / values["picks"], 4),
                }
            )
        players_by_team[team].append(
            {
                "player": player,
                "games": totals["games"],
                "wins": totals["wins"],
                "losses": totals["games"] - totals["wins"],
                "win_rate": round(totals["wins"] / totals["games"], 4),
                "top_picks": top_picks,
            }
        )

    for players in players_by_team.values():
        players.sort(key=lambda item: (-item["games"], item["player"].lower()))

    player_history: dict[str, dict] = {}
    for player, totals in global_player_totals.items():
        picks = []
        for hero, values in sorted(
            global_player_heroes[player].items(),
            key=lambda item: (-item[1]["picks"], item[0].lower()),
        ):
            picks.append(
                {
                    "hero": hero,
                    "picks": values["picks"],
                    "wins": values["wins"],
                    "win_rate": round(values["wins"] / values["picks"], 4),
                }
            )
        player_history[player] = {
            "player": player,
            "games": totals["games"],
            "wins": totals["wins"],
            "losses": totals["games"] - totals["wins"],
            "win_rate": round(totals["wins"] / totals["games"], 4),
            "historical_teams": sorted(global_player_teams[player]),
            "top_picks": picks,
        }

    team_hero_payload: dict[str, list[dict]] = {}
    for team, heroes in team_heroes.items():
        hero_rows = []
        for hero, values in sorted(
            heroes.items(), key=lambda item: (-item[1]["picks"], item[0].lower())
        ):
            hero_rows.append(
                {
                    "hero": hero,
                    "picks": values["picks"],
                    "wins": values["wins"],
                    "win_rate": round(values["wins"] / values["picks"], 4),
                    "team_pick_rate": round(values["picks"] / team_games[team], 4),
                }
            )
        team_hero_payload[team] = hero_rows

    return {
        "hero_pool": sorted(hero_pool, key=str.lower),
        "players": dict(players_by_team),
        "player_history": player_history,
        "team_heroes": team_hero_payload,
        "team_games": dict(team_games),
        "player_source": "Kaggle MPL Philippines S13",
        "team_hero_sources": "Kaggle MPL PH S13 and Hugging Face MPL ID/PH S14",
    }


def write_context_data(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def read_context_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
