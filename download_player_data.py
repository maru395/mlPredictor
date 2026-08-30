"""Download MPL PH Season 17 game lineups for the player-Elo model."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data" / "processed" / "mpl_ph_s17_player_games.json"
BASE_URL = "https://strivemlbb.com"
TOURNAMENT_URL = (
    BASE_URL
    + "/tournament/mobile-legends-professional-league-philippines/s17"
)
FIXTURE_URLS = (
    TOURNAMENT_URL + "/fixtures/regular-season",
    TOURNAMENT_URL + "/fixtures/playoffs",
)
TEAM_CODES = {
    "AP.Bren": "APBR",
    "Aurora Gaming PH": "RORA",
    "Omega Esports": "OMG",
    "ONIC PH": "FNOP",
    "Team Falcons PH": "FLCN",
    "Team Liquid PH": "TLPH",
    "TNC Pro Team": "TNC",
    "Twisted Minds PH": "TWIS",
}
TAG_RE = re.compile(r"<[^>]+>")


def fetch(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mlbb-player-elo/1.0 (+local research app)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def clean_text(value: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub("", value)).split())


def match_urls() -> list[str]:
    ordered: dict[str, str] = {}
    for fixture_url in FIXTURE_URLS:
        anchors = re.findall(r"<a\b[^>]*>", fetch(fixture_url), flags=re.IGNORECASE)
        for anchor in anchors:
            class_match = re.search(r'class="([^"]*)"', anchor, flags=re.IGNORECASE)
            href_match = re.search(r'href="([^"]*)"', anchor, flags=re.IGNORECASE)
            if (
                not class_match
                or "match-card" not in class_match.group(1).split()
                or not href_match
                or "match_" not in href_match.group(1)
            ):
                continue
            href = href_match.group(1)
            absolute = urllib.parse.urljoin(BASE_URL, html.unescape(href))
            match_id = re.search(r"match_[^/?#]+", absolute)
            if match_id:
                ordered.setdefault(
                    match_id.group(0), TOURNAMENT_URL + "/match/" + match_id.group(0)
                )
    if len(ordered) != 64:
        raise ValueError(f"Expected 64 Season 17 series, found {len(ordered)}.")
    return list(ordered.values())


def parse_series(page: str, url: str) -> list[dict]:
    date_match = re.search(r'data-match-datetime="([^"]+)"', page)
    team_names = [
        clean_text(value)
        for value in re.findall(
            r'<a[^>]+class="[^"]*\bteam-name\b[^"]*"[^>]*>(.*?)</a>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )[:2]
    ]
    if not date_match or len(team_names) != 2:
        raise ValueError(f"Could not read date or teams from {url}.")
    try:
        team_a, team_b = (TEAM_CODES[name] for name in team_names)
    except KeyError as error:
        raise ValueError(f"Unknown MPL PH team {error.args[0]!r} in {url}.") from error

    card_pattern = re.compile(
        r'<a[^>]+class="game-summary-card"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    rows: list[dict] = []
    for game_number, card in enumerate(card_pattern.findall(page), start=1):
        first_side = re.search(
            r'class="gs-team-mini\s+([^\"]+)"', card, flags=re.IGNORECASE
        )
        player_names = [
            clean_text(value)
            for value in re.findall(
                r'<span[^>]+class="gs-players__name"[^>]*>(.*?)</span>',
                card,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        players_a = player_names[0::2]
        players_b = player_names[1::2]
        hero_names = [
            clean_text(value)
            for value in re.findall(
                r'<span[^>]+class="gs-players__hero-name"[^>]*>(.*?)</span>',
                card, flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        if (
            not first_side
            or len(players_a) != 5
            or len(players_b) != 5
            or len(set(players_a)) != 5
            or len(set(players_b)) != 5
            or not ({"is-winner", "is-loser"} & set(first_side.group(1).split()))
        ):
            raise ValueError(f"Invalid Game {game_number} lineup in {url}.")
        winner = team_a if "is-winner" in first_side.group(1) else team_b
        row = {
                "date": date_match.group(1)[:10],
                "team_a": team_a,
                "team_b": team_b,
                "players_a": players_a,
                "players_b": players_b,
                "heroes_a": hero_names[0::2],
                "heroes_b": hero_names[1::2],
                "season": "MPL Philippines Season 17",
                "winner": winner,
                "series_url": url,
                "game": game_number,
            }
        row["picks_verified"] = len(hero_names) == 10 and len(set(hero_names)) == 10
        if not row["picks_verified"]:
            row["reported_heroes_a"] = row.pop("heroes_a")
            row["reported_heroes_b"] = row.pop("heroes_b")
            row["pick_data_issue"] = "Published summary has missing or duplicate heroes; excluded from pick counts."
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed games found in {url}.")
    return rows


def main() -> None:
    urls = match_urls()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, series_rows in enumerate(pool.map(lambda url: parse_series(fetch(url), url), urls), start=1):
            rows.extend(series_rows)
            print(f"[{index:02d}/{len(urls)}] {len(series_rows)} games", flush=True)

    if len(rows) != 169:
        raise ValueError(f"Expected 169 Season 17 games, found {len(rows)}.")
    payload = {
        "season": "MPL Philippines Season 17",
        "scope": "Regular season and playoffs",
        "series": len(urls),
        "games": len(rows),
        "games_with_valid_picks": sum(row["picks_verified"] for row in rows),
        "source": TOURNAMENT_URL,
        "method": (
            "Extracted from every completed game breakdown; each row contains "
            "both five-player lineups and the game winner. Played heroes are retained "
            "when the published draft has ten unique heroes; inconsistent drafts "
            "are flagged and excluded from pick counts."
        ),
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} games to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
