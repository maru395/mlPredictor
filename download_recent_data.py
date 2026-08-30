"""Verify the nine user-supplied S18 series against MLDB game and draft records.

Fail closed on missing results or mismatched bans; flag incomplete lineups/picks.
Only the submitted series are imported; this is not an automatic live feed.
"""

from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from download_player_data import clean_text

ROOT = Path(__file__).resolve().parent
MANUAL_PATH = ROOT / "data/manual/mpl_ph_s18_aug21_28.json"
OUTPUT_PATH = ROOT / "data/processed/mpl_ph_s18_player_games.json"
EVENT_URL = "https://mldb.gg/event/mpl-philippines-season-18"
FIXTURES_URL = "https://mldb.gg/ajax/event/mpl-philippines-season-18/section/matches"
SEASON = "MPL Philippines Season 18"
TEAM_SLUGS = {
    "team-liquid-ph": "TLPH", "team-falcons-ph": "FLCN",
    "aurora-gaming-ph": "RORA", "onic-ph": "FNOP", "onic-philippines": "FNOP",
    "apbren": "APBR", "ap-bren": "APBR", "ap-bren-ph": "APBR",
    "omega-esports": "OMG", "smart-omega": "OMG",
    "tnc-pro-team": "TNC", "twisted-minds-ph": "TWIS",
}
SHORT_CODES = {"ONIC": "FNOP", "ONPH": "FNOP", "FLCP": "FLCN", "TWPH": "TWIS"}
FLAGS = re.IGNORECASE | re.DOTALL


def fetch(url: str, ajax: bool = False) -> str:
    headers = {"User-Agent": "mlbb-player-elo/1.0 (+local research app)"}
    if ajax:
        headers.update({"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
        return response.read().decode("utf-8")


def local_date(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp).astimezone(timezone(timedelta(hours=8))).date().isoformat()


def fixture_index(page: str) -> dict[tuple, str]:
    index = {}
    for url, card in re.findall(r'<a class="matches-item" href="([^"]+)">(.*?)</a>', page, FLAGS):
        timestamp = re.search(r'data-utc="([^"]+)"', card)
        codes = re.findall(r'<span class="team-info-short-name">(.*?)</span>', card, FLAGS)
        if timestamp and len(codes) == 2:
            codes = [SHORT_CODES.get(clean_text(code), clean_text(code)) for code in codes]
            if "TBD" in codes:
                continue
            key = (local_date(timestamp.group(1)), *sorted(codes))
            if key in index and index[key] != url:
                raise ValueError(f"Ambiguous fixture: {key}")
            index[key] = url
    return index


def parse_series(page: str, url: str) -> list[dict]:
    timestamp = re.search(r'<div class="summary-info" data-utc="([^"]+)"', page)
    if not timestamp:
        raise ValueError(f"Missing match timestamp: {url}")
    chunks = re.split(r'<div class="tab-pane[^\"]*"\s+id="game-(\d+)"', page, flags=FLAGS)
    rows = []
    for offset in range(1, len(chunks), 2):
        game, card = int(chunks[offset]), chunks[offset + 1]
        teams = re.findall(r'<a href="https://mldb.gg/team/([^\"]+)" class="team-name">', card, FLAGS)[:2]
        if len(teams) != 2 or any(team not in TEAM_SLUGS for team in teams):
            raise ValueError(f"Unknown game teams {teams} in {url}")
        team_a, team_b = [TEAM_SLUGS[team] for team in teams]
        statuses = re.findall(r'class="mdc-header-team (win|lose) team-(?:one|two)"', card)
        names = [clean_text(value) for value in re.findall(
            r'<a[^>]+class="mdc-pf-player-name[^\"]*"[^>]*>(.*?)</a>', card, FLAGS)]
        heroes = [clean_text(value) for value in re.findall(
            r'<a[^>]+class="mdc-pf-hero-name[^\"]*"[^>]*>(.*?)</a>', card, FLAGS)]
        if sorted(statuses) != ["lose", "win"] or len(names) != len(heroes) or len(names) % 2:
            raise ValueError(f"Incomplete Game {game} record: {url}")
        if len(set(name.casefold() for name in names)) != len(names) or len(set(heroes)) != len(heroes):
            raise ValueError(f"Duplicate players/heroes in Game {game}: {url}")
        rows.append({
            "date": local_date(timestamp.group(1)), "series_started_at": timestamp.group(1),
            "season": SEASON, "team_a": team_a, "team_b": team_b,
            "players_a": names[0::2], "players_b": names[1::2],
            "heroes_a": heroes[0::2], "heroes_b": heroes[1::2],
            "lineups_verified": len(names) == 10, "picks_verified": len(heroes) == 10,
            "winner": team_a if statuses[0] == "win" else team_b,
            "series_url": url, "game": game,
        })
    if not rows:
        raise ValueError(f"No game records in {url}")
    return sorted(rows, key=lambda row: row["game"])


def parse_draft(page: str) -> tuple[list[str], list[str], list[str], list[str]]:
    # The mobile carousel repeats the desktop draft. Never count it twice.
    desktop = page.split('<div id="mobileDraftSlider-', 1)[0]
    portraits = re.findall(r'<div class="hero-portrait([^\"]*)">(.*?)</div>', desktop, FLAGS)
    bans, picks = [], []
    for classes, content in portraits:
        name = re.search(r'class="hero-img"[^>]*\balt="([^"]+)"', content, FLAGS)
        if not name:
            raise ValueError("Missing draft hero name")
        (bans if "banned-hero" in classes.split() else picks).append(clean_text(name.group(1)))
    if len(bans) != 10 or len(picks) != 10:
        raise ValueError(f"Expected ten bans and ten picks in the desktop draft; found {len(bans)} bans and {len(picks)} picks")
    return bans[:5], bans[5:], picks[:5], picks[5:]


def verify_series(expected: dict, rows: list[dict]) -> None:
    if len(rows) != len(expected["games"]):
        raise ValueError("Submitted game count differs from published results")
    for side in ("a", "b"):
        team = expected[f"team_{side}"]
        if sum(row["winner"] == team for row in rows) != expected[f"score_{side}"]:
            raise ValueError(f"Submitted series score does not match published winners for {team}")
    for game, row in enumerate(rows, start=1):
        if row["date"] != expected["date"] or row["game"] != game:
            raise ValueError("Date or game order mismatch")
        for side in ("a", "b"):
            team = row[f"team_{side}"]
            if set(row[f"bans_{side}"]) != set(expected["games"][game - 1][team]):
                raise ValueError(f"Submitted heroes differ from published bans: {team}, Game {game}")
        if set(row["bans_a"] + row["bans_b"]) & set(row.get("heroes_a", row.get("reported_heroes_a", [])) + row.get("heroes_b", row.get("reported_heroes_b", []))):
            raise ValueError("A banned hero was also recorded as played")


def import_series(expected: dict, url: str) -> list[dict]:
    rows = parse_series(fetch(url), url)
    for row in rows:
        draft_url = f"{url}?load_draft=false&game_seq={row['game']}"
        draft_html = json.loads(fetch(draft_url, ajax=True))["html"]
        if 'class="hero-portrait' not in draft_html:
            # Game player records still contain actual picks. Preserve the user
            # lists separately with an explicit unverified-ban flag.
            bans_a = expected["games"][row["game"] - 1][row["team_a"]]
            bans_b = expected["games"][row["game"] - 1][row["team_b"]]
            row["bans_verified"] = False
        else:
            try:
                bans_a, bans_b, picks_a, picks_b = parse_draft(draft_html)
            except ValueError as error:
                raise ValueError(f"{draft_url}: {error}") from error
            if set(picks_a) != set(row["heroes_a"]) or set(picks_b) != set(row["heroes_b"]):
                raise ValueError(f"Draft and player-record picks disagree: {draft_url}")
            row["bans_verified"] = True
        row.update(bans_a=bans_a, bans_b=bans_b, draft_source=draft_url,
                   supplied_hero_lists_verified_as="bans" if row["bans_verified"] else "unverified; treated as reported bans, not picks")
        if not row["lineups_verified"]:
            row["lineup_data_issue"] = "Published record has fewer than five players per side; excluded from player Elo."
        if not row["picks_verified"]:
            row["reported_heroes_a"] = row.pop("heroes_a")
            row["reported_heroes_b"] = row.pop("heroes_b")
            row["pick_data_issue"] = "Published record has fewer than five played heroes per side; excluded from pick counts."
    verify_series(expected, rows)
    print(f"Verified {expected['date']} {expected['team_a']} {expected['score_a']}-{expected['score_b']} {expected['team_b']}: {len(rows)} games", flush=True)
    return rows


def main() -> None:
    supplied = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    fixtures = fixture_index(json.loads(fetch(FIXTURES_URL))["html"])
    jobs = []
    for series in supplied["series"]:
        key = (series["date"], *sorted((series["team_a"], series["team_b"])))
        if key not in fixtures:
            raise ValueError(f"Submitted series not found in published fixtures: {key}")
        jobs.append((series, fixtures[key]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(lambda job: import_series(*job), jobs))
    rows = sorted([row for group in groups for row in group], key=lambda row: (row["series_started_at"], row["game"]))
    profiles = json.loads((ROOT / "config/mpl_ph_teams.json").read_text(encoding="utf-8"))
    normalize_players(rows, profiles)
    payload = {
        "season": SEASON, "scope": "Nine submitted regular-season series, August 21–28, 2026",
        "source": EVENT_URL, "series": len(groups), "games": len(rows),
        "supplied_data": str(MANUAL_PATH.relative_to(ROOT)),
        "verified_ban_games": sum(row["bans_verified"] for row in rows),
        "games_with_valid_lineups": sum(row["lineups_verified"] for row in rows),
        "games_with_valid_picks": sum(row["picks_verified"] for row in rows),
        "method": "Published game winners, actual players and played heroes. Supplied hero lists are stored as bans, with per-game verification flags where the published draft is available. Player/hero order is paired, not lane order.",
        "rows": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(groups)} series / {len(rows)} games to {OUTPUT_PATH}")


def normalize_players(rows: list[dict], profiles: dict) -> None:
    # Identity aliases are explicit, never inferred from a player's team slot.
    aliases = {member["player"].casefold(): member.get("history_alias", member["player"])
               for team in profiles["teams"] for member in team["roster"]}
    aliases.update({"supermarco": "Super Marco", "superfrince": "Super Frince",
                    "superyoshi": "Super Yoshi", "sensu1": "Sensui",
                    "jmpinkman": "JIMPINKMAN", "vinnn": "Vin", "teddyqt": "Teddy"})
    for row in rows:
        for side in ("a", "b"):
            row[f"source_players_{side}"] = row[f"players_{side}"][:]
            row[f"players_{side}"] = [aliases.get(name.casefold(), name) for name in row[f"players_{side}"]]
if __name__ == "__main__":
    main()
