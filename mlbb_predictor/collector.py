"""Incremental MPL PH collection with fail-closed validation and atomic publishing."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

from download_recent_data import (
    EVENT_URL, FIXTURES_URL, SEASON, SHORT_CODES, clean_text, local_date,
    normalize_players, parse_draft, parse_series,
)
from .history import merge_game_history, series_key, valid_lineups, valid_picks
from .live_data import snapshot_path, train_collected_model
from .player_model import player_key
from .teams import rating_lineup

TEAM_CODES = {"APBR", "RORA", "FNOP", "OMG", "FLCN", "TLPH", "TNC", "TWIS"}
DEFAULT_SETTINGS = {"enabled": True, "mode": "day_after_completion", "completion_delay_hours": 24}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()  # Only the exact temporary file created above.


def settings_path(root: Path) -> Path:
    return root / "config/data_collection.json"


def status_path(root: Path) -> Path:
    return root / "data/processed/collection_status.json"


def load_settings(root: Path) -> dict:
    saved = read_json(settings_path(root), {})
    # Old interval settings migrate to the requested match-aware policy. Only
    # the user's enabled/paused preference carries over; no recurring import.
    settings = {**DEFAULT_SETTINGS, "enabled": saved.get("enabled", True)}
    if not isinstance(settings["enabled"], bool):
        raise ValueError("Collection enabled setting must be true or false")
    return settings


def save_settings(root: Path, enabled: bool) -> None:
    atomic_json(settings_path(root), {**DEFAULT_SETTINGS, "enabled": bool(enabled)})


@contextmanager
def collection_lock(root: Path, name: str = "collection.lock"):
    """OS lock: shared by CLI and app workers; automatically released on crashes."""
    if name not in {"collection.lock", "schedule.lock", "profiles.lock"}:
        raise ValueError("Unknown collector lock")
    path = root / "data/processed" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass
            yield acquired
        finally:
            if acquired:
                file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(file, fcntl.LOCK_UN)


def safe_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "mldb.gg" or not (
        parsed.path == "/ajax/event/mpl-philippines-season-18/section/matches"
        or parsed.path.startswith("/match/MPL_Philippines_Season_18-")
    ):
        raise ValueError("Collector only reads the configured public MPL PH S18 source")
    return url


def fetch_public(url: str, ajax: bool = False) -> str:
    safe_url(url)
    headers = {"User-Agent": "mlbb-local-collector/1.1 (personal research; match-aware collection)"}
    if ajax:
        headers.update({"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as response:
        safe_url(response.geturl())
        return response.read().decode("utf-8")


class _FixtureStages(HTMLParser):
    """Stage belongs to a containing section, never the preceding sibling tab."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.stages = {}
        self.weeks = {}
        self.week = None
        self.result_week = None
        self.result_stage = "Unknown"

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "div":
            self.stack.append(attrs.get("data-stage", self.stack[-1] if self.stack else "Unknown"))
        if tag == "a" and "matches-item" in classes:
            self.stages[attrs.get("href")] = self.stack[-1] if self.stack else "Unknown"
        if tag == "span" and "td-dr-week-title" in classes:
            self.week = []
        if tag == "tr" and "td-dr-row" in classes and self.result_stage != "Unknown":
            self.stages[attrs.get("data-href")] = self.result_stage
            if self.result_week:
                self.weeks[attrs.get("data-href")] = self.result_week

    def handle_data(self, data):
        if self.week is not None:
            self.week.append(data)

    def handle_endtag(self, tag):
        if tag == "div" and self.stack:
            self.stack.pop()
        if tag == "span" and self.week is not None:
            title = " ".join("".join(self.week).split())
            number = re.search(r"\bWeek (\d+)\b", title, re.I)
            self.result_week = int(number.group(1)) if number else None
            self.result_stage = ("Playoffs" if "playoffs" in title.casefold() else
                                 "Regular Season" if re.fullmatch(r"Week \d+", title, re.I) else "Unknown")
            self.week = None


def discover_schedule(page: str, now: datetime) -> list[dict]:
    """Read starts/statuses; data-utc is a scheduled start, NOT a finish time."""
    found = {}
    stages = _FixtureStages()
    stages.feed(page)
    pattern = r'<a class="matches-item" href="([^"]+)">(.*?)</a>'
    for match in re.finditer(pattern, page, re.I | re.S):
        url, card = html.unescape(match.group(1)), match.group(2)
        status = re.search(r'class="match-status\s+([a-z-]+)"', card)
        timestamp = re.search(r'data-utc="([^"]+)"', card)
        codes = re.findall(r'class="team-info-short-name">(.*?)</span>', card, re.S)
        scores = re.findall(r'class="team-info-score">(.*?)</span>', card, re.S)
        best_of = re.search(r'\bBO([357])\b', card)
        if len(codes) != 2:
            raise ValueError("A fixture is missing its teams")
        codes = [SHORT_CODES.get(clean_text(code), clean_text(code)) for code in codes]
        if not set(codes) <= TEAM_CODES or len(set(codes)) != 2:
            continue
        if not timestamp or not best_of or not status:
            raise ValueError("A fixture is missing its date, status or format")
        state = status.group(1)
        if state not in {"completed", "live", "upcoming", "postponed", "cancelled", "canceled"}:
            raise ValueError(f"Unrecognized fixture status: {state}")
        when = datetime.fromisoformat(timestamp.group(1))
        if when.tzinfo is None:
            raise ValueError("Fixture timestamp is missing its timezone")
        if state == "completed" and when > now:
            continue
        if state == "completed":
            if len(scores) != 2:
                raise ValueError("A completed fixture is missing its scores")
            scores = [int(clean_text(score)) for score in scores]
            needed = int(best_of.group(1)) // 2 + 1
            if max(scores) != needed or min(scores) < 0 or min(scores) >= needed:
                raise ValueError("Completed fixture does not contain a finished series score")
        else:
            scores = [None, None]  # Partial/live scores must never become final results.
        stage = stages.stages.get(url, "Unknown")
        fixture = {
            "date": local_date(timestamp.group(1)), "series_started_at": timestamp.group(1),
            "team_a": codes[0], "team_b": codes[1], "score_a": scores[0], "score_b": scores[1],
            "best_of": int(best_of.group(1)), "stage": stage, "url": safe_url(url), "status": state,
        }
        if url in stages.weeks:
            fixture["week"] = stages.weeks[url]
        key = series_key(fixture)
        if key in found and found[key] != fixture:
            raise ValueError(f"Conflicting fixture: {key}")
        found[key] = fixture
    if not found:
        raise ValueError("No recognizable MPL PH S18 fixtures found; keeping the last-good schedule")
    return sorted(found.values(), key=lambda item: item["series_started_at"])


def discover_completed(page: str, now: datetime) -> list[dict]:
    fixtures = [item for item in discover_schedule(page, now) if item["status"] == "completed"]
    if not fixtures:
        raise ValueError("No completed MPL PH S18 fixtures found; keeping the last-good data")
    return fixtures


def collect_series(fixture: dict, profiles: dict, fetcher=fetch_public) -> list[dict]:
    rows = parse_series(fetcher(fixture["url"]), fixture["url"])
    if len(rows) != fixture["score_a"] + fixture["score_b"]:
        raise ValueError("Game count differs from completed fixture score")
    for side in ("a", "b"):
        if sum(row["winner"] == fixture[f"team_{side}"] for row in rows) != fixture[f"score_{side}"]:
            raise ValueError("Game winners differ from completed fixture score")
    for number, row in enumerate(rows, 1):
        if row["game"] != number or series_key(row) != series_key(fixture):
            raise ValueError("Game order, date or teams differ from the fixture")
        row["stage"] = fixture["stage"]
        if fixture.get("week"):
            row["week"] = fixture["week"]
        row["bans_a"], row["bans_b"], row["bans_verified"] = [], [], False
        row["draft_source"] = f"{fixture['url']}?load_draft=false&game_seq={number}"
        try:
            draft = json.loads(fetcher(row["draft_source"], ajax=True))["html"]
        except (OSError, ValueError, KeyError) as error:
            draft = ""
            row["draft_data_issue"] = f"Draft unavailable: {type(error).__name__}"
        if 'class="hero-portrait' in draft:
            bans_a, bans_b, picks_a, picks_b = parse_draft(draft)
            if len(set(picks_a + picks_b)) != 10 or set(bans_a + bans_b) & set(picks_a + picks_b):
                raise ValueError("Published draft repeats a hero or contains a banned pick")
            for side, picks in (("a", picks_a), ("b", picks_b)):
                if not set(row[f"heroes_{side}"]) <= set(picks):
                    raise ValueError("Player picks disagree with the published draft")
            row.update(bans_a=bans_a, bans_b=bans_b, bans_verified=True)
            if not valid_picks(row):
                # Full team picks can be known without a complete player mapping.
                row.update(heroes_a=picks_a, heroes_b=picks_b, picks_verified=True)
        if not valid_lineups(row):
            row["lineup_data_issue"] = "Incomplete published lineup; excluded from player Elo."
        if not valid_picks(row):
            row["reported_heroes_a"] = row.pop("heroes_a")
            row["reported_heroes_b"] = row.pop("heroes_b")
            row["pick_data_issue"] = "Incomplete published draft; excluded from pick counts."
    normalize_players(rows, profiles)
    return rows


def merge_series(previous: list[dict], incoming: list[dict]) -> list[dict]:
    """Upgrade incomplete records but quarantine conflicting or worse source data."""
    if not previous:
        return incoming
    if len(previous) != len(incoming):
        raise ValueError("Source changed the number of games in an already saved series")
    merged = []
    for old, candidate in zip(previous, incoming):
        new = deepcopy(candidate)
        if series_key(old) != series_key(new) or old["game"] != new["game"] or old["winner"] != new["winner"]:
            raise ValueError("Source conflicts with a saved game result; review required")
        if old["team_a"] != new["team_a"]:
            for field in ("team", "players", "source_players", "heroes", "reported_heroes", "bans"):
                a, b = f"{field}_a", f"{field}_b"
                if a in new and b in new:
                    new[a], new[b] = new[b], new[a]
        if valid_lineups(old) and valid_lineups(new):
            for side in ("a", "b"):
                if {player_key(n) for n in old[f"players_{side}"]} != {player_key(n) for n in new[f"players_{side}"]}:
                    raise ValueError("Source conflicts with a saved complete lineup; review required")
                if valid_picks(old) and valid_picks(new):
                    old_pairs = dict(zip(map(player_key, old[f"players_{side}"]), old[f"heroes_{side}"]))
                    new_pairs = dict(zip(map(player_key, new[f"players_{side}"]), new[f"heroes_{side}"]))
                    if old_pairs != new_pairs:
                        raise ValueError("Source conflicts with saved player-to-hero mappings; review required")
        elif valid_picks(old) and valid_picks(new):
            if any(set(old[f"heroes_{s}"]) != set(new[f"heroes_{s}"]) for s in ("a", "b")):
                raise ValueError("Source conflicts with saved picks; review required")
        if ((valid_lineups(old) or valid_lineups(new)) and
                (valid_picks(old) or valid_picks(new)) and
                not (valid_lineups(old) and valid_picks(old)) and
                not (valid_lineups(new) and valid_picks(new))):
            raise ValueError("Cannot combine independently ordered lineups and picks into a player-to-hero mapping")
        # Keep an existing verified player/hero mapping as a unit. Never reorder
        # its players without also reordering the paired heroes.
        if valid_lineups(old) and (not valid_lineups(new) or (valid_picks(old) and not valid_picks(new))):
            for field in ("players_a", "players_b", "source_players_a", "source_players_b", "heroes_a", "heroes_b", "picks_verified", "lineups_verified"):
                if field in old:
                    new[field] = old[field]
        if valid_picks(old) and not valid_picks(new):
            new.update(heroes_a=old["heroes_a"], heroes_b=old["heroes_b"], picks_verified=True)
        if old.get("bans_verified") and new.get("bans_verified"):
            if any(set(old[f"bans_{s}"]) != set(new[f"bans_{s}"]) for s in ("a", "b")):
                raise ValueError("Source conflicts with saved verified bans; review required")
        if not new.get("bans_verified") and old.get("bans_a"):
            new.update(bans_a=old["bans_a"], bans_b=old["bans_b"], bans_verified=old.get("bans_verified", False))
        result = {**old, **new}
        if valid_lineups(result):
            result.pop("lineup_data_issue", None)
        if valid_picks(result):
            for field in ("pick_data_issue", "reported_heroes_a", "reported_heroes_b"):
                result.pop(field, None)
        merged.append(result)
    return merged


def roster_alerts(rows: list[dict], profiles: dict) -> list[dict]:
    latest = {}
    for row in rows:
        if valid_lineups(row):
            for side in ("a", "b"):
                latest[row[f"team_{side}"]] = (row["date"], row[f"players_{side}"])
    alerts = []
    for profile in profiles["teams"]:
        if profile["code"] in latest:
            date, players = latest[profile["code"]]
            if set(map(player_key, players)) != set(map(player_key, rating_lineup(profile))):
                alerts.append({"team": profile["name"], "date": date, "observed_players": players,
                               "message": "Latest recorded lineup differs from configured starters; review substitutions/transfers before changing the prediction roster."})
    return alerts


def run_collection(root: Path, *, fetcher=fetch_public, now: datetime | None = None,
                   selected_urls: set[str] | None = None, fixture_feed: list[dict] | None = None) -> dict:
    root = root.resolve()
    now = now or utc_now()
    with collection_lock(root) as acquired:
        if not acquired:
            return {"state": "busy", "message": "Another collection is already running"}
        previous_status = read_json(status_path(root), {})
        status = {**previous_status, "state": "running", "last_attempt_at": now.isoformat(), "errors": []}
        atomic_json(status_path(root), status)
        try:
            fixtures = (discover_completed(json.loads(fetcher(FIXTURES_URL))["html"], now)
                        if fixture_feed is None else [item for item in fixture_feed if item["status"] == "completed"])
            profiles = read_json(root / "config/mpl_ph_teams.json")
            live_profiles = read_json(root / "data/processed/team_profiles.json", {})
            # Keep historical aliases too, including players who left a team.
            profiles = {**profiles, "teams": profiles["teams"] + live_profiles.get("teams", [])}
            old_payload = read_json(root / "data/processed/mpl_ph_s17_player_games.json")
            seed = read_json(root / "data/processed/mpl_ph_s18_player_games.json")
            previous_snapshot = read_json(snapshot_path(root), {})
            existing = previous_snapshot.get("season_rows", seed["rows"])
            groups = defaultdict(list)
            for row in existing:
                groups[series_key(row)].append(row)
            latest_keys = {series_key(item) for item in fixtures[-4:]}
            candidates = [item for item in fixtures if series_key(item) not in groups or
                          series_key(item) in latest_keys or any(
                              not valid_lineups(row) or not valid_picks(row) or not row.get("bans_verified")
                              for row in groups[series_key(item)])]
            if selected_urls is not None:
                # Scheduled runs import ONLY due fixtures, not the most recent
                # four unrelated games or any newly finished match too early.
                candidates = [item for item in fixtures if item["url"] in selected_urls]
            checked_series = dict(previous_status.get("checked_series", {}))
            candidates.sort(key=lambda item: (series_key(item) in groups,
                                              checked_series.get("|".join(series_key(item)), ""),
                                              item["series_started_at"]))
            changed_series, errors, outcomes = 0, [], []
            for fixture in candidates[:16]:
                key = series_key(fixture)
                checked_series["|".join(key)] = now.isoformat()
                try:
                    incoming = collect_series(fixture, profiles, fetcher)
                    updated = merge_series(groups.get(key, []), incoming)
                    changed_series += int(groups.get(key) != updated)
                    groups[key] = updated
                    complete = all(valid_lineups(row) and valid_picks(row) and row.get("bans_verified") for row in updated)
                    outcomes.append({"url": fixture["url"], "state": "complete" if complete else "incomplete"})
                except Exception as error:
                    outcomes.append({"url": fixture["url"], "state": "error"})
                    errors.append({"series": f"{fixture['team_a']}–{fixture['team_b']} {fixture['date']}",
                                   "source": fixture["url"], "error": str(error)[:500]})
            season_rows = merge_game_history([row for group in groups.values() for row in group])
            changed = season_rows != existing or not previous_snapshot
            if changed:
                model, metrics = train_collected_model(old_payload["rows"], season_rows)
                combined = merge_game_history(old_payload["rows"], season_rows)
                live_metadata = {
                    "season": SEASON, "source": EVENT_URL, "scope": "Automatically collected completed S18 series plus preserved user imports",
                    "games": len(season_rows), "series": len(groups),
                    "games_with_valid_lineups": sum(valid_lineups(row) for row in season_rows),
                    "games_with_valid_picks": sum(valid_picks(row) for row in season_rows),
                }
                snapshot = {
                    "version": 1, "published_at": utc_now().isoformat(), "season_rows": season_rows,
                    "history_rows": combined,
                    "history_metadata": {"sources": [{k: v for k, v in old_payload.items() if k != "rows"}, live_metadata],
                                         "games": len(combined), "series": len({series_key(row) for row in combined}),
                                         "cutoff": max(row["date"] for row in combined)},
                    "model": model.to_dict(metrics),
                }
                if previous_snapshot:
                    atomic_json(root / "data/processed/live_snapshot.previous.json", previous_snapshot)
                atomic_json(snapshot_path(root), snapshot)
            status.update(
                state="partial" if errors else "success", finished_at=utc_now().isoformat(),
                last_success_at=now.isoformat() if not errors else previous_status.get("last_success_at"),
                discovered_series=len(fixtures), collected_series=len(groups), collected_games=len(season_rows),
                new_games=len(season_rows) - len(existing), changed_series=changed_series,
                pending_series=max(0, len(candidates) - 16), errors=errors, series_outcomes=outcomes,
                checked_series=checked_series,
                games_with_lineups=sum(valid_lineups(row) for row in season_rows),
                games_with_picks=sum(valid_picks(row) for row in season_rows),
                cutoff=max(row["date"] for row in season_rows), roster_alerts=roster_alerts(season_rows, profiles),
            )
        except Exception as error:
            status.update(state="error", finished_at=utc_now().isoformat(),
                          errors=[{"error": str(error)[:700]}], new_games=0)
        atomic_json(status_path(root), status)
        return status
