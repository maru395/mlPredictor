"""Persisted schedule discovery and delayed, per-series collection.

MLDB exposes a start and a status, not an actual finish timestamp. The delay
therefore starts when this app first observes Completed, never at kickoff.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

from .collector import (
    FIXTURES_URL, atomic_json, collection_lock, discover_schedule, fetch_public,
    read_json, run_collection, utc_now,
)
from .history import series_key, valid_lineups, valid_picks
from .live_data import snapshot_path

DAY = timedelta(hours=24)
STATUS_RETRY = timedelta(hours=1)
FIRST_STATUS_CHECK = timedelta(hours=3)


def schedule_path(root: Path) -> Path:
    return root / "data/processed/match_schedule.json"


def timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("Scheduler timestamps must include their timezone")
    return moment


def collection_due(item: dict) -> datetime | None:
    if item.get("status") != "completed" or item.get("last_collected_at") or not item.get("present", True):
        return None
    eligible = timestamp(item.get("collect_after"))
    retry = timestamp(item.get("retry_after"))
    return max(eligible, retry) if eligible and retry else eligible


def next_collection(schedule: dict) -> datetime | None:
    dates = [due for item in schedule.get("matches", {}).values() if (due := collection_due(item))]
    return min(dates) if dates else None


def next_check(settings: dict, schedule: dict, now: datetime) -> datetime | None:
    """Next network action, not a fixed interval for downloading all matches."""
    if not settings["enabled"]:
        return None
    if not schedule.get("last_checked_at"):
        retry = timestamp(schedule.get("retry_at"))
        return retry or now
    if schedule.get("state") in {"error", "busy"}:
        return timestamp(schedule.get("retry_at")) or now
    last = timestamp(schedule["last_checked_at"])
    if last > now:  # Recover from a backwards system clock adjustment.
        return now
    dates = [timestamp(schedule.get("next_discovery_at")) or now]
    for item in schedule.get("matches", {}).values():
        if not item.get("present", True):
            continue
        if (due := collection_due(item)):
            dates.append(due)
        if item.get("status") in {"live", "upcoming"} and item.get("check_after"):
            dates.append(timestamp(item["check_after"]))
    return min(dates)


def _saved_complete(fixture: dict, rows: list[dict]) -> bool:
    games = [row for row in rows if series_key(row) == series_key(fixture)]
    return (len(games) == fixture["score_a"] + fixture["score_b"] and
            all(valid_lineups(row) and valid_picks(row) and row.get("bans_verified") for row in games) and
            sum(row["winner"] == fixture["team_a"] for row in games) == fixture["score_a"])


def observe_schedule(previous: dict, fixtures: list[dict], now: datetime,
                     saved_rows: list[dict] | None = None) -> dict:
    """Pure transition: preserve completion timers through restarts and polls."""
    result = deepcopy(previous)
    matches = result.setdefault("matches", {})
    old_urls = set(matches)
    for item in matches.values():
        item["present"] = False
    for fixture in fixtures:
        url = fixture["url"]  # Stable across a postponement's date/time change.
        old = matches.get(url, {})
        item = {**old, **fixture, "present": True, "last_seen_at": now.isoformat()}
        if fixture["status"] == "completed":
            changed_identity = old and (series_key(old) != series_key(fixture) or
                                       old.get("series_started_at") != fixture["series_started_at"])
            if old.get("status") != "completed" or changed_identity or not old.get("first_seen_completed_at"):
                item.update(first_seen_completed_at=now.isoformat(), collect_after=(now + DAY).isoformat(),
                            last_collected_at=None, retry_after=None, collection_state="waiting_24h")
            elif old.get("last_collected_at") and (old.get("score_a"), old.get("score_b")) != (fixture["score_a"], fixture["score_b"]):
                # A corrected score deserves review through the normal safe merge.
                item.update(last_collected_at=None, retry_after=(now + DAY).isoformat(), collection_state="source_changed")
            item.pop("check_after", None)
            if url not in old_urls and saved_rows and _saved_complete(fixture, saved_rows):
                item.update(last_collected_at=now.isoformat(), collection_state="already_saved")
        else:
            # If a completion is withdrawn or a match postponed, cancel its old
            # timer. A later confirmed completion starts a fresh 24-hour delay.
            for field in ("first_seen_completed_at", "collect_after", "retry_after", "last_collected_at"):
                item.pop(field, None)
            item["collection_state"] = "awaiting_completion"
            start = timestamp(fixture["series_started_at"])
            if fixture["status"] in {"upcoming", "live"}:
                first_check = start + FIRST_STATUS_CHECK
                # A stale/postponed listing must not cause endless hourly traffic.
                retry = STATUS_RETRY if now - start < DAY else DAY
                item["check_after"] = max(first_check, now + retry).isoformat()
            else:
                item.pop("check_after", None)
        matches[url] = item
    result.update(version=1, state="success", last_checked_at=now.isoformat(),
                  next_discovery_at=(now + DAY).isoformat(), retry_at=None, error=None)
    return result


def run_scheduled_collection(root: Path, *, fetcher=fetch_public, now: datetime | None = None,
                             collector=run_collection) -> dict:
    """One lightweight discovery/check; full downloads only for eligible jobs."""
    root = root.resolve()
    now = now or utc_now()
    with collection_lock(root, "schedule.lock") as acquired:
        if not acquired:
            return {"state": "busy", "message": "Another schedule check is running"}
        previous = read_json(schedule_path(root), {})
        # Persist the old queue before networking; an interrupted process can
        # resume it rather than resetting the completion delay.
        current = {**previous, "state": "checking", "last_attempt_at": now.isoformat()}
        atomic_json(schedule_path(root), current)
        try:
            fixtures = discover_schedule(json.loads(fetcher(FIXTURES_URL))["html"], now)
            snapshot = read_json(snapshot_path(root), {})
            seed = read_json(root / "data/processed/mpl_ph_s18_player_games.json", {"rows": []})
            current = observe_schedule(current, fixtures, now, snapshot.get("season_rows", seed["rows"]))
            # Persist first-observed completion times even if the following
            # download/model rebuild fails or the app closes midway through it.
            atomic_json(schedule_path(root), current)
            due = sorted((item for item in current["matches"].values()
                          if (when := collection_due(item)) and when <= now), key=collection_due)
            if due:
                batch = due[:16]
                report = collector(root, fetcher=fetcher, now=now,
                                   selected_urls={item["url"] for item in batch}, fixture_feed=fixtures)
                outcomes = {item["url"]: item["state"] for item in report.get("series_outcomes", [])}
                for item in batch:
                    item["last_collection_attempt_at"] = now.isoformat()
                    state = outcomes.get(item["url"], "error") if report["state"] in {"success", "partial"} else "error"
                    item["collection_state"] = state
                    if state == "complete":
                        item.update(last_collected_at=now.isoformat(), retry_after=None)
                    else:
                        item["retry_after"] = (now + DAY).isoformat()
                if report["state"] == "busy":
                    for item in batch:
                        item["retry_after"] = (now + STATUS_RETRY).isoformat()
                current["last_collection_report"] = {"state": report["state"], "new_games": report.get("new_games", 0),
                                                      "at": now.isoformat()}
                # Large catch-ups are bounded and continue one hour later.
                for item in due[16:]:
                    item["retry_after"] = (now + STATUS_RETRY).isoformat()
            atomic_json(schedule_path(root), current)
        except Exception as error:
            # Keep the queue and last-good model. Network errors retry in one
            # hour; incomplete/conflicting game details retry the following day.
            current.update(state="error", error=f"{type(error).__name__}: {error}"[:700],
                           retry_at=(now + STATUS_RETRY).isoformat())
            atomic_json(schedule_path(root), current)
        return current
