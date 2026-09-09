"""Coordinate match and profile collection without coupling source failures."""

from .collector import load_settings, read_json, utc_now
from .match_schedule import next_check as next_match_check, run_scheduled_collection, schedule_path
from .profile_collection import next_profile_check, refresh_profiles


def next_check(settings, schedule, now, root=None):
    match_due = next_match_check(settings, schedule, now)
    profile_due = next_profile_check(root, now) if root is not None and settings["enabled"] else None
    dates = [value for value in (match_due, profile_due) if value is not None]
    return min(dates) if dates else None


def run_automatic_collection(root):
    now = utc_now()
    schedule = read_json(schedule_path(root), {})
    # A profile retry must not bypass the match source's independent backoff.
    match_due = next_match_check(load_settings(root), schedule, now)
    if schedule.get("state") != "error" or match_due is None or match_due <= now:
        schedule = run_scheduled_collection(root)
    profiles = refresh_profiles(root)
    return {**schedule, "profile_report": profiles}
