"""Collect newly completed MPL PH matches once; the app handles periodic runs."""

import json
import argparse
from pathlib import Path

from mlbb_predictor.collector import run_collection
from mlbb_predictor.automatic_collection import run_automatic_collection
from mlbb_predictor.profile_collection import refresh_profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the match-aware schedule; or explicitly collect completed games immediately.")
    parser.add_argument("--check-schedule", action="store_true", help="Check fixtures and collect only matches whose 24-hour delay has elapsed")
    args = parser.parse_args()
    collector = run_automatic_collection if args.check_schedule else run_collection
    result = collector(Path(__file__).resolve().parent)
    if not args.check_schedule:
        result["profile_report"] = refresh_profiles(Path(__file__).resolve().parent)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    raise SystemExit(1 if result["state"] == "error" else 0)
