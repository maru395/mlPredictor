"""Load and normalize the Kaggle and Hugging Face MLBB datasets."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


TEAM_ALIASES = {
    # Kaggle MPL PH S13 names mapped to the same franchises in S14.
    "AP.BREN": "FCAP",
    "FALCONS AP.BREN": "FCAP",
    "BLACKLIST INTL.": "BLCK",
    "BLACKLIST INTERNATIONAL": "BLCK",
    "ECHO": "TLPH",
    "TEAM LIQUID PHILIPPINES": "TLPH",
    "MINANA EVOS": "MNNE",
    "ONIC PHILIPPINES": "FNOP",
    "FNATIC ONIC PHILIPPINES": "FNOP",
    "OMEGA ESPORTS": "OMG",
    "RSG PHILIPPINES": "RSG",
    "TNC PRO TEAM": "TNC",
    # Friendly full-name inputs for current dataset codes.
    "ALTER EGO": "AE",
    "BIGETRON ALPHA": "BTR",
    "DEWA UNITED": "DEWA",
    "EVOS GLORY": "EVOS",
    "FNATIC ONIC": "FNOC",
    "GEEK FAM": "GEEK",
    "REBELLION ESPORTS": "RBL",
    "RRQ HOSHI": "RRQ",
    "TEAM LIQUID ID": "TLID",
    "AURORA GAMING": "RORA",
}

DISPLAY_NAMES = {
    "AE": "Alter Ego",
    "BLCK": "Blacklist International",
    "BTR": "Bigetron Alpha",
    "DEWA": "Dewa United",
    "EVOS": "EVOS Glory",
    "FCAP": "Falcons AP.Bren",
    "FNOC": "Fnatic ONIC (Indonesia)",
    "FNOP": "Fnatic ONIC PH",
    "GEEK": "Geek Fam",
    "MNNE": "Minana EVOS",
    "OMG": "Smart Omega",
    "RBL": "Rebellion Esports",
    "RORA": "Aurora Gaming PH",
    "RRQ": "RRQ Hoshi",
    "RSG": "RSG Philippines",
    "TLID": "Team Liquid Indonesia",
    "TLPH": "Team Liquid PH",
    "TNC": "TNC Pro Team",
    "APBR": "AP BREN",
    "FLCN": "Team Falcons PH",
    "TWIS": "Twisted Minds PH",
}


def canonical_team(raw_name: str) -> str:
    """Return a stable team code while preserving unknown codes."""
    cleaned = " ".join(raw_name.strip().split())
    upper = cleaned.upper()
    return TEAM_ALIASES.get(upper, upper)


def display_name(team_code: str) -> str:
    return DISPLAY_NAMES.get(team_code, team_code)


def _iso_kaggle_date(value: str) -> str:
    return datetime.strptime(value.strip(), "%m/%d/%Y").date().isoformat()


def _iso_hf_date(value: str) -> str:
    # Both S14 files represent the 2024 season and omit the year.
    # The Indonesia CSV uses the Indonesian abbreviation "okt" for October.
    normalized = value.strip().lower().replace("okt-", "oct-", 1)
    parsed = datetime.strptime(f"{normalized}-2024", "%b-%d-%Y")
    return parsed.date().isoformat()


def load_kaggle_boxmatch(path: Path) -> list[dict[str, str | int]]:
    """Collapse the Kaggle player-level file into one row per game.

    The source stores five consecutive rows per team and ten consecutive rows
    per game. The ordering is more reliable than the Time column, which has one
    known typo in the source file.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or len(rows) % 10:
        raise ValueError(
            f"Expected a non-empty multiple of 10 player rows in {path}; "
            f"found {len(rows)}."
        )

    matches: list[dict[str, str | int]] = []
    for offset in range(0, len(rows), 10):
        chunk = rows[offset : offset + 10]
        team_counts = Counter(canonical_team(row["Team"]) for row in chunk)
        if sorted(team_counts.values()) != [5, 5]:
            raise ValueError(f"Invalid game block at player row {offset + 2}: {team_counts}")

        teams: list[str] = []
        for row in chunk:
            code = canonical_team(row["Team"])
            if code not in teams:
                teams.append(code)

        outcomes = {
            canonical_team(row["Team"]): row["Win"].strip().lower() for row in chunk
        }
        winners = [team for team, result in outcomes.items() if result == "win"]
        if len(winners) != 1:
            raise ValueError(f"Expected one winner at player row {offset + 2}: {outcomes}")

        matches.append(
            {
                "date": _iso_kaggle_date(chunk[0]["Date"]),
                "order": offset // 10,
                "team_a": teams[0],
                "team_b": teams[1],
                "winner": winners[0],
                "source": "Kaggle",
                "league": "MPL Philippines",
                "season": "S13",
            }
        )
    return matches


def load_huggingface_games(
    path: Path, *, league: str
) -> list[dict[str, str | int]]:
    """Load one Hugging Face S14 game-level CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    matches: list[dict[str, str | int]] = []
    for index, row in enumerate(rows):
        team_a = canonical_team(row["blue_team"])
        team_b = canonical_team(row["red_team"])
        result = row["result"].strip().upper()
        if result not in {"BLUE", "RED"}:
            raise ValueError(f"Unexpected result {result!r} on row {index + 2} of {path}")
        matches.append(
            {
                "date": _iso_hf_date(row["date"]),
                "order": int(row.get("no") or index),
                "team_a": team_a,
                "team_b": team_b,
                "winner": team_a if result == "BLUE" else team_b,
                "source": "Hugging Face",
                "league": league,
                "season": "S14",
            }
        )
    return matches


def find_kaggle_boxmatch(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "kaggle_mpl_ph_s13_boxmatch.csv",
        raw_dir
        / "kaggle_mpl_ph_s13"
        / "MPL Philippines Season 13 - BoxMatch.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Kaggle BoxMatch CSV not found. Run `python download_data.py` first."
    )


def load_all_matches(raw_dir: Path) -> list[dict[str, str | int]]:
    required_hf = {
        "MPL Indonesia": raw_dir / "hf_mpl_id_s14.csv",
        "MPL Philippines": raw_dir / "hf_mpl_ph_s14.csv",
    }
    missing = [str(path) for path in required_hf.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Hugging Face data: " + ", ".join(missing) + ". Run `python download_data.py`."
        )

    matches = load_kaggle_boxmatch(find_kaggle_boxmatch(raw_dir))
    for league, path in required_hf.items():
        matches.extend(load_huggingface_games(path, league=league))

    # The two leagues have no overlapping teams on the same day, so source/order
    # is only a deterministic tie-breaker and cannot leak results between them.
    matches.sort(key=lambda row: (str(row["date"]), str(row["league"]), int(row["order"])))
    return matches


def write_normalized_matches(
    matches: Iterable[dict[str, str | int]], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "order", "team_a", "team_b", "winner", "source", "league", "season"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def read_normalized_matches(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
