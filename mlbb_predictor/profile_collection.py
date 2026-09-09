"""Daily official profiles, with independent retries and last-good publication."""

from copy import deepcopy
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
import re
import urllib.parse
import urllib.request

from .collector import atomic_json, collection_lock, read_json, utc_now
from .teams import load_team_profiles, starting_roster

ROLES = ("EXP Lane", "Jungle", "Mid Lane", "Gold Lane", "Roam")


def profile_snapshot_path(root: Path) -> Path:
    return root / "data/processed/team_profiles.json"


def profile_status_path(root: Path) -> Path:
    return root / "data/processed/profile_collection_status.json"


def next_profile_check(root: Path, now):
    # Empty test/utility workspaces do not have a profile source to refresh.
    if not (root / "config/mpl_ph_teams.json").exists():
        return None
    status = read_json(profile_status_path(root), {})
    from .match_schedule import timestamp
    last = timestamp(status.get("last_attempt_at"))
    return now if last and last > now else timestamp(status.get("next_check_at")) or now


def fetch_profile(url: str) -> str:
    def validate(value):
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme != "https" or parsed.netloc != "ph-mpl.com" or
                not re.fullmatch(r"/team/[a-z0-9-]+", parsed.path) or parsed.query):
            raise ValueError("Only public official MPL PH team profiles are allowed")
    validate(url)
    request = urllib.request.Request(url, headers={"User-Agent": "mlbb-local-collector/1.2"})
    with urllib.request.urlopen(request, timeout=20) as response:
        validate(response.geturl())
        return response.read().decode("utf-8")


class ProfileHTML(HTMLParser):
    """Read only public player names/roles and visible profile text."""

    def __init__(self):
        super().__init__()
        self.players = {}
        self.text = []
        self.summary = []
        self.description_depth = 0
        self.achievement_depth = 0
        self.achievement_row_depth = 0
        self.achievement_row = []
        self.achievements = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input":
            match = re.fullmatch(r"player-(\d+)-(ign|role)", attrs.get("name", ""))
            if match:
                fields = self.players.setdefault(match[1], {})
                value = " ".join(attrs.get("value", "").split())
                if match[2] in fields and fields[match[2]] != value:
                    raise ValueError("Conflicting official player fields")
                fields[match[2]] = value
        if tag == "div":
            classes = attrs.get("class", "").split()
            if self.achievement_depth:
                self.achievement_depth += 1
            elif "achievements" in classes:
                self.achievement_depth = 1
            if self.achievement_row_depth:
                self.achievement_row_depth += 1
            elif self.achievement_depth and "justify-content-between" in classes:
                self.achievement_row_depth = 1
                self.achievement_row = []
            if self.description_depth:
                self.description_depth += 1
            elif "team-description" in attrs.get("class", "").split():
                self.description_depth = 1

    def handle_endtag(self, tag):
        if tag == "div" and self.achievement_row_depth:
            self.achievement_row_depth -= 1
            if not self.achievement_row_depth:
                self.achievements.append(" — ".join(self.achievement_row))
        if tag == "div" and self.achievement_depth:
            self.achievement_depth -= 1
        if tag == "div" and self.description_depth:
            self.description_depth -= 1

    def handle_data(self, data):
        value = " ".join(data.split())
        if value:
            self.text.append(value)
            if self.description_depth:
                self.summary.append(value)
            if self.achievement_row_depth:
                self.achievement_row.append(value)


def parse_profile(page: str, previous: dict, aliases: dict, season="Season 18") -> dict:
    parser = ProfileHTML()
    parser.feed(page)
    marker = f"Roster for {season}"
    if marker not in parser.text or sum(text.startswith("Roster for Season") for text in parser.text) != 1:
        raise ValueError("Official roster season is missing or changed; keeping saved profile")
    roster, staff = [], []
    roles = {role.casefold().replace(" ", ""): role for role in ROLES}
    for fields in parser.players.values():
        name, role = fields.get("ign", ""), fields.get("role", "")
        if not name or not role:
            raise ValueError("Official player name or role missing")
        if role.casefold().replace(" ", "") in roles:
            canonical = aliases.get(name.casefold(), name)
            roster.append({"player": name, "history_alias": canonical,
                           "role": roles[role.casefold().replace(" ", "")], "status": "Registered"})
        elif any(word in role.casefold() for word in ("coach", "manager", "analyst")):
            staff.append({"name": name, "role": role})
        else:
            raise ValueError(f"Unrecognized official role: {role}")
    identities = [member["history_alias"].casefold() for member in roster]
    if (not 5 <= len(roster) <= 12 or len(set(identities)) != len(identities) or
            set(member["role"] for member in roster) != set(ROLES)):
        raise ValueError("Official roster is incomplete or repeats players")
    if not staff or not parser.summary or "Achievements" not in parser.text:
        raise ValueError("Official profile sections are incomplete")
    if not parser.achievements or not all(parser.achievements):
        raise ValueError("Official achievements could not be parsed")
    result = deepcopy(previous)
    result.update(roster=roster, staff=staff, summary=" ".join(parser.summary),
                  achievements=parser.achievements,
                  prediction_roster=deepcopy(starting_roster(previous)))
    return result


def refresh_profiles(root: Path, *, fetcher=fetch_profile, now=None, force=False) -> dict:
    now = now or utc_now()
    with collection_lock(root, "profiles.lock") as acquired:
        if not acquired:
            return {"state": "busy"}
        due = next_profile_check(root, now)
        if due is None or (due > now and not force):
            return read_json(profile_status_path(root), {"state": "not_due"})
        old_status = read_json(profile_status_path(root), {})
        status = {**old_status, "state": "checking", "last_attempt_at": now.isoformat(),
                  "next_check_at": (now + timedelta(hours=1)).isoformat(), "errors": []}
        atomic_json(profile_status_path(root), status)
        try:
            seed = load_team_profiles(root / "config/mpl_ph_teams.json")
            saved = read_json(profile_snapshot_path(root), seed)
            aliases = {member["player"].casefold(): member.get("history_alias", member["player"])
                       for payload in (seed, saved) for team in payload["teams"] for member in team["roster"]}
            # Verified source spellings; punctuation is not a new player identity.
            aliases.update({"jamespangks.": "JamesPangks", "ryota.": "Ryota", "supermarco": "Super Marco",
                            "superfrince": "Super Frince", "superyoshi": "Super Yoshi"})
            result = deepcopy(saved)
            for index, team in enumerate(saved["teams"]):
                try:
                    updated = parse_profile(fetcher(team["official_url"]), team, aliases, seed["season"])
                    updated["profile_checked_at"] = now.isoformat()
                    result["teams"][index] = updated
                except Exception as error:
                    status["errors"].append({"team": team["name"], "error": str(error)[:500]})
            players = [member.get("history_alias", member["player"]).casefold()
                       for team in result["teams"] for member in team["roster"]]
            if len(players) != len(set(players)):
                raise ValueError("Official profiles list a player on multiple teams; keeping saved profiles until resolved")
            if result != saved:
                result["published_at"] = now.isoformat()
                atomic_json(profile_snapshot_path(root), result)
            status.update(state="partial" if status["errors"] else "success",
                          last_success_at=old_status.get("last_success_at") if status["errors"] else now.isoformat(),
                          next_check_at=(now + timedelta(hours=1 if status["errors"] else 24)).isoformat())
        except Exception as error:
            status.update(state="error", errors=[{"error": str(error)[:500]}])
        atomic_json(profile_status_path(root), status)
        return status
