"""User-editable hero tiers and transparent draft adjustments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .model import EloPredictor, elo_probability
from .player_model import player_key


TIER_ORDER = ("S", "A", "B", "C", "D", "F")
TIER_SCORES = {"S": 5.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
UNRATED_SCORE = 2.5
AUTO_META_PRIOR_GAMES = 5.0
ROLE_ALIASES = {
    "exp": "EXP Lane",
    "exp lane": "EXP Lane",
    "jungle": "Jungle",
    "jungler": "Jungle",
    "mid": "Mid Lane",
    "mid lane": "Mid Lane",
    "gold": "Gold Lane",
    "gold lane": "Gold Lane",
    "roam": "Roam",
    "roamer": "Roam",
}


def normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    return ROLE_ALIASES.get(" ".join(str(role).strip().casefold().split()))


def hero_tier(
    hero: str,
    tiers: dict[str, str],
    *,
    role: str | None = None,
    role_tiers: dict[str, dict[str, str]] | None = None,
) -> str:
    """Return a role-specific tier when present, then fall back to the global tier."""
    canonical_role = normalize_role(role)
    if canonical_role and role_tiers:
        assigned = role_tiers.get(canonical_role, {}).get(hero)
        if assigned in TIER_SCORES:
            return assigned
    return tiers.get(hero, "Unrated")


def roster_meta_profile(
    players: list[str], player_picks: dict, tiers: dict[str, str],
    *, roles: list[str] | None = None,
    role_tiers: dict[str, dict[str, str]] | None = None,
    prior_games: float = AUTO_META_PRIOR_GAMES,
) -> dict:
    """Estimate current-roster tier fit from each player's empirical hero pool.

    Each starter has equal influence. Five neutral pseudo-games per player
    soften small samples; missing/unrated heroes are neutral. Pick frequency is
    already the weight, so automatic mode must not add a second comfort bonus.
    These hand-set weights are an explanatory heuristic, not learned effects.
    """
    keys = [player_key(player) for player in players]
    if len(keys) != 5 or len(set(keys)) != 5 or not all(keys):
        raise ValueError("Automatic meta requires five unique current starters")
    if roles is not None and len(roles) != len(players):
        raise ValueError("Automatic meta roles must match the current starters")
    if prior_games <= 0:
        raise ValueError("Neutral prior games must be positive")
    summaries = []
    player_roles = roles if roles is not None else [None] * len(players)
    for player, key, role in zip(players, keys, player_roles):
        history = player_picks.get(key, {})
        hero_rows = history.get("heroes", [])
        if any(int(row["picks"]) <= 0 for row in hero_rows):
            raise ValueError("Hero pick counts must be positive")
        count = sum(int(row["picks"]) for row in hero_rows)
        resolved_tiers = {
            row["hero"]: hero_tier(
                row["hero"], tiers, role=role, role_tiers=role_tiers,
            )
            for row in hero_rows
        }
        weighted_total = sum(
            int(row["picks"]) * TIER_SCORES.get(resolved_tiers[row["hero"]], UNRATED_SCORE)
            for row in hero_rows
        )
        raw_score = weighted_total / count if count else UNRATED_SCORE
        score = (weighted_total + prior_games * UNRATED_SCORE) / (count + prior_games)
        rated_picks = sum(int(row["picks"]) for row in hero_rows if resolved_tiers[row["hero"]] in TIER_SCORES)
        top_tier_picks = sum(int(row["picks"]) for row in hero_rows if resolved_tiers[row["hero"]] in {"S", "A"})
        summaries.append({
            "player": player, "role": normalize_role(role), "games_with_picks": count,
            "series": history.get("series", 0), "missing_pick_games": history.get("missing_pick_games", 0),
            "from": history.get("from"), "through": history.get("through"),
            "raw_tier_score": raw_score, "tier_score": score,
            "rated_picks": rated_picks, "unrated_picks": count - rated_picks,
            "sa_pick_share": top_tier_picks / count if count else None,
            "sample_weight": count / (count + prior_games),
            "heroes": [{**row, "tier": resolved_tiers[row["hero"]]} for row in hero_rows],
        })
    return {
        "tier_score": sum(row["tier_score"] for row in summaries) / 5,
        "raw_tier_score": sum(row["raw_tier_score"] for row in summaries) / 5,
        "players_with_picks": sum(row["games_with_picks"] > 0 for row in summaries),
        "players_with_rated_picks": sum(row["rated_picks"] > 0 for row in summaries),
        "prior_games": prior_games, "players": summaries,
    }


def default_meta_config() -> dict:
    return {
        "season": "Custom season / patch",
        "tiers": {},
        "role_tiers": {},
        "custom_heroes": [],
        "updated_at": None,
    }


def load_meta_config(path: Path) -> dict:
    if not path.exists():
        return default_meta_config()
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = default_meta_config()
    config.update(payload)
    config["tiers"] = {
        str(hero): str(tier).upper()
        for hero, tier in dict(config["tiers"]).items()
        if str(tier).upper() in TIER_SCORES
    }
    clean_role_tiers: dict[str, dict[str, str]] = {}
    for role, assignments in dict(config.get("role_tiers", {})).items():
        canonical_role = normalize_role(str(role))
        if canonical_role is None or not isinstance(assignments, dict):
            continue
        clean_role_tiers[canonical_role] = {
            str(hero): str(tier).upper()
            for hero, tier in assignments.items()
            if str(tier).upper() in TIER_SCORES
        }
    config["role_tiers"] = clean_role_tiers
    config["custom_heroes"] = sorted(
        {" ".join(str(hero).strip().split()) for hero in config["custom_heroes"] if str(hero).strip()},
        key=str.lower,
    )
    return config


def save_meta_config(path: Path, config: dict) -> None:
    payload = default_meta_config()
    payload.update(config)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def draft_tier_score(draft: list[str], tiers: dict[str, str]) -> float:
    if not draft:
        return UNRATED_SCORE
    return sum(TIER_SCORES.get(tiers.get(hero, ""), UNRATED_SCORE) for hero in draft) / len(draft)


def team_comfort_score(team: str, draft: list[str], context: dict) -> float:
    """Return 0..1 comfort relative to the team's most-picked supplied hero."""
    if not draft:
        return 0.5
    hero_rows = context.get("team_heroes", {}).get(team, [])
    counts = {row["hero"]: int(row["picks"]) for row in hero_rows}
    if not counts:
        return 0.5
    maximum = max(counts.values())
    return sum(counts.get(hero, 0) / maximum for hero in draft) / len(draft)


def analyze_matchup(
    model: EloPredictor,
    team_a: str,
    team_b: str,
    draft_a: list[str],
    draft_b: list[str],
    tiers: dict[str, str],
    context: dict,
    *,
    context_team_a: str | None = None,
    context_team_b: str | None = None,
    tier_elo_per_step: float = 25.0,
    comfort_elo_range: float = 40.0,
) -> dict[str, float | bool]:
    """Blend optional draft context into the trained team ratings.

    The base Elo remains untouched. Draft adjustments are applied only when
    both sides have five unique, non-overlapping heroes.
    """
    base_probability = model.predict(team_a, team_b)
    valid_draft = (
        len(draft_a) == 5
        and len(draft_b) == 5
        and len(set(draft_a)) == 5
        and len(set(draft_b)) == 5
        and not set(draft_a).intersection(draft_b)
    )
    result: dict[str, float | bool] = {
        "base_probability_a": base_probability,
        "final_probability_a": base_probability,
        "draft_applied": valid_draft,
        "tier_score_a": UNRATED_SCORE,
        "tier_score_b": UNRATED_SCORE,
        "comfort_score_a": 0.5,
        "comfort_score_b": 0.5,
        "tier_adjustment_a": 0.0,
        "tier_adjustment_b": 0.0,
        "comfort_adjustment_a": 0.0,
        "comfort_adjustment_b": 0.0,
        "total_adjustment_a": 0.0,
        "total_adjustment_b": 0.0,
    }
    if not valid_draft:
        return result

    tier_a = draft_tier_score(draft_a, tiers)
    tier_b = draft_tier_score(draft_b, tiers)
    comfort_a = team_comfort_score(context_team_a or team_a, draft_a, context)
    comfort_b = team_comfort_score(context_team_b or team_b, draft_b, context)
    tier_adjustment_a = (tier_a - UNRATED_SCORE) * tier_elo_per_step
    tier_adjustment_b = (tier_b - UNRATED_SCORE) * tier_elo_per_step
    comfort_adjustment_a = (comfort_a - 0.5) * comfort_elo_range
    comfort_adjustment_b = (comfort_b - 0.5) * comfort_elo_range
    total_a = tier_adjustment_a + comfort_adjustment_a
    total_b = tier_adjustment_b + comfort_adjustment_b
    final_probability = elo_probability(
        model.ratings[team_a] + total_a,
        model.ratings[team_b] + total_b,
        model.scale,
    )
    result.update(
        {
            "final_probability_a": final_probability,
            "tier_score_a": tier_a,
            "tier_score_b": tier_b,
            "comfort_score_a": comfort_a,
            "comfort_score_b": comfort_b,
            "tier_adjustment_a": tier_adjustment_a,
            "tier_adjustment_b": tier_adjustment_b,
            "comfort_adjustment_a": comfort_adjustment_a,
            "comfort_adjustment_b": comfort_adjustment_b,
            "total_adjustment_a": total_a,
            "total_adjustment_b": total_b,
        }
    )
    return result


def analyze_rating_matchup(
    rating_a: float,
    rating_b: float,
    draft_a: list[str],
    draft_b: list[str],
    tiers: dict[str, str],
    context: dict,
    *,
    context_team_a: str,
    context_team_b: str,
    scale: float = 400.0,
    tier_elo_per_step: float = 25.0,
    comfort_elo_range: float = 40.0,
    automatic_profile_a: dict | None = None,
    automatic_profile_b: dict | None = None,
    automatic_meta_enabled: bool = True,
) -> dict:
    """Use a complete draft, otherwise optional automatic current-player meta fit.

    Manual drafts replace (never stack with) the automatic tier estimate.
    Without profiles this retains the original unadjusted-Elo behavior.
    """
    valid_draft = (
        len(draft_a) == 5
        and len(draft_b) == 5
        and len(set(draft_a)) == 5
        and len(set(draft_b)) == 5
        and not set(draft_a).intersection(draft_b)
    )
    base_probability = elo_probability(rating_a, rating_b, scale)
    result: dict = {
        "base_probability_a": base_probability,
        "final_probability_a": base_probability,
        "draft_applied": valid_draft,
        "automatic_meta_applied": False,
        "adjustment_mode": "draft" if valid_draft else "base",
        "tier_score_a": UNRATED_SCORE,
        "tier_score_b": UNRATED_SCORE,
        "comfort_score_a": 0.5,
        "comfort_score_b": 0.5,
        "tier_adjustment_a": 0.0,
        "tier_adjustment_b": 0.0,
        "comfort_adjustment_a": 0.0,
        "comfort_adjustment_b": 0.0,
        "total_adjustment_a": 0.0,
        "total_adjustment_b": 0.0,
    }
    if not valid_draft:
        if automatic_meta_enabled and automatic_profile_a is not None and automatic_profile_b is not None:
            tier_a = float(automatic_profile_a["tier_score"])
            tier_b = float(automatic_profile_b["tier_score"])
            total_a = (tier_a - UNRATED_SCORE) * tier_elo_per_step
            total_b = (tier_b - UNRATED_SCORE) * tier_elo_per_step
            result.update(
                automatic_meta_applied=True, adjustment_mode="automatic",
                tier_score_a=tier_a, tier_score_b=tier_b,
                tier_adjustment_a=total_a, tier_adjustment_b=total_b,
                total_adjustment_a=total_a, total_adjustment_b=total_b,
                final_probability_a=elo_probability(rating_a + total_a, rating_b + total_b, scale),
            )
        return result

    tier_a = draft_tier_score(draft_a, tiers)
    tier_b = draft_tier_score(draft_b, tiers)
    comfort_a = team_comfort_score(context_team_a, draft_a, context)
    comfort_b = team_comfort_score(context_team_b, draft_b, context)
    tier_adjustment_a = (tier_a - UNRATED_SCORE) * tier_elo_per_step
    tier_adjustment_b = (tier_b - UNRATED_SCORE) * tier_elo_per_step
    comfort_adjustment_a = (comfort_a - 0.5) * comfort_elo_range
    comfort_adjustment_b = (comfort_b - 0.5) * comfort_elo_range
    total_a = tier_adjustment_a + comfort_adjustment_a
    total_b = tier_adjustment_b + comfort_adjustment_b
    result.update(
        {
            "final_probability_a": elo_probability(
                rating_a + total_a, rating_b + total_b, scale
            ),
            "tier_score_a": tier_a,
            "tier_score_b": tier_b,
            "comfort_score_a": comfort_a,
            "comfort_score_b": comfort_b,
            "tier_adjustment_a": tier_adjustment_a,
            "tier_adjustment_b": tier_adjustment_b,
            "comfort_adjustment_a": comfort_adjustment_a,
            "comfort_adjustment_b": comfort_adjustment_b,
            "total_adjustment_a": total_a,
            "total_adjustment_b": total_b,
        }
    )
    return result
