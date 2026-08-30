"""Presentation-only HTML for the match desk. No prediction or data writes.

Native Streamlit controls own all interaction. Every interpolated source string
is escaped here; imported names and season labels are never trusted as markup.
"""

from __future__ import annotations

from html import escape
from math import isclose


def display_text(value: object) -> str:
    """Normalize presentation punctuation without modifying stored records."""
    return str(value).replace("\u2014", ":").replace("\u2013", "-")


def safe_text(value: object) -> str:
    return escape(display_text(value), quote=True)


def masthead_html() -> str:
    return (
        '<header class="matchdesk-masthead">'
        '<p class="matchdesk-league">MPL Philippines</p>'
        '<h1>MLBB Match Predictor</h1>'
        '<p class="matchdesk-intro">Current players. Recent hero pools. A clearer view of the matchup.</p>'
        '</header>'
    )


def data_strip_html(*, season: str, teams: int, games: int, cutoff: str) -> str:
    facts = (
        ("Competition", season),
        ("Teams", str(teams)),
        ("Verified Elo games", str(games)),
        ("Results through", cutoff),
    )
    return '<dl class="matchdesk-data-strip">' + ''.join(
        f'<div><dt>{safe_text(label)}</dt><dd>{safe_text(value)}</dd></div>'
        for label, value in facts
    ) + '</dl>'


def forecast_html(
    *, team_a: str, team_b: str, name_a: str, name_b: str,
    probability_a: float, base_a: float, label: str, adjusted: bool,
    rating_a: float, rating_b: float, known_a: int, known_b: int,
) -> str:
    """Display the model output directly, including the base/adjusted delta."""
    probability_b = 1.0 - probability_a
    tied = isclose(probability_a, 0.5, abs_tol=1e-9)
    favorite = name_a if probability_a > 0.5 else name_b
    verdict = "Even matchup" if tied else f"{favorite} favored"
    change = (probability_a - base_a) * 100
    sides = []
    for code, name, probability, base, delta, rating, known in (
        (team_a, name_a, probability_a, base_a, change, rating_a, known_a),
        (team_b, name_b, probability_b, 1.0 - base_a, -change, rating_b, known_b),
    ):
        leading = probability > 0.5 and not tied
        shift = f"{delta:+.1f} pp with meta" if adjusted else "Player Elo only"
        sides.append(
            f'<section class="matchdesk-side{" is-leading" if leading else ""}">'
            f'<div class="matchdesk-teamcode">{safe_text(code)}</div>'
            f'<h3>{safe_text(name)}</h3>'
            f'<div class="matchdesk-chance">{probability * 100:.1f}<span>%</span></div>'
            f'<p class="matchdesk-shift">{shift}</p>'
            f'<div class="matchdesk-baseline"><span>Player-Elo baseline</span><strong>{base:.1%}</strong></div>'
            f'<div class="matchdesk-baseline"><span>Starting-five Elo</span><strong>{rating:.0f}</strong></div>'
            f'<p class="matchdesk-coverage">{known} of 5 starters with verified game data</p>'
            '</section>'
        )
    return (
        '<section class="matchdesk-forecast" aria-label="Match forecast" aria-live="polite" aria-atomic="true">'
        '<div class="matchdesk-forecast-heading">'
        f'<h2>{safe_text(verdict)}</h2><p>{safe_text(label)}</p></div>'
        '<div class="matchdesk-scoreboard">'
        + sides[0] + '<div class="matchdesk-versus" aria-hidden="true">VS</div>' + sides[1]
        + '</div><div class="matchdesk-forecast-foot">'
        '<span>Estimated chance to win <strong>one game</strong></span>'
        '<span>Estimate, not a guarantee</span></div></section>'
    )


def roster_html(starters: list[dict], *, team: str) -> str:
    return f'<ul class="matchdesk-roster" aria-label="{safe_text(team)} current starters">' + ''.join(
        f'<li><span>{safe_text(member["role"])}</span><strong>{safe_text(member["player"])}</strong></li>'
        for member in starters
    ) + '</ul>'


def tier_summary_html(tiers: dict[str, str], order: tuple | list) -> str:
    return '<dl class="matchdesk-tier-summary" aria-label="Saved hero tier counts">' + ''.join(
        f'<div><dt>{safe_text(tier)} <span>tier</span></dt>'
        f'<dd>{sum(value == tier for value in tiers.values())}<span> heroes</span></dd></div>'
        for tier in order
    ) + '</dl>'
