"""Offline presentation checks. These do not substitute for browser visual QA."""

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from mlbb_predictor.ui import (
    data_strip_html, display_text, forecast_html, masthead_html,
    roster_html, safe_text, tier_summary_html,
)


ROOT = Path(__file__).resolve().parents[1]


class ParsedMarkup(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.tags = []
        self.attributes = []
        self.content = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.append(dict(attrs))

    def handle_data(self, data):
        self.content.append(data)


def luminance(color):
    rgb = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in rgb]
    return sum(weight * value for weight, value in zip((.2126, .7152, .0722), linear))


def contrast(first, second):
    values = sorted((luminance(first), luminance(second)))
    return (values[1] + .05) / (values[0] + .05)


class PresentationTests(unittest.TestCase):
    def forecast(self, **changes):
        defaults = dict(team_a="TLPH", team_b="RORA", name_a="Team Liquid PH",
                        name_b="Aurora Gaming PH", probability_a=.63, base_a=.60,
                        label="Automatic meta favorite", adjusted=True,
                        rating_a=1550, rating_b=1500, known_a=5, known_b=4)
        return forecast_html(**(defaults | changes))

    def test_forecast_renders_both_sides_baselines_and_real_delta(self):
        parsed = ParsedMarkup(self.forecast())
        text = ' '.join(parsed.content)
        for expected in ("63.0", "37.0", "+3.0 pp", "-3.0 pp", "60.0%", "40.0%",
                         "4 of 5 starters", "Team Liquid PH favored", "one game"):
            self.assertIn(expected, text)
        self.assertEqual(self.forecast().count('class="matchdesk-chance"'), 2)
        self.assertIn('aria-live="polite"', self.forecast())

    def test_tie_does_not_invent_a_favorite(self):
        markup = self.forecast(probability_a=.5)
        self.assertIn("Even matchup", markup)
        self.assertNotIn("is-leading", markup)
        self.assertNotIn("PH favored", markup)

    def test_base_only_removes_meta_delta_label(self):
        markup = self.forecast(adjusted=False)
        self.assertNotIn("pp with meta", markup)
        self.assertEqual(markup.count("Player Elo only"), 2)

    def test_zero_and_one_probability_are_displayable(self):
        for probability in (0.0, 1.0):
            parsed = ParsedMarkup(self.forecast(probability_a=probability))
            self.assertIn("100.0", parsed.content)
            self.assertIn("0.0", parsed.content)
            self.assertEqual(self.forecast(probability_a=probability).count("is-leading"), 1)

    def test_imported_names_and_labels_are_escaped(self):
        injection = '<img src=x onerror="alert(1)">'
        markup = self.forecast(name_a=injection, team_a=injection, label=injection)
        self.assertNotIn("img", ParsedMarkup(markup).tags)
        self.assertNotIn("script", ParsedMarkup(markup).tags)
        roster = roster_html([{"role": injection, "player": injection}], team=injection)
        self.assertNotIn("img", ParsedMarkup(roster).tags)
        self.assertIn("&lt;img", roster)

    def test_header_and_saved_tier_counts_use_supplied_values(self):
        strip = data_strip_html(season="Season 18", teams=8, games=198, cutoff="2026-08-29")
        self.assertIn("198", strip)
        self.assertIn("2026-08-29", strip)
        self.assertEqual(masthead_html().count("<h1>"), 1)
        markup = tier_summary_html({"Claude": "S", "Ruby": "S", "Akai": "A"}, ["S", "A", "B", "C", "D", "F"])
        self.assertEqual(markup.count("<dt>"), 6)
        self.assertIn("<dd>2<span> heroes", markup)
        self.assertEqual(markup.count("<dd>0<span> heroes"), 4)

    def test_source_punctuation_normalized_only_for_display(self):
        original = "Season 17\u201318 \u2014 source"
        self.assertEqual(display_text(original), "Season 17-18 : source")
        self.assertIn("\u2014", original)
        self.assertEqual(safe_text('A&B'), 'A&amp;B')

    def test_both_theme_palettes_have_readable_text_and_buttons(self):
        css = (ROOT / "ui/matchdesk.css").read_text(encoding="utf-8")
        tokens = {
            name: (light, dark)
            for name, light, dark in re.findall(r"--desk-([\w-]+): light-dark\((#[\da-f]{6}), (#[\da-f]{6})\)", css)
        }
        for index, mode in enumerate(("light", "dark")):
            for text in ("ink", "muted", "accent"):
                for surface in ("paper", "surface"):
                    with self.subTest(mode=mode, text=text, surface=surface):
                        self.assertGreaterEqual(contrast(tokens[text][index], tokens[surface][index]), 4.5)
            self.assertGreaterEqual(contrast(tokens["accent-ink"][index], tokens["accent"][index]), 4.5)

    def test_responsive_and_reduced_motion_rules_are_present(self):
        css = (ROOT / "ui/matchdesk.css").read_text(encoding="utf-8")
        for rule in ("@media (max-width: 640px)", "@media (max-width: 900px)",
                     "prefers-reduced-motion: reduce", ":focus-visible", "overflow-wrap: anywhere"):
            self.assertIn(rule, css)
        self.assertNotIn("@import", css)
        self.assertNotIn("url(", css)
        self.assertNotIn("linear-gradient", css)
        self.assertNotIn("probability-track", css)

    def test_native_form_borders_and_text_contrast_in_both_themes(self):
        config = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
        for mode in ("light", "dark"):
            section = config.split(f"[theme.{mode}]", 1)[1].split("[", 1)[0]
            values = dict(re.findall(r'(\w+) = "(#[\da-f]{6})"', section))
            with self.subTest(mode=mode):
                self.assertGreaterEqual(contrast(values["borderColor"], values["secondaryBackgroundColor"]), 3)
                self.assertGreaterEqual(contrast(values["textColor"], values["secondaryBackgroundColor"]), 4.5)
                self.assertGreaterEqual(contrast(values["linkColor"], values["backgroundColor"]), 4.5)

    def test_tabs_use_native_roles_and_keep_every_destination_visible(self):
        css = (ROOT / "ui/matchdesk.css").read_text(encoding="utf-8")
        self.assertNotIn('[data-baseweb="tab-list"]', css)
        self.assertNotIn('button[role="tab"]', css)
        tab_list = re.search(r'\[data-testid="stTabs"\] \[role="tablist"\] \{([^}]+)\}', css).group(1)
        for declaration in ("flex-wrap: wrap", "overflow: visible", "background: var(--desk-surface)"):
            self.assertIn(declaration, tab_list)
        active = re.search(r'\[role="tab"\]\[aria-selected="true"\] \{([^}]+)\}', css).group(1)
        self.assertIn("background: var(--desk-accent)", active)
        self.assertIn("color: var(--desk-accent-ink)", active)
        self.assertIn("box-shadow:", active)
        mobile = css.split("@media (max-width: 640px)", 1)[1]
        self.assertIn('[role="tablist"] { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr))', mobile)
        self.assertIn("white-space: normal", mobile)

    def test_tab_boundaries_have_non_text_contrast_in_both_themes(self):
        css = (ROOT / "ui/matchdesk.css").read_text(encoding="utf-8")
        tokens = dict((name, (light, dark)) for name, light, dark in re.findall(
            r"--desk-([\w-]+): light-dark\((#[\da-f]{6}), (#[\da-f]{6})\)", css))
        for index in (0, 1):
            for background in ("paper", "surface"):
                self.assertGreaterEqual(contrast(tokens["control-line"][index], tokens[background][index]), 3)


if __name__ == "__main__":
    unittest.main()
