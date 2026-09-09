"""Streamlit UI for MLBB team, draft, meta, and player-pick analysis."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from mlbb_predictor.context import read_context_data
from mlbb_predictor.collector import load_settings, read_json, save_settings, status_path, utc_now
from mlbb_predictor.collection_service import CollectionService, next_check
from mlbb_predictor.startup import startup_notice
from mlbb_predictor.profile_collection import profile_status_path
from mlbb_predictor.player_model import player_key
from mlbb_predictor.data import display_name
from mlbb_predictor.history import season_player_picks, season_player_pick_records, season_team_picks, matches_by_week, season_games, series_key, valid_lineups, valid_picks
from mlbb_predictor.live_data import data_revision, load_prediction_state
from mlbb_predictor.match_schedule import collection_due, next_collection, schedule_path
from mlbb_predictor.meta import (
    TIER_ORDER,
    analyze_rating_matchup,
    hero_tier,
    load_meta_config,
    roster_meta_profile,
    save_meta_config,
)
from mlbb_predictor.teams import (
    load_team_profiles,
    current_team_profiles,
    profiles_by_code,
    rating_lineup,
    starting_roster,
)
from mlbb_predictor.ui import (
    data_strip_html,
    display_text,
    forecast_html,
    masthead_html,
    roster_html,
    safe_text,
    tier_summary_html,
)


ROOT = Path(__file__).resolve().parent
PLAYER_MODEL_PATH = ROOT / "models" / "player_elo_model.json"
GAME_PATHS = [
    ROOT / "data/processed/mpl_ph_s17_player_games.json",
    ROOT / "data/processed/mpl_ph_s18_player_games.json",
]
CONTEXT_PATH = ROOT / "data" / "processed" / "context.json"
META_PATH = ROOT / "config" / "meta_tiers.json"
TEAM_PROFILES_PATH = ROOT / "config" / "mpl_ph_teams.json"
OFFLINE_TEST_MODE = os.environ.get("MLBB_OFFLINE_TEST_MODE") == "1"

st.set_page_config(page_title="MLBB Match Predictor", page_icon="⚔️", layout="wide")

@st.cache_resource(max_entries=4)
def load_current_prediction_data(modified_at: tuple, offline: bool) -> tuple:
    del modified_at
    return load_prediction_state(ROOT, offline=offline)


@st.cache_resource
def get_collection_service() -> CollectionService:
    return CollectionService(ROOT)


@st.cache_data
def load_context(modified_at: float) -> dict:
    del modified_at
    return read_context_data(CONTEXT_PATH)


def matchup_history(matches: list[dict[str, str]], team_a: str, team_b: str) -> tuple[int, int, int]:
    relevant = [
        row for row in matches if {row["team_a"], row["team_b"]} == {team_a, team_b}
    ]
    wins_a = sum(row["winner"] == team_a for row in relevant)
    wins_b = sum(row["winner"] == team_b for row in relevant)
    return len(relevant), wins_a, wins_b


def team_option(team: str) -> str:
    return f"{display_name(team)}  ·  {team}"


def top_team_heroes(team_picks: dict[str, list[dict]], team: str, count: int = 5) -> list[str]:
    return [row["hero"] for row in team_picks.get(team, [])[:count]]


def signed_rating(value: float) -> str:
    return f"{value:+.0f}"


st.markdown(masthead_html(), unsafe_allow_html=True)

required_artifacts = (
    PLAYER_MODEL_PATH,
    *GAME_PATHS,
    CONTEXT_PATH,
    TEAM_PROFILES_PATH,
    META_PATH,
    ROOT / "ui" / "matchdesk.css",
)
missing_artifacts = [path for path in required_artifacts if not path.is_file()]
if missing_artifacts:
    st.error("This deployment is missing required model, data, or configuration files.")
    st.write("Missing files, relative to app.py:")
    st.code("\n".join(path.relative_to(ROOT).as_posix() for path in missing_artifacts), language="text")
    st.info(
        "On Streamlit Community Cloud, these files must be committed to the GitHub repository, "
        "not just saved on your PC. Upload them at the paths above, push to the deployed branch, "
        "then reboot the app. The README has the deployment commands. "
        "If the saved files already exist locally, you do not need to download datasets or retrain."
    )
    with st.expander("Only if the saved files are also missing on your PC"):
        st.write("Run the data-preparation commands locally, then commit the generated model and data files. "
                 "These commands require the external sources to be available. Restore configuration and UI files from the project.")
        st.code("python download_data.py\npython download_player_data.py\npython download_recent_data.py\npython train_model.py", language="bash")
    st.stop()

st.html(ROOT / "ui" / "matchdesk.css")

collection_service = None if OFFLINE_TEST_MODE else get_collection_service()
if collection_service is not None and not getattr(collection_service, "profile_updates_enabled", False):
    # Replace a cached match-only worker during Streamlit's hot reload.
    collection_service.stop()
    get_collection_service.clear()
    collection_service = get_collection_service()
# A new browser session checks once. Widget reruns and fragment refreshes must
# not download again. The shared service coalesces concurrent visitors.
if collection_service is not None and not st.session_state.get("opening_data_checked", False):
    with st.spinner("Checking the match schedule and loading data...", show_time=True):
        opening_result = collection_service.refresh_on_open()
    st.session_state["opening_data_checked"] = True
    notice_kind, notice_text = startup_notice(opening_result)
    getattr(st, notice_kind)(notice_text)

# Read the revision only after the opening check, so a newly published model
# and its matching history are used together on the first prediction.
loaded_revision = data_revision(ROOT, offline=OFFLINE_TEST_MODE)
try:
    player_model, metrics, player_game_payload, player_games = load_current_prediction_data(loaded_revision, OFFLINE_TEST_MODE)
except (ValueError, KeyError, OSError) as error:
    st.error(f"Could not load a consistent prediction snapshot: {error}. Check the data files before continuing.")
    st.stop()
if sum(valid_lineups(row) for row in player_games) != metrics["match_count"]:
    st.warning("Game data changed since the model was trained. Run `python train_model.py` to update player Elo.")
context = load_context(CONTEXT_PATH.stat().st_mtime)
team_profile_payload = current_team_profiles(ROOT, player_games, offline=OFFLINE_TEST_MODE)
team_profiles = team_profile_payload["teams"]
team_profile_by_code = profiles_by_code(team_profile_payload)
recent_payload = season_team_picks(player_games, set(team_profile_by_code))
recent_picks: dict[str, list[dict]] = recent_payload["teams"]
recent_context = {**context, "team_heroes": recent_picks}
meta_config = load_meta_config(META_PATH)
tiers: dict[str, str] = dict(meta_config["tiers"])
role_tiers: dict[str, dict[str, str]] = dict(meta_config["role_tiers"])
all_heroes = sorted(
    set(context["hero_pool"])
    | {
        hero
        for row in player_games if valid_picks(row)
        for side in ("a", "b") for hero in row.get(f"heroes_{side}", [])
    }
    | set(meta_config["custom_heroes"])
    | set(tiers)
    | {hero for assignments in role_tiers.values() for hero in assignments},
    key=str.lower,
)
teams = [profile["code"] for profile in team_profiles]

st.markdown(data_strip_html(
    season=team_profile_payload["season"], teams=len(teams),
    games=int(metrics["match_count"]), cutoff=player_game_payload["cutoff"],
), unsafe_allow_html=True)

predict_tab, players_tab, meta_tab, history_tab, collection_tab = st.tabs(
    ["Predict", "MPL PH team profiles", "Season meta tiers", "Collected matches", "Data collection"]
)

with predict_tab:
    with st.container(key="matchup_controls"):
        left, spacer, right = st.columns([1, 0.12, 1])
        with left:
            team_a = st.selectbox(
                "Team 1",
                teams,
                index=teams.index("TLPH") if "TLPH" in teams else 0,
                format_func=team_option,
                key="predict_team_a",
            )
        spacer.markdown('<div class="matchdesk-selector-versus" aria-hidden="true">VS</div>', unsafe_allow_html=True)
        with right:
            default_b = "FNOP" if "FNOP" in teams else teams[min(1, len(teams) - 1)]
            team_b = st.selectbox(
                "Team 2",
                teams,
                index=teams.index(default_b),
                format_func=team_option,
                key="predict_team_b",
            )

    if team_a == team_b:
        st.warning("Choose two different teams to make a matchup.")
    elif set(map(player_key, rating_lineup(team_profile_by_code[team_a]))) & set(map(player_key, rating_lineup(team_profile_by_code[team_b]))):
        st.warning("These prediction lineups overlap after a roster change. A complete verified lineup is needed before predicting this matchup.")
    else:
        profile_a = team_profile_by_code[team_a]
        profile_b = team_profile_by_code[team_b]
        starters_a = starting_roster(profile_a)
        starters_b = starting_roster(profile_b)
        rating_names_a = rating_lineup(profile_a)
        rating_names_b = rating_lineup(profile_b)
        lineup_a = player_model.lineup_summary(rating_names_a)
        lineup_b = player_model.lineup_summary(rating_names_b)
        player_pools = season_player_picks(player_games, rating_names_a + rating_names_b)
        automatic_a = roster_meta_profile(
            rating_names_a, player_pools, tiers,
            roles=[member["role"] for member in starters_a], role_tiers=role_tiers,
        )
        automatic_b = roster_meta_profile(
            rating_names_b, player_pools, tiers,
            roles=[member["role"] for member in starters_b], role_tiers=role_tiers,
        )

        for profile, starters, lineup in (
            (profile_a, starters_a, lineup_a),
            (profile_b, starters_b, lineup_b),
        ):
            if profile.get("lineup_warning"):
                st.warning(f"{profile['name']}: {profile['lineup_warning']}")
            unknown = [
                starters[index]["player"]
                for index, row in enumerate(lineup["players"])
                if not row["known"]
            ]
            if unknown:
                st.warning(
                    f"{profile['name']}: {', '.join(unknown)} have no verified games in the loaded history, "
                    "so each starts at neutral 1500 player Elo."
                )

        # Reserve the forecast above its optional inputs. Streamlit still reads
        # every native control before calculating and filling this container.
        forecast_slot = st.container(key="forecast_output")
        with st.expander("Automatic meta and optional drafts", expanded=False):
            automatic_enabled = st.toggle(
                "Automatically adjust for current players' meta heroes",
                value=True, key="automatic_meta_enabled",
                help="Use the current five starters' recent hero frequencies and your saved tiers, without entering a draft. Complete drafts replace this estimate.",
            )
            st.caption(
                f"Active tier list: {meta_config['season']} · {len(tiers)} rated heroes. "
                "Automatic mode uses each starter's lane and collected Season 18 games, including games for previous teams. "
                "More frequently played heroes contribute more. No draft input is needed."
            )
            st.caption("Optional: select five unique, non-overlapping heroes per side to replace the automatic estimate with that actual draft.")
            draft_left, draft_right = st.columns(2)
            with draft_left:
                draft_a = st.multiselect(
                    f"{display_name(team_a)} draft",
                    all_heroes,
                    max_selections=5,
                    format_func=lambda hero: f"{hero} · {hero_tier(hero, tiers)}",
                    key="draft_a",
                )
                suggested_a = top_team_heroes(recent_picks, team_a)
                if suggested_a:
                    st.caption("Season 18 favorite picks: " + ", ".join(suggested_a))
            with draft_right:
                draft_b = st.multiselect(
                    f"{display_name(team_b)} draft",
                    all_heroes,
                    max_selections=5,
                    format_func=lambda hero: f"{hero} · {hero_tier(hero, tiers)}",
                    key="draft_b",
                )
                suggested_b = top_team_heroes(recent_picks, team_b)
                if suggested_b:
                    st.caption("Season 18 favorite picks: " + ", ".join(suggested_b))

            for code in (team_a, team_b):
                coverage = recent_payload["coverage"][code]
                st.caption(
                    f"{display_name(code)}: {coverage['series']} series, "
                    f"{coverage['games_with_picks']}/{coverage['games']} games with usable picks · "
                    f"{coverage['from']} to {coverage['through']}."
                )
                if coverage["missing_pick_games"]:
                    st.warning(f"{display_name(code)}: {len(coverage['missing_pick_games'])} incomplete or inconsistent drafts are excluded from favorite-pick counts.")
            st.caption("A match means a completed series, including all its games. Only Season 18 games count toward favorite picks and meta fit.")

            with st.expander("Adjustment strength"):
                weight_left, weight_right = st.columns(2)
                tier_weight = weight_left.slider(
                    "Elo points per tier step",
                    min_value=0,
                    max_value=50,
                    value=25,
                    key="tier_elo_weight",
                    help="How much one average tier grade changes the matchup rating in automatic or complete-draft mode. This is a manual heuristic weight.",
                )
                comfort_weight = weight_right.slider(
                    "Season 18 comfort range",
                    min_value=0,
                    max_value=80,
                    value=40,
                    key="comfort_elo_weight",
                    help="Only for complete manual drafts. Automatic mode already weights heroes by pick frequency and does not add a second comfort bonus.",
                )

            repeated_heroes = set(draft_a).intersection(draft_b)
            if repeated_heroes:
                repeated = ", ".join(sorted(repeated_heroes))
                fallback = "Automatic meta remains active" if automatic_enabled else "Player Elo remains active"
                st.warning(f"The same hero cannot be on both sides: {repeated}. {fallback}; the invalid draft is not applied.")
            elif draft_a or draft_b:
                if len(draft_a) != 5 or len(draft_b) != 5:
                    fallback = "Automatic meta" if automatic_enabled else "Current-roster player Elo"
                    st.info(f"{fallback} remains active until both drafts contain five unique, non-overlapping heroes.")

        analysis = analyze_rating_matchup(
            float(lineup_a["rating"]),
            float(lineup_b["rating"]),
            draft_a,
            draft_b,
            tiers,
            recent_context,
            context_team_a=team_a,
            context_team_b=team_b,
            scale=player_model.scale,
            tier_elo_per_step=float(tier_weight),
            comfort_elo_range=float(comfort_weight),
            automatic_profile_a=automatic_a,
            automatic_profile_b=automatic_b,
            automatic_meta_enabled=automatic_enabled,
        )
        probability_a = float(analysis["final_probability_a"])
        prediction_label = {
            "draft": "Meta-adjusted favorite · selected draft",
            "automatic": "Automatic meta favorite",
            "base": "Current-roster favorite",
        }[analysis["adjustment_mode"]]

        base_a = float(analysis["base_probability_a"])
        adjusted = analysis["draft_applied"] or analysis["automatic_meta_applied"]
        with forecast_slot:
            st.markdown(forecast_html(
                team_a=team_a, team_b=team_b,
                name_a=display_name(team_a), name_b=display_name(team_b),
                probability_a=probability_a, base_a=base_a,
                label=prediction_label, adjusted=adjusted,
                rating_a=float(lineup_a["rating"]), rating_b=float(lineup_b["rating"]),
                known_a=int(lineup_a["known_players"]), known_b=int(lineup_b["known_players"]),
            ), unsafe_allow_html=True)
            if adjusted:
                method = "the selected draft" if analysis["draft_applied"] else "current players' Season 18 hero pools"
                st.caption(
                    f"Meta fit uses {method}. pp = percentage points versus player Elo. "
                    "This adjustment is a heuristic, not a validated increase in win rate."
                )

        if analysis["automatic_meta_applied"]:
            st.subheader("Automatic meta contribution")
            st.dataframe([
                {
                    "Team": display_name(code),
                    "Weighted meta score": f"{profile['tier_score']:.2f} / 5",
                    "Starters with rated picks": f"{profile['players_with_rated_picks']} / 5",
                    "Automatic meta Elo": f"{analysis[f'total_adjustment_{side}']:+.1f}",
                }
                for code, profile, side in ((team_a, automatic_a, "a"), (team_b, automatic_b, "b"))
            ], hide_index=True, width="stretch")
            st.caption(
                "Each current starter contributes equally. Their heroes are weighted by actual pick count across their collected Season 18 games; "
                "five neutral pseudo-games per player soften small samples. Unrated heroes and missing history are neutral. "
                "The teams' meta-adjusted ratings determine the percentage change. There is no fixed win-percent bonus per S-tier hero."
            )
            for code, profile in ((team_a, automatic_a), (team_b, automatic_b)):
                no_data = [row["player"] for row in profile["players"] if not row["rated_picks"]]
                if no_data:
                    st.warning(f"{display_name(code)}: {', '.join(no_data)} have no usable rated hero history; their automatic meta contribution is neutral.")
            with st.expander("Which players and heroes drive the automatic adjustment?"):
                st.caption(f"Season 18 only · collected results through {player_game_payload['cutoff']}. "
                           "Both this table and automatic meta scoring count every usable pick this season; S17 and S13 picks are excluded.")
                evidence_rows = []
                for code, profile, starters in ((team_a, automatic_a, starters_a), (team_b, automatic_b, starters_b)):
                    for member, player in zip(starters, profile["players"]):
                        picks = ", ".join(f"{row['hero']} ({row['tier']}, {row['picks']}×)" for row in player["heroes"][:3])
                        evidence_rows.append({
                            "Team": display_name(code), "Player": member["player"], "Role": member["role"],
                            "Most-played heroes · S18": picks or "No usable Season 18 picks · neutral",
                            "Series": player["series"], "Usable games": player["games_with_picks"],
                            "Excluded games": player["missing_pick_games"], "Unrated picks": player["unrated_picks"],
                            "S/A pick share": f"{player['sa_pick_share']:.0%}" if player["sa_pick_share"] is not None else "No data",
                            "Meta Elo share": f"{(player['tier_score'] - 2.5) * tier_weight / 5:+.1f}",
                            "Season data from": player["from"], "Last recorded series": player["through"],
                        })
                st.dataframe(evidence_rows, hide_index=True, width="stretch")
                st.caption("Top three Season 18 heroes are displayed; scoring uses the full Season 18 pool. "
                           "Counts describe this season's played heroes, not a prediction of the next draft. "
                           "Bans, invalid drafts and incomplete player mappings are excluded. No older-season picks are used as a fallback.")
                audit_members = {member.get("history_alias", member["player"]): member["player"]
                                 for member in starters_a + starters_b}
                audit_player = st.selectbox("Verify a player's Season 18 picks", list(audit_members),
                                            format_func=audit_members.get, key="meta_pick_audit_player")
                audit_rows = season_player_pick_records(player_games, audit_player)
                st.dataframe(audit_rows, hide_index=True, width="stretch",
                             column_config={"Source": st.column_config.LinkColumn("Match source")})
                st.caption(f"{len(audit_rows)} verified played-hero appearances for {audit_members[audit_player]} this season. "
                           "Each row contributes exactly one pick to the counts and meta evidence above.")
        elif analysis["draft_applied"]:
            st.caption("Selected-draft mode: the complete draft replaces automatic meta fit. The two adjustments are never added together.")

        st.subheader("Why the model leans this way")
        st.caption(f"Current starting fives, configured {team_profile_payload['as_of']}. Ratings follow players, not the team name.")
        stats_a, stats_b = st.columns(2)
        for column, lineup, starters, profile in (
            (stats_a, lineup_a, starters_a, profile_a),
            (stats_b, lineup_b, starters_b, profile_b),
        ):
            with column:
                st.markdown(f"**{profile['name']}**")
                st.markdown(roster_html(starters, team=profile["name"]), unsafe_allow_html=True)
                player_rows = []
                for member, player in zip(starters, lineup["players"]):
                    player_rows.append(
                        {
                            "Player": member["player"],
                            "Role": member["role"],
                            "Player Elo": f"{float(player['rating']):.0f}",
                            "Recorded games W-L": (
                                f"{player['wins']}-{player['losses']}"
                                if player["known"]
                                else "No data · neutral"
                            ),
                        }
                    )
                with st.expander("Player ratings and game records"):
                    st.dataframe(player_rows, hide_index=True, width="stretch")
                standing = profile["standing"]
                st.caption(
                    f"Collected standings ({team_profile_payload['standings_as_of']}): #{standing['position']} · series "
                    f"{standing['matches_won']}-{standing['matches_lost']} · games "
                    f"{standing['games_won']}-{standing['games_lost']}"
                )

        if analysis["draft_applied"]:
            adjustment_rows = [
                {
                    "Team": display_name(team_a),
                    "Average tier score": f"{float(analysis['tier_score_a']):.2f} / 5",
                    "Season 18 comfort": f"{float(analysis['comfort_score_a']):.0%}",
                    "Tier rating change": signed_rating(float(analysis["tier_adjustment_a"])),
                    "Comfort rating change": signed_rating(float(analysis["comfort_adjustment_a"])),
                    "Total draft change": signed_rating(float(analysis["total_adjustment_a"])),
                },
                {
                    "Team": display_name(team_b),
                    "Average tier score": f"{float(analysis['tier_score_b']):.2f} / 5",
                    "Season 18 comfort": f"{float(analysis['comfort_score_b']):.0%}",
                    "Tier rating change": signed_rating(float(analysis["tier_adjustment_b"])),
                    "Comfort rating change": signed_rating(float(analysis["comfort_adjustment_b"])),
                    "Total draft change": signed_rating(float(analysis["total_adjustment_b"])),
                },
            ]
            st.dataframe(adjustment_rows, hide_index=True, width="stretch")
            st.caption(
                "Tier strength is your input. Comfort is each selected hero's recent pick count relative to that team's most-picked hero in its collected Season 18 series."
            )

        h2h_games, h2h_a, h2h_b = matchup_history(player_games, team_a, team_b)
        if h2h_games:
            st.info(
                f"Loaded-history game head-to-head: {display_name(team_a)} {h2h_a}-{h2h_b} "
                f"{display_name(team_b)} across {h2h_games} game{'s' if h2h_games != 1 else ''}. "
                "Context only; player Elo uses the subset with verified ten-player lineups."
            )
        else:
            st.info(
                "No head-to-head result is loaded; the prediction still follows each current starter's learned rating."
            )

        with st.expander("Model quality, data, and limitations"):
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Games", int(metrics["match_count"]))
            q2.metric("Rated players", int(metrics["player_count"]))
            q3.metric("Validation accuracy", f"{float(metrics['validation_accuracy']):.1%}")
            q4.metric("Validation log loss", f"{float(metrics['validation_log_loss']):.3f}")
            st.markdown(
                f"""
                - **Player-Elo source:** {int(metrics['match_count'])} games with actual ten-player lineups from S17 and the collected S18 results, through {metrics.get('rating_cutoff', player_game_payload['cutoff'])}.
                - **How transfers work:** the team score is the mean Elo of its five current starters. A rating follows a player to a new team; it does not stay with the franchise.
                - **New players:** anyone without a verified game starts at neutral 1500. That uncertainty is shown above instead of borrowing a departed player's rating.
                - **Validated part:** K-factor selection and the displayed validation metrics use a chronological final-25% holdout of Season 17 games.
                - **S18 updates:** the S17-selected K-factor stays fixed; {metrics.get('update_game_count', 0)} new games update only the actual participants. {metrics.get('excluded_lineup_games', 0)} games with incomplete lineups do not update Elo.
                - **Meta heuristic:** enabled automatically by default. Each current player's collected Season 18 games provides their pick-weighted tier score, softened toward neutral by five pseudo-games. The five starters contribute equally; no departed player's pool is inherited. These weights are not learned or calibrated win-rate effects.
                - **Draft override:** a complete draft replaces automatic fit and can add Season 18 team comfort. Both modes use your saved tier list; neither changes stored player Elo. Season totals can include earlier patches and rosters within Season 18.
                - **Missing context:** the collector updates completed-match history, not live in-game information. Patches, scrims, player condition, and side selection are not modeled. Stored bans are not scored as picks.
                - **Do not use as betting advice.** A percentage is an estimate, not a promise.
                """
            )

with players_tab:
    st.subheader("MPL Philippines Season 18 teams")
    st.caption(
        f"Official team profiles refresh daily. Collected standings through {team_profile_payload['standings_as_of']}. "
        + team_profile_payload["standings_note"]
    )

    standings_rows = []
    for profile in sorted(team_profiles, key=lambda item: item["standing"]["position"]):
        standing = profile["standing"]
        standings_rows.append(
            {
                "#": standing["position"],
                "Team": profile["name"],
                "Match points": standing["match_points"],
                "Match W-L": f"{standing['matches_won']}-{standing['matches_lost']}",
                "Net game wins": f"{standing['game_diff']:+d}",
                "Game W-L": f"{standing['games_won']}-{standing['games_lost']}",
            }
        )
    st.dataframe(standings_rows, hide_index=True, width="stretch")
    if not OFFLINE_TEST_MODE:
        saved_schedule = read_json(schedule_path(ROOT), {})
        loaded_series = {series_key(row) for row in season_games(player_games)}
        uncollected = [item for item in saved_schedule.get("matches", {}).values()
                       if item.get("present") and item.get("status") == "completed" and series_key(item) not in loaded_series]
        if uncollected:
            st.warning(f"{len(uncollected)} completed series from the source are not in the loaded results yet. "
                       "Standings, player Elo, and pick counts show the same saved data until collection finishes. See Data collection for waiting times or errors.")

    inspector_team = st.selectbox(
        "Inspect team", teams, format_func=team_option, key="inspector_team"
    )
    profile = team_profile_by_code[inspector_team]
    standing = profile["standing"]
    roster_summary = player_model.lineup_summary(rating_lineup(profile))

    st.markdown(f"### {profile['name']} · {profile['code']}")
    if profile["official_name"] != profile["name"]:
        st.caption(f"Official Season 18 listing: {profile['official_name']}")
    st.write(display_text(profile["summary"]))
    st.caption("Official profile last checked: " + profile.get("profile_checked_at", team_profile_payload["as_of"] + " (seed profile)"))
    st.caption(profile.get("lineup_basis", "Saved prediction lineup"))
    if profile.get("lineup_warning"):
        st.warning(profile["lineup_warning"])
    st.link_button("Open official MPL PH profile", profile["official_url"])

    info_1, info_2, info_3, info_4 = st.columns(4)
    info_1.metric("Collected S18 rank", f"#{standing['position']}")
    info_2.metric("Series record", f"{standing['matches_won']}-{standing['matches_lost']}")
    info_3.metric("Game record", f"{standing['games_won']}-{standing['games_lost']}")
    info_4.metric("Starting-five Elo", f"{roster_summary['rating']:.0f}")
    st.caption(f"Match points: {standing['match_points']} · Net game wins: {standing['game_diff']:+d} · "
               f"Results through {team_profile_payload['standings_as_of']}")
    st.info(
        f"The current starting five has verified game data for {roster_summary['known_players']} of 5 players. "
        "Each missing player is held at the neutral 1500 rating."
    )

    st.subheader("Season 18 roster, player Elo, and most-used picks")
    roster_rows = []
    profile_player_picks = season_player_picks(player_games, [member.get("history_alias", member["player"]) for member in profile["roster"]])
    for member in profile["roster"]:
        lookup_name = member.get("history_alias", member["player"])
        player_elo = player_model.player_summary(lookup_name)
        pick_stats = profile_player_picks[player_key(lookup_name)]
        roster_rows.append(
            {
                "Player": member["player"],
                "Role": member["role"],
                "Status": member["status"],
                "Player Elo": f"{float(player_elo['rating']):.0f}",
                "Most-used picks this season": ", ".join(f"{pick['hero']} ({pick['picks']})" for pick in pick_stats["heroes"][:5]) or "No usable Season 18 picks",
                "S18 games with picks": pick_stats["games_with_picks"],
                "S18 missing picks": pick_stats["missing_pick_games"],
                "S18 game W-L": f"{pick_stats['wins']}-{pick_stats['losses']}",
                "Elo game record": (
                    f"{player_elo['wins']}-{player_elo['losses']}"
                    if player_elo["known"]
                    else "No data · neutral"
                ),
            }
        )
    st.dataframe(roster_rows, hide_index=True, width="stretch")
    st.caption(
        "Roles come from the latest saved official profile. Main marks the five used for prediction; registered reserves are other listed players. Player Elo and its record use verified S17 and collected S18 appearances. "
        "Most-used picks count every verified Season 18 appearance, including former teams, with no rolling-window limit. "
        "Each new played hero adds one pick; bans and incomplete player-to-hero mappings are excluded. Elo retains its S17 training baseline and applies collected S18 games once."
    )
    with st.expander("All player hero counts this season"):
        st.dataframe([
            {"Player": member["player"], "Hero": pick["hero"], "Season picks": pick["picks"]}
            for member in profile["roster"]
            for pick in profile_player_picks[player_key(member.get("history_alias", member["player"]))]["heroes"]
        ], hide_index=True, width="stretch")

    staff_left, achievement_right = st.columns(2)
    with staff_left:
        st.subheader("Staff")
        st.dataframe(profile["staff"], hide_index=True, width="stretch")
    with achievement_right:
        st.subheader("Achievements")
        for achievement in profile["achievements"]:
            st.markdown(f"- {display_text(achievement)}")

    st.subheader("Most-used team heroes · Season 18")
    coverage = recent_payload["coverage"][profile["code"]]
    season_pick_games = coverage["games_with_picks"]
    hero_rows = [
        {
            "Rank": rank,
            "Hero": row["hero"],
            "Your tier": hero_tier(row["hero"], tiers),
            "Season picks": row["picks"],
            "Pick rate per game": f"{row['picks'] / season_pick_games:.1%}",
        }
        for rank, row in enumerate(recent_picks[profile["code"]][:10], start=1)
    ]
    st.dataframe(hero_rows, hide_index=True, width="stretch")
    st.caption(
        f"All {coverage['series']} loaded Season 18 series. {season_pick_games}/{coverage['games']} games with usable picks, "
        f"{coverage['from']} to {coverage['through']}. Pick rates use only games with valid drafts. "
        "Ties sort alphabetically. All counted heroes, not just the top ten shown here, power draft comfort."
    )
    if coverage["missing_pick_games"]:
        st.warning(f"{len(coverage['missing_pick_games'])} drafts have missing or inconsistent source data and are excluded from Season 18 pick counts.")
    st.caption("Only this season counts. Team totals include every lineup that played for the team this season.")
    with st.expander("Show the series included in favorite picks"):
        visible_series = [
            {**row, "score": display_text(row["score"])} for row in coverage["series_rows"]
        ]
        st.dataframe(visible_series, hide_index=True, width="stretch",
                     column_config={"source": st.column_config.LinkColumn("Source")})

with history_tab:
    st.subheader("Collected Season 18 matches")
    imported = [row for row in player_games if row.get("season") == "MPL Philippines Season 18"]
    st.caption(
        f"{len({series_key(row) for row in imported})} loaded series / {len(imported)} games · "
        f"{min((row['date'] for row in imported), default='Not available')} to {max((row['date'] for row in imported), default='Not available')}. "
        "Results are saved separately from picks and bans."
    )
    st.info("The supplied hero lists match the published bans where draft records are available. Favorite picks use actual played heroes from game records, never these ban lists.")
    count1, count2, count3 = st.columns(3)
    count1.metric("Games updating player Elo", sum(valid_lineups(row) for row in imported))
    count2.metric("Games with usable picks", sum(valid_picks(row) for row in imported))
    count3.metric("Published bans matched", sum(row.get("bans_verified", False) for row in imported))
    incomplete = [row for row in imported if not valid_lineups(row) or not valid_picks(row)]
    if incomplete:
        affected = sorted({f"{row['team_a']}-{row['team_b']} ({row['date']})" for row in incomplete})
        st.warning(
            f"{len(incomplete)} games have incomplete published lineups or picks: {', '.join(affected)}. "
            "Their results are saved, but incomplete lineups do not update Elo and incomplete drafts do not count as picks. "
            "The collector retries these records; complete post-game scoreboards or an updated source are needed to fill the gaps."
        )
    unverified_bans = sum(not row.get("bans_verified", False) for row in imported)
    if unverified_bans:
        st.caption(f"Ban lists for {unverified_bans} games are unavailable or user-reported, not verified against a published draft.")
    week_groups = matches_by_week(imported)
    selected_week = st.selectbox("Match week", ["All weeks"] + [group["label"] for group in week_groups], key="history_week")
    for group in week_groups:
        if selected_week != "All weeks" and selected_week != group["label"]:
            continue
        st.subheader(group["label"])
        st.caption(f"{group['from']} to {group['through']} · {len({series_key(row) for row in group['rows']})} series / {len(group['rows'])} games")
        records = [{
            "Date": row["date"], "Match": f"{row['team_a']} vs {row['team_b']}",
            "Game": row["game"], "Winner": row["winner"],
            "Team A picks": ", ".join(row.get("heroes_a", [])) or "Incomplete · excluded",
            "Team B picks": ", ".join(row.get("heroes_b", [])) or "Incomplete · excluded",
            "Team A bans": ", ".join(row.get("bans_a", [])),
            "Team B bans": ", ".join(row.get("bans_b", [])),
            "Bans verified": bool(row.get("bans_verified")), "Used for Elo": valid_lineups(row),
            "Source": row["series_url"],
        } for row in group["rows"]]
        st.dataframe(records, hide_index=True, width="stretch",
                 column_config={"Source": st.column_config.LinkColumn("Source")})

with meta_tab:
    st.subheader("Your season hero tiers")
    st.write(
        "Assign heroes to S, A, B, C, D, or F. Automatic scoring uses a player's lane when a lane-specific assignment exists; unrated heroes are neutral."
    )
    import_details = meta_config.get("import")
    if isinstance(import_details, dict):
        chart_names = ", ".join(import_details.get("charts", []))
        st.info(
            f"Loaded {meta_config['season']} from the supplied {chart_names} charts "
            f"credited to {import_details.get('attribution', 'the supplied source')}. "
            f"{import_details.get('duplicate_rule', '')}"
        )
        with st.expander("Imported chart-tier conversion"):
            conversion = import_details.get("tier_conversion", {})
            st.dataframe(
                [
                    {"Chart tier": chart_tier, "App tier": app_tier}
                    for chart_tier, app_tier in conversion.items()
                ],
                hide_index=True,
                width="stretch",
            )
    tier_metric_1, tier_metric_2, tier_metric_3 = st.columns(3)
    tier_metric_1.metric("Hero pool", len(all_heroes))
    tier_metric_2.metric("Rated", len(tiers))
    tier_metric_3.metric("Unrated", len(all_heroes) - len(tiers))
    st.markdown(tier_summary_html(tiers, TIER_ORDER), unsafe_allow_html=True)
    if role_tiers:
        with st.expander("Lane-specific meta assignments", expanded=True):
            lane_filter = st.selectbox("Filter by role", ["All roles"] + list(role_tiers), key="meta_role_filter")
            st.dataframe(
                [
                    {"Lane": role, "Hero": hero, "Tier": tier}
                    for role, assignments in role_tiers.items()
                    if lane_filter == "All roles" or role == lane_filter
                    for hero, tier in sorted(assignments.items(), key=lambda item: item[0].lower())
                ],
                hide_index=True,
                width="stretch",
            )

    with st.form("add_custom_heroes", clear_on_submit=True):
        custom_text = st.text_input(
            "Add heroes missing from the hero pool",
            placeholder="Example: Hero One, Hero Two",
        )
        add_submitted = st.form_submit_button("Add to hero pool")
    if add_submitted:
        requested = [" ".join(item.strip().split()) for item in custom_text.split(",") if item.strip()]
        known_by_lower = {hero.lower(): hero for hero in all_heroes}
        additions = [hero for hero in requested if hero.lower() not in known_by_lower]
        if additions:
            updated = dict(meta_config)
            updated["custom_heroes"] = sorted(
                set(meta_config["custom_heroes"]) | set(additions), key=str.lower
            )
            save_meta_config(META_PATH, updated)
            st.success(f"Added {len(additions)} hero{'es' if len(additions) != 1 else ''}.")
            st.rerun()
        elif requested:
            st.info("Those heroes are already in the pool.")
        else:
            st.warning("Enter at least one hero name.")

    with st.form("tier_editor"):
        season_label = st.text_input("Season / patch label", value=meta_config["season"])
        selections: dict[str, list[str]] = {}
        tier_columns = st.columns(3)
        for index, tier in enumerate(TIER_ORDER):
            with tier_columns[index % 3]:
                selections[tier] = st.multiselect(
                    f"{tier} tier",
                    all_heroes,
                    default=[hero for hero, assigned in tiers.items() if assigned == tier],
                    key=f"tier_editor_{tier}",
                )
        save_submitted = st.form_submit_button("Save season tiers", type="primary")

    if save_submitted:
        assignments: dict[str, list[str]] = {}
        for tier, heroes in selections.items():
            for hero in heroes:
                assignments.setdefault(hero, []).append(tier)
        duplicates = {hero: values for hero, values in assignments.items() if len(values) > 1}
        if duplicates:
            details = ", ".join(
                f"{hero} ({'/'.join(values)})" for hero, values in sorted(duplicates.items())
            )
            st.error("Each hero can have only one tier. Remove duplicates: " + details)
        else:
            updated = dict(meta_config)
            updated["season"] = season_label.strip() or "Custom season / patch"
            updated["tiers"] = {hero: assigned[0] for hero, assigned in assignments.items()}
            updated["role_tiers"] = {}
            save_meta_config(META_PATH, updated)
            st.success(f"Saved {len(updated['tiers'])} tier assignments for {updated['season']}.")
            st.rerun()

    st.caption(
        "The tier file is saved locally at config/meta_tiers.json. Saving updates automatic meta fit and selected-draft adjustments immediately; retraining is not required. "
        "Saving manual edits keeps the import notes while replacing the hero assignments and lane-specific overrides."
    )

def manila_time(value: str | datetime | None) -> str:
    if not value:
        return "Not yet"
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    return moment.astimezone(timezone(timedelta(hours=8))).strftime("%b %d, %Y · %I:%M %p PHT")


@st.fragment(run_every="10s" if not OFFLINE_TEST_MODE else None)
def collection_status_panel():
    # A single atomic file contains both the new history and its trained model.
    # Refresh all prediction/profile tables together when that file is published.
    if data_revision(ROOT, offline=OFFLINE_TEST_MODE) != loaded_revision:
        st.rerun()
    status = {} if OFFLINE_TEST_MODE else read_json(status_path(ROOT), {})
    settings = load_settings(ROOT)
    running = collection_service is not None and collection_service.running
    pending = collection_service is not None and collection_service.pending
    schedule = {} if OFFLINE_TEST_MODE else read_json(schedule_path(ROOT), {})
    if st.button("Check schedule now", key="collect_now", type="primary", disabled=OFFLINE_TEST_MODE or running or pending):
        if collection_service.request_refresh():
            st.info("Schedule check requested. This does not bypass the 24-hour waiting period.")
    state = schedule.get("state", "not_started")
    if running or state == "checking":
        st.info("Checking the match schedule; game details are downloaded only for due matches…")
    elif state == "success":
        st.success("Match schedule checked. Waiting for each match's collection time; saved predictions remain available.")
    elif state == "error":
        st.warning("The schedule source could not be checked. Saved predictions and queued completion times are preserved.")
        st.caption(schedule.get("error", "Unknown schedule error"))
    else:
        st.info("The app will discover upcoming matches when automatic collection is enabled.")
    if collection_service is not None and collection_service.last_error:
        st.error("Collector worker: " + collection_service.last_error)
    when = next_check(settings, schedule, utc_now(), ROOT)
    due = next_collection(schedule)
    one, two = st.columns(2)
    one.metric("Last schedule check", manila_time(schedule.get("last_checked_at")))
    two.metric("Next schedule/status check", "Paused" if when is None else ("After current run" if running else manila_time(when)))
    st.write("Next game-data update: " + ("Paused" if not settings["enabled"] else manila_time(due) if due else "Waiting for a newly completed match"))
    st.caption("Game-data updates use 24 hours after first observed Completed, not 24 hours after the scheduled start. "
               "A source outage or sleeping PC can delay collection. All times below are Philippine time.")
    scheduled_matches = list(schedule.get("matches", {}).values())
    pending_matches = [item for item in scheduled_matches if item.get("present") and not item.get("last_collected_at")]
    if pending_matches:
        st.subheader("Upcoming matches and collection queue")
        st.dataframe([{
            "Match": f"{item['team_a']} vs {item['team_b']}",
            "Scheduled start (PHT)": manila_time(item["series_started_at"]),
            "Source status": item["status"].title(),
            "First seen completed (PHT)": manila_time(item.get("first_seen_completed_at")),
            "Collect / retry after (PHT)": manila_time(collection_due(item)) if collection_due(item) else "Awaiting completion",
            "Collection state": item.get("collection_state", "awaiting_completion").replace("_", " "),
            "Source": item["url"],
        } for item in sorted(pending_matches, key=lambda row: row["series_started_at"])],
            hide_index=True, width="stretch", column_config={"Source": st.column_config.LinkColumn("Source")})
    st.caption("Latest fully successful game-data collection: " + manila_time(status.get("last_success_at")))
    if status.get("collected_games") is not None:
        st.write(f"Season 18: {status['collected_series']} series / {status['collected_games']} games. "
                 f"{status.get('games_with_lineups', 0)} complete lineups, {status.get('games_with_picks', 0)} usable drafts, "
                 f"results through {status.get('cutoff', 'unknown')}.")
    if status.get("pending_series"):
        st.caption(f"{status['pending_series']} additional series await a later check (16-series safety limit per run).")
    if status.get("errors"):
        with st.expander("Last game-data collection errors (saved data remains available)"):
            st.caption("Attempt: " + manila_time(status.get("last_attempt_at")))
            for issue in status["errors"]:
                st.warning(f"{issue.get('series', 'Collection')}: {issue['error']}")
                if issue.get("source"):
                    st.link_button("Review source record", issue["source"])
    profile_status = {} if OFFLINE_TEST_MODE else read_json(profile_status_path(ROOT), {})
    st.subheader("Official team profile updates")
    st.caption("Latest fully successful profile check: " + manila_time(profile_status.get("last_success_at")))
    st.caption("Next profile check: " + ("Paused" if not settings["enabled"] else manila_time(profile_status.get("next_check_at"))))
    for issue in profile_status.get("errors", []):
        st.warning(f"{issue.get('team', 'Profiles')}: {issue['error']} — retaining the saved profile.")
    for team in team_profiles:
        if team.get("lineup_warning"):
            st.warning(f"{team['name']}: {team['lineup_warning']}")


with collection_tab:
    st.subheader("Automatic MPL PH data collection")
    st.caption("Opening the app checks for due data before showing predictions. Recent checks are shared for five minutes; "
               "the opening wait is capped at 15 seconds, after which saved data remains usable while collection continues.")
    st.write("Collect each Season 18 match one day after the app first sees it marked Completed on MLDB. "
             "Player Elo, standings, collected matches, favorite heroes, and automatic meta fit update together after collection. "
             "Official rosters, staff, summaries, and achievements refresh daily, with hourly retries on source failures.")
    st.warning("MLDB exposes scheduled start times and completion status, but no reliable finish timestamp was found. "
               "The 24-hour delay starts at first observed completion, so collection may be later than exactly one day after the actual finish.")
    st.info("Runs in the background while this Streamlit server is running and the PC is awake with internet access. "
            "Closing a browser tab does not stop the server. Stopping the server stops collection; reopening the app resumes overdue checks.")
    current_settings = load_settings(ROOT)
    with st.form("collection_settings"):
        enabled = st.toggle("Automatically collect new matches", value=current_settings["enabled"], key="collection_enabled")
        st.caption("Timing: 24 hours after first observed completion. There is no 30-minute game-data refresh.")
        if st.form_submit_button("Save collection settings", disabled=OFFLINE_TEST_MODE):
            save_settings(ROOT, enabled)
            collection_service.settings_changed()
            st.success("Collection settings saved. Pausing stops future checks; a run already in progress can finish.")
    collection_status_panel()
    st.markdown("Source: [MLDB · MPL Philippines Season 18](https://mldb.gg/event/mpl-philippines-season-18). "
                "Upcoming/live matches are used only for scheduling; only completed series whose waiting period has elapsed update predictions.")
    st.caption("The lightweight match list is checked daily for schedule changes and around scheduled games "
               "(first planned check three hours after the listed start, then hourly while awaiting completion). "
               "Stale listings older than a day and incomplete game details retry daily. Successful complete games are not repeatedly downloaded.")
    st.caption("Duplicate games do not earn Elo twice. Incomplete records are retried, and conflicting records are held for review. "
               "Collected lineups update the prediction five when all players match the official roster. "
               "Your saved hero tiers and original imports are preserved. New collected heroes enter the tier editor automatically; "
               "tier grades remain your settings.")

st.markdown(
    '<footer class="matchdesk-footnote">'
    f"Loaded-result cutoff: {safe_text(player_game_payload['cutoff'])}. Prediction lineups use official rosters and collected appearances. "
    "Favorites count all collected Season 18 games; source gaps are shown. "
    "Appearance: choose light, dark, or system in the app menu's Settings."
    '</footer>', unsafe_allow_html=True,
)
