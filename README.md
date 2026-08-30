# MLBB Professional Match Predictor

A local Streamlit app that estimates which of the **eight MPL Philippines Season 18 teams** would win a single game. The visible team list is limited to AP BREN, Aurora Gaming PH, Fnatic ONIC PH, Smart Omega, Team Falcons PH, Team Liquid PH, TNC Pro Team, and Twisted Minds PH. Its prediction core is a transparent **player-level Elo model using 169 MPL PH Season 17 games plus automatically collected, lineup-complete Season 18 games**. A team is scored from its current five starters, so ratings follow transferred players and departed players no longer strengthen their old franchise.

The project still includes the older Kaggle and Hugging Face datasets for S13 player hero history and additional historical context:

- [Kaggle — Mobile Legend Tournament Match MPL Philippines S13](https://www.kaggle.com/datasets/bcakra/mobile-legend-tournament-match-mpl-philippines): 160 games reconstructed from 1,600 player rows; MIT license.
- [Hugging Face — MPL ID & PH Season 14](https://huggingface.co/datasets/z4fL/mpl_s14_dataset): 383 game-level results (212 MPL ID + 171 MPL PH); the dataset card does not specify a license.

The app does not use kills, gold, match duration, or any other information that would only be known after a match. It learns only from earlier winners and losers.

## Quick start (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
cd C:\Users\maru\ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python download_player_data.py
python download_recent_data.py
python train_model.py
python -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open it and select Team 1 and Team 2. Automatic meta adjustment is enabled by default; a draft is not required. Schedule discovery starts when the app is opened; game data is collected 24 hours after the app first observes a match marked Completed.

## Deploy to Streamlit Community Cloud

Community Cloud starts from the files in your GitHub repository, not the files on your PC. See [Streamlit's file organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

**Fix for "An artifact is missing":** the original `.gitignore` excluded all of `models/` and `data/processed/`. The updated rules allow these five small saved files while continuing to ignore locks, collection queues, backup snapshots, and other generated outputs:

```text
models/player_elo_model.json
data/processed/context.json
data/processed/mpl_ph_s17_player_games.json
data/processed/mpl_ph_s18_player_games.json
data/processed/live_snapshot.json
```

The first four files are required to start the app. `live_snapshot.json` is optional, but including it preserves the latest already-collected games and their matching player ratings. Without it, the app starts from the original S17/S18 imports. Keep `app.py`, `mlbb_predictor/`, `download_recent_data.py`, `download_player_data.py`, `requirements.txt`, `ui/matchdesk.css`, `.streamlit/config.toml`, and the existing `config/` files in the repository too. Raw datasets are not needed to serve the saved predictor.

For this existing checkout, run these commands in PowerShell. They upload the deployment fix and saved files; do not rerun the download/training scripts when the saved files already exist:

```powershell
cd C:\Users\maru\ml
git add .gitignore app.py README.md tests/test_deployment.py
git add models/player_elo_model.json data/processed/context.json data/processed/mpl_ph_s17_player_games.json data/processed/mpl_ph_s18_player_games.json data/processed/live_snapshot.json
git diff --cached --stat
git commit -m "Include saved prediction files for Streamlit deployment"
git push
```

Check that the files appear on GitHub in the branch used by your Streamlit app, preserving the exact folder names. The entrypoint remains `app.py` and Python dependencies come from `requirements.txt`. If the old error remains, use your Streamlit workspace's app menu > **Reboot**, then confirm. See [Streamlit's reboot instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app). Rebooting interrupts active users briefly.

If you use GitHub's upload page instead of Git, upload the listed files into their matching repository folders, plus the updated `.gitignore` and `app.py`. Do not upload `.venv`, secrets, locks, or your local collection queue.

This change ships a startup snapshot; it is not cloud persistence. The app still writes new collection results and tier edits to its running server's local files, not back to GitHub. Keep a separate durable backup of important changes, and do not rely on the worker running while the hosting service has stopped or suspended the app. A public deployment also shares the current tier/settings controls among visitors; this fix does not add administrator authentication.

## Match desk interface

The predictor uses an editorial scoreboard layout, with the forecast above the optional controls. Both teams' single-game probabilities, their player-Elo baselines, the meta change in percentage points, and starter coverage appear together. An exactly even prediction is labeled **Even matchup**, not assigned an arbitrary favorite.

- Open **Automatic meta and optional drafts** to switch automatic fit off, enter a complete draft, or change heuristic weights.
- Below the forecast, inspect the meta contribution and current starting fives. **Player ratings and game records** expands the per-player detail.
- The other four tabs retain team profiles, editable season tiers, imported match records, and match-aware data collection. The tier editor shows a compact count for each saved tier.
- Choose **Settings > Theme** from Streamlit's app menu for light, dark, or system appearance. Both palettes use the same teal accent, native accessible controls, and local system fonts. No font service, JavaScript framework, or image downloads are needed.
- Navigation uses large bordered tabs with a filled selected state. All five destinations wrap into view on smaller screens, with a two-column tab layout on mobile. Narrow screens also use a compact two-sided scoreboard and stacked controls. Motion is limited to button feedback and respects reduced-motion preferences.

Presentation lives in `mlbb_predictor/ui.py`, `ui/matchdesk.css`, and `.streamlit/config.toml`. Streamlit 1.62 or newer is required for the native theme options. The redesign does not change Elo, meta weights, imports, current starters, or collection timing. The presentation tests cover escaping, probability rendering, theme contrast, and control regressions; browser layout and performance still require a visual check in an allowed browser.

## Automatic data collection

No API key or Kaggle login is needed for new S18 matches. The collector reads the public [MLDB MPL PH S18 match schedule](https://mldb.gg/event/mpl-philippines-season-18), including upcoming, live and completed statuses.

### Data check when the site opens

Each new browser session shows **Checking the match schedule and loading data...** before the predictor is loaded. With collection enabled, the app joins an existing check or requests one from the shared background worker. Any matches already eligible under the 24-hour rule can be collected before the first prediction; the model and matching history are then read together.

- A successful check less than five minutes old can be reused across visitors, unless a scheduled check or game-data update is already due.
- Team changes, tab interactions, and automatic page reruns do not trigger another opening check in the same browser session.
- The page waits at most 15 seconds for collection. If it takes longer, saved predictions are shown and the worker continues; the page's revision watcher loads the new snapshot when ready.
- Source errors, incomplete collections, and paused collection are reported without claiming that saved data is newly downloaded. Existing source retry backoff is respected.
- Opening does **not** bypass the 24-hour wait after first observed completion. If cloud storage loses the completion queue, a newly observed match starts a new timer. This feature does not add durable storage or update GitHub automatically.

For the deployed app, push the changed `app.py`, `mlbb_predictor/collection_service.py`, and new `mlbb_predictor/startup.py` alongside the existing model/data files. Reboot the deployed app if needed to load the new worker code. Offline UI tests intentionally disable network checks.

### Background collection

1. For local use, keep the Streamlit server running on an awake, internet-connected PC. For a deployed app, the cloud server must be running; your PC can be off. No terminal commands are needed for each match.
2. In **Data collection**, leave **Automatically collect new matches** enabled and click **Save collection settings**. The policy is one day after first observed completion; the old 30-minute interval has been removed.
3. Use **Check schedule now** to refresh the match list, even when automatic checks are paused. It still respects the 24-hour delay. Upcoming matches, completion-observation times, collection/retry times, errors and roster-review alerts appear in that tab.
4. Predictions, player ratings, recent favorite picks and meta fit refresh together after successful collection. An open page checks for newly published data every ten seconds.

Closing a browser tab does not stop an already running server. Stopping Streamlit, sleeping the PC, or losing internet prevents collection. Reopen the app after restarting Streamlit to resume overdue checks. This does not install a Windows startup task or a cloud service, and it does not collect live in-game data.

**Timing limitation:** the inspected MLDB feed/pages expose a scheduled start and Completed status, but no reliable actual finish timestamp. A start time is never treated as an end time. The app conservatively starts the 24-hour countdown when it first sees Completed, which can be later than the real finish. Example: first observed completed Sunday at 10 PM PHT → collect Monday at 10 PM PHT or later if the PC/source is unavailable.

The lightweight match list is checked at least once per 24 hours while the app is running, and around scheduled matches: the first planned status check is three hours after the listed start, then roughly hourly until completion. These are discovery/status requests, not full game downloads or rating updates. Listings older than a day fall back to daily checks. A rescheduled start or withdrawn completion cancels the old countdown; a later observed completion starts a new one. Each due match is rechecked against the fresh feed before importing.

Only explicit completed series for the eight configured S18 teams whose delay has elapsed are automatically imported. The collector checks per-game winners against the series score, dates, team identities, distinct players, and played heroes. Bans remain separate. Successfully collected complete series are not repeatedly downloaded. Incomplete/conflicting details retry the next day, schedule-source failures retry after an hour, and catch-up batches are capped at 16 series with remaining due work deferred an hour. Conflicts retain the saved record and appear as warnings. A source outage leaves the last-good predictions usable.

The combined game history and rebuilt player model are published atomically to `data/processed/live_snapshot.json`; its previous version is retained as `live_snapshot.previous.json`. Original seed imports and `models/player_elo_model.json` remain untouched. The model rebuild starts from the S17 baseline, so repeat collection never awards Elo twice. Settings live in `config/data_collection.json`, the persistent schedule/queue in `data/processed/match_schedule.json`, and the latest full-download report in `data/processed/collection_status.json`. Queue timestamps survive app restarts. Previously saved complete imports remain available when switching to this policy; they are not removed or delayed retroactively.

Your season tiers and configured starting rosters remain manual. If the latest recorded five differs from the saved starters, the app flags it for review instead of treating a substitute as a permanent transfer. Older Kaggle/Hugging Face context, static team-profile information, and dated standings are not automatically refreshed. Source/season rollover requires an explicit configuration/code update.

The initial automatic check on August 30, 2026 collected **12 S18 series / 29 games** through August 29, including eight new games and repaired player/hero records for two previously incomplete games. At that check, all 29 S18 games had complete lineups and picks, giving 198 player-Elo games including S17. The live status tab is authoritative for later counts.

## Meta, drafts, and player picks

The app has five tabs:

- **Predict:** select two teams. Player Elo supplies the base probability; automatic meta fit adjusts it using the current five starters' recent hero pools and your saved tiers. The app shows the change in percentage points and player-level evidence. An optional complete draft replaces the automatic estimate and uses that draft's tier ratings and team pick comfort.
- **MPL PH team profiles:** inspect the current Season 18 roster, individual player Elo, recorded game results, roles, substitutes, staff, achievements, an August 28, 2026 standings snapshot, recent team-favorite heroes, and older S13 player top picks where available.
- **Season meta tiers:** enter a season/patch name and assign heroes to S, A, B, C, D, or F. You can also add a new hero that is missing from the historical hero pool.
- **Collected matches:** audit loaded S18 series, game winners, played heroes, bans, source links, and which games update Elo.
- **Data collection:** enable/pause match-aware collection, check the schedule, view the upcoming/next-day queue, and review errors or lineup changes.

Tier settings are saved locally in `config/meta_tiers.json` and apply immediately to automatic and selected-draft modes; model retraining is not required. Unrated heroes receive a neutral score. Meta adjustments are transparent, hand-weighted heuristics, while the reported validation metrics apply only to the historical Elo model.

### Automatic meta calculation

1. Select only the current five starters from the roster, resolving their existing player aliases.
2. For each starter, use their latest 10 recorded series, even if they played for another team. More frequently played heroes receive more weight. Departed players' pools are not inherited by the old team.
3. Grade heroes using your saved tiers: S=5, A=4, B=3, C=2, D=1, F=0; unrated=2.5.
4. Compute each player's score as `(sum(pick count × tier score) + 5 × 2.5) / (usable games + 5)`. The five neutral pseudo-games soften small samples; no usable history gives exactly 2.5.
5. Average the five player scores equally. With the default slider, each team's meta adjustment is `25 × (average score − 2.5)` Elo points. Missing player history stays neutral rather than copying the team's former players.
6. Recalculate Elo probability from the two adjusted lineup ratings and display the difference from base probability in percentage points. Equal meta fit gives no relative advantage; this is not a fixed win-percent bonus for each S-tier hero.

Frequency already contributes to automatic fit, so no extra comfort bonus is added in that mode. Selecting five unique, non-overlapping heroes on both sides **replaces** automatic fit with the existing draft tier/comfort calculation. Partial or invalid drafts leave automatic mode active. Turn off **Automatically adjust for current players' meta heroes** to see pure player Elo without a complete draft.

The automatic evidence table shows the current players, top heroes, S/A pick share, sample sizes, excluded games, history dates, and each player's share of the meta Elo adjustment. It displays the top three heroes per player but calculates from their entire recent pool. Bans, invalid drafts, and incomplete player-to-hero mappings are excluded. Older-patch history can still be present. The scores describe historical tendencies, not a guaranteed future draft or a statistically validated increase in win rate.

The included tier file is preloaded from the five July 2026 Season 41 role charts supplied by the user and credited in the images to `@UomiPH`. Because the predictor uses one tier per hero, chart grades are converted as follows: S+ → S, S- and A+ → A, A → B, B → C, C → D, and D → F. When a flexible hero appears in multiple roles, its strongest converted grade is retained. All assignments remain editable in the **Season meta tiers** tab.

Team-favorite heroes are calculated from each team's **latest 10 loaded completed series**, including every game within each selected series. This is not the last ten individual games. Windows cross from S18 into S17 as needed and can include previous rosters. Counts rank descending, with alphabetical tie-breaking. Five heroes are suggested beside each draft, ten are shown on the profile, and all counted heroes are used by the comfort heuristic. The old `config/mpl_ph_s17_hero_picks.json` is retained as a historical snapshot but no longer drives favorites or draft comfort.

Seed game history lives in `data/processed/mpl_ph_s17_player_games.json` and `data/processed/mpl_ph_s18_player_games.json`. Sources: [Strive S17](https://strivemlbb.com/tournament/mobile-legends-professional-league-philippines/s17) and [MLDB S18](https://mldb.gg/event/mpl-philippines-season-18), with per-series links in the data and UI. Once collected, `data/processed/live_snapshot.json` supplies the app's combined history and matching model.

The original user submission is preserved in `data/manual/mpl_ph_s18_aug21_28.json`. The hero lists matched **bans**, not played picks, in all 17 games with a published draft panel at initial import. For example, [Liquid–Falcons Game 1](https://mldb.gg/match/MPL_Philippines_Season_18-TLPHvsFLCN-3452) records Liquid playing Rafaela, Paquito, Novaria, Clint, and Uranus. Four August 28 ban lists were initially user-reported because their draft panels were unavailable; the collector retries missing panels. Bans are stored separately and never counted as picks.

The seed import had only four published player/hero entries per side for two Twisted Minds–ONIC games on August 28. Collection on August 30 recovered complete source records, so both now contribute to Elo and favorites in the live snapshot. The seed files preserve the original missing-data flags for auditability. Nine older S17 drafts contain duplicate/missing heroes and are excluded from pick counts only; their complete lineup results still update Elo. Each team profile shows the series window, usable-game denominator, and missing draft count. Missing drafts do not extend the window or count as zero-pick games. Absent players are never inferred from the current roster.

Current team and roster information is stored in `config/mpl_ph_teams.json`. Every current starter is looked up individually. Players without a verified game start at a neutral 1500 rating and are visibly labeled; no team inherits another lineup's rating. Explicit source-name aliases preserve continuity (including Domeng/Domengkite, Kyle/KyleTzy, Flap/FlapTzy, and Teddyqt/Teddy).

The current-team profiles link back to the [official MPL Philippines team directory](https://ph-mpl.com/teams). The standings shown in the app are a dated snapshot from the [2026 MPL Philippines season page](https://en.wikipedia.org/wiki/2026_MPL_Philippines_season), not a live feed.

The repository already contains the downloaded source files and a trained model after the initial setup, so for this copy you only need to install the requirement and run the final Streamlit command.

## Rebuild or update

For a one-shot schedule check that respects the 24-hour waiting period without opening Streamlit:

```powershell
python collect_data.py --check-schedule
```

For an explicit **manual override** that immediately collects completed games without waiting (including rechecks of the most recent four series):

```powershell
python collect_data.py
```

This uses the same collector and cross-process lock as the app. Check its printed `state` and `errors`; partial success keeps valid updates and reports quarantined records. Internet access to `https://mldb.gg` is required.

Use `python download_data.py --force` to fetch the older Kaggle/Hugging Face context again. To rebuild the original historical seed artifacts (not necessary for routine updates):

```powershell
python download_player_data.py
python download_recent_data.py
python train_model.py
```

The legacy S18 downloader looks up only the series in the manual submission file. It does not replace the collected snapshot. `train_model.py` rebuilds the seed model, while `collect_data.py` builds the active snapshot model. To intentionally reset to seed data, first pause collection and move the current snapshot to a separate backup; do not delete the manual source or tier files.

## How the probability is made

1. Every player starts at 1500 Elo.
2. Before each S17 game, each team's strength is the average Elo of the five players who actually played that game.
3. After the winner is known, only those ten players are updated.
4. The Elo K-factor is chosen by lowest log loss on the final 25% of S17 games in chronological order.
5. The model is refit on all 169 S17 games, then updated chronologically with all collected complete S18 lineups using that same K-factor. S18 outcomes are not used to select K. The active model is saved together with history in the atomic live snapshot; `models/player_elo_model.json` remains the original seed model.
6. At prediction time, the app averages the five current starters from `config/mpl_ph_teams.json`. New or previously unseen players remain at 1500.
7. Automatic meta fit from the current starters' recent hero pools adjusts the displayed probability by default. A complete draft replaces it with the selected-draft tier/comfort heuristic. Neither mode changes saved player ratings.

Elo is used because the available history is a small sample, and the player-rating logic makes roster changes inspectable. Validation metrics still describe the S17 tuning holdout, not independent S18 accuracy or calibrated draft-adjusted probabilities.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Limitations

- Automatic collection depends on the public MLDB S18 source and may lag or contain gaps. Incomplete player lineups do not update Elo; incomplete picks do not affect favorites or meta fit. Conflicting records require review.
- The app converts roster changes directly through the current starting five, but an unseen player receives a neutral 1500 rather than a guessed rating.
- Recent wins/losses affect player Elo; patches, scrim results, player condition, bans, side selection, and tournament format are not automatic model features.
- Favorite-pick windows may span older rosters and contain flagged source gaps. They describe team tendencies, not individual mastery.
- The older Kaggle S13 and Hugging Face S14 files remain context sources; they do not drive the current player-Elo probability.
- Automatic meta fit and selected-draft adjustments have not been validated as win-rate effects; their weights are editable in the app.
- Probabilities are estimates for learning/demo use and are not betting advice.
