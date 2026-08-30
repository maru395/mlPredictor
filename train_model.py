"""Normalize the downloaded matches, tune Elo, and save the model artifact."""

from pathlib import Path

from mlbb_predictor.context import build_context_data, write_context_data
from mlbb_predictor.data import load_all_matches, write_normalized_matches
from mlbb_predictor.model import train_and_evaluate
from mlbb_predictor.player_model import load_player_games, train_player_elo
from mlbb_predictor.history import load_game_history, valid_lineups


ROOT = Path(__file__).resolve().parent


def main() -> None:
    matches = load_all_matches(ROOT / "data" / "raw")
    model, metrics = train_and_evaluate(matches)
    write_normalized_matches(matches, ROOT / "data" / "processed" / "matches.csv")
    context = build_context_data(ROOT / "data" / "raw")
    write_context_data(context, ROOT / "data" / "processed" / "context.json")
    model.save(ROOT / "models" / "elo_model.json", metrics)
    _, player_games = load_player_games(
        ROOT / "data" / "processed" / "mpl_ph_s17_player_games.json"
    )
    player_model, player_metrics = train_player_elo(player_games)
    _, combined = load_game_history([
        ROOT / "data/processed/mpl_ph_s17_player_games.json",
        ROOT / "data/processed/mpl_ph_s18_player_games.json",
    ])
    new_games = [row for row in combined if row.get("season") == "MPL Philippines Season 18" and valid_lineups(row)]
    # Freeze the K selected on S17; S18 is an update, not a tuning target.
    player_model.fit(new_games)
    player_metrics.update(
        match_count=len(player_games) + len(new_games), player_count=len(player_model.ratings),
        imported_game_count=len(combined), excluded_lineup_games=sum(not valid_lineups(row) for row in combined),
        update_game_count=len(new_games), rating_cutoff=max(row["date"] for row in combined),
        season="MPL Philippines Seasons 17–18",
        validation_season="MPL Philippines Season 17",
    )
    player_model.save(ROOT / "models" / "player_elo_model.json", player_metrics)

    print("Training complete")
    print(f"  matches: {metrics['match_count']}")
    print(f"  teams: {metrics['team_count']}")
    print(f"  selected K-factor: {metrics['selected_k_factor']}")
    print(f"  chronological validation accuracy: {metrics['validation_accuracy']:.1%}")
    print(f"  validation log loss: {metrics['validation_log_loss']:.4f}")
    print(f"  heroes with pick history: {len(context['hero_pool'])}")
    print(f"  teams with player rows: {len(context['players'])}")
    print(f"  player-Elo games: {player_metrics['match_count']} ({len(new_games)} new S18 games)")
    print(f"  rated players: {player_metrics['player_count']}")
    print(f"  selected player K-factor: {player_metrics['selected_k_factor']}")
    print(f"  player-Elo validation accuracy: {player_metrics['validation_accuracy']:.1%}")


if __name__ == "__main__":
    main()
