from understatapi import UnderstatClient
import logging

logging.basicConfig(level=logging.INFO)

print("Testing understatAPI...")
try:
    with UnderstatClient() as understat:
        # Get data for players in the Premier League for 2024
        print("Requesting 2024 EPL player data...")
        league_player_data = understat.league(league="EPL").get_player_data(season="2024")
        if league_player_data:
            print(f"Success! Found {len(league_player_data)} players.")
            print(f"Sample Player: {league_player_data[0]['player_name']} (id: {league_player_data[0]['id']})")
        else:
            print("No data returned.")
except Exception as e:
    print(f"understatAPI test failed with error: {e}")
