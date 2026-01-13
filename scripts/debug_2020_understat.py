from understatapi import UnderstatClient
import logging
import json

logging.basicConfig(level=logging.INFO)

season = "2020"
print(f"Testing understatAPI for season {season}...")
try:
    with UnderstatClient() as understat:
        league = understat.league(league="EPL")
        
        print("Fetching Match Data...")
        matches = league.get_match_data(season=season)
        print(f"Matches count: {len(matches)}")
        
        print("Fetching Player Data...")
        players = league.get_player_data(season=season)
        print(f"Players count: {len(players)}")

        print("Fetching Team Data...")
        teams = league.get_team_data(season=season)
        print(f"Teams count: {len(teams)}")

except Exception as e:
    print(f"Failed for season {season}: {e}")
    # Try to see if it's a specific method failing
