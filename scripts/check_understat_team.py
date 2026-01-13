from understatapi import UnderstatClient
import json

with UnderstatClient() as u:
    print("Checking Team Manchester_United...")
    t = u.team(team="Manchester_United")
    print("TEAM METHODS:", dir(t))
    
    # Try to get data
    print("Fetching Match Data for 2024...")
    matches = t.get_match_data(season="2024")
    if matches:
        print("SAMPLE MATCH KEYS:", matches[0].keys())
        
    print("Fetching Player Data for 2024...")
    players = t.get_player_data(season="2024")
    if players:
        print("SAMPLE PLAYER KEYS:", players[0].keys())
