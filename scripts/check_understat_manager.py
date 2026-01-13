from understatapi import UnderstatClient
import json

with UnderstatClient() as u:
    match_id = "26602"
    print(f"Checking Match {match_id}...")
    
    # Check roster
    roster = u.match(match=match_id).get_roster_data()
    print("ROSTER KEYS:", roster.keys())
    
    # Check shots
    shots = u.match(match=match_id).get_shot_data()
    print("SHOTS COUNT:", len(shots))
    
    # Check match object itself if it has other methods
    m = u.match(match=match_id)
    print("MATCH METHODS:", dir(m))
