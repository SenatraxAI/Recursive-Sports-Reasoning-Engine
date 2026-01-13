from understatapi import UnderstatClient
import logging

logging.basicConfig(level=logging.ERROR)

with UnderstatClient() as understat:
    league = understat.league(league="EPL")
    print(f"Methods available for league: {dir(league)}")
    
    # Also check if get_team_data works or exists
    try:
        data = league.get_team_data(season="2024")
        print(f"get_team_data returned {len(data) if data else 0} items.")
    except Exception as e:
        print(f"get_team_data failed: {e}")

    try:
        data = league.get_match_data(season="2024")
        print(f"get_match_data returned {len(data) if data else 0} items.")
    except Exception as e:
        print(f"get_match_data failed: {e}")
