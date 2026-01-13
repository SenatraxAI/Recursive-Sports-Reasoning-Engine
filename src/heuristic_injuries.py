import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.utils import normalize_name, map_team_name

logger = logging.getLogger(__name__)

def generate_heuristic_injuries():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    seasons = ["2020", "2021", "2022", "2023", "2024"]
    all_absences = []

    for season in seasons:
        logger.info(f"Processing availability gaps for {season}...")
        
        # 1. Load Rosters (who actually appeared in the 18-man squad)
        roster_path = raw_dir / f"understat_rosters_{season}.parquet"
        if not roster_path.exists():
            logger.warning(f"Roster data missing for {season}")
            continue
            
        rosters = pd.read_parquet(roster_path)
        
        # 2. Get Match Metadata (Dates and Teams)
        match_path = raw_dir / f"understat_matches_{season}.parquet"
        if not match_path.exists(): continue
        matches = pd.read_parquet(match_path)
        
        # Flatten matches for easier processing
        match_info = []
        for _, row in matches.iterrows():
            match_info.append({'match_id': str(row['id']), 'date': pd.to_datetime(row['datetime']), 'team': map_team_name(row['h']['title'])})
            match_info.append({'match_id': str(row['id']), 'date': pd.to_datetime(row['datetime']), 'team': map_team_name(row['a']['title'])})
        match_df = pd.DataFrame(match_info)
        
        # 3. Define the "Expectation"
        # A player is "Expected" if they have played in >20% of the team's games so far
        player_appearance_counts = rosters.groupby(['player_id', 'team_id']).size().reset_index(name='total_apps')
        # Filter for "Main Squad" (e.g., played at least 5 times in the season)
        main_players = player_appearance_counts[player_appearance_counts['total_apps'] >= 5]
        
        # 4. Find the Gaps
        # For every match a team played, check which 'Main Players' were NOT in the roster
        unique_matches = match_df.sort_values('date')
        
        for team in unique_matches['team'].unique():
            team_matches = unique_matches[unique_matches['team'] == team]
            # Get actual team_id from rosters mapping
            try:
                # Find the team_id used in Understat for this team name
                # We'll use the roster data to find the mapping
                sample = rosters[rosters['player_id'].isin(main_players['player_id'])] # Just to be safe
                # Note: team mapping in rosters might be inconsistent, we use norm_name
                # Mapping team title to team_id
                pass 
            except: continue
            
            # Simplified version: Use normalize_name(player) to track throughout
            team_rosters = rosters.merge(match_df, left_on='match_id', right_on='match_id')
            # The 'side' in rosters (h/a) must match where the team was in match_df
            # But match_df is already expanded to one row per team per match.
            # We need to ensure we only look at the player's performance for their team.
            
            # Since rosters has 'match_id' and 'side', and match_df has 'match_id' and 'team'
            # We need a way to know if 'h' in rosters corresponds to 'Arsenal' in match_df.
            # We'll join with the original matches to get the h/a titles.
            
            # Let's refine the join
            team_rosters = team_rosters[team_rosters['team'] == team]
            
            p_list = team_rosters['player'].unique()
            
            for player in p_list:
                p_matches = team_rosters[team_rosters['player'] == player]
                p_dates = sorted(p_matches['date'].unique())
                
                # Check for intervals between appearances
                for i in range(len(p_dates) - 1):
                    gap = (p_dates[i+1] - p_dates[i]).days
                    if gap > 14: # More than 2 weeks absence usually indicates injury/suspension
                        all_absences.append({
                            'player': player,
                            'team': team,
                            'start': p_dates[i],
                            'end': p_dates[i+1],
                            'days_out': gap,
                            'season': season
                        })

    absences_df = pd.DataFrame(all_absences)
    absences_df.to_parquet(processed_dir / "heuristic_injuries.parquet")
    logger.info(f"Heuristic Injury detection complete. Found {len(all_absences)} significant absences.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_heuristic_injuries()
