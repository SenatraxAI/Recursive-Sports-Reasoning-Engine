import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from src.utils import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("team_dna")

def build_team_dna():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    logger.info("Loading Data...")
    
    # 1. Matches (Maps ID -> Team)
    matches = pd.read_parquet(processed_dir / "processed_matches.parquet")
    
    # 2. Rosters (Maps MatchID -> Players)
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = pd.concat([pd.read_parquet(f) for f in roster_files], ignore_index=True)
    all_rosters['norm_name'] = all_rosters['player'].apply(normalize_name)
    all_rosters['match_id'] = all_rosters['match_id'].astype(str)
    
    # 3. Player DNA (L0 Features)
    l0_features = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    
    # helper to build DNA dict
    dna_lookup = {}
    for _, row in l0_features.iterrows():
        # DNA Mapping (Using available proxies)
        # Normalize roughly based on max values observed in data
        dna = {
            'dna_stamina': min(1.0, row.get('last_season_90s', 0) / 30.0), 
            'dna_passing': min(1.0, row.get('career_avg_progressive_passes', 0) / 500.0), # Approx max
            'dna_defense': min(1.0, row.get('career_avg_tackles_won', 0) / 50.0) 
        }
        dna_lookup[row['norm_name']] = dna
        
    logger.info("Aggregating Team Statistics...")
    
    team_stats = {} # Team -> List of DNAs
    
    for idx, m in matches.iterrows():
        m_id = str(m['id'])
        h_team = str(m['home_team']).lower().strip()
        a_team = str(m['away_team']).lower().strip()
        
        roster_m = all_rosters[all_rosters['match_id'] == m_id]
        
        # Home Starters
        h_players = roster_m[(roster_m['side'] == 'h') & (roster_m['positionOrder'].astype(int) <= 11)]['norm_name']
        for p in h_players:
            if p in dna_lookup:
                if h_team not in team_stats: team_stats[h_team] = []
                team_stats[h_team].append(dna_lookup[p])

        # Away Starters
        a_players = roster_m[(roster_m['side'] == 'a') & (roster_m['positionOrder'].astype(int) <= 11)]['norm_name']
        for p in a_players:
            if p in dna_lookup:
                if a_team not in team_stats: team_stats[a_team] = []
                team_stats[a_team].append(dna_lookup[p])
                
    logger.info("Calculating Averages...")
    
    final_db = {}
    for team, history_dna in team_stats.items():
        # history_dna is list of dicts
        df_dna = pd.DataFrame(history_dna)
        avg = df_dna.mean().to_dict()
        final_db[team] = {
            'dna_stamina': float(round(avg['dna_stamina'], 2)),
            'dna_passing': float(round(avg['dna_passing'], 2)),
            'dna_defense': float(round(avg['dna_defense'], 2))
        }
        
    # Manual Injections for Promoted Teams (if missing)
    manual_dna = {
        "sunderland": {'dna_stamina': 0.85, 'dna_passing': 0.60, 'dna_defense': 0.55}, # Championship Playoff Winner Profile
        "leeds": {'dna_stamina': 0.90, 'dna_passing': 0.75, 'dna_defense': 0.65},       # Strong Championship
        "leicester": {'dna_stamina': 0.88, 'dna_passing': 0.70, 'dna_defense': 0.60},
        "ipswich": {'dna_stamina': 0.95, 'dna_passing': 0.65, 'dna_defense': 0.50}      # High Energy, Low Tech
    }
    
    for team, features in manual_dna.items():
        if team not in final_db:
            logger.info(f"Injecting Manual DNA for {team}")
            final_db[team] = features
            
    out_path = processed_dir / "team_dna_agg.json"
    with open(out_path, "w") as f:
        json.dump(final_db, f, indent=4)
        
    logger.info(f"Saved DNA profiles for {len(final_db)} teams to {out_path}")
    logger.info(f"Example (arsenal): {final_db.get('arsenal', 'Not Found')}")

if __name__ == "__main__":
    build_team_dna()
