import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def train_efficiency_model():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 1: EXPERIMENTAL (Opponent Efficiency)...")
    
    trainer = HierarchicalTrainer()
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # 1. Load Data
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = []
    for f in roster_files:
        season_year = f.name.split('_')[-1].split('.')[0]
        rdf = pd.read_parquet(f)
        rdf['season_year'] = season_year
        all_rosters.append(rdf)
        
    roster_df = pd.concat(all_rosters, ignore_index=True)
    # roster_df = roster_df.sort_values(['player_id', 'match_id']) # Optional, safer to sort by date later
    
    # Normalize player names for merging
    roster_df['norm_name'] = roster_df['player'].apply(normalize_name)
    
    # --- NEW: CALCULATE OPPONENT DEFENSIVE STRENGTH ---
    match_files = list(raw_dir.glob("understat_matches_*.parquet"))
    all_matches = []
    for f in match_files:
        all_matches.append(pd.read_parquet(f))
    matches_df = pd.concat(all_matches)
    matches_df = matches_df.rename(columns={'id': 'match_id'})
    
    # Fix Datetime
    try:
        matches_df['date'] = pd.to_datetime(matches_df['datetime'])
    except KeyError:
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
    matches_df = matches_df.sort_values('date')
    
    # Extract Team Names & xG
    def extract_val(val, key=None):
        if isinstance(val, dict):
            if key: return val.get(key)
            return val.get('title')
        return None

    matches_df['home_team'] = matches_df['h'].apply(lambda x: extract_val(x))
    matches_df['away_team'] = matches_df['a'].apply(lambda x: extract_val(x))
    matches_df['home_team'] = matches_df['home_team'].apply(normalize_name)
    matches_df['away_team'] = matches_df['away_team'].apply(normalize_name)
    
    matches_df['home_xg'] = matches_df['xG'].apply(lambda x: pd.to_numeric(extract_val(x, 'h'), errors='coerce'))
    matches_df['away_xg'] = matches_df['xG'].apply(lambda x: pd.to_numeric(extract_val(x, 'a'), errors='coerce'))
    matches_df['home_goals'] = matches_df['goals'].apply(lambda x: pd.to_numeric(extract_val(x, 'h'), errors='coerce'))
    matches_df['away_goals'] = matches_df['goals'].apply(lambda x: pd.to_numeric(extract_val(x, 'a'), errors='coerce'))

    # Merge Roster with Matches to get Context
    full_roster = pd.merge(roster_df, matches_df[['match_id', 'date', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_goals', 'away_goals']], on='match_id', how='left')
    full_roster = full_roster.sort_values(['player_id', 'date'])

    # --- FEATURE ENGINEERING ---
    # 1. Player Efficiency (G - xG)
    full_roster['goals'] = pd.to_numeric(full_roster['goals'], errors='coerce').fillna(0)
    full_roster['xG'] = pd.to_numeric(full_roster['xG'], errors='coerce').fillna(0)
    full_roster['assists'] = pd.to_numeric(full_roster['assists'], errors='coerce').fillna(0)
    full_roster['xA'] = pd.to_numeric(full_roster['xA'], errors='coerce').fillna(0)
    
    full_roster['goal_efficiency'] = full_roster['goals'] - full_roster['xG']
    full_roster['assist_efficiency'] = full_roster['assists'] - full_roster['xA']

    # 2. Defensive Efficiency (Team xG Faced - Team Goals Conceded)
    def calc_def_eff(row):
        if row['side'] == 'h':
            # Home player: conceded goals are away_goals, faced xG is away_xg
            return row['away_xg'] - row['away_goals']
        else:
            return row['home_xg'] - row['home_goals']
            
    full_roster['defensive_efficiency'] = full_roster.apply(calc_def_eff, axis=1).fillna(0)

    # 3. Rolling Efficiency Features
    eff_metrics = ['goal_efficiency', 'assist_efficiency', 'defensive_efficiency']
    for m in eff_metrics:
        # Shift(1) to predict future based on past efficiency
        full_roster[f'prev_{m}_5'] = full_roster.groupby('player_id')[m].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    
    metrics = ['goals', 'xG', 'assists', 'xA', 'xGChain', 'xGBuildup']
    # Ensure numeric
    for m in metrics:
        full_roster[m] = pd.to_numeric(full_roster[m], errors='coerce').fillna(0)
    
    # 1. Base Features (Standard Model uses these)
    for m in metrics:
        full_roster[f'prev_{m}_5'] = full_roster.groupby('player_id')[m].shift(1).rolling(5, min_periods=1).mean().fillna(0)
        
    # 2. Variance Features (Experimental Model uses these)
    for m in metrics:
        full_roster[f'var_{m}_5'] = full_roster.groupby('player_id')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=2).var()).fillna(0)

    # Merge Identity Attributes
    full_matrix = pd.merge(
        full_roster,
        identity_df.drop_duplicates(subset=['norm_name']),
        on='norm_name',
        how='left',
        suffixes=('', '_attr')
    )
    
    full_matrix['Position_Clean'] = full_matrix.get('Position', full_matrix.get('pos', full_matrix.get('position', 'Unknown')))
    
    # 5. TEMPORAL SPLIT (Full data for 2026 usage)
    train_mask = full_matrix['season_year'].isin(['2020', '2021', '2022', '2023', '2024', '2025'])
    test_mask = full_matrix['season_year'].isin(['2026'])
    
    pos_map = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    
    for pos_key, pos_val in pos_map.items():
        pos_mask = full_matrix['Position_Clean'].str.contains(pos_val, na=False, case=False)
        target_col = 'xG'
        if pos_key == 'GK': target_col = 'xGBuildup' # Use xGBuildup if available, else xG
        
        # Check if target exists
        if target_col not in full_matrix.columns:
            full_matrix[target_col] = 0.0
            
        train_data = full_matrix[pos_mask & train_mask].dropna(subset=[target_col])
        
        if train_data.empty: 
            logger.warning(f"No training data for {pos_key}")
            continue
        
        # Define Features
        base_features = [f'prev_{m}_5' for m in metrics]
        eff_features = [f'prev_{m}_5' for m in eff_metrics]
        id_features = ['age']
        
        final_features = base_features + eff_features + id_features
        
        # Verify Features Exist
        missing = [f for f in final_features if f not in train_data.columns]
        if missing:
            logger.error(f"Missing features: {missing}")
            for f in missing: train_data[f] = 0.0
            
        X_train = train_data[final_features]
        y_train = train_data[target_col]
        
        logger.info(f"Training Efficiency-Aware Model ({pos_key}) on {len(X_train)} rows with {len(final_features)} features...")
        
        model = trainer.train_level1_player_models(X_train, y_train, pos_key, save_name=f"level1_{pos_key}_efficiency.json")
        logger.info(f"Saved Experimental Model B to data/models/level1_{pos_key}_efficiency.json")

if __name__ == "__main__":
    train_efficiency_model()
