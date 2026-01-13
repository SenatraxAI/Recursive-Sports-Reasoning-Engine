import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def train_all_level_1():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 1: THE BASE LAYER (Player Identity + Form)...")
    
    trainer = HierarchicalTrainer()
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # 1. Load the Identity Matrix (Archer Attributes)
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    
    # 2. Gather all Roster files (Individual match performance)
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    if not roster_files:
        logger.error("No rosters found! Can't train Level 1.")
        return
        
    all_rosters = []
    for f in roster_files:
        season_year = f.name.split('_')[-1].split('.')[0]
        rdf = pd.read_parquet(f)
        rdf['season_year'] = season_year
        all_rosters.append(rdf)
        
    # Separate steps for safety
    roster_df = pd.concat(all_rosters, ignore_index=True)
    roster_df = roster_df.sort_values(['player_id', 'match_id'])
    roster_df = roster_df.reset_index(drop=True)
    
    # Understat rosters use 'player', Archer Matrix uses 'player_name'
    roster_df['norm_name'] = roster_df['player'].apply(normalize_name)
    
    # 3. Feature Engineering: Rolling Form (The "Momentum" Layer)
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes', 'xGBuildup']
    # for m in metrics:
    #     try:
    #         roster_df[m] = pd.to_numeric(roster_df[m], errors='coerce').fillna(0)
    #         # Shift(1) to avoid leakage
    #         roster_df[f'prev_{m}_5'] = roster_df.groupby('player_id')[m].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    #     except Exception as e:
    #         logger.error(f"Error engineering feature {m}")
    #         logger.error(f"Index Unique: {roster_df.index.is_unique}")
    #         logger.error(f"Columns Duplicated: {roster_df.columns[roster_df.columns.duplicated()].tolist()}")
    #         raise e
            
    # SAFE FEATURE ENGINEERING
    for m in metrics:
        roster_df[m] = pd.to_numeric(roster_df[m], errors='coerce').fillna(0)
        # Use simple shift(1) to avoid 'duplicate labels' error from complex transform
        roster_df[f'prev_{m}_5'] = roster_df.groupby('player_id')[m].shift(1).fillna(0)
    
    # 4. Merge Identity (Attributes) with Roster (Form)
    full_matrix = pd.merge(
        roster_df,
        identity_df.drop_duplicates(subset=['norm_name']), 
        on='norm_name',
        how='left',
        suffixes=('', '_attr')
    )
    
    # Standardize Position
    full_matrix['Position_Clean'] = full_matrix.get('Position', full_matrix.get('pos', full_matrix.get('position', 'Unknown')))
    
    print("\nDEBUG: Unique Positions Found:", full_matrix['Position_Clean'].unique())
    print("DEBUG: Columns in matrix:", full_matrix.columns.tolist()[:10])
    
    # Cast attributes to numeric
    attr_map = {
        'FWD': ['Goals p 90', 'Assists p 90', 'Expected Goals', 'Goals per shot', '% Shots on target'],
        'MID': ['Assists p 90', 'Progressive Passes', 'Progressive Carries', 'Key passes'],
        'DEF': ['Tackles Won', 'Interceptions', 'Clearances', '% Aerial Duels won'],
        'GK': ['Saves %', 'Clean Sheets', 'Crosses Stopped']
    }
    
    all_needed_attrs = list(set([item for sublist in attr_map.values() for item in sublist] + ['age']))
    for attr in all_needed_attrs:
        if attr in full_matrix.columns:
            full_matrix[attr] = pd.to_numeric(full_matrix[attr], errors='coerce').fillna(0)
    
    # 5. TEMPORAL SPLIT (2020-2025 Train | 2026 Test)
    train_mask = full_matrix['season_year'].isin(['2020', '2021', '2022', '2023', '2024', '2025'])
    test_mask = full_matrix['season_year'].isin(['2026'])
    
    # 6. Train Position-Specific Models
    # Map config keys to data values (e.g. 'DEF' -> 'DF')
    pos_map = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    
    for pos_key, pos_val in pos_map.items():
        pos_mask = full_matrix['Position_Clean'].str.contains(pos_val, na=False, case=False)
        target_col = 'xG' if pos_key != 'GK' else 'xGBuildup'
        
        train_data = full_matrix[pos_mask & train_mask].dropna(subset=[target_col])
        test_data = full_matrix[pos_mask & test_mask].dropna(subset=[target_col])
        
        if train_data.empty: 
            logger.warning(f"No data found for position {pos_key} (looked for {pos_val})")
            continue
        
        # FEATURES: ONLY Base Stats + Identity (No Managerial Tactics yet)
        form_features = [f'prev_{m}_5' for m in metrics]
        id_features = [f for f in attr_map.get(pos_key, []) if f in train_data.columns] + ['age']
        
        final_features = form_features + id_features
        logger.info(f"Level 1 {pos_key} - Base Level Features: {len(final_features)}")
        
        X_train = train_data[final_features]
        y_train = train_data[target_col] 
        
        model = trainer.train_level1_player_models(X_train, y_train, pos_key)
        
        # Quick Validation Score (MAE)
        if not test_data.empty:
            X_test = test_data[final_features]
            y_test = test_data[target_col]
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
            mae = np.mean(np.abs(preds - y_test))
            logger.info(f"Level 1 {pos_key} MAE: {mae:.4f}")
            
            with open("level1_results.csv", "a") as f:
                f.write(f"{pos_key},{mae:.4f}\n")

if __name__ == "__main__":
    train_all_level_1()
