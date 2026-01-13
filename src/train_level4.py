import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.trainer import HierarchicalTrainer

import json

logger = logging.getLogger(__name__)

def train_all_level_4():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 4: THE FINAL VERDICT (Outcome Layer)...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # THIS LAYER SEES EVERYTHING.
    # Ideally, it takes inputs from L2 (Team Strength) and L3 (Tactical Advantage).
    
    # Target: Match Result (0=Home, 1=Draw, 2=Away)
    # We need to encode W/D/L
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
        
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # Features:
    # 1. Team Strength (We'll use actual xG history as proxy for L2 predictions to save inference time)
    # 2. Tactical Metadata (Formations)
    # 3. Context (Home/Away)
    
    matches_df['h_recent_xg'] = matches_df.sort_values('date').groupby('home_team')['home_xg'].transform(lambda x: x.shift(1).rolling(5).mean()).fillna(1.0)
    matches_df['a_recent_xg'] = matches_df.sort_values('date').groupby('away_team')['away_xg'].transform(lambda x: x.shift(1).rolling(5).mean()).fillna(1.0)
    
    # Load L3 for Stacking
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    with open(model_dir / "level3_features.json", "r") as f:
        l3_cols = json.load(f)
    
    # Prepare L3 inputs
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 'away_formation', 'away_pressing', 'away_style']
    existing_cols_l3 = [c for c in tactical_cols if c in matches_df.columns]
    l3_df_raw = pd.get_dummies(matches_df[existing_cols_l3], columns=existing_cols_l3, drop_first=True)
    l3_df_aligned = l3_df_raw.reindex(columns=l3_cols, fill_value=0)
    
    # Inject L3 prediction into L4 frame
    matches_df['l3_pred_tactical_control'] = model_l3.predict(xgb.DMatrix(l3_df_aligned.astype(float)))
    
    # Re-encode formations for L4
    cols_to_encode = [c for c in ['home_formation', 'away_formation'] if c in matches_df.columns]
    l4_df = pd.get_dummies(matches_df, columns=cols_to_encode, drop_first=True)
    
    train_l4 = l4_df[l4_df['date'] < '2026-01-01']
    features = ['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control'] + [c for c in l4_df.columns if 'formation_' in c]
    
    X = train_l4[features].astype(float)
    y = train_l4['result'].astype(int)
    
    # Multi-class Classification
    trainer.train_level4_outcome_model(X, y)
    
    # Save Feature Names for Inference Alignment
    with open(model_dir / "level4_features.json", "w") as f:
        json.dump(features, f)
        
    logger.info("Level 4 Training Complete. The Final Brain is Ready.")

if __name__ == "__main__":
    train_all_level_4()
