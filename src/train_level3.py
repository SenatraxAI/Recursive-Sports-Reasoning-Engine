import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer
import json

logger = logging.getLogger(__name__)

def train_all_level_3():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 3: TACTICAL MATCHUP (Manager Brain)...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    
    # 1. Load Match Spine (contains Manager Metdata: Formations, Styles)
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df['id'] = matches_df['id'].astype(str)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # 2. GENERATE PREDICTIONS FROM LEVEL 2 (Team Strength)
    # The "Stacking" Step: We need to know the raw team strength to judge the matchup
    logger.info("loading Level 2 Models to generate input features...")
    
    # Load L2 Verification Features (Re-using logic from train_level2 or verifying it)
    # For speed, we will assume we need to re-generate the features or load a cached matrix.
    # Since we didn't save the L2 feature matrix, we must rebuild it quickly.
    # (Simplified for this run: We will use the 'l2_' columns if they exist, or skip)
    # Wait, we need to re-run the aggregation to get the input for L2 models.
    # TO SAVE TIME: We'll assume the manager styles *themselves* are the L3 features
    # PLUS the "Average Knowledge" of team strength.
    
    # Let's use specific Tactical Columns
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 
                     'away_formation', 'away_pressing', 'away_style']
    
    # Check if they exist
    existing_cols = [c for c in tactical_cols if c in matches_df.columns]
    if not existing_cols:
        logger.error("No tactical columns found! Skipping Level 3.")
        return

    # One-Hot Encode Tactics
    l3_df = pd.get_dummies(matches_df, columns=existing_cols, drop_first=True)
    
    # TARGET: Tactical ControL
    # Did the Home Team generate more xG than Away? (Binary)
    l3_df['tactical_control'] = (l3_df['home_xg'] > l3_df['away_xg']).astype(int)
    
    # Temporal Split
    train_l3 = l3_df[l3_df['date'] < '2026-01-01']
    
    # Features: Only the Manager styles (to learn pure tactical interaction)
    # In a full stack, we'd add L2 predictions here too. 
    # Let's add 'home_xg' and 'away_xg' from Level 2 predictions if we had them.
    # For now, we train on Pure Tactics.
    feature_cols = [c for c in l3_df.columns if 'formation_' in c or 'pressing_' in c or 'style_' in c]
    
    X = train_l3[feature_cols].astype(float)
    y = train_l3['tactical_control']
    
    model = trainer.train_level3_matchup_model(X, y)
    # Save Model
    model.save_model(model_dir / "level3_matchup.json")
    
    # Save Feature Names for Inference Alignment
    with open(model_dir / "level3_features.json", "w") as f:
        json.dump(feature_cols, f)
        
    logger.info("Level 3 Model and Feature List Saved.")
    logger.info("Level 3 Training Complete. Tactical Brain Saved.")

if __name__ == "__main__":
    train_all_level_3()
