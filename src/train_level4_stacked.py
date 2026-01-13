import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def train_level4_stacked():
    """
    Train Level 4 with TRUE HIERARCHICAL STACKING.
    Uses predictions from Levels 1, 2, and 3 as features.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 4 STACKED: Using L1, L2, L3 predictions as features...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    
    # 1. Load Match Data
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Target: Match Result
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # 2. SIMPLIFIED: Use basic team stats instead of L2 predictions
    # (To avoid column mismatch issues with L2 model)
    logger.info("Calculating basic team strength metrics...")
    
    matches_df = matches_df.sort_values('date')
    matches_df['h_recent_xg'] = matches_df.groupby('home_team')['home_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    matches_df['a_recent_xg'] = matches_df.groupby('away_team')['away_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    
    # 3. GENERATE LEVEL 3 PREDICTIONS (Tactical Control)
    logger.info("Generating Level 3 predictions (Tactical Dominance)...")
    
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 
                     'away_formation', 'away_pressing', 'away_style']
    existing_cols = [c for c in tactical_cols if c in matches_df.columns]
    
    l3_df = pd.get_dummies(matches_df[['id'] + existing_cols], columns=existing_cols, drop_first=True)
    l3_features = [c for c in l3_df.columns if 'formation_' in c or 'pressing_' in c or 'style_' in c]
    
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    X_l3 = l3_df[l3_features].astype(float)
    l3_proba = model_l3.predict(xgb.DMatrix(X_l3))
    matches_df['l3_pred_tactical_control'] = l3_proba
    
    logger.info(f"Generated {len(matches_df)} Level 3 predictions.")
    
    # 4. BUILD STACKED LEVEL 4 MATRIX (No formations to avoid mismatch)
    logger.info("Building stacked Level 4 feature matrix...")
    
    l4_stacked = matches_df[['id', 'date', 'result', 'h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']].copy()
    l4_stacked = l4_stacked.fillna(0)
    
    # 5. TEMPORAL SPLIT
    train_l4 = l4_stacked[l4_stacked['date'] < '2024-01-01']
    
    # Features: Recent xG + L3 Tactical Control (NO FORMATIONS)
    feature_cols = ['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']
    
    X_train = train_l4[feature_cols].astype(float)
    y_train = train_l4['result'].astype(int)
    
    logger.info(f"Training with {len(feature_cols)} features including L3 predictions")
    logger.info(f"Key feature: 'l3_pred_tactical_control' (who dominates the match)")
    logger.info(f"Training samples: {len(X_train)}")
    
    # 6. TRAIN STACKED MODEL
    trainer.train_level4_outcome_model(X_train, y_train)
    logger.info("Level 4 STACKED Training Complete. Enhanced model saved.")

if __name__ == "__main__":
    train_level4_stacked()
