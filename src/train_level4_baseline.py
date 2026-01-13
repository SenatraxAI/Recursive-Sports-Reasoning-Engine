import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def train_baseline_simple():
    """
    Train a SIMPLE BASELINE Level 4 model.
    Features: Just Recent xG (No formations, No L3 predictions).
    Purpose: Clean comparison against the Stacked model.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 4 SIMPLE BASELINE Training...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Target
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # 2. Features: Recent xG ONLY
    matches_df = matches_df.sort_values('date')
    matches_df['h_recent_xg'] = matches_df.groupby('home_team')['home_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    matches_df['a_recent_xg'] = matches_df.groupby('away_team')['away_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    
    # 3. Temporal Split
    train_df = matches_df[matches_df['date'] < '2024-01-01']
    
    feature_cols = ['h_recent_xg', 'a_recent_xg']
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df['result'].astype(int)
    
    logger.info(f"Training Baseline with features: {feature_cols}")
    logger.info(f"Training samples: {len(X_train)}")
    
    # 4. Train and Save manually to avoid overwriting the 'outcome' slot if using default trainer
    # We use XGBoost directly here to save to specific 'baseline' filename
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,
        'eta': 0.1,
        'eval_metric': 'mlogloss',
        'device': 'cuda',
        'tree_method': 'hist'
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    model.save_model(model_dir / "level4_baseline.json")
    logger.info("Baseline model saved to data/models/level4_baseline.json")

if __name__ == "__main__":
    train_baseline_simple()
