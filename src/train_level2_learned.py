import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
from src.trainer import HierarchicalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_l2_learned")

def train_learned_level2():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models/experimental")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading Enriched Training Data...")
    df = pd.read_parquet(processed_dir / "level2_enriched_training.parquet")
    
    # We need to train two models: Home Offense and Away Offense (or a shared one)
    # Let's train a shared Team Offense model for better generalization.
    
    # 1. Prepare Dataset (Stack Home and Away rows)
    h_df = df[['h_system_fit', 'h_density_central', 'h_density_attack', 'h_width_balance', 'home_xg', 'home_goals']].copy()
    h_df.columns = ['system_fit', 'density_central', 'density_attack', 'width_balance', 'team_xg_sum', 'actual_goals']
    
    a_df = df[['a_system_fit', 'a_density_central', 'a_density_attack', 'a_width_balance', 'away_xg', 'away_goals']].copy()
    a_df.columns = ['system_fit', 'density_central', 'density_attack', 'width_balance', 'team_xg_sum', 'actual_goals']
    
    train_df = pd.concat([h_df, a_df], ignore_index=True)
    
    # 2. Add Interaction Features (e.g., Density * Width)
    train_df['traffic_jam_score'] = train_df['density_central'] * (1.0 / (train_df['width_balance'] + 0.1))
    
    # 3. Define Features and Target
    features = ['team_xg_sum', 'system_fit', 'density_central', 'density_attack', 'width_balance', 'traffic_jam_score']
    target = 'actual_goals' # We want to predict ACTUAL outcomes, not just xG, to capture efficiency
    
    X = train_df[features].astype(float)
    y = train_df[target].astype(float)
    
    # 4. Train XGBoost
    logger.info(f"Training Learned Aggregator on {len(train_df)} team-matches...")
    
    model = xgb.train(
        params={
            'objective': 'reg:absoluteerror', # Robust to outliers (like 7-0 wins)
            'max_depth': 4,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mae'
        },
        dtrain=xgb.DMatrix(X, label=y),
        num_boost_round=500
    )
    
    # 5. Analyze Feature Importance
    importance = model.get_score(importance_type='gain')
    logger.info("\n--- FEATURE IMPORTANCE (What matters for Team Output?) ---")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for f, score in sorted_imp:
        logger.info(f"{f}: {score:.2f}")
        
    # 6. Save
    out_path = model_dir / "level2_learned_aggregator.json"
    model.save_model(out_path)
    logger.info(f"Model saved to {out_path}")
    
    # Save feature names
    with open(model_dir / "level2_features.json", "w") as f:
        import json
        json.dump(features, f)

if __name__ == "__main__":
    train_learned_level2()
