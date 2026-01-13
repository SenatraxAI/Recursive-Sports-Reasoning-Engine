import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_l4_learned")

def train_learned_level4():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models/experimental")
    
    logger.info("Loading enriched training data...")
    df = pd.read_parquet(processed_dir / "level2_enriched_training.parquet")
    
    # 1. LOAD UPSTREAM MODELS (L3)
    # We need to generate L3 predictions to feed into L4
    l3_model = xgb.Booster()
    l3_model.load_model(model_dir / "level3_learned_tactician.json")
    with open(model_dir / "level3_features.json", "r") as f:
        l3_features = json.load(f)
        
    # 2. GENERATE L3 SIGNALS (The Tactical Context)
    # Re-engineer the L3 features exactly as in train_level3_learned.py
    df['fit_delta'] = df['h_system_fit'] - df['a_system_fit']
    df['h_traffic'] = df['h_density_central'] / (df['h_width_balance'] + 0.1)
    df['a_traffic'] = df['a_density_central'] / (df['a_width_balance'] + 0.1)
    df['traffic_delta'] = df['h_traffic'] - df['a_traffic']
    
    X_l3 = df[l3_features].astype(float)
    l3_classes = l3_model.predict(xgb.DMatrix(X_l3))
    
    # 3. Add L3 Class as a Categorical Feature
    # Since we trained with multi:softmax, we get 0, 1, 2, 3
    df['tactical_state'] = l3_classes.astype(int)
    
    # One-hot encode for L4
    df['is_deadlock'] = (df['tactical_state'] == 0).astype(int)
    df['is_h_control'] = (df['tactical_state'] == 1).astype(int)
    df['is_a_control'] = (df['tactical_state'] == 2).astype(int)
    df['is_chaos'] = (df['tactical_state'] == 3).astype(int)
    
    # 4. DEFINE TARGET
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0 # Home
        if row['home_goals'] < row['away_goals']: return 2 # Away
        return 1 # Draw
        
    df['result'] = df.apply(get_result, axis=1)
    
    # 4. TRAIN LEVEL 4 (Outcomes)
    # Features: The Best of Every Layer
    features = [
        # L3 Signals (Tactical Flow)
        'is_deadlock', 'is_h_control', 'is_a_control', 'is_chaos',
        # L2 Signals (Structure & Fit)
        'fit_delta', 'traffic_delta', 'h_system_fit', 'a_system_fit',
        # Raw Power Signals (Base xG)
        'home_xg', 'away_xg' 
    ]
    
    X = df[features].astype(float)
    y = df['result'].astype(int)
    
    logger.info(f"Training Final Outcome Model on {len(df)} matches...")
    
    model = xgb.train(
        params={
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 4,
            'eta': 0.02, # Slow learner to prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss'
        },
        dtrain=xgb.DMatrix(X, label=y),
        num_boost_round=1000,
        verbose_eval=50
    )
    
    # 5. SAVE
    out_path = model_dir / "level4_learned_outcome.json"
    model.save_model(out_path)
    
    with open(model_dir / "level4_features.json", "w") as f:
        json.dump(features, f)
        
    logger.info(f"Final Brain saved to {out_path}")
    
    # Feature Importance
    imp = model.get_score(importance_type='gain')
    logger.info("\n--- FINAL DECISION DRIVERS ---")
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    for f, score in sorted_imp:
        logger.info(f"{f}: {score:.2f}")

if __name__ == "__main__":
    train_learned_level4()
