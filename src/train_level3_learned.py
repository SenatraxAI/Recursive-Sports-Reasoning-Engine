import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path
from src.trainer import HierarchicalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_l3_learned")

def train_learned_level3():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models/experimental")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading Enriched Data for Tactical Analysis...")
    df = pd.read_parquet(processed_dir / "level2_enriched_training.parquet")
    
    # 1. Feature Engineering: The Tactical Deltas
    # We want features that describe the *difference* in tactical execution
    
    df['fit_delta'] = df['h_system_fit'] - df['a_system_fit']
    
    # Traffic Jam / Density Deltas
    df['h_traffic'] = df['h_density_central'] / (df['h_width_balance'] + 0.1)
    df['a_traffic'] = df['a_density_central'] / (df['a_width_balance'] + 0.1)
    df['traffic_delta'] = df['h_traffic'] - df['a_traffic']
    
    # We also need Manager Styles if available in the enriched set.
    # The enriched set currently focused on L2 features. 
    # To include Styles, we would need to merge back with 'processed_matches.parquet'.
    # For this Experimental Run, let's test if "System Fit" alone drives the tactical prediction.
    # (Hypothesis: Fit is a proxy for how well the style is interacting).
    
    # 2. Define The Target: Game State Classification
    # 0 = Deadlock (Low xG both sides)
    # 1 = Home Control (Home xG >> Away xG)
    # 2 = Away Control (Away xG >> Home xG)
    # 3 = Chaos (High xG both sides)
    
    def classify_game(row):
        h_xg, a_xg = row['home_xg'], row['away_xg']
        total_xg = h_xg + a_xg
        diff = h_xg - a_xg
        
        if total_xg < 1.5:
            return 0 # Deadlock (Cagey)
        
        if diff > 0.5:
            return 1 # Home Control
        elif diff < -0.5:
            return 2 # Away Control
        else:
            return 3 # Chaos (High scoring draw/exchange)
            
    df['game_state'] = df.apply(classify_game, axis=1)
    
    # 3. Training
    features = ['h_system_fit', 'a_system_fit', 'fit_delta', 
                'h_traffic', 'a_traffic', 'traffic_delta']
                
    X = df[features].astype(float)
    y = df['game_state'].astype(int)
    
    logger.info(f"Training Tactical Classifier (Classes: Deadlock, H-Control, A-Control, Chaos)...")
    
    model = xgb.train(
        params={
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8
        },
        dtrain=xgb.DMatrix(X, label=y),
        num_boost_round=500
    )
    
    # 4. Save
    out_path = model_dir / "level3_learned_tactician.json"
    model.save_model(out_path)
    
    with open(model_dir / "level3_features.json", "w") as f:
        json.dump(features, f)
        
    logger.info(f"Tactical Model saved to {out_path}")
    
    # Quick feature importance check
    imp = model.get_score(importance_type='gain')
    logger.info("Top Tactical Drivers:")
    print(sorted(imp.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    train_learned_level3()
