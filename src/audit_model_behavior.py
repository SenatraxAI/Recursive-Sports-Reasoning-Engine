import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

def debug_player_prediction():
    model_dir = Path("data/models/experiments")
    
    # Load FWD model (The most sensitive to xG)
    mod_path = model_dir / "new_exp_level1_FWD.json"
    if not mod_path.exists():
        print("Model not found.")
        return
        
    model = xgb.Booster()
    model.load_model(mod_path)
    feats = model.feature_names
    
    print(f"DEBUGGING MODEL: {mod_path.name}")
    print(f"Features expected: {feats}")
    
    # TEST SCENE: An elite striker
    # DNA Prior = 0.5 xG/90 (Historical Average)
    # Recent Form = 0.3 xG/5 (Slightly out of form)
    
    test_X = pd.DataFrame(0.0, index=[0], columns=feats)
    test_X['dna_prior_xG'] = 0.5
    test_X['prev_xG_5'] = 0.3
    test_X['time'] = 90.0
    
    pred = model.predict(xgb.DMatrix(test_X.values, feature_names=feats))[0]
    print(f"\nELITE STRIKER (DNA=0.5, Form=0.3) -> Predicted xG: {pred:.4f}")
    
    # TEST SCENE: A Bench warmer
    test_X['dna_prior_xG'] = 0.05
    test_X['prev_xG_5'] = 0.0
    pred_low = model.predict(xgb.DMatrix(test_X.values, feature_names=feats))[0]
    print(f"BENCH WARMER (DNA=0.05, Form=0.0) -> Predicted xG: {pred_low:.4f}")

    # TEST SCENE: DNA Only
    test_X = pd.DataFrame(0.0, index=[0], columns=feats)
    test_X['dna_prior_xG'] = 1.0 # Unrealistically high DNA
    test_X['time'] = 90.0
    pred_hot = model.predict(xgb.DMatrix(test_X.values, feature_names=feats))[0]
    print(f"GOD-TIER (DNA=1.0, Form=0.0) -> Predicted xG: {pred_hot:.4f}")

if __name__ == "__main__":
    debug_player_prediction()
