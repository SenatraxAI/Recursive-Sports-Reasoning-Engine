import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import logging
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate")

def train_calibration():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models/experimental")
    
    logger.info("Loading Data and L4 Model...")
    df = pd.read_parquet(processed_dir / "level2_enriched_training.parquet")
    
    # Load L4 features and model
    with open(model_dir / "level4_features.json", "r") as f:
        l4_features = json.load(f)
        
    model = xgb.Booster()
    model.load_model(model_dir / "level4_learned_outcome.json")
    
    # Need L3 signals first? Yes, L4 depends on them.
    # Re-generate L3 signals (copy-paste logic from train L4)
    # Ideally this feature generation should be in a shared function, but for now we duplicate for speed.
    l3_model = xgb.Booster()
    l3_model.load_model(model_dir / "level3_learned_tactician.json")
    with open(model_dir / "level3_features.json", "r") as f:
        l3_feats = json.load(f)
        
    df['fit_delta'] = df['h_system_fit'] - df['a_system_fit']
    df['h_traffic'] = df['h_density_central'] / (df['h_width_balance'] + 0.1)
    df['a_traffic'] = df['a_density_central'] / (df['a_width_balance'] + 0.1)
    df['traffic_delta'] = df['h_traffic'] - df['a_traffic']
    
    X_l3 = df[l3_feats].astype(float)
    l3_classes = l3_model.predict(xgb.DMatrix(X_l3))
    df['tactical_state'] = l3_classes.astype(int)
    df['is_deadlock'] = (df['tactical_state'] == 0).astype(int)
    df['is_h_control'] = (df['tactical_state'] == 1).astype(int)
    df['is_a_control'] = (df['tactical_state'] == 2).astype(int)
    df['is_chaos'] = (df['tactical_state'] == 3).astype(int)
    
    # Predict Raw Scores with L4
    logger.info("Generating raw model scores...")
    X_l4 = df[l4_features].astype(float)
    raw_preds = model.predict(xgb.DMatrix(X_l4)) # (N, 3) array [Home, Draw, Away]
    
    # Prepare Targets
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0 # Home
        if row['home_goals'] < row['away_goals']: return 2 # Away
        return 1 # Draw
    y_true = df.apply(get_result, axis=1).values
    
    # Train Calibrator for each class
    # Isotonic Regression is non-parametric (flexible)
    calibrators = {}
    
    # Class 0: Home
    logger.info("Calibrating Home Probabilities...")
    iso_h = IsotonicRegression(out_of_bounds='clip')
    y_h_binary = (y_true == 0).astype(int)
    iso_h.fit(raw_preds[:, 0], y_h_binary)
    calibrators['home'] = iso_h
    
    # Class 1: Draw
    logger.info("Calibrating Draw Probabilities...")
    iso_d = IsotonicRegression(out_of_bounds='clip')
    y_d_binary = (y_true == 1).astype(int)
    iso_d.fit(raw_preds[:, 1], y_d_binary)
    calibrators['draw'] = iso_d
    
    # Class 2: Away
    logger.info("Calibrating Away Probabilities...")
    iso_a = IsotonicRegression(out_of_bounds='clip')
    y_a_binary = (y_true == 2).astype(int)
    iso_a.fit(raw_preds[:, 2], y_a_binary)
    calibrators['away'] = iso_a
    
    # Save
    out_path = model_dir / "calibration_models.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(calibrators, f)
        
    logger.info(f"Calibration models saved to {out_path}")

if __name__ == "__main__":
    train_calibration()
