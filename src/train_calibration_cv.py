import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import logging
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate_cv")

def train_calibration_cv():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models/experimental")
    
    logger.info("Loading Data...")
    df = pd.read_parquet(processed_dir / "level2_enriched_training.parquet")
    
    # Load Feature Defs
    with open(model_dir / "level4_features.json", "r") as f: l4_feats = json.load(f)
    with open(model_dir / "level3_features.json", "r") as f: l3_feats = json.load(f)
    
    # Helper: L3 Signal Generation (Need to do this on the fly or pre-calc?)
    # We should pre-calc L3 signals for the whole dataset first for speed.
    l3_model = xgb.Booster()
    l3_model.load_model(model_dir / "level3_learned_tactician.json")
    
    # Feature Engineering (Same as before)
    df['fit_delta'] = df['h_system_fit'] - df['a_system_fit']
    df['h_traffic'] = df['h_density_central'] / (df['h_width_balance'] + 0.1)
    df['a_traffic'] = df['a_density_central'] / (df['a_width_balance'] + 0.1)
    df['traffic_delta'] = df['h_traffic'] - df['a_traffic']
    
    X_l3 = df[l3_feats].astype(float)
    l3_classes = l3_model.predict(xgb.DMatrix(X_l3))
    df['tactical_state'] = l3_classes.astype(int)
    # One-hot
    for i, name in enumerate(['is_deadlock', 'is_h_control', 'is_a_control', 'is_chaos']):
        df[name] = (df['tactical_state'] == i).astype(int)
        
    X = df[l4_feats].astype(float).values
    
    # Target
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        if row['home_goals'] < row['away_goals']: return 2
        return 1
    y = df.apply(get_result, axis=1).values
    
    # START CROSS-VALIDATION FOR OOF PREDICTIONS
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(df), 3)) # [Home, Draw, Away]
    
    logger.info("Generating OOF Predictions (5 Folds)...")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,
        'eta': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Train a TEMPORARY L4 model on this fold
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val)
        
        tmp_model = xgb.train(params, dtrain, num_boost_round=1000)
        
        # Predict on Val
        preds = tmp_model.predict(dval)
        oof_preds[val_idx] = preds
        
        logger.info(f"Fold {fold+1} complete.")
        
    # NOW TRAIN CALIBRATORS ON OOF PREDICTIONS
    logger.info("Fitting Isotonic Regression on OOF Predictions...")
    
    calibrators = {}
    
    # Class 0: Home
    iso_h = IsotonicRegression(out_of_bounds='clip')
    iso_h.fit(oof_preds[:, 0], (y == 0).astype(int))
    calibrators['home'] = iso_h
    
    # Class 1: Draw
    iso_d = IsotonicRegression(out_of_bounds='clip')
    iso_d.fit(oof_preds[:, 1], (y == 1).astype(int))
    calibrators['draw'] = iso_d
    
    # Class 2: Away
    iso_a = IsotonicRegression(out_of_bounds='clip')
    iso_a.fit(oof_preds[:, 2], (y == 2).astype(int))
    calibrators['away'] = iso_a
    
    # Save
    out_path = model_dir / "calibration_models_cv.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(calibrators, f)
        
    logger.info(f"Correct CV-Calibration models saved to {out_path}")

if __name__ == "__main__":
    train_calibration_cv()
