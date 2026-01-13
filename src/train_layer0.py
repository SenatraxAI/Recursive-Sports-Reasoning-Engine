import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("layer0_training")

def train_layer0():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading high-res career history...")
    df = pd.read_parquet(processed_dir / "high_res_career_history.parquet")
    
    # Standardize types and fill NaNs
    for col in df.columns:
        if df[col].dtype == object and col not in ['player', 'norm_name', 'data_season', 'pos']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Sort by player and season
    df = df.sort_values(['norm_name', 'data_season'])
    
    # 1. Feature Engineering: Career Averages (Prior to current season)
    # We want to predict Season N using everything from [Start...N-1]
    
    # Metrics to profile
    core_metrics = [
        'expected_goals', 'assists', 'progressive_carries', 'progressive_passes',
        'key_passes', 'shot_creating_actions_p_90', 'tackles_won', 'interceptions',
        'clearances', 'saves_%%'
    ]
    
    records = []
    
    for name, group in df.groupby('norm_name'):
        group = group.sort_values('data_season')
        for i in range(1, len(group)):
            # Current season (Target)
            target_row = group.iloc[i]
            
            # Historical seasons (Features)
            history = group.iloc[:i]
            
            # Weighted average (Recent history counts more)
            feat_dict = {
                'norm_name': name,
                'target_season': target_row['data_season'],
                'pos': target_row['pos'],
                'age': target_row['age']
            }
            
            for m in core_metrics:
                if m in history.columns:
                    # Simple career mean for now, but could be exponentially weighted
                    feat_dict[f'career_avg_{m}'] = history[m].mean()
                    feat_dict[f'last_season_{m}'] = history[m].iloc[-1]
            
            # Targets (Standardized to Per 90)
            mp = target_row.get('matches_played', 1)
            if mp == 0: mp = 1
            
            feat_dict['target_xG'] = target_row['expected_goals'] / mp
            feat_dict['target_SCA'] = target_row.get('shot_creating_actions_p_90', 0)
            
            records.append(feat_dict)
            
    profile_df = pd.DataFrame(records)
    
    if profile_df.empty:
        logger.error("No multi-season history found for Layer 0 training!")
        return

    # 2. Train DNA Models (One for Offense, One for Defense/Playmaking)
    # For simplicity, we train one global "DNA Profiler" for xG
    
    X = profile_df[[c for c in profile_df.columns if 'career_avg' in c or 'last_season' in c or c == 'age']]
    y = profile_df['target_xG']
    
    logger.info(f"Training Layer 0 DNA Profiler on {len(profile_df)} samples...")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    model_path = model_dir / "layer0_dna.json"
    model.save_model(model_path)
    logger.info(f"Layer 0 DNA Profiler saved to {model_path}")
    
    # Save the feature names for inference
    profile_df.to_parquet(processed_dir / "layer0_training_features.parquet")

if __name__ == "__main__":
    train_layer0()
