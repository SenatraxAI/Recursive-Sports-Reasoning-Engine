import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_layer1")

def train_new_prod_layer1():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. Load Layer 0 DNA model
    logger.info("Loading Layer 0 DNA Profiler...")
    dna_model = xgb.Booster()
    dna_model.load_model(model_dir / "layer0_dna.json")
    
    # 2. Get the Career Baseline stats for DNA generation
    # We use the training features we saved during Layer 0 training
    logger.info("Loading Layer 0 DNA features...")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feature_names = dna_model.feature_names
    
    # 3. Load Roster Data for Layer 1
    logger.info("Loading match rosters...")
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = []
    for f in roster_files:
        season_year = f.name.split('_')[-1].split('.')[0]
        rdf = pd.read_parquet(f)
        rdf['season_year'] = season_year
        all_rosters.append(rdf)
    roster_df = pd.concat(all_rosters, ignore_index=True)
    roster_df['norm_name'] = roster_df['player'].apply(normalize_name)
    
    # Feature Engineering (Standard Rolling Form)
    logger.info("Engineering rolling form...")
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes']
    for m in metrics:
        roster_df[m] = pd.to_numeric(roster_df[m], errors='coerce').fillna(0)
        roster_df[f'prev_{m}_5'] = roster_df.groupby('player_id')[m].shift(1).fillna(0)

    # 4. ENRICH WITH LAYER 0 DNA
    logger.info("Interpreting player DNA for match context...")
    # Map L0 features to roster
    # L0 needs 'career_avg_x', etc. 
    # For training L1, we use the L0 output for that specific player in that specific season.
    
    # Create a lookup for DNA Predictions per player-season
    dna_lookup = l0_features_df[['norm_name', 'target_season']].copy()
    X_l0 = l0_features_df[l0_feature_names]
    dna_lookup['dna_prior_xG'] = dna_model.predict(xgb.DMatrix(X_l0))
    
    # Merge DNA Prior into Layer 1 matrix
    # Note: 'target_season' in DNA lookup corresponds to 'season_year' in rosters
    roster_df = roster_df.merge(
        dna_lookup,
        left_on=['norm_name', 'season_year'],
        right_on=['norm_name', 'target_season'],
        how='left'
    )
    roster_df['dna_prior_xG'] = roster_df['dna_prior_xG'].fillna(0)
    
    # 5. Train New Prod Models
    pos_map = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    trainer = HierarchicalTrainer()
    trainer.experimental_mode = True # Use 'experiments' subfolder for New Prod (can rename later)
    
    # Filter up to 2023 for training, so we can test on 2024/25
    train_mask = roster_df['season_year'].isin(['2020', '2021', '2022', '2023'])
    
    for pos_key, pos_val in pos_map.items():
        pos_mask = roster_df['position'].str.contains(pos_val, na=False, case=False)
        train_data = roster_df[pos_mask & train_mask]
        
        if train_data.empty: continue
        
        # Features: Rolling Form + DNA Prior
        form_feats = [f'prev_{m}_5' for m in metrics]
        final_features = form_feats + ['dna_prior_xG', 'time']
        
        X = train_data[final_features].fillna(0).astype(float)
        y = train_data['xG']
        
        logger.info(f"Training New Prod Layer 1 ({pos_key}) with DNA Priors...")
        trainer.train_level1_player_models(X, y, pos_key, save_name=f"new_prod_level1_{pos_key}.json")

if __name__ == "__main__":
    train_new_prod_layer1()
