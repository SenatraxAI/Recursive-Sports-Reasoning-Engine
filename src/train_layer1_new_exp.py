import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage3_layer1_exp")

def train_new_exp_layer1():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. Load DNA Layer
    logger.info("Loading Layer 0 DNA Profiler...")
    dna_model = xgb.Booster()
    dna_model.load_model(model_dir / "layer0_dna.json")
    
    # 2. Get DNA features
    logger.info("Loading DNA features...")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feature_names = dna_model.feature_names
    
    # 3. Load Roster Data
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
    
    # ADVANCED FEATURE ENGINEERING: Efficiency + Form
    logger.info("Engineering clinicality and rolling form...")
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes']
    for m in metrics:
        roster_df[m] = pd.to_numeric(roster_df[m], errors='coerce').fillna(0)
        
    # G-xG Efficiency
    roster_df['goal_efficiency'] = roster_df['goals'] - roster_df['xG']
    roster_df['assist_efficiency'] = roster_df['assists'] - roster_df['xA']
    
    # Rolling stats (Momentum)
    for m in metrics + ['goal_efficiency', 'assist_efficiency']:
        roster_df[f'prev_{m}_5'] = roster_df.groupby('player_id')[m].shift(1).fillna(0)

    # 4. ENRICH WITH LAYER 0 DNA
    logger.info("Generating DNA Priors...")
    dna_lookup = l0_features_df[['norm_name', 'target_season']].copy()
    X_l0 = l0_features_df[l0_feature_names]
    dna_lookup['dna_prior_xG'] = dna_model.predict(xgb.DMatrix(X_l0))
    
    # 5. Train New Exp Models
    # Updated pos_map for Understat codes: DF captures D-prefixed, MF captures M-prefixed/AMC/etc.
    pos_map = {
        'GK': 'GK',
        'DEF': 'D',    # Matches DR, DC, DL, DMC
        'MID': 'M',    # Matches MC, AMC, MR, ML, DMR
        'FWD': 'FW'    # Matches FW, FWR, FWL
    }
    
    # Merge DNA Prior into Layer 1 matrix
    roster_df = roster_df.merge(
        dna_lookup,
        left_on=['norm_name', 'season_year'],
        right_on=['norm_name', 'target_season'],
        how='left'
    )

    # Impute missing DNA priors by position instead of global zero
    # This prevents the model from assuming every unknown player has the same baseline.
    logger.info("Imputing missing DNA priors by position...")
    for pos_key, pos_val in pos_map.items():
        pos_mask = roster_df['position'].str.contains(pos_val, na=False, case=False)
        median_dna = roster_df[pos_mask]['dna_prior_xG'].median()
        if pd.isna(median_dna): median_dna = 0.0
        roster_df.loc[pos_mask & roster_df['dna_prior_xG'].isna(), 'dna_prior_xG'] = median_dna
    
    roster_df['dna_prior_xG'] = roster_df['dna_prior_xG'].fillna(0.0)
    
    # 5. Train New Exp Models
    trainer = HierarchicalTrainer()
    trainer.experimental_mode = True # models go to /experiments/
    
    # Train up to 2025 for final production deployment
    train_mask = roster_df['season_year'].isin(['2020', '2021', '2022', '2023', '2024', '2025'])
    
    for pos_key, pos_val in pos_map.items():
        pos_mask = roster_df['position'].str.contains(pos_val, na=False, case=False)
        train_data = roster_df[pos_mask & train_mask]
        
        if train_data.empty: continue
        
        # Features: DNA Prior + Form + Clinicality
        form_feats = [f'prev_{m}_5' for m in metrics]
        eff_feats = ['prev_goal_efficiency_5', 'prev_assist_efficiency_5']
        final_features = form_feats + eff_feats + ['dna_prior_xG', 'time']
        
        X = train_data[final_features].fillna(0).astype(float)
        y = train_data['xG'] # Target still xG (volume prediction), but aware of efficiency
        
        logger.info(f"Training New Exp Layer 1 ({pos_key}) [DNA + EFF]...")
        trainer.train_level1_player_models(X, y, pos_key, save_name=f"new_exp_level1_{pos_key}.json")

if __name__ == "__main__":
    train_new_exp_layer1()
