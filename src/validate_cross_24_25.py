import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validation_24_25")

def run_cross_validation_24_25():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. LOAD 24/25 DATA (THE TEST SLICE)
    logger.info("Loading 24/25 match rosters for cross-validation...")
    roster_24_25 = pd.read_parquet(raw_dir / "understat_rosters_2024.parquet")
    roster_24_25['norm_name'] = roster_24_25['player'].apply(normalize_name)
    roster_24_25['season_year'] = '2024'
    
    # Feature Engineering for 24/25
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes']
    for m in metrics:
        roster_24_25[m] = pd.to_numeric(roster_24_25[m], errors='coerce').fillna(0)
        roster_24_25[f'prev_{m}_5'] = roster_24_25.groupby('player_id')[m].shift(1).fillna(0)
    
    roster_24_25['goal_efficiency'] = roster_24_25['goals'] - roster_24_25['xG']
    roster_24_25['assist_efficiency'] = roster_24_25['assists'] - roster_24_25['xA']
    for m in ['goal_efficiency', 'assist_efficiency']:
        roster_24_25[f'prev_{m}_5'] = roster_24_25.groupby('player_id')[m].shift(1).fillna(0)

    # 2. GENERATE DNA PRIORS
    logger.info("Generating Career DNA Priors...")
    dna_model = xgb.Booster(); dna_model.load_model(model_dir / "layer0_dna.json")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feats = dna_model.feature_names
    
    # Use DNA from before 2024
    dna_lookup = l0_features_df[l0_features_df['target_season'] < '2024'].sort_values('target_season').groupby('norm_name').tail(1)
    if not dna_lookup.empty:
        dna_lookup['dna_prior_xG'] = dna_model.predict(xgb.DMatrix(dna_lookup[l0_feats]))
    
    roster_24_25 = roster_24_25.merge(dna_lookup[['norm_name', 'dna_prior_xG']], on='norm_name', how='left')
    roster_24_25['dna_prior_xG'] = roster_24_25['dna_prior_xG'].fillna(0)

    # 3. DEFINE MODELS
    gen_configs = {
        'Legacy Standard': {'path': model_dir, 'prefix': 'level1_', 'dna': False, 'eff': False},
        'Legacy Efficiency': {'path': model_dir / 'experiments', 'prefix': 'level1_', 'suffix': '_efficiency', 'dna': False, 'eff': True},
        'New Prod (DNA)': {'path': model_dir / 'experiments', 'prefix': 'new_prod_level1_', 'dna': True, 'eff': False},
        'New Exp (DNA+EFF)': {'path': model_dir / 'experiments', 'prefix': 'new_exp_level1_', 'dna': True, 'eff': True}
    }
    
    positions = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    results = []

    for gen_name, cfg in gen_configs.items():
        logger.info(f"Evaluating {gen_name}...")
        
        for pos_key, pos_val in positions.items():
            pos_mask = roster_24_25['position'].str.contains(pos_val, na=False, case=False)
            pos_df = roster_24_25[pos_mask].dropna(subset=['xG'])
            if pos_df.empty: continue
            
            # Special case for Legacy naming
            m_key = pos_key
            if gen_name.startswith('Legacy'):
                if pos_key == 'DEF': m_key = 'DF'
                if pos_key == 'MID': m_key = 'MF'
                if pos_key == 'FWD': m_key = 'FW'
            
            m_name = f"{cfg['prefix']}{m_key}{cfg.get('suffix', '')}.json"
            m_path = cfg['path'] / m_name
            
            if not m_path.exists():
                continue
                
            model = xgb.Booster(); model.load_model(m_path)
            feats = model.feature_names
            
            # Predict (Safe feature mapping)
            X = pos_df.reindex(columns=feats, fill_value=0.0).astype(float)
            dtest = xgb.DMatrix(X.values, feature_names=feats)
            preds = model.predict(dtest)
            mae = mean_absolute_error(pos_df['xG'], preds)
            
            results.append({
                'Generation': gen_name,
                'Position': pos_key,
                'MAE': round(mae, 4),
                'Samples': len(pos_df)
            })

    # Summary Report
    res_df = pd.DataFrame(results)
    pivot = res_df.pivot(index='Generation', columns='Position', values='MAE')
    pivot['Global Avg'] = pivot.mean(axis=1)
    
    print("\n" + "="*60)
    print("THE 4-GENERATION CROSS-VALIDATION RESULTS (SEASON 24/25)")
    print("="*60)
    print(pivot.to_string())
    print("="*60)
    
if __name__ == "__main__":
    run_cross_validation_24_25()
