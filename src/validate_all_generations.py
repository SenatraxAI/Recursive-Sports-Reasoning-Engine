import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validation_battle")

def run_battle_2026():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    model_dir = Path("data/models")
    
    # 1. LOAD 2026 DATA
    logger.info("Loading January 2026 validation slice...")
    roster_2026 = pd.read_parquet(raw_dir / "understat_rosters_2026.parquet")
    roster_2026['norm_name'] = roster_2026['player'].apply(normalize_name)
    roster_2026['season_year'] = '2026'
    
    # Feature Engineering for 2026
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes']
    for m in metrics:
        roster_2026[m] = pd.to_numeric(roster_2026[m], errors='coerce').fillna(0)
        # For simplicity in validation, we use a global shift
        roster_2026[f'prev_{m}_5'] = roster_2026.groupby('player_id')[m].shift(1).fillna(0)
    
    roster_2026['goal_efficiency'] = roster_2026['goals'] - roster_2026['xG']
    roster_2026['assist_efficiency'] = roster_2026['assists'] - roster_2026['xA']
    for m in ['goal_efficiency', 'assist_efficiency']:
        roster_2026[f'prev_{m}_5'] = roster_2026.groupby('player_id')[m].shift(1).fillna(0)

    # 2. GENERATE DNA PRIORS FOR 2026
    logger.info("Generating Career DNA Priors for 2026 context...")
    dna_model = xgb.Booster(); dna_model.load_model(model_dir / "layer0_dna.json")
    l0_features_df = pd.read_parquet(processed_dir / "layer0_training_features.parquet")
    l0_feats = dna_model.feature_names
    
    # Simple DNA lookup: use player's most recent DNA vector from training
    latest_dna = l0_features_df.sort_values('target_season').groupby('norm_name').tail(1)
    latest_dna['dna_prior_xG'] = dna_model.predict(xgb.DMatrix(latest_dna[l0_feats]))
    
    roster_2026 = roster_2026.merge(latest_dna[['norm_name', 'dna_prior_xG']], on='norm_name', how='left')
    roster_2026['dna_prior_xG'] = roster_2026['dna_prior_xG'].fillna(0)

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
        total_mae = []
        
        for pos_key, pos_val in positions.items():
            pos_mask = roster_2026['position'].str.contains(pos_val, na=False, case=False)
            pos_df = roster_2026[pos_mask].dropna(subset=['xG'])
            if pos_df.empty: continue
            
            # Load specific model
            m_name = f"{cfg['prefix']}{pos_key}{cfg.get('suffix', '')}.json"
            m_path = cfg['path'] / m_name
            
            if not m_path.exists():
                logger.warning(f"Model missing: {m_path}")
                continue
                
            model = xgb.Booster(); model.load_model(m_path)
            feats = model.feature_names
            
            # Predict
            X = pos_df.reindex(columns=feats, fill_value=0.0).astype(float)
            preds = model.predict(xgb.DMatrix(X))
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
    
    print("\n" + "="*50)
    print("THE 4-GENERATION BATTLE RESULTS (JAN 2026)")
    print("="*50)
    print(pivot.to_string())
    print("="*50)
    
    report_path = Path("C:/Users/Asus/.gemini/antigravity/brain/6d169247-1db8-4d8d-b9d7-5fc958f4b13f/final_battle_report.md")
    with open(report_path, "w") as f:
        f.write("# 🏆 Final 4-Generation Validation Report (2026)\n\n")
        f.write(pivot.to_markdown() + "\n\n")
        f.write("## Observations\n")
        f.write("- **Legacy Standard:** Baseline performance.\n")
        f.write("- **New Prod:** DNA integration for context-aware volume prediction.\n")
        f.write("- **New Exp:** Combined DNA priors with clinicality weightings.\n")
        
if __name__ == "__main__":
    run_battle_2026()
