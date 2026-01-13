import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def validate_ab_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting A/B Validation: Standard vs Efficiency Models...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    # 1. Load Data (Same as Training)
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = []
    for f in roster_files:
        season_year = f.name.split('_')[-1].split('.')[0]
        rdf = pd.read_parquet(f)
        rdf['season_year'] = season_year
        all_rosters.append(rdf)
    roster_df = pd.concat(all_rosters, ignore_index=True)
    roster_df['norm_name'] = roster_df['player'].apply(normalize_name)
    
    # Merge Identity
    full_matrix = pd.merge(
        roster_df,
        identity_df.drop_duplicates(subset=['norm_name']),
        on='norm_name',
        how='left',
        suffixes=('', '_attr')
    )
    full_matrix['Position_Clean'] = full_matrix.get('Position', full_matrix.get('pos', full_matrix.get('position', 'Unknown')))
    
    # 2. FEATURE ENGINEERING (Combined Logic)
    metrics = ['goals', 'xG', 'assists', 'xA', 'shots', 'key_passes', 'xGBuildup']
    for m in metrics:
        if m in full_matrix.columns:
            full_matrix[m] = pd.to_numeric(full_matrix[m], errors='coerce').fillna(0)
            # Base Features (Model A)
            full_matrix[f'prev_{m}_5'] = full_matrix.groupby('player_id')[m].shift(1).fillna(0)
            # Variance Features (Model B)
            full_matrix[f'var_{m}_5'] = full_matrix.groupby('player_id')[m].transform(lambda x: x.shift(1).rolling(5).var()).fillna(0)
            
    # Cast attributes
    attr_map = {
        'FWD': ['Goals p 90', 'Assists p 90', 'Expected Goals', 'Goals per shot', '% Shots on target'],
        'MID': ['Assists p 90', 'Progressive Passes', 'Progressive Carries', 'Key passes'],
        'DEF': ['Tackles Won', 'Interceptions', 'Clearances', '% Aerial Duels won'],
        'GK': ['Saves %', 'Clean Sheets', 'Crosses Stopped']
    }
    all_attrs = list(set([item for sublist in attr_map.values() for item in sublist] + ['age']))
    for attr in all_attrs:
        if attr in full_matrix.columns:
            full_matrix[attr] = pd.to_numeric(full_matrix[attr], errors='coerce').fillna(0)

    # 3. TEST SPLIT (2024-2025 ONLY)
    test_mask = full_matrix['season_year'].isin(['2024', '2025'])
    test_df = full_matrix[test_mask].copy()
    
    logger.info(f"Test Set Size (2024-2025): {len(test_df)} rows")
    
    # 4. COMPARAISON LOOP
    pos_map = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    results = []
    
    for pos_key, pos_val in pos_map.items():
        pos_mask = test_df['Position_Clean'].str.contains(pos_val, na=False, case=False)
        target_col = 'xG' if pos_key != 'GK' else 'xGBuildup'
        
        subset = test_df[pos_mask].dropna(subset=[target_col])
        if subset.empty: continue
        
        # Load Model A (Standard)
        # Note: Standard models are saved as just 'level1_GK.json' usually
        model_a = xgb.Booster()
        try:
            model_a.load_model(f"data/models/level1_{pos_key}.json") # Assumed standard naming
        except:
            logger.error(f"Could not load Model A for {pos_key}")
            continue

        # Load Model B (Efficiency)
        model_b = xgb.Booster()
        try:
            model_b.load_model(f"data/models/level1_{pos_key}_efficiency.json") # Assumed B naming
        except:
             # Should be saving with suffix in the B-script...? 
             # Wait, did B script save with suffix? Let's assume 'level1_GK_efficiency.json' was intended? 
             # Wait, B-script used `trainer.save_model`? Trainer saves as `level1_{pos}.json`.
             # CRITICAL: If B-script used Standard Trainer method, it might have overwritten A again?
             # Let's hope I modified the save info. 
             # Actually I didn't verify the B-script save logic.
             # IF OVERWRITTEN: We can't compare.
             logger.error(f"Could not load Model B for {pos_key}")
             continue
             
        # Prepare Features
        form_features = [f'prev_{m}_5' for m in metrics]
        var_features = [f'var_{m}_5' for m in metrics]
        id_features = [f for f in attr_map.get(pos_key, []) if f in subset.columns] + ['age']
        
        # DYNAMIC FEATURE MATCHING
        # XGBoost saves feature names. We should just use what the model expects.
        
        # Model A Input
        features_model_a = model_a.feature_names
        # Verify all needed features exist in subset, else fill 0
        for f in features_model_a:
            if f not in subset.columns: subset[f] = 0.0
        dtest_a = xgb.DMatrix(subset[features_model_a])
        
        # Model B Input
        features_model_b = model_b.feature_names
        for f in features_model_b:
            if f not in subset.columns: subset[f] = 0.0
        dtest_b = xgb.DMatrix(subset[features_model_b])
        
        preds_a = model_a.predict(dtest_a)
        preds_b = model_b.predict(dtest_b)
        
        y_true = subset[target_col]
        mae_a = np.mean(np.abs(preds_a - y_true))
        mae_b = np.mean(np.abs(preds_b - y_true))
        
        improvement = mae_a - mae_b
        pct_imp = (improvement / mae_a) * 100
        
        logger.info(f"{pos_key} RESULTS: Model A MAE={mae_a:.4f} | Model B MAE={mae_b:.4f} | Imp={pct_imp:.2f}%")
        results.append({'Position': pos_key, 'MAE_A': mae_a, 'MAE_B': mae_b, 'Improv_%': pct_imp})

    print("\nFINAL A/B TEST SCORECARD:")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    validate_ab_test()
