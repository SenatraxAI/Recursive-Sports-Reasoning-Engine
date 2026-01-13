import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def run_ab_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting A/B Test: Standard vs Efficiency Level 1 Models...")
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    
    # 1. Load Data
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = []
    for f in roster_files:
        season = f.name.split('_')[-1].split('.')[0]
        if season in ['2024', '2025']:
            all_rosters.append(pd.read_parquet(f))
    
    test_df = pd.concat(all_rosters)
    test_df['norm_name'] = test_df['player'].apply(normalize_name)
    test_df = test_df.sort_values(['player_id', 'match_id'])
    
    # Load Identity for A
    identity_df = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    identity_df['norm_name'] = identity_df['player_name'].apply(normalize_name)
    
    # 2. Feature Engineering (Common + Specific)
    metrics = ['goals', 'xG', 'assists', 'xA', 'xGChain', 'xGBuildup']
    for m in metrics:
        if m in test_df.columns:
            test_df[m] = pd.to_numeric(test_df[m], errors='coerce').fillna(0)
        else:
            test_df[m] = 0.0
        test_df[f'prev_{m}_5'] = test_df.groupby('player_id')[m].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    
    # Efficiency Specific
    test_df['goal_efficiency'] = test_df['goals'] - test_df['xG']
    test_df['assist_efficiency'] = test_df['assists'] - test_df['xA']
    test_df[f'prev_goal_efficiency_5'] = test_df.groupby('player_id')['goal_efficiency'].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    test_df[f'prev_assist_efficiency_5'] = test_df.groupby('player_id')['assist_efficiency'].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    test_df['prev_defensive_efficiency_5'] = 0.0 # Neutral placeholder for AB
    
    # Merge for A (Attributes)
    full_matrix = pd.merge(test_df, identity_df.drop_duplicates(subset=['norm_name']), on='norm_name', how='left')
    full_matrix['Position_Clean'] = full_matrix.get('Position', full_matrix.get('pos', 'MF'))
    
    # DEBUG: Print columns to verify case
    print("DEBUG: Columns in full_matrix:", full_matrix.columns.tolist()[:20])
    
    # Ensure canonical 'xG'
    if 'xg' in full_matrix.columns and 'xG' not in full_matrix.columns:
        full_matrix = full_matrix.rename(columns={'xg': 'xG'})
    elif 'xG' not in full_matrix.columns:
        full_matrix['xG'] = 0.0

    # Production Mapping (Jan 10)
    # Standard Model A was trained using different codes
    std_map = {'GK': 'level1_GK.json', 'DEF': 'level1_DF.json', 'MID': 'level1_MF.json', 'FWD': 'level1_FW.json'}
    # Experimental Mapping
    exp_map = {'GK': 'level1_GK_efficiency.json', 'DEF': 'level1_DEF_efficiency.json', 'MID': 'level1_MID_efficiency.json', 'FWD': 'level1_FWD_efficiency.json'}

    positions = {'GK': 'GK', 'DEF': 'DF', 'MID': 'MF', 'FWD': 'FW'}
    results = []

    for label, code in positions.items():
        pos_df = full_matrix[full_matrix['Position_Clean'].str.contains(code, na=False, case=False)].copy()
        if pos_df.empty: continue
        
        # Ensure column exists before dropna
        if 'xG' in pos_df.columns:
            pos_df = pos_df.dropna(subset=['xG'])
        else:
            continue
            
        y_true = pos_df['xG'].values
        
        # --- MODEL A (Standard Production) ---
        path_a = model_dir / std_map.get(label, f"level1_{label}.json")
        if path_a.exists():
            mod_a = xgb.Booster(); mod_a.load_model(path_a)
            a_feats = mod_a.feature_names
            if a_feats:
                X_a = pos_df.reindex(columns=a_feats, fill_value=0.0).astype(float)
                preds_a = mod_a.predict(xgb.DMatrix(X_a))
                mae_a = mean_absolute_error(y_true, preds_a)
                r2_a = r2_score(y_true, preds_a)
            else:
                mae_a, r2_a = 0, 0
        else:
            mae_a, r2_a = 0, 0

        # --- MODEL B (Efficiency Experiment) ---
        path_b = model_dir / "experiments" / exp_map.get(label, f"level1_{label}_efficiency.json")
        if path_b.exists():
            mod_b = xgb.Booster(); mod_b.load_model(path_b)
            b_feats = mod_b.feature_names
            if b_feats:
                if 'age' in b_feats and 'age' not in pos_df.columns:
                    pos_df['age'] = 27
                elif 'age' in b_feats:
                    pos_df['age'] = pd.to_numeric(pos_df['age'], errors='coerce').fillna(27)
                    
                X_b = pos_df.reindex(columns=b_feats, fill_value=0.0).astype(float)
                preds_b = mod_b.predict(xgb.DMatrix(X_b))
                mae_b = mean_absolute_error(y_true, preds_b)
                r2_b = r2_score(y_true, preds_b)
            else:
                mae_b, r2_b = 0, 0
        else:
            mae_b, r2_b = 0, 0
            
        results.append({
            'Pos': label,
            'MAE A': mae_a, 'MAE B': mae_b,
            'R2 A': r2_a, 'R2 B': r2_b,
            'Winner': 'B' if mae_b < mae_a and mae_b > 0 else 'A'
        })

    print("\n" + "="*70)
    print(f"{'POSITION':<10} | {'MAE A (Std)':<12} | {'MAE B (Eff)':<12} | {'IMPROVEMENT'}")
    print("="*70)
    for r in results:
        imp = ((r['MAE A'] - r['MAE B']) / r['MAE A'] * 100) if r['MAE A'] > 0 else 0
        print(f"{r['Pos']:<10} | {r['MAE A']:<12.4f} | {r['MAE B']:<12.4f} | {imp:>10.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_ab_test()
