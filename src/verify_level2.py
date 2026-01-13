import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import normalize_name
from sklearn.metrics import mean_squared_error, r2_score

def run_level2_audit():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- 🏟️ LEVEL 2 TEAM ACCURACY AUDIT (2024-2025 DATA) ---")
    
    # 1. Load Data (Replicating Logic from train_level2.py)
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df['id'] = matches_df['id'].astype(str)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    l1_matrix = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    l1_matrix['norm_name'] = l1_matrix['player_name'].apply(normalize_name)
    # Fix Position
    l1_matrix['Position_Clean'] = l1_matrix.get('Position', l1_matrix.get('pos', 'Unknown'))

    injury_df = pd.read_parquet(processed_dir / "heuristic_injuries.parquet")
    
    # Explode Injuries
    injury_df = injury_df[injury_df['end'] >= injury_df['start']]
    injury_expanded = []
    for _, row in injury_df.iterrows():
        dates = pd.date_range(start=row['start'], end=row['end'], freq='D')
        temp_df = pd.DataFrame({'team': row['team'], 'date': dates})
        injury_expanded.append(temp_df)
    injury_daily = pd.concat(injury_expanded).drop_duplicates()
    injury_daily['date'] = injury_daily['date'].dt.normalize()

    # Load Routines
    raw_dir = Path("data/raw")
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = pd.concat([pd.read_parquet(f) for f in roster_files])
    all_rosters['norm_name'] = all_rosters['player'].apply(normalize_name)
    all_rosters['match_id'] = all_rosters['match_id'].astype(str)
    
    # Merge for Features
    roster_agg = pd.merge(
        all_rosters,
        l1_matrix[['norm_name', 'Position_Clean', 'age', 'Goals p 90', 'Assists p 90', 'Goals per shot', 'Progressive Passes', 'Tackles Won']].drop_duplicates('norm_name'),
        on='norm_name',
        how='left'
    )
    
    # 2. Vectorized Feature Engineering
    starters = roster_agg[roster_agg['positionOrder'].astype(int) <= 11].copy()
    
    team_stats = starters.groupby(['match_id', 'side']).agg({
        'Goals p 90': 'sum',
        'Assists p 90': 'sum',
        'age': 'mean'
    }).reset_index()
    
    defenders = starters[starters['Position_Clean'].str.contains('DF', na=False)]
    df_solidity = defenders.groupby(['match_id', 'side'])['Tackles Won'].sum().reset_index().rename(columns={'Tackles Won': 'l2_df_solidity'})
    
    attackers = starters[starters['Position_Clean'].str.contains('FW|MF', na=False)]
    fw_efficiency = attackers.groupby(['match_id', 'side'])['Goals per shot'].mean().reset_index().rename(columns={'Goals per shot': 'l2_fw_efficiency'})

    l2_df = pd.merge(team_stats, df_solidity, on=['match_id', 'side'], how='left')
    l2_df = pd.merge(l2_df, fw_efficiency, on=['match_id', 'side'], how='left')
    
    l2_df = l2_df.rename(columns={
        'Goals p 90': 'l2_agg_goals_p90',
        'Assists p 90': 'l2_agg_assists_p90',
        'age': 'l2_avg_age'
    })
    
    # Injury Integration
    matches_df['date_norm'] = matches_df['date'].dt.normalize()
    inj_counts = injury_daily.groupby(['team', 'date']).size().reset_index(name='l2_injury_volume')
    
    h_inj = pd.merge(matches_df[['id', 'home_team', 'date_norm']], inj_counts, left_on=['home_team', 'date_norm'], right_on=['team', 'date'], how='left')
    h_inj = h_inj[['id', 'l2_injury_volume']].rename(columns={'id': 'match_id'}).assign(side='h')
    
    a_inj = pd.merge(matches_df[['id', 'away_team', 'date_norm']], inj_counts, left_on=['away_team', 'date_norm'], right_on=['team', 'date'], how='left')
    a_inj = a_inj[['id', 'l2_injury_volume']].rename(columns={'id': 'match_id'}).assign(side='a')
    
    all_inj_vol = pd.concat([h_inj, a_inj])
    l2_df = pd.merge(l2_df, all_inj_vol, on=['match_id', 'side'], how='left').fillna(0)

    home_l2 = l2_df[l2_df['side'] == 'h'].drop(columns='side').add_prefix('h_')
    away_l2 = l2_df[l2_df['side'] == 'a'].drop(columns='side').add_prefix('a_')
    
    final_l2_matrix = pd.merge(matches_df, home_l2, left_on='id', right_on='h_match_id')
    final_l2_matrix = pd.merge(final_l2_matrix, away_l2, left_on='id', right_on='a_match_id')
    
    # 3. Test on 2024-2025
    test_l2 = final_l2_matrix[final_l2_matrix['date'] >= '2024-01-01']
    l2_features = [c for c in test_l2.columns if c.startswith('h_l2') or c.startswith('a_l2')]
    
    # Evaluate Home
    print("Evaluating Home Offense...")
    model_h = xgb.Booster()
    model_h.load_model(model_dir / "level2_home_offensive_power.json")
    
    X_h = test_l2[l2_features].astype(float)
    y_h = test_l2['home_xg'].astype(float)
    preds_h = model_h.predict(xgb.DMatrix(X_h))
    
    rmse_h = np.sqrt(mean_squared_error(y_h, preds_h))
    r2_h = r2_score(y_h, preds_h)
    print(f"🏠 Home Team xG | RMSE: {rmse_h:.4f} | R²: {r2_h:.4f}")
    
    # Evaluate Away
    print("Evaluating Away Offense...")
    model_a = xgb.Booster()
    model_a.load_model(model_dir / "level2_away_offensive_power.json")
    
    X_a = test_l2[l2_features].astype(float)
    y_a = test_l2['away_xg'].astype(float)
    preds_a = model_a.predict(xgb.DMatrix(X_a))
    
    rmse_a = np.sqrt(mean_squared_error(y_a, preds_a))
    r2_a = r2_score(y_a, preds_a)
    print(f"✈️ Away Team xG | RMSE: {rmse_a:.4f} | R²: {r2_a:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_h, y=preds_h, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"Home xG: Actual vs Predicted (R²: {r2_h:.2f})")
    plt.xlabel("Actual Home xG")
    plt.ylabel("Predicted Home xG")
    
    plt.subplot(1, 2, 2)
    sns.regplot(x=y_a, y=preds_a, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"Away xG: Actual vs Predicted (R²: {r2_a:.2f})")
    plt.xlabel("Actual Away xG")
    plt.ylabel("Predicted Away xG")
    
    plt.tight_layout()
    plt.savefig(visual_dir / "l2_accuracy_plot.png")
    print(f"\n✅ Level 2 Accuracy Plot saved to: {visual_dir / 'l2_accuracy_plot.png'}")

if __name__ == "__main__":
    run_level2_audit()
