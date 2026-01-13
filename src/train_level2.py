import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import logging
from src.utils import normalize_name
from src.trainer import HierarchicalTrainer

logger = logging.getLogger(__name__)

def train_all_level_2():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Level 2: TEAM IDENTITY (Aggregation Layer)...")
    
    trainer = HierarchicalTrainer()
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    # 1. Load the foundation
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df['id'] = matches_df['id'].astype(str)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    l1_matrix = pd.read_parquet(processed_dir / "trinity_player_matrix.parquet")
    l1_matrix['norm_name'] = l1_matrix['player_name'].apply(normalize_name)
    # Fix: Create Position_Clean before merging
    l1_matrix['Position_Clean'] = l1_matrix.get('Position', l1_matrix.get('pos', 'Unknown'))
    
    # --- DATA PROTECTION: Loading Injury Volume ---
    logger.info("Integrating Injury/Absence signals into Team Brain...")
    injury_df = pd.read_parquet(processed_dir / "heuristic_injuries.parquet")
    
    # Explode Date Ranges into Daily Records (The "Absence Expansion" Rule)
    # This turns [Start: Jan 1, End: Jan 3] into [Jan 1, Jan 2, Jan 3]
    logger.info(f"Expanding {len(injury_df)} injury records into daily snapshots...")
    
    # Filter out entries where end is before start just in case
    injury_df = injury_df[injury_df['end'] >= injury_df['start']]
    
    # Create daily records
    injury_expanded = []
    for _, row in injury_df.iterrows():
        dates = pd.date_range(start=row['start'], end=row['end'], freq='D')
        temp_df = pd.DataFrame({'team': row['team'], 'date': dates})
        injury_expanded.append(temp_df)
    
    injury_daily = pd.concat(injury_expanded, ignore_index=True).drop_duplicates()
    injury_daily['date'] = injury_daily['date'].dt.normalize()
    
    # Load all rosters
    roster_files = list(raw_dir.glob("understat_rosters_*.parquet"))
    all_rosters = pd.concat([pd.read_parquet(f) for f in roster_files], ignore_index=True)
    all_rosters['norm_name'] = all_rosters['player'].apply(normalize_name)
    all_rosters['match_id'] = all_rosters['match_id'].astype(str)
    
    # 2. Join rosters with position attributes
    logger.info("Merging Roster data with Player Identity Matrix...")
    roster_agg = pd.merge(
        all_rosters,
        l1_matrix[['norm_name', 'Position_Clean', 'age', 'Goals p 90', 'Assists p 90', 'Goals per shot', 'Progressive Passes', 'Tackles Won']].drop_duplicates('norm_name'),
        on='norm_name',
        how='left'
    )
    
    # 3. VECTORIZED AGGREGATION (High Speed)
    logger.info("Aggregating individual player threat into team scores...")
    
    # Filter for starters
    starters = roster_agg[roster_agg['positionOrder'].astype(int) <= 11].copy()
    
    # Calculate Team Aggregates per Match and Side
    team_stats = starters.groupby(['match_id', 'side']).agg({
        'Goals p 90': 'sum',
        'Assists p 90': 'sum',
        'age': 'mean'
    }).reset_index()
    
    # Advanced Aggregates (Position-Specific)
    defenders = starters[starters['Position_Clean'].str.contains('DF', na=False)]
    df_solidity = defenders.groupby(['match_id', 'side'])['Tackles Won'].sum().reset_index().rename(columns={'Tackles Won': 'l2_df_solidity'})
    
    attackers = starters[starters['Position_Clean'].str.contains('FW|MF', na=False)]
    fw_efficiency = attackers.groupby(['match_id', 'side'])['Goals per shot'].mean().reset_index().rename(columns={'Goals per shot': 'l2_fw_efficiency'})

    # Merge aggregates
    l2_df = pd.merge(team_stats, df_solidity, on=['match_id', 'side'], how='left')
    l2_df = pd.merge(l2_df, fw_efficiency, on=['match_id', 'side'], how='left')
    
    # Rename for consistency
    l2_df = l2_df.rename(columns={
        'Goals p 90': 'l2_agg_goals_p90',
        'Assists p 90': 'l2_agg_assists_p90',
        'age': 'l2_avg_age'
    })
    
    # Add Injury Volume (ABSENCE LAYER)
    # Standardize match dates for joining
    matches_df['date_norm'] = matches_df['date'].dt.normalize()
    
    inj_counts = injury_daily.groupby(['team', 'date']).size().reset_index(name='l2_injury_volume')
    
    h_inj = pd.merge(matches_df[['id', 'home_team', 'date_norm']], inj_counts, left_on=['home_team', 'date_norm'], right_on=['team', 'date'], how='left')
    h_inj = h_inj[['id', 'l2_injury_volume']].rename(columns={'id': 'match_id'}).assign(side='h')
    
    a_inj = pd.merge(matches_df[['id', 'away_team', 'date_norm']], inj_counts, left_on=['away_team', 'date_norm'], right_on=['team', 'date'], how='left')
    a_inj = a_inj[['id', 'l2_injury_volume']].rename(columns={'id': 'match_id'}).assign(side='a')
    
    all_inj_vol = pd.concat([h_inj, a_inj], ignore_index=True)
    l2_df = pd.merge(l2_df, all_inj_vol, on=['match_id', 'side'], how='left').fillna(0)

    # 4. Final Matrix for Training
    # Pivot to match format (Home features alongside Away features)
    home_l2 = l2_df[l2_df['side'] == 'h'].drop(columns='side').add_prefix('h_')
    away_l2 = l2_df[l2_df['side'] == 'a'].drop(columns='side').add_prefix('a_')
    
    final_l2_matrix = pd.merge(matches_df, home_l2, left_on='id', right_on='h_match_id')
    final_l2_matrix = pd.merge(final_l2_matrix, away_l2, left_on='id', right_on='a_match_id')
    
    # 5. Train Level 2 Models
    final_l2_matrix['date'] = pd.to_datetime(final_l2_matrix['date'])
    train_l2 = final_l2_matrix[final_l2_matrix['date'] < '2026-01-01']
    
    l2_features = [c for c in train_l2.columns if c.startswith('h_l2') or c.startswith('a_l2')]
    
    # Offense
    X_h = train_l2[l2_features].astype(float)
    y_h = train_l2['home_xg'].astype(float)
    trainer.train_level2_team_models(X_h, y_h, "home_offensive_power")
    
    X_a = train_l2[l2_features].astype(float)
    y_a = train_l2['away_xg'].astype(float)
    trainer.train_level2_team_models(X_a, y_a, "away_offensive_power")
    
    logger.info("Level 2 Training Complete. Team Identity Models Saved.")

if __name__ == "__main__":
    train_all_level_2()
