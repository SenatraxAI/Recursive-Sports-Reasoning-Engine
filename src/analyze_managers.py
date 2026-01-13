import pandas as pd
import json
from pathlib import Path
from src.processor import PremierLeagueProcessor
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_managers():
    processor = PremierLeagueProcessor()
    
    # 1. Process all matches to get the master dataset
    print("Processing all matches (2020-2025)...")
    df = processor.process_match_logic(seasons=["2020", "2021", "2022", "2023", "2024"])
    
    if df.empty:
        print("No match data found!")
        return

    # 2. Calculate results
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 'H'
        if row['home_goals'] < row['away_goals']: return 'A'
        return 'D'
    
    df['result'] = df.apply(get_result, axis=1)

    # 3. Extract Manager stats
    # Prepare lists to expand home/away into a single manager list
    manager_records = []
    
    for _, row in df.iterrows():
        # Home side
        h_res = 'W' if row['result'] == 'H' else ('L' if row['result'] == 'A' else 'D')
        manager_records.append({
            'manager': row['home_manager'],
            'team': row['home_team'],
            'res': h_res,
            'xg': row['home_xg'],
            'goals': row['home_goals']
        })
        # Away side
        a_res = 'W' if row['result'] == 'A' else ('L' if row['result'] == 'H' else 'D')
        manager_records.append({
            'manager': row['away_manager'],
            'team': row['away_team'],
            'res': a_res,
            'xg': row['away_xg'],
            'goals': row['away_goals']
        })
    
    m_df = pd.DataFrame(manager_records)
    
    stats = m_df.groupby('manager').agg({
        'res': [lambda x: (x=='W').sum(), lambda x: (x=='L').sum(), lambda x: (x=='D').sum(), 'count'],
        'xg': 'mean',
        'goals': 'mean'
    })
    stats.columns = ['wins', 'losses', 'draws', 'games', 'avg_xg', 'avg_goals']
    stats['win_rate'] = (stats['wins'] / stats['games']) * 100
    stats = stats.sort_values('win_rate', ascending=False)
    
    # 3. Correlation with Tactical Style
    # Get tactics for each manager
    with open('data/raw/manager_master.json', 'r') as f:
        master = json.load(f)
    
    tactics_df = pd.DataFrame.from_dict(master['tactics'], orient='index')
    
    # Merge stats to tactics
    merged = stats.merge(tactics_df, left_index=True, right_index=True)
    
    # Group by Style and Pressing to see "Style win rates"
    style_analysis = merged.groupby('style').agg({
        'win_rate': 'mean',
        'games': 'sum'
    }).sort_values('win_rate', ascending=False)
    
    pressing_analysis = merged.groupby('pressing').agg({
        'win_rate': 'mean',
        'games': 'sum'
    }).sort_values('win_rate', ascending=False)

    print("\n--- TOP MANAGERS BY WIN RATE (2020-2025) ---")
    print(stats.head(15))
    
    print("\n--- TACTICAL STYLE WIN RATES ---")
    print(style_analysis)
    
    print("\n--- PRESSING INTENSITY WIN RATES ---")
    print(pressing_analysis)
    
    # Save results
    merged.to_csv("data/processed/manager_correlation_report.csv")
    print("\nFull report saved to data/processed/manager_correlation_report.csv")

if __name__ == "__main__":
    analyze_managers()
