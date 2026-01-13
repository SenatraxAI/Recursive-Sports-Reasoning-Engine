import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from src.utils import map_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def man_united_deep_dive():
    processed_dir = Path("data/processed")
    output_dir = Path("data/visuals/man_utd")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="darkgrid")
    team_name = "man_utd"
    
    # 1. Load Data
    matches = pd.read_parquet(processed_dir / "processed_matches.parquet")
    injuries = pd.read_parquet(processed_dir / "heuristic_injuries.parquet")
    
    # 2. Filter for Man Utd
    utd_home = matches[matches['home_team'] == team_name].copy()
    utd_away = matches[matches['away_team'] == team_name].copy()
    utd_matches = pd.concat([utd_home, utd_away]).sort_values('date')
    
    # 3. Manager Performance at Man Utd
    logger.info("Analyzing Man Utd Managers...")
    def get_utd_manager(row):
        return row['home_manager'] if row['home_team'] == team_name else row['away_manager']
    
    def get_utd_res(row):
        if row['home_team'] == team_name:
            return 'W' if row['home_goals'] > row['away_goals'] else ('L' if row['home_goals'] < row['away_goals'] else 'D')
        else:
            return 'W' if row['away_goals'] > row['home_goals'] else ('L' if row['away_goals'] < row['home_goals'] else 'D')

    utd_matches['manager'] = utd_matches.apply(get_utd_manager, axis=1)
    utd_matches['result'] = utd_matches.apply(get_utd_res, axis=1)
    
    mgr_stats = utd_matches.groupby('manager').agg({
        'result': [lambda x: (x=='W').sum(), 'count']
    })
    mgr_stats.columns = ['wins', 'games']
    mgr_stats['win_rate'] = (mgr_stats['wins'] / mgr_stats['games']) * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=mgr_stats.index, y=mgr_stats['win_rate'], palette="Reds_r")
    plt.title("Manchester United Manager Win Rates (2020-2025)")
    plt.ylabel("Win %")
    plt.savefig(output_dir / "utd_manager_performance.png")
    plt.close()

    # 4. Injury Trend at Man Utd
    logger.info("Analyzing Man Utd Injuries...")
    utd_injuries = injuries[injuries['team'] == team_name].copy()
    seasonal_injuries = utd_injuries.groupby('season').size()
    
    plt.figure(figsize=(10, 6))
    seasonal_injuries.plot(kind='line', marker='o', color='red', linewidth=2)
    plt.title("Manchester United Injury Volume Trend (2020-2025)")
    plt.ylabel("Significant Absences Detected")
    plt.grid(True)
    plt.savefig(output_dir / "utd_injury_trend.png")
    plt.close()

    # 5. Over/Under Performance (xG vs Goals)
    logger.info("Analyzing xG Performance...")
    utd_matches['goals'] = utd_matches.apply(lambda r: r['home_goals'] if r['home_team'] == team_name else r['away_goals'], axis=1)
    utd_matches['xg'] = utd_matches.apply(lambda r: r['home_xg'] if r['home_team'] == team_name else r['away_xg'], axis=1)
    
    # Cumulative xG vs Goals
    utd_matches['cum_goals'] = utd_matches['goals'].cumsum()
    utd_matches['cum_xg'] = utd_matches['xg'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(utd_matches['date'], utd_matches['cum_goals'], label="Actual Goals", color='red')
    plt.plot(utd_matches['date'], utd_matches['cum_xg'], label="Expected Goals (xG)", color='black', linestyle='--')
    plt.title("Man Utd: Cumulative Goals vs xG (The Efficiency Gap)")
    plt.legend()
    plt.savefig(output_dir / "utd_efficiency.png")
    plt.close()

    logger.info(f"Deep dive complete. Visuals in {output_dir}")
    
    # Print a small text summary
    print("\n--- MANCHESTER UNITED DEEP DIVE (2020-2025) ---")
    print(mgr_stats.sort_values('win_rate', ascending=False))
    print("\nInjury Peak:", seasonal_injuries.idxmax(), "with", seasonal_injuries.max(), "absences.")

if __name__ == "__main__":
    man_united_deep_dive()
