import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_visuals():
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    output_dir = Path("data/visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Injury Distribution (Data Quality Check)
    logger.info("Visualizing Injury Distribution...")
    if (processed_dir / "heuristic_injuries.parquet").exists():
        injuries = pd.read_parquet(processed_dir / "heuristic_injuries.parquet")
        plt.figure(figsize=(10, 6))
        sns.histplot(injuries['days_out'], bins=50, kde=True, color='red')
        plt.title("Distribution of Player Absences (2020-2025)")
        plt.xlabel("Days Out")
        plt.ylabel("Frequency")
        plt.savefig(output_dir / "injury_distribution.png")
        plt.close()

    # 2. Manager Tactical DNA Breakdown
    logger.info("Visualizing Tactical DNA...")
    if (processed_dir / "processed_matches.parquet").exists():
        matches = pd.read_parquet(processed_dir / "processed_matches.parquet")
        style_counts = matches['home_style'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=style_counts.index, y=style_counts.values, palette="viridis")
        plt.title("Premier League Tactical Styles (Match Frequency 2020-2025)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "tactical_styles.png")
        plt.close()

    # 3. xG vs Actual Goals (Reliability Check)
    logger.info("Visualizing xG vs Actual...")
    if (processed_dir / "processed_matches.parquet").exists():
        matches = pd.read_parquet(processed_dir / "processed_matches.parquet")
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=matches, x='home_xg', y='home_goals', alpha=0.3, label="Home")
        sns.scatterplot(data=matches, x='away_xg', y='away_goals', alpha=0.3, label="Away")
        plt.plot([0, 6], [0, 6], 'r--', label="Perfect Correlation")
        plt.title("xG vs. Real Goals (Model Truth Check)")
        plt.legend()
        plt.savefig(output_dir / "xg_vs_goals.png")
        plt.close()

    # 4. Data Completeness by Season
    logger.info("Visualizing Completeness...")
    seasons = ["2020", "2021", "2022", "2023", "2024"]
    counts = []
    for s in seasons:
        p_path = raw_dir / f"understat_players_{s}.parquet"
        if p_path.exists():
            df = pd.read_parquet(p_path)
            counts.append({'Season': s, 'Records': len(df)})
    
    completeness_df = pd.DataFrame(counts)
    if not completeness_df.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=completeness_df, x='Season', y='Records', marker='o')
        plt.title("Player Record Completeness per Season")
        plt.savefig(output_dir / "data_completeness.png")
        plt.close()

    logger.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    generate_visuals()
