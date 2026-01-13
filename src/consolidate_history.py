import pandas as pd
from pathlib import Path
from src.utils import normalize_name
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("historical_consolidation")

def consolidate_history():
    archive_dir = Path("archive data 2")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(archive_dir.glob("cleaned_*.csv"))
    all_seasons = []
    
    for f in csv_files:
        logger.info(f"Processing {f.name}...")
        df = pd.read_csv(f)
        df['norm_name'] = df['player'].apply(normalize_name)
        # Ensure season is consistent
        season_str = f.name.replace("cleaned_", "").replace(".csv", "")
        df['data_season'] = season_str
        all_seasons.append(df)
        
    master_history = pd.concat(all_seasons, ignore_index=True)
    
    # Standardize column names to lowercase for consistency
    master_history.columns = [c.lower().replace(" ", "_") for c in master_history.columns]
    
    # Handle the '2024-2025' CSV separately if needed, but it was already merged into identity.
    # For Layer 0 training, we strictly need seasons up to 2023 to predict 2024.
    
    save_path = output_dir / "high_res_career_history.parquet"
    master_history.to_parquet(save_path)
    logger.info(f"Consolidated {len(master_history)} seasonal records. Saved to {save_path}")

if __name__ == "__main__":
    consolidate_history()
