import pandas as pd
import requests
from io import StringIO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEASONS = {
    "2020-21": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2020-21/gws/merged_gw.csv",
    "2021-22": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv",
    "2022-23": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv",
    "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
    "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"
}

def collect_injuries():
    output_dir = Path("data/raw/injuries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_injury_data = []

    for season, url in SEASONS.items():
        logger.info(f"Downloading injury data for {season}...")
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    df['season'] = season
                    cols_to_keep = ['name', 'kickoff_time', 'minutes', 'season', 'team']
                    available_cols = [c for c in cols_to_keep if c in df.columns]
                    df_filtered = df[available_cols].copy()
                    df_filtered.to_parquet(output_dir / f"fpl_availability_{season}.parquet")
                    all_injury_data.append(df_filtered)
                    logger.info(f"Saved {len(df_filtered)} records for {season}")
                    break
                else:
                    logger.error(f"Failed to download {season}: {response.status_code}")
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {season}: {e}")
                if attempt == 2: logger.error(f"Giving up on {season}")

    if all_injury_data:
        master_injury = pd.concat(all_injury_data)
        master_injury.to_parquet("data/raw/injury_master.parquet")
        logger.info("Master Injury Availability file created.")

if __name__ == "__main__":
    collect_injuries()
