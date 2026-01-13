# Robust Download Script for FPL Data
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

def download_files():
    data_dir = "data/raw/fpl"
    os.makedirs(data_dir, exist_ok=True)
    
    for season in SEASONS:
        url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
        target = f"{data_dir}/merged_gw_{season}.csv"
        
        if os.path.exists(target):
            logger.info(f"File {target} already exists. Skipping.")
            continue
            
        logger.info(f"Downloading {season} via curl...")
        try:
            # -L follows redirects, -o specifies output, -s is silent but we want progress
            subprocess.run(["curl", "-L", url, "-o", target], check=True)
            logger.info(f"Successfully downloaded {season}.")
        except Exception as e:
            logger.error(f"Failed to download {season}: {e}")

if __name__ == "__main__":
    download_files()
