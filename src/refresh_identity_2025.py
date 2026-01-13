import pandas as pd
from src.utils import normalize_name
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("identity_refresh")

def refresh_identity():
    processed_dir = Path("data/processed")
    archive_file = Path("archive/players_data-2024_2025.csv")
    master_file = processed_dir / "trinity_player_matrix.parquet"

    logger.info("Loading master identity matrix...")
    master_df = pd.read_parquet(master_file)
    master_df['norm_name'] = master_df['player_name'].apply(normalize_name)

    logger.info("Loading 24/25 CSV data...")
    csv_df = pd.read_csv(archive_file)
    csv_df['norm_name'] = csv_df['Player'].apply(normalize_name)
    csv_df['season'] = '2024-2025'

    # Map CSV columns to Master columns
    mapping = {
        'Player': 'player_name',
        'Age': 'age',
        'MP': 'games',
        'Min': 'time',
        'Gls': 'goals',
        'xG': 'xG',
        'Ast': 'assists',
        'Pos': 'Position'
    }
    csv_aligned = csv_df.rename(columns=mapping)

    # 1. Build Final CSV from scratch to match Master Columns
    final_csv = pd.DataFrame(index=csv_aligned.index)
    
    for col in master_df.columns:
        if col in csv_aligned.columns:
            # Transfer and Cast
            target_dtype = master_df[col].dtype
            if target_dtype != object:
                final_csv[col] = pd.to_numeric(csv_aligned[col], errors='coerce').fillna(0).astype(target_dtype)
            else:
                final_csv[col] = csv_aligned[col].astype(str)
        else:
            # Missing in CSV: Use default from Master or 0/""
            if master_df[col].dtype != object:
                final_csv[col] = 0.0
            else:
                final_csv[col] = ""

    # 2. Add structural columns
    final_csv['norm_name'] = csv_aligned['norm_name']
    final_csv['season'] = '2024-2025'

    logger.info(f"Appending {len(final_csv)} records from 24/25 season...")
    
    combined_df = pd.concat([master_df, final_csv], ignore_index=True)
    
    # 3. Final unique check and drop
    combined_df = combined_df.drop_duplicates(subset=['norm_name', 'season'])
    
    save_path = processed_dir / "trinity_player_matrix.parquet"
    combined_df.to_parquet(save_path)
    logger.info(f"Master Identity refreshed. Total records: {len(combined_df)}")

if __name__ == "__main__":
    refresh_identity()
