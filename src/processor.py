import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from src.utils import normalize_name, map_team_name

logger = logging.getLogger(__name__)

class PremierLeagueProcessor:
    """
    The BRAIN of the data pipeline. 
    Merges Results (Spine), Attributes (Identity), and Form (Momentum).
    """
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.managers = self.load_manager_data()

    def load_manager_data(self):
        """Loads hand-curated manager tenure and tactics data."""
        manager_file = self.raw_dir / "manager_master.json"
        if not manager_file.exists():
            logger.warning("Manager Master file not found!")
            return {"tenures": [], "tactics": {}}
        with open(manager_file, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def load_trinity_players(self, seasons: list = ["2020", "2021", "2022", "2023", "2024"]):
        """
        Merges Understat stats with Archive Attributes.
        """
        logger.info("Merging Player Trinity (Stats + Attributes)...")
        
        # 1. Load Attribute Master
        attr_path = self.raw_dir / "master_player_attributes.parquet"
        if not attr_path.exists():
            logger.error("Attribute Master missing!")
            return pd.DataFrame()
            
        attr_df = pd.read_parquet(attr_path)
        attr_df['norm_name'] = attr_df['player'].apply(normalize_name)
        attr_df['norm_team'] = attr_df['squad'].apply(map_team_name)
        
        all_season_stats = []
        for season in seasons:
            stat_file = self.raw_dir / f"understat_players_{season}.parquet"
            if not stat_file.exists(): continue
            
            df = pd.read_parquet(stat_file)
            df['norm_name'] = df['player_name'].apply(normalize_name)
            df['norm_team'] = df['team_title'].apply(map_team_name)
            df['season_year'] = season
            
            # Merge with Attributes
            # Map season to Archer's format if possible
            target_season = f"{season}-{str(int(season)+1)[-2:]}" # e.g. 2024-25
            
            merged = pd.merge(
                df, 
                attr_df, # Relaxing season match to get any available attributes
                on=['norm_name'], 
                how='left',
                suffixes=('', '_attr')
            )
            all_season_stats.append(merged)
            
        if not all_season_stats:
            return pd.DataFrame()
            
        final_df = pd.concat(all_season_stats, ignore_index=True)
        logger.info(f"Trinity Player Matrix Complete: {len(final_df)} records.")
        return final_df

    def process_match_logic(self, seasons: list = ["2020", "2021", "2022", "2023", "2024"]):
        """
        Prepares matches and calculates team form.
        """
        logger.info("Processing Match Spine and Form...")
        all_matches = []
        
        for season in seasons:
            match_file = self.raw_dir / f"understat_matches_{season}.parquet"
            if not match_file.exists(): continue
            
            df = pd.read_parquet(match_file)
            # Flatten Understat nested JSON
            df['home_team'] = df['h'].apply(lambda x: map_team_name(x['title']))
            df['away_team'] = df['a'].apply(lambda x: map_team_name(x['title']))
            
            # Understat nested goals and xG
            df['home_xg'] = df['xG'].apply(lambda x: float(x['h']) if isinstance(x, dict) else 0)
            df['away_xg'] = df['xG'].apply(lambda x: float(x['a']) if isinstance(x, dict) else 0)
            df['home_goals'] = df['goals'].apply(lambda x: int(x['h']) if isinstance(x, dict) else 0)
            df['away_goals'] = df['goals'].apply(lambda x: int(x['a']) if isinstance(x, dict) else 0)
            df['date'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            
            all_matches.append(df[['id', 'date', 'home_team', 'away_team', 'home_xg', 'away_xg', 'home_goals', 'away_goals']])
            
        if not all_matches:
            return pd.DataFrame()
            
        matches_df = pd.concat(all_matches).sort_values('date')
        
        # 3. LINK MANAGERS
        matches_df = self.link_managers(matches_df)
        
        return matches_df

    def link_managers(self, df: pd.DataFrame):
        """Links each match to the manager in charge at that date."""
        logger.info("Linking managers and tactical DNA to matches...")
        
        def get_manager(team, match_date):
            norm_team = map_team_name(team)
            for t in self.managers['tenures']:
                if map_team_name(t['team']) == norm_team:
                    start = pd.to_datetime(t['start']).tz_localize(None)
                    end = pd.to_datetime(t['end']).tz_localize(None) if t['end'] != "Present" else pd.to_datetime('2099-01-01')
                    if start <= match_date.replace(tzinfo=None) <= end:
                        return t['manager']
            return "Unknown"

        df['home_manager'] = df.apply(lambda row: get_manager(row['home_team'], row['date']), axis=1)
        df['away_manager'] = df.apply(lambda row: get_manager(row['away_team'], row['date']), axis=1)
        
        # Link Tactics
        tactics = self.managers['tactics']
        for side in ['home', 'away']:
            df[f'{side}_formation'] = df[f'{side}_manager'].apply(lambda x: tactics.get(x, tactics['Generic'])['formation'])
            df[f'{side}_pressing'] = df[f'{side}_manager'].apply(lambda x: tactics.get(x, tactics['Generic'])['pressing'])
            df[f'{side}_style'] = df[f'{side}_manager'].apply(lambda x: tactics.get(x, tactics['Generic'])['style'])
            
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = PremierLeagueProcessor()
    
    logger.info("Starting Processing Pipeline...")
    players = processor.load_trinity_players()
    if not players.empty:
        players.to_parquet("data/processed/trinity_player_matrix.parquet")
    
    matches = processor.process_match_logic()
    if not matches.empty:
        matches.to_parquet("data/processed/processed_matches.parquet")
    
    logger.info("Processing Complete. Data ready in 'data/processed/'.")
