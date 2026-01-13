"""
This is the 'Data Collector' for our Super-Quality Pipeline. 

To get that "blink-level" detail you asked for, we aren't just looking at 
who scored. We are looking for 'Advanced Metrics' like:
- Progressive Carries (Moving the ball while running)
- Shot Coordinates (Exactly where on the pitch a shot was taken)
- Pressing Actions (How hard they worked to get the ball back)

What is 'soccerdata'?
It is our "Robot Scout". It goes to sites like FBref and Understat to fetch 
these three full seasons of high-quality data automatically.
"""

import soccerdata as sd
try:
    from soccerdata.understat import Understat
except ImportError:
    pass
import pandas as pd
from pathlib import Path
from src.schema import PlayerMatchStats
import logging
import json
import requests
import os
import re
from understatapi import UnderstatClient

# Setup logging: This prints helpful messages to the console so we know what the script is doing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremierLeagueCollector:
    """
    This class handles the connection to football data websites.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        When we create the collector, it checks if we have a place to save the data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_fbref_data(self, seasons: list = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]):
        """
        Downloads data from FBref. 
        CRITICAL UPDATE: Now fetching PER-MATCH logs, not just season totals.
        """
        logger.info(f"Collecting FBref data for seasons: {seasons}")
        try:
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=seasons)
            
            # 1. Match Schedule (The Spine)
            schedule = fbref.read_schedule()
            schedule.to_parquet(self.data_dir / "fbref_schedule.parquet")
            
            # 2. Player Match Logs (The "Blink" Details per 90)
            # This gives us xG, Shots, Carries for EVERY single match.
            # Warning: This is a large download!
            match_stats = fbref.read_player_match_stats(stat_type="summary")
            match_stats.to_parquet(self.data_dir / "fbref_player_match_logs.parquet")
            
            # 3. Lineups (Squad Composition)
            # Who started? Who was on the bench?
            lineups = fbref.read_lineup()
            lineups.to_parquet(self.data_dir / "fbref_lineups.parquet")
                
            logger.info("FBref 'Super-Quality' Per-Match collection successful.")
        except Exception as e:
            logger.error(f"Error collecting FBref data: {e}")

    def collect_understat_data(self, seasons: list = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]):
        """
        Downloads data from Understat.
        Understat gives us the "WHERE did it happen?" data.
        We are pulling FIVE FULL SEASONS (from 2020).
        """
        logger.info(f"Collecting Understat data for seasons: {seasons}")
        try:
            print("DEBUG: Importing Understat...")
            # FIX: Use the class directly if not available in top-level 'sd'
            UnderstatClass = getattr(sd, 'Understat', None)
            if UnderstatClass is None:
                from soccerdata.understat import Understat as UnderstatClass
                
            print(f"DEBUG: Initializing Understat for {seasons}...")
            understat = UnderstatClass(leagues="ENG-Premier League", seasons=seasons)
            
            # --- GRANULAR METRIC 4: Team-Level xG Dynamics ---
            print("DEBUG: Reading Team Stats...")
            team_stats = understat.read_team_stats()
            print(f"DEBUG: Team Stats Shape: {team_stats.shape}")
            team_stats.to_parquet(self.data_dir / "understat_team_stats.parquet")
            
            # --- GRANULAR METRIC 5: Shot-by-Shot Logic ---
            print("DEBUG: Reading Shot Data...")
            shots = understat.read_shot_data()
            print(f"DEBUG: Shots Shape: {shots.shape}")
            shots.to_parquet(self.data_dir / "understat_shots.parquet")
            
            logger.info("Understat high-quality data collection successful.")
        except Exception as e:
            print(f"DEBUG: ERROR: {e}")
            logger.error(f"Error collecting Understat data: {e}")

    def load_openfootball_data(self, repo_path: str = "data/openfootball_repo"):
        """
        LOADER: Ingests match results from the local openfootball clone.
        """
        repo_path = Path(repo_path)
        all_matches = []
        seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
        
        for season in seasons:
            file_path = repo_path / season / "en.1.json"
            if file_path.exists():
                logger.info(f"Loading OpenFootball: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    matches = data.get('matches', [])
                    for m in matches:
                        m['season'] = season
                        if 'score' in m and 'ft' in m['score']:
                            m['home_score'] = m['score']['ft'][0]
                            m['away_score'] = m['score']['ft'][1]
                        all_matches.append(m)
            else:
                logger.warning(f"File not found: {file_path}")
        
        df = pd.DataFrame(all_matches)
        df.to_parquet(self.data_dir / "openfootball_matches.parquet")
        return df

    def load_archive_data(self, archive_path: str = "archive data 2"):
        """
        LOADER: Ingests the deep player attributes (Physical/Identity).
        """
        archive_path = Path(archive_path)
        all_player_data = []
        
        for file in archive_path.glob("cleaned_*.csv"):
            logger.info(f"Loading Archive Data: {file}")
            df = pd.read_csv(file)
            all_player_data.append(df)
            
        if all_player_data:
            master_player_df = pd.concat(all_player_data, ignore_index=True)
            master_player_df.to_parquet(self.data_dir / "master_player_attributes.parquet")
            return master_player_df
        return pd.DataFrame()

    def collect_understat_api_data(self, seasons: list = ["2020", "2021", "2022", "2023", "2024"]):
        """
        NEW PRIMARY SOURCE: understatAPI.
        Fetches granular player and team data using AJAX endpoints.
        """
        logger.info(f"Collecting Understat data via API for seasons: {seasons}")
        
        with UnderstatClient() as understat:
            for season in seasons:
                # 1. Season Player Data (Form Baseline)
                try:
                    logger.info(f"Fetching EPL Player Data for {season}...")
                    players = understat.league(league="EPL").get_player_data(season=season)
                    df_players = pd.DataFrame(players)
                    df_players.to_parquet(self.data_dir / f"understat_players_{season}.parquet")
                except Exception as e:
                    logger.error(f"Failed Player Data for {season}: {e}")
                
                # 2. Season Team Data (Season Totals)
                try:
                    logger.info(f"Fetching EPL Team Data for {season}...")
                    teams_data = understat.league(league="EPL").get_team_data(season=season)
                    df_teams = pd.DataFrame(teams_data)
                    df_teams.to_parquet(self.data_dir / f"understat_teams_{season}.parquet")
                except Exception as e:
                    logger.error(f"Failed Team Data for {season}: {e}")

                # 3. Match Data (Per-Match results/xG)
                try:
                    logger.info(f"Fetching EPL Match Data for {season}...")
                    matches_data = understat.league(league="EPL").get_match_data(season=season)
                    df_matches = pd.DataFrame(matches_data)
                    df_matches.to_parquet(self.data_dir / f"understat_matches_{season}.parquet")
                    
                    # 4. ROSTERS (The Granular Key)
                    # We pull rosters for every match to get per-match player stats.
                    logger.info(f"Fetching Rosters for {len(matches_data)} matches in {season}...")
                    all_rosters = []
                    for i, match in enumerate(matches_data):
                        match_id = match['id']
                        if i % 100 == 0: logger.info(f"Progress: {i}/{len(matches_data)}")
                        roster = understat.match(match=match_id).get_roster_data()
                        # Flatten the roster (it's usually { 'h': [...], 'a': [...] })
                        for side in ['h', 'a']:
                            for p_id, p_stats in roster[side].items():
                                p_stats['match_id'] = match_id
                                p_stats['side'] = side
                                all_rosters.append(p_stats)
                    
                    df_rosters = pd.DataFrame(all_rosters)
                    df_rosters.to_parquet(self.data_dir / f"understat_rosters_{season}.parquet")
                    logger.info(f"Successfully saved {len(df_rosters)} roster entries for {season}.")

                except Exception as e:
                    logger.error(f"Failed Match/Roster Data for {season}: {e}")
        
        logger.info("Understat API collection complete.")

    def fetch_understat_manual(self, year: str = "2024"):
        """DEPRECATED: Use collect_understat_api_data instead."""
        pass

if __name__ == "__main__":
    # This part only runs if we execute this file directly.
    collector = PremierLeagueCollector()
    # collector.collect_fbref_data()
    # collector.collect_understat_data()
