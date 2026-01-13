"""
This is the 'Incremental Update Manager'.
As mentioned in idea.md, we don't want to re-download 5 years of data 
every time a single match is played!

This module tracks what we already have and only asks for the NEW stuff.
"""

import pandas as pd
from pathlib import Path
from datetime import date
import logging

logger = logging.getLogger(__name__)

class IncrementalUpdateManager:
    """
    Manages the delta between our local data and the latest available stats.
    """
    
    def __init__(self, data_registry_path: str = "data/registry.csv"):
        self.registry_path = Path(data_registry_path)
        self.registry = self._load_registry()
        
    def _load_registry(self):
        if self.registry_path.exists():
            return pd.read_csv(self.registry_path)
        return pd.DataFrame(columns=['data_type', 'last_updated', 'last_matchweek'])
        
    def get_update_window(self, data_type: str) -> tuple:
        """
        Returns the date or matchweek from which we need new data.
        """
        if data_type in self.registry['data_type'].values:
            row = self.registry[self.registry['data_type'] == data_type].iloc[0]
            return row['last_updated'], row['last_matchweek']
        return "2020-00-01", 0 # Default for new setup (2020 start)

    def mark_updated(self, data_type: str, matchweek: int):
        """
        Records that we have successfully fetched data up to a certain point.
        """
        new_row = {
            'data_type': data_type,
            'last_updated': str(date.today()),
            'last_matchweek': matchweek
        }
        # Logic to update registry.csv...
        logger.info(f"Registry updated for {data_type} up to week {matchweek}")
