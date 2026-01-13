"""
This is the 'Quality Assurance' (QA) Module.
It ensures our data is 'Super Quality' by checking for mistakes or weird numbers.

As mentioned in idea.md, we need to know "when they blink". This module 
flags anything that looks humanly impossible or technically wrong.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataQualityAssurance:
    """
    Checks the data for anomalies, missing values, and inconsistencies.
    """
    
    def __init__(self, benchmark_file: str = None):
        self.benchmarks = benchmark_file # Placeholder for historical averages
        
    def check_player_stats(self, df: pd.DataFrame) -> dict:
        """
        Runs a series of 'Sanity Checks' on player stats.
        """
        issues = {
            "missing_values": [],
            "outliers": [],
            "inconsistencies": []
        }
        
        # 1. Check for Missing Values (NaNs)
        missing = df.isnull().sum()
        if missing.any():
            issues["missing_values"].append(missing[missing > 0].to_dict())
            
        # 2. Outlier Detection (Self-Correction Logic)
        # If someone played 90 mins and has 50 goals... that's probably a data error!
        if 'goals' in df.columns:
            impossible_goals = df[df['goals'] > 10]
            if not impossible_goals.empty:
                issues["outliers"].append(f"Impossible goal count detected for: {impossible_goals['player_id'].tolist()}")
        
        # 3. Cross-Source Validation (Idea from idea.md)
        # If we had two sources, we would compare them here.
        
        logger.info(f"QA Check completed. Found {len(issues['outliers'])} outliers.")
        return issues

    def validate_ids(self, df: pd.DataFrame, id_column: str, reference_list: list):
        """
        Ensures player or team IDs match our 'Standard List'.
        Prevents the system from thinking 'L. Messi' and 'Lionel Messi' are different people.
        """
        unknown_ids = df[~df[id_column].isin(reference_list)][id_column].unique()
        return list(unknown_ids)
