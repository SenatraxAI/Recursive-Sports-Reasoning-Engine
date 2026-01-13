"""
This is the 'Mock Data Generator'.

Wait, what is 'Mock Data'?
It's 'Fake Data' that looks real!

Why do we need it?
Because downloading data from the internet can be slow or fail (like soccerdata install issues).
Mock data lets us TEST our code and logic immediately to make sure it works before 
we connect it to real Premier League data.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict

def generate_mock_player_stats(player_id: str, n_matches: int = 10) -> pd.DataFrame:
    """Creates a fake history of matches for a single player."""
    dates = [date.today() - timedelta(days=7*i) for i in range(n_matches)]
    data = {
        'player_id': [player_id] * n_matches,
        'match_date': dates,
        'possession_pct': np.random.uniform(40, 60, n_matches),
        'pass_completion_pct': np.random.uniform(70, 95, n_matches),
        'key_passes': np.random.randint(0, 5, n_matches),
        'dribbles_successful': np.random.randint(0, 3, n_matches),
        'tackles_won': np.random.randint(0, 4, n_matches),
        'aerial_duels_won': np.random.randint(0, 4, n_matches),
        'xg': np.random.uniform(0, 0.5, n_matches),
        'xa': np.random.uniform(0, 0.3, n_matches),
    }
    return pd.DataFrame(data)

def generate_mock_match_history(manager_id: str, n_matches: int = 15) -> pd.DataFrame:
    """Creates a fake history of matches for a manager/team."""
    data = {
        'manager_id': [manager_id] * n_matches,
        'manager_name': ["Mock Manager"] * n_matches,
        'formation': ["4-3-3"] * n_matches,
        'possession_pct': np.random.uniform(45, 55, n_matches),
        'ppda': np.random.uniform(8, 15, n_matches),
    }
    return pd.DataFrame(data)

def generate_mock_squad(team_id: str, size: int = 11) -> List[str]:
    """Creates a fake list of 11 player IDs for a team."""
    return [f"player_{team_id}_{i}" for i in range(size)]
