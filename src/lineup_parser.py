import re
import json
from pathlib import Path
import pandas as pd
# In a real deployed app, we would use requests/bs4. 
# For this demo, we assume the user PASTE the text from the website.

import unicodedata

class LineupParser:
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.team_rosters = {} # Team -> Set of Normalized Names
        self.load_rosters()
    
    def normalize(self, text):
        """
        Aggressive normalization: Lowercase, ASCII only, no dots/hyphens.
        """
        text = str(text).lower()
        # Remove accents
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.replace('.', '').replace('-', ' ').strip()

    def load_rosters(self):
        try:
             # Load all roster files
             roster_files = list(self.raw_dir.glob("understat_rosters_*.parquet"))
             if not roster_files:
                 return
                 
             all_rosters = pd.concat([pd.read_parquet(f) for f in roster_files], ignore_index=True)
             
             # Normalized column
             all_rosters['clean_name'] = all_rosters['player'].apply(self.normalize)
             
             matches = pd.read_parquet(Path("data/processed/processed_matches.parquet"))
             id_map = {}
             for _, m in matches.iterrows():
                 id_map[str(m['id'])] = {'h': str(m['home_team']).lower().strip(), 'a': str(m['away_team']).lower().strip()}
             
             for _, row in all_rosters.iterrows():
                 mid = str(row['match_id'])
                 side = row['side'] # 'h' or 'a'
                 if mid in id_map:
                     team = id_map[mid][side]
                     if team not in self.team_rosters: self.team_rosters[team] = set()
                     self.team_rosters[team].add(row['clean_name'])
                     
        except Exception as e:
            print(f"Error loading rosters: {e}")

    def parse_text(self, text):
        """
        Extracts candidate names.
        """
        # 1. Truncate bench
        text_lower = text.lower()
        for kw in ["subs:", "substitutes:", "bench:", "reserves:"]:
            if kw in text_lower: text = text[:text_lower.find(kw)]; break
        
        # 2. Extract and Normalize
        text = text.replace('\n', ',').replace(';', ',').replace('-', ' ')
        raw_parts = [p.strip() for p in text.split(',') if len(p.strip()) > 3]
        return [self.normalize(p) for p in raw_parts]

    def calculate_strength(self, team_name, lineup_text):
        """
        Returns (score, details_string).
        """
        team_clean = team_name.lower().strip()
        if team_clean not in self.team_rosters:
            return 100, "Team not in database."
            
        roster = self.team_rosters[team_clean]
        if not roster: return 100, "Empty Roster in DB."
        
        input_names = self.parse_text(lineup_text)
        if not input_names: return 100, "No valid names found."
        
        # Count matches
        matched_names = []
        unmatched_names = []
        
        matches = 0
        for name in input_names:
            found = False
            for r_name in roster:
                if name in r_name or r_name in name:
                    matches += 1
                    matched_names.append(name)
                    found = True
                    break
            if not found:
                unmatched_names.append(name)
        
        score = min(100, int((matches / 11.0) * 100))
        
        # Debug Details
        msg = f"Matched {matches}/11: {', '.join(matched_names[:3])}..."
        if unmatched_names:
            msg += f" | Unmatched: {', '.join(unmatched_names)}"
            
        return score, msg
