import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from src.manager_profiles import ManagerTacticalSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_profiler")

class DynamicManagerProfiler:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.exp_model_dir = Path("data/models/experimental")
        self.exp_model_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        logger.info("Starting Dynamic Manager Profiling (Learning from History)...")
        
        # 1. Load Match Spine with L2 Aggregates (Actual Performance)
        # We need the matches dataframe which has manager names + team stats
        # For this prototype, we'll re-load matches and merge with calculated team stats if available,
        # OR we infer from the 'home_xg', 'home_possession' (if we had it).
        
        # Since we don't have detailed 'distance_run' in our current dataset,
        # we will infer 'Tactical Intensity' from 'Goals', 'xG', and 'Injuries'.
        # AND we will use the 'home_style' columns if they exist.
        
        matches_df = pd.read_parquet(self.processed_dir / "processed_matches.parquet")
        
        # 2. Extract Manager Stats
        manager_stats = []
        
        for _, row in matches_df.iterrows():
            # Home
            manager_stats.append({
                'manager': row['home_manager'],
                'xg': row['home_xg'],
                'goals': row['home_goals'],
                # In a real full-data scenario, we'd add 'possession', 'passes', 'distance' here
                'intensity_proxy': row['home_xg'] / max(0.1, row['home_goals']) # Efficiency as proxy for discipline?
            })
            # Away
            manager_stats.append({
                'manager': row['away_manager'],
                'xg': row['away_xg'],
                'goals': row['away_goals'],
                'intensity_proxy': row['away_xg'] / max(0.1, row['away_goals'])
            })
            
        m_df = pd.DataFrame(manager_stats)
        
        # 3. Calculate Archetypes
        # Group by Manager
        profiles = m_df.groupby('manager').agg({
            'xg': 'mean',
            'goals': 'mean',
            'intensity_proxy': 'mean'
        }).reset_index()
        
        # Normalize to 0-1 for Constraints
        # e.g., Highest xG Manager = 1.0 Progressive Action Requirement
        max_xg = profiles['xg'].max()
        max_intensity = profiles['intensity_proxy'].max()
        
        learned_db = {}
        
        for _, row in profiles.iterrows():
            name = row['manager']
            
            # Heuristic Mapping (The Logic Bridge)
            # High xG -> Demands Progressive Actions
            req_prog = min(1.0, row['xg'] / max_xg)
            
            # Low Variance (Goals match xG) -> Demands Discipline
            # This is a simplification; in production we'd use variance columns.
            req_disc = 0.5 
            
            # High Intensity (Proxy) -> Demands Stamina
            req_stam = min(1.0, row['intensity_proxy'] / max_intensity)
            
            setup = ManagerTacticalSetup(
                name=name,
                style="Dynamic", # We let the numbers speak
                primary_formation="Dynamic", 
                req_progressive_actions=float(round(req_prog, 2)),
                req_positional_discipline=float(round(req_disc, 2)),
                req_stamina=float(round(req_stam, 2)),
                # Defaults for others for now
                req_pass_completion=0.5,
                req_defensive_workrate=0.5,
                req_sprint_speed=0.5
            )
            
            # Store as dictionary for JSON serialization
            learned_db[name] = {
                'req_progressive_actions': setup.req_progressive_actions,
                'req_positional_discipline': setup.req_positional_discipline,
                'req_stamina': setup.req_stamina,
                'req_pass_completion': setup.req_pass_completion,
                'req_defensive_workrate': setup.req_defensive_workrate,
                'req_sprint_speed': setup.req_sprint_speed
            }
            
        # 4. Save
        out_path = self.exp_model_dir / "manager_profiles_learned.json"
        with open(out_path, "w") as f:
            json.dump(learned_db, f, indent=4)
            
        logger.info(f"Learned profiles for {len(learned_db)} managers saved to {out_path}")

if __name__ == "__main__":
    DynamicManagerProfiler().run()
