import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.system_fit_calculator import SystemFitCalculator
from src.manager_profiles import Formation
from src.utils import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enricher")

class SystemFitEnricher:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.raw_dir = Path("data/raw")
        
        logger.info("Initializing System Fit Calculator...")
        self.calculator = SystemFitCalculator()
        
        # Load Player L0 Features (Pseudo-DNA for now)
        # In a real run, this would be the output of Layer 0.
        # For this prototype, we mock the DNA from available stats.
        self.l0_features = pd.read_parquet(self.processed_dir / "layer0_training_features.parquet")
        self.dna_lookup = self._build_dna_lookup(self.l0_features)
        
    def _build_dna_lookup(self, df):
        """Converts L0 dataframe into a lookup dictionary of DNA traits."""
        lookup = {}
        # We need to normalize these features to 0-1 range for the fit calculator
        # Simple Min-Max scaling for the prototype
        
        # Proxies for DNA traits
        # Stamina -> '90s' (Availability)
        # Passing -> 'Cmp%' (Pass Completion)
        # Defense -> 'Tkl+Int' (Defensive Actions)
        
        # Check available columns
        cols = df.columns.tolist()
        
        # Fallback if specific columns aren't there (using standardized names)
        # Assuming 'prog_passes', 'tackles', etc.
        
        for _, row in df.iterrows():
            # Mock DNA extraction
            # In production, this comes from the L0 XGBoost output or SHAP values
            dna = {
                'dna_stamina': min(1.0, row.get('90s', 0) / 30.0), # Normalized availability
                'dna_passing': min(1.0, row.get('pass_completion', 0.8)), 
                'dna_defense': min(1.0, row.get('tackles_won', 0) / 5.0) 
            }
            lookup[row['norm_name']] = dna
        return lookup

    def run(self):
        logger.info("Enriching Match Data with System Intelligence...")
        
        matches = pd.read_parquet(self.processed_dir / "processed_matches.parquet")
        rosters = pd.read_parquet(self.processed_dir / "trinity_player_matrix.parquet") # Or master roster file
        
        # We need per-match rosters.
        # Let's load the raw rosters which have match_id
        roster_files = list(self.raw_dir.glob("understat_rosters_*.parquet"))
        all_rosters = pd.concat([pd.read_parquet(f) for f in roster_files], ignore_index=True)
        all_rosters['norm_name'] = all_rosters['player'].apply(normalize_name)
        all_rosters['match_id'] = all_rosters['match_id'].astype(str)
        
        enriched_records = []
        
        for idx, m in matches.iterrows():
            if idx % 500 == 0: logger.info(f"Processed {idx} matches...")
            
            m_id = str(m['id'])
            
            # Get Home & Away Lineups (Starters)
            roster_m = all_rosters[all_rosters['match_id'] == m_id]
            h_starters = roster_m[(roster_m['side'] == 'h') & (roster_m['positionOrder'].astype(int) <= 11)]
            a_starters = roster_m[(roster_m['side'] == 'a') & (roster_m['positionOrder'].astype(int) <= 11)]
            
            # --- CALCULATE SYSTEM FIT ---
            # Home
            h_dna_list = [self.dna_lookup.get(n, {}) for n in h_starters['norm_name']]
            h_fit = self.calculator.batch_calculate_team_fit(h_dna_list, m['home_manager'])
            
            # Away
            a_dna_list = [self.dna_lookup.get(n, {}) for n in a_starters['norm_name']]
            a_fit = self.calculator.batch_calculate_team_fit(a_dna_list, m['away_manager'])
            
            # --- CALCULATE FORMATION DENSITY (The "Traffic Jam" Features) ---
            # Retrieve formation structure
            h_fmt = Formation.get_structure(m.get('home_formation', '4-4-2'))
            a_fmt = Formation.get_structure(m.get('away_formation', '4-4-2'))
            
            record = {
                'match_id': m_id,
                'h_system_fit': h_fit,
                'a_system_fit': a_fit,
                
                # Structural Features (Home)
                'h_density_central': h_fmt['density_central'],
                'h_density_attack': h_fmt['density_attack'],
                'h_width_balance':  h_fmt['wide_att'] / max(1, h_fmt['density_central']), # Ratio
                
                # Structural Features (Away)
                'a_density_central': a_fmt['density_central'],
                'a_density_attack': a_fmt['density_attack'],
                'a_width_balance': a_fmt['wide_att'] / max(1, a_fmt['density_central']),
                
                # Target Labels (Existing)
                'home_goals': m['home_goals'],
                'away_goals': m['away_goals'],
                'home_xg': m['home_xg'],
                'away_xg': m['away_xg']
            }
            enriched_records.append(record)
            
        # Save Enriched Dataset
        df_out = pd.DataFrame(enriched_records)
        out_path = self.processed_dir / "level2_enriched_training.parquet"
        df_out.to_parquet(out_path)
        logger.info(f"Enrichment Complete. Saved {len(df_out)} records to {out_path}")

if __name__ == "__main__":
    SystemFitEnricher().run()
