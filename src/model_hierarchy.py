"""
This is the 'Main Brain' or Orchestrator. 
It ties EVERYTHING together. 

It uses:
1. Collector (to get data)
2. Processor (to clean it)
3. Profiler (to understand managers/players)
4. Injury Model (to account for missing stars)
5. Trainer (to build the AI models)

This matches your blueprint's 4-level architecture!
"""

from src.schema import PlayerMatchStats, GameState
from src.collector import PremierLeagueCollector
from src.processor import PremierLeagueProcessor
from src.trainer import HierarchicalTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PremierLeaguePredictor:
    """
    The main control center for our Prediction Engine.
    """
    
    def __init__(self):
        self.collector = PremierLeagueCollector()
        self.processor = PremierLeagueProcessor()
        self.trainer = HierarchicalTrainer()
        
    def run_pipeline(self):
        """
        Runs the whole process from A to Z.
        """
        logger.info("Starting Premier League Prediction Pipeline...")
        
        # Level 0: Data Sourcing
        # Step 1: Collect Data (Currently commented out until soccerdata is installed)
        # self.collector.collect_fbref_data()
        
        # Level 1: Player Processing
        # Step 2: Clean player stats and calculate values
        # self.processor.process_player_stats()
        
        # Level 2: Team Training
        # Step 3: Train the Team-level brains
        
        logger.info("Pipeline run complete.")

    def predict_match(self, home_team, away_team):
        """
        This is the final goal: Predict a specific match!
        
        It flows through the hierarchy:
        Level 1 -> Predict 22 player performances.
        Level 2 -> Aggregate into 2 team strengths.
        Level 3 -> Adjust for Matchup Context (Who is home? Who is tired?)
        Level 4 -> Produce the final Win/Draw/Loss probabilities.
        """
        pass

if __name__ == "__main__":
    predictor = PremierLeaguePredictor()
    predictor.run_pipeline()
