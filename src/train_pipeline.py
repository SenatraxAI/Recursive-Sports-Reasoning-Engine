import logging
import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import HierarchicalTrainer
from src.train_level1 import train_all_level_1
from src.train_level2 import train_all_level_2
from src.train_level3 import train_all_level_3
from src.train_level4 import train_all_level_4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline():
    logger.info("INITIATING 4-LAYER HIERARCHICAL TRAINING...")
    
    # 1. PLAYER LEVEL (Individual Threat)
    # This builds the GK, DEF, MID, FWD models.
    logger.info("--- LEVEL 1: TRAINING PLAYER DNA ---")
    train_all_level_1() 
    
    # 2. TEAM LEVEL (Availability & Aggregation)
    logger.info("--- LEVEL 2: TRAINING TEAM IDENTITY ---")
    train_all_level_2()
    
    # 3. MATCHUP LEVEL (Managerial Tactics)
    logger.info("--- LEVEL 3: TRAINING TACTICAL MATCHUPS ---")
    train_all_level_3()
    
    # 4. OUTCOME LEVEL (The Verdict)
    logger.info("--- LEVEL 4: TRAINING FINAL OUTCOME ENGINE ---")
    train_all_level_4()
    
    logger.info("PHASE 2: HIERARCHICAL TRAINING COMPLETE.")

if __name__ == "__main__":
    import traceback
    try:
        run_full_pipeline()
    except Exception:
        traceback.print_exc()
