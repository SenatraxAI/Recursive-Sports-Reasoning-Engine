from src.collector import PremierLeagueCollector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    collector = PremierLeagueCollector()
    
    # We are pulling the 5 full seasons requested (2020-2025)
    seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
    
    logger.info("Starting Trinity Data Ingestion (Results + Attributes)...")
    
    # TRINITY STEP 1: Load Results from OpenFootball clone
    logger.info("Step 1: Ingesting match spine from openfootball...")
    res_df = collector.load_openfootball_data()
    logger.info(f"Loaded {len(res_df)} match records.")
    
    # TRINITY STEP 3: Load Understat Form (API Fetch)
    logger.info("Step 3: Fetching high-granularity xG and Player Form from Understat...")
    collector.collect_understat_api_data(seasons=["2020", "2021", "2022", "2023", "2024"])
    
    logger.info("Trinity Data Collection (Spine, Identity, and Form) Complete.")

if __name__ == "__main__":
    main()
