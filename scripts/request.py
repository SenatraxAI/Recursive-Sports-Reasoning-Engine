import sys
import argparse
from src.predict_match import MatchPredictor

def main():
    parser = argparse.ArgumentParser(description="Request a Match Prediction from the Hierarchical 2.0 Engine")
    parser.add_argument("home", help="Name of the Home Team (e.g. man_city)")
    parser.add_argument("away", help="Name of the Away Team (e.g. arsenal)")
    
    # Use -h for help if needed
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Initialize the machine
    predictor = MatchPredictor()
    
    # Run prediction
    try:
        predictor.request_prediction(args.home, args.away)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
