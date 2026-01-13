"""
This is the 'Hierarchical Trainer'. 
This is where the 'AI Brain' is built. 

We use XGBoost, which is one of the most powerful algorithms for structured data (like table rows).
Our strategy has 4 distinct levels of training.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class HierarchicalTrainer:
    """
    Manages the training of multiple XGBoost models.
    Each model handles a different part of the game.
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Creates a 'models' folder to store the trained AI brains.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # We store models in dictionaries so we can easily pick the right one.
        self.level1_models = {}
        self.level2_models = {}
        self.level3_model = None
        self.level4_model = None
        self.experimental_mode = False # Guardrail

    def train_level1_player_models(self, X, y, position, save_name=None):
        """
        TRAINING LEVEL 1: Player Performance.
        - X = The input stats (passes, dribbles, etc).
        - y = The result we want to predict (e.g., how many goals they'll score next).
        - position = Which position we are training for.
        """
        logger.info(f"Training Level 1 model for position: {position}")
        
        # Settings for the XGBoost 'Algorithm'
        # GPU Acceleration: RTX 3060 Detected
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 0.05,
            'tree_method': 'hist',       # Use RTX 3060 for hardware acceleration
            'device': 'cuda',
            'predictor': 'gpu_predictor'     # Fast predictions on GPU
        }
        
        # Convert data into XGBoost's favorite format
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train the model!
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Save the brain
        self.level1_models[position] = model
        
        target_dir = self.model_dir
        if self.experimental_mode:
            target_dir = self.model_dir / "experiments"
            target_dir.mkdir(parents=True, exist_ok=True)
            
        name = save_name if save_name else f"level1_{position}.json"
        model.save_model(target_dir / name)
        return model

    def train_level2_team_models(self, X, y, stat_type):
        """
        TRAINING LEVEL 2: Team Aggregation.
        Predicts team-level metrics by looking at all 11 players.
        """
        logger.info(f"Training Level 2 model for stat: {stat_type}")
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.08,
            'tree_method': 'hist',
            'device': 'cuda'
        }
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        self.level2_models[stat_type] = model
        model.save_model(self.model_dir / f"level2_{stat_type}.json")
        return model

    def train_level3_matchup_model(self, X, y):
        """
        TRAINING LEVEL 3: Matchup Context.
        - Predicts specific tactical outcomes (e.g., 'Will Home Team dominate possession?').
        """
        logger.info("Training Level 3 Matchup Model...")
        params = {
            'objective': 'binary:logistic', # Yes/No prediction
            'max_depth': 6,
            'learning_rate': 0.1,
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'logloss'
        }
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=150)
        
        self.level3_model = model
        model.save_model(self.model_dir / "level3_matchup.json")
        return model

    def train_level4_outcome_model(self, X, y):
        """
        TRAINING LEVEL 4: Final Outcome Integration.
        - The 'Wisdom of the Crowd' layer. 
        - Combines Level 1/2/3 inputs to predict: Home Win (0), Draw (1), Away Win (2).
        """
        logger.info("Training Level 4 Final Outcome Model...")
        params = {
            'objective': 'multi:softprob', # Predict probabilities for 3 classes
            'num_class': 3,                # Home, Draw, Away
            'max_depth': 4,                # Prevent overfitting
            'learning_rate': 0.05,
            'subsample': 0.8,              # Use 80% of data per tree to vary perspective
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'mlogloss'
        }
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain, num_boost_round=200)
        
        self.level4_model = model
        model.save_model(self.model_dir / "level4_outcome.json")
        return model

if __name__ == "__main__":
    trainer = HierarchicalTrainer()
