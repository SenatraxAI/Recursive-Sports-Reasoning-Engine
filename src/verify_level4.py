import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_level4_audit():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- 🎯 LEVEL 4 OUTCOME PREDICTION AUDIT (2024-2025 DATA) ---")
    
    # Load Match Data
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Target: Match Result (0=Home Win, 1=Draw, 2=Away Win)
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # Feature Engineering (Recent xG as proxy for team strength)
    matches_df = matches_df.sort_values('date')
    matches_df['h_recent_xg'] = matches_df.groupby('home_team')['home_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    matches_df['a_recent_xg'] = matches_df.groupby('away_team')['away_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    
    # One-Hot Encode Formations
    cols_to_encode = [c for c in ['home_formation', 'away_formation'] if c in matches_df.columns]
    l4_df = pd.get_dummies(matches_df, columns=cols_to_encode, drop_first=True)
    
    # Test Set
    test_l4 = l4_df[l4_df['date'] >= '2024-01-01']
    
    # Features
    features = ['h_recent_xg', 'a_recent_xg'] + [c for c in l4_df.columns if 'formation_' in c]
    
    X_test = test_l4[features].astype(float)
    y_test = test_l4['result'].astype(int)
    
    # Load Model
    model = xgb.Booster()
    model.load_model(model_dir / "level4_outcome.json")
    
    # Predict
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    
    # Multi-class: Get argmax
    if len(y_pred_proba.shape) > 1:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Binary fallback
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Level 4 Outcome Prediction Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Home Win', 'Draw', 'Away Win'],
                                zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Home Win', 'Draw', 'Away Win'],
                yticklabels=['Home Win', 'Draw', 'Away Win'])
    plt.title(f'Level 4: Match Outcome Prediction (Acc: {acc:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(visual_dir / "l4_confusion_matrix.png")
    print(f"\n✅ Level 4 Confusion Matrix saved to: {visual_dir / 'l4_confusion_matrix.png'}")

if __name__ == "__main__":
    run_level4_audit()
