import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_level3_audit():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- ⚔️ LEVEL 3 TACTICAL MATCHUP AUDIT (2024-2025 DATA) ---")
    
    # Load Match Data
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    # Tactical Columns
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 
                     'away_formation', 'away_pressing', 'away_style']
    
    existing_cols = [c for c in tactical_cols if c in matches_df.columns]
    if not existing_cols:
        print("❌ No tactical columns found. Skipping Level 3 audit.")
        return
    
    # One-Hot Encode
    l3_df = pd.get_dummies(matches_df, columns=existing_cols, drop_first=True)
    
    # Target: Tactical Control (Home xG > Away xG)
    l3_df['tactical_control'] = (l3_df['home_xg'] > l3_df['away_xg']).astype(int)
    
    # Test Set (2024-2025)
    test_l3 = l3_df[l3_df['date'] >= '2024-01-01']
    
    # Features
    feature_cols = [c for c in l3_df.columns if 'formation_' in c or 'pressing_' in c or 'style_' in c]
    
    if not feature_cols:
        print("❌ No tactical features found after encoding. Skipping.")
        return
    
    X_test = test_l3[feature_cols].astype(float)
    y_test = test_l3['tactical_control']
    
    # Load Model
    model = xgb.Booster()
    model.load_model(model_dir / "level3_matchup.json")
    
    # Predict
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\n⚔️ Level 3 Tactical Control Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Dominance', 'Home Dominance']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Away Dom', 'Home Dom'],
                yticklabels=['Away Dom', 'Home Dom'])
    plt.title(f'Level 3: Tactical Control Prediction (Acc: {acc:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(visual_dir / "l3_confusion_matrix.png")
    print(f"\n✅ Level 3 Confusion Matrix saved to: {visual_dir / 'l3_confusion_matrix.png'}")

if __name__ == "__main__":
    run_level3_audit()
