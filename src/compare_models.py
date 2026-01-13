import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

def compare_models():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🏆 MODEL SHOWDOWN: BASELINE vs STACKED (Hierarchical)")
    print("="*80)
    
    # Load test data
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    test_df = matches_df[matches_df['date'] >= '2024-01-01'].copy()
    y_test = test_df['result'].astype(int)
    
    print(f"\nTest Set: {len(test_df)} matches from 2024-2025 season\n")
    
    # ==================== MODEL 1: BASELINE ====================
    print("📊 MODEL 1: BASELINE (No Hierarchical Stacking)")
    print("-" * 80)
    
    # Baseline features (simplified - no formations to avoid mismatch)
    test_df = test_df.sort_values('date')
    test_df['h_recent_xg'] = test_df.groupby('home_team')['home_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    test_df['a_recent_xg'] = test_df.groupby('away_team')['away_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    
    baseline_features = ['h_recent_xg', 'a_recent_xg']
    X_baseline = test_df[baseline_features].astype(float)
    
    model_baseline = xgb.Booster()
    model_baseline.load_model(model_dir / "level4_baseline.json")
    
    y_pred_proba_baseline = model_baseline.predict(xgb.DMatrix(X_baseline))
    if len(y_pred_proba_baseline.shape) > 1:
        y_pred_baseline = np.argmax(y_pred_proba_baseline, axis=1)
        conf_baseline = np.max(y_pred_proba_baseline, axis=1)
    else:
        y_pred_baseline = (y_pred_proba_baseline > 0.5).astype(int)
        conf_baseline = np.where(y_pred_baseline == 1, y_pred_proba_baseline, 1 - y_pred_proba_baseline)
    
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    print(f"Accuracy: {acc_baseline:.2%}")
    print(f"Features: {len(baseline_features)} (Recent xG + Formations)")
    
    # ==================== MODEL 2: STACKED ====================
    print("\n🔥 MODEL 2: STACKED (With Level 3 Tactical Predictions)")
    print("-" * 80)
    
    # Generate L3 predictions
    tactical_cols = ['home_formation', 'home_pressing', 'home_style', 
                     'away_formation', 'away_pressing', 'away_style']
    existing_cols = [c for c in tactical_cols if c in test_df.columns]
    
    model_l3 = xgb.Booster()
    model_l3.load_model(model_dir / "level3_matchup.json")
    
    # Load L3 Features to ensure alignment
    import json
    with open(model_dir / "level3_features.json", "r") as f:
        l3_feature_cols = json.load(f)
    
    # Align columns: Add missing as 0, remove extra, ensure order
    # First, get dummies usually
    l3_df_raw = pd.get_dummies(test_df[['id'] + existing_cols], columns=existing_cols, drop_first=True)
    
    # Reindex to match training features exactly
    l3_df_aligned = l3_df_raw.reindex(columns=l3_feature_cols, fill_value=0)
    
    X_l3 = l3_df_aligned.astype(float)
    l3_proba = model_l3.predict(xgb.DMatrix(X_l3))
    test_df['l3_pred_tactical_control'] = l3_proba
    
    # Stacked features (simplified - no formations)
    stacked_features = ['h_recent_xg', 'a_recent_xg', 'l3_pred_tactical_control']
    X_stacked = test_df[stacked_features].astype(float)
    
    model_stacked = xgb.Booster()
    model_stacked.load_model(model_dir / "level4_outcome.json")
    
    y_pred_proba_stacked = model_stacked.predict(xgb.DMatrix(X_stacked))
    if len(y_pred_proba_stacked.shape) > 1:
        y_pred_stacked = np.argmax(y_pred_proba_stacked, axis=1)
        conf_stacked = np.max(y_pred_proba_stacked, axis=1)
    else:
        y_pred_stacked = (y_pred_proba_stacked > 0.5).astype(int)
        conf_stacked = np.where(y_pred_stacked == 1, y_pred_proba_stacked, 1 - y_pred_proba_stacked)
    
    acc_stacked = accuracy_score(y_test, y_pred_stacked)
    print(f"Accuracy: {acc_stacked:.2%}")
    print(f"Features: {len(stacked_features)} (Recent xG + Formations + L3 Tactical)")
    
    # ==================== COMPARISON ====================
    print("\n" + "="*80)
    print("🏆 FINAL VERDICT")
    print("="*80)
    
    improvement = (acc_stacked - acc_baseline) * 100
    improvement_pct = (acc_stacked / acc_baseline - 1) * 100
    
    print(f"\nBaseline Model:  {acc_baseline:.2%}")
    print(f"Stacked Model:   {acc_stacked:.2%}")
    print(f"\nImprovement:     +{improvement:.2f} percentage points ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✅ WINNER: STACKED MODEL")
        print(f"   Hierarchical stacking IMPROVED accuracy!")
    elif improvement == 0:
        print(f"\n🤝 TIE: Both models performed equally")
    else:
        print(f"\n⚠️  BASELINE performed better (unexpected)")
    
    # Classification reports
    print("\n" + "-"*80)
    print("BASELINE Classification Report:")
    print(classification_report(y_test, y_pred_baseline, 
                                target_names=['Home Win', 'Draw', 'Away Win'],
                                zero_division=0))
    
    print("\n" + "-"*80)
    print("STACKED Classification Report:")
    print(classification_report(y_test, y_pred_stacked, 
                                target_names=['Home Win', 'Draw', 'Away Win'],
                                zero_division=0))
    
    # Confidence comparison
    print("\n" + "="*80)
    print("📊 CONFIDENCE ANALYSIS (70%+ threshold)")
    print("="*80)
    
    # Baseline @ 70%
    mask_baseline_70 = conf_baseline >= 0.70
    if mask_baseline_70.sum() > 0:
        acc_baseline_70 = accuracy_score(y_test[mask_baseline_70], y_pred_baseline[mask_baseline_70])
        print(f"\nBaseline @ 70%+: {acc_baseline_70:.2%} ({mask_baseline_70.sum()} matches)")
    
    # Stacked @ 70%
    mask_stacked_70 = conf_stacked >= 0.70
    if mask_stacked_70.sum() > 0:
        acc_stacked_70 = accuracy_score(y_test[mask_stacked_70], y_pred_stacked[mask_stacked_70])
        print(f"Stacked @ 70%+:  {acc_stacked_70:.2%} ({mask_stacked_70.sum()} matches)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    models = ['Baseline\n(No Stacking)', 'Stacked\n(with L3)']
    accuracies = [acc_baseline * 100, acc_stacked * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.axhline(y=33.33, color='red', linestyle='--', label='Random Baseline (33%)', linewidth=2)
    ax1.axhline(y=52, color='green', linestyle='--', label='Profitable Threshold (52%)', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(30, max(accuracies) + 5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Confidence distribution
    ax2.hist([conf_baseline, conf_stacked], bins=20, label=['Baseline', 'Stacked'], 
             alpha=0.7, color=colors, edgecolor='black')
    ax2.axvline(x=0.70, color='black', linestyle='--', label='70% Threshold', linewidth=2)
    ax2.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(visual_dir / "model_comparison.png", dpi=150)
    print(f"\n✅ Comparison chart saved to: {visual_dir / 'model_comparison.png'}")

if __name__ == "__main__":
    compare_models()
