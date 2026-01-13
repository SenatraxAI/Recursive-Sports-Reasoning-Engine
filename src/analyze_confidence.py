import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score

def analyze_confidence_thresholds():
    processed_dir = Path("data/processed")
    model_dir = Path("data/models")
    visual_dir = Path("data/visuals/validation")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🎯 CONFIDENCE FILTERING ANALYSIS - LEVEL 4")
    print("="*70)
    print("\nTesting: Should you bet on ALL matches or only HIGH-CONFIDENCE ones?\n")
    
    # Load and prepare data (same as verify_level4.py)
    matches_df = pd.read_parquet(processed_dir / "processed_matches.parquet")
    matches_df = matches_df.reset_index(drop=True)
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    def get_result(row):
        if row['home_goals'] > row['away_goals']: return 0
        elif row['home_goals'] == row['away_goals']: return 1
        else: return 2
    
    matches_df['result'] = matches_df.apply(get_result, axis=1)
    
    # Feature engineering
    matches_df = matches_df.sort_values('date')
    matches_df['h_recent_xg'] = matches_df.groupby('home_team')['home_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    matches_df['a_recent_xg'] = matches_df.groupby('away_team')['away_xg'].transform(
        lambda x: x.shift(1).fillna(1.0))
    
    cols_to_encode = [c for c in ['home_formation', 'away_formation'] if c in matches_df.columns]
    l4_df = pd.get_dummies(matches_df, columns=cols_to_encode, drop_first=True)
    
    # Test set
    test_l4 = l4_df[l4_df['date'] >= '2024-01-01'].copy()
    
    features = ['h_recent_xg', 'a_recent_xg'] + [c for c in l4_df.columns if 'formation_' in c]
    X_test = test_l4[features].astype(float)
    y_test = test_l4['result'].astype(int)
    
    # Load model and get predictions
    model = xgb.Booster()
    model.load_model(model_dir / "level4_outcome.json")
    
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    
    # Get predicted class and confidence
    if len(y_pred_proba.shape) > 1:
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidence = np.max(y_pred_proba, axis=1)  # Max probability = confidence
    else:
        y_pred = (y_pred_proba > 0.5).astype(int)
        confidence = np.where(y_pred == 1, y_pred_proba, 1 - y_pred_proba)
    
    # Test different confidence thresholds
    thresholds = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []
    
    print("\n📊 RESULTS BY CONFIDENCE THRESHOLD:\n")
    print(f"{'Threshold':<12} {'Matches':<10} {'% of Total':<12} {'Accuracy':<10} {'vs Baseline':<12}")
    print("-" * 70)
    
    for threshold in thresholds:
        mask = confidence >= threshold
        if mask.sum() == 0:
            continue
            
        filtered_preds = y_pred[mask]
        filtered_actual = y_test.values[mask]
        
        acc = accuracy_score(filtered_actual, filtered_preds)
        num_matches = mask.sum()
        pct_total = (num_matches / len(y_test)) * 100
        vs_baseline = acc - 0.3333  # vs random (33%)
        
        results.append({
            'threshold': threshold,
            'matches': num_matches,
            'pct_total': pct_total,
            'accuracy': acc,
            'vs_baseline': vs_baseline
        })
        
        # Visual indicator for good thresholds
        indicator = "⭐" if acc > 0.52 else "✅" if acc > 0.45 else ""
        
        print(f"{threshold:.2f}+ {indicator:<5} {num_matches:<10} {pct_total:>6.1f}%      {acc:>6.1%}     {vs_baseline:>+6.1%}")
    
    print("-" * 70)
    print("\nLEGEND:")
    print("⭐ = Potentially profitable (>52% accuracy beats bookmaker margins)")
    print("✅ = Above current baseline (45.74%)")
    print()
    
    # Convert to DataFrame for plotting
    results_df = pd.DataFrame(results)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Threshold
    ax1.plot(results_df['threshold'], results_df['accuracy'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=0.52, color='green', linestyle='--', label='Profitable Threshold (52%)', linewidth=2)
    ax1.axhline(y=0.4574, color='orange', linestyle='--', label='Current Baseline (45.74%)', linewidth=2)
    ax1.axhline(y=0.3333, color='red', linestyle='--', label='Random Guess (33%)', linewidth=2)
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, max(results_df['accuracy'].max() + 0.05, 0.65))
    
    # Plot 2: Number of Matches vs Threshold
    ax2.bar(results_df['threshold'], results_df['matches'], 
            color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Number of Matches', fontsize=12)
    ax2.set_title('Available Bets vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(visual_dir / "confidence_analysis.png", dpi=150)
    print(f"✅ Visualization saved to: {visual_dir / 'confidence_analysis.png'}\n")
    
    # Find optimal threshold (highest accuracy above 100 matches)
    viable = results_df[results_df['matches'] >= 100]
    if not viable.empty:
        optimal = viable.loc[viable['accuracy'].idxmax()]
        print("🎯 RECOMMENDED STRATEGY:")
        print(f"   Threshold: {optimal['threshold']:.0%}+ confidence")
        print(f"   Bet on: {optimal['matches']:.0f} matches ({optimal['pct_total']:.1f}% of total)")
        print(f"   Expected accuracy: {optimal['accuracy']:.1%}")
        print(f"   Improvement: +{(optimal['accuracy'] - 0.4574)*100:.1f} percentage points\n")
    
    return results_df

if __name__ == "__main__":
    analyze_confidence_thresholds()
