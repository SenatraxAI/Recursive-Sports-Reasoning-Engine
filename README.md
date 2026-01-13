# Recursive Sports Reasoning Engine (The Council)

> **A Hierarchical AI System for Premier League Tactical Prediction**
> *Achieved +37.8% ROI in Out-of-Sample Jan 2026 Backtesting.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## 🧠 The Architecture

This is not a simple "stats model". It is a **Compound AI System** that mimics how a human syndicate analyzes a match.

### Layer 1: The Expert System ("System Fit")
- **Inputs:** Manager Tactics (e.g. "Gegenpress"), Player Physical Exams (DNA).
- **Logic:** Calculates a `System Compatibility Score` (0.0 - 1.0).
- **Example:** Determines that *Archie Gray (Leeds)* is a perfect fit for *Daniel Farke's Possession*, but would fail in *Sean Dyche's Low Block*.

### Layer 2: The Tactical Learner (XGBoost)
- **Inputs:** Formation Density, Central Overload, Width Balance, Team Power Ratings.
- **Logic:** Predicts raw goal expectancy based on **Tactical Mismatches** (e.g. 4-4-2 vs 3-5-2).
- **Feature:** Recursive Feature Elimination (RFE) used to identify key tactical drivers.

### Layer 3: The State Management ("The Narrative")
- **Inputs:** Layer 2 Predictions, Recent Form, Momentum.
- **Logic:** Classifies the "Game Script":
    - `DEADLOCK`: High Draw probability, Low Goals.
    - `CHAOS`: High Variance, End-to-End.
    - `CONTROL`: One team dominates possession.

### Layer 4: The Calibration Ensemble ("The Council")
- **The Veteran:** A static model trained on long-term averages (removes bias).
- **The Scout:** A dynamic recursive model that updates ratings match-by-match (catches streaks).
- **Output:** A Probability Matrix calibrated via Isotonic Regression.

---

## � How It Works (Simple Explanation)
Imagine you are betting on a match, but before you place your money, you consult a **Council of Experts**:

1.  **The Scout:** "I've watched every game this month. Newcastle is exhausted and their press is broken."
2.  **The Tactician:** "Leeds plays a Possession style, which is kryptonite for Newcastle's current formation."
3.  **The Veteran:** "Historically, home teams with this specific rating advantage win 65% of the time."

The AI acts as the **Chairman**, listening to all these experts. If they argue (e.g., Scout says "Lose", Veteran says "Win"), the AI skips the bet. If they **agree**, it gives you a green light.

---

## 🔬 Technical Deep Dive (The Math)
This project moves beyond standard "Goal Averages" by implementing **High-Dimensional Vector Mismatches**.

### 1. Vector Space Embedding (The "Fit" Score)
We map every Manager and Player into a shared latent space ($R^6$):
$$ \text{Fit}(P, M) = 1.0 - \alpha \cdot || \vec{v}_P - \vec{v}_M || $$
Where $\vec{v}_P$ is the Player's Attribute Vector (Stamina, Passing, etc.) and $\vec{v}_M$ is the Manager's Requirement Vector. This allows us to quantify "Tactical Friction" even for players who have never played for that manager before.

### 2. Recursive State Estimation
Unlike static models, this engine processes matches sequentially. The Team Strength $S_t$ at time $t$ is a function of the previous state and the prediction error:
$$ S_{t+1} = S_t + \eta \cdot (Y_{actual} - Y_{predicted}) $$
This allows the model to "learn" from a team's over/underperformance in real-time.

### 3. Gradient Boosted Decision Trees (XGBoost)
We utilize `XGBoost` with **Recursive Feature Elimination (RFE)**. The model learned that heavily engineered features like `central_density_ratio` (3 mid vs 2 mid) are 4x more predictive than simple "Last 5 Games" form.

---

## �🚀 Getting Started

### Prerequisites
*   Python 3.10+

### Installation
```bash
git clone https://github.com/SenatraxAI/Recursive-Sports-Reasoning-Engine.git
cd Recursive-Sports-Reasoning-Engine
pip install -r requirements.txt
```

### Running the Dashboard
The "Council" UI allows for interactive betting analysis:
```bash
streamlit run src/app.py
```
*   **Features:** Lineup Parsing (Paste text), Strength Sliders, Dual-Model visualizer.

### Validating the Model
To run the simulation on the test set (Jan 2026):
```bash
python src/sim_betting_calibrated.py
```

---

## 📂 Project Structure
*   `src/` - Core source code (Predictors, Parsers, Models).
*   `data/` - (GitIgnored) Raw Parquet files and Match History.
*   `scripts/` - Auxiliary tools for auditing and data collection.
*   `notebooks/` - Experimental research.

## 🤝 Contributing
Built by **SenatraxAI** specializing in Agentic Coding and High-Dimensional Reasoning.
