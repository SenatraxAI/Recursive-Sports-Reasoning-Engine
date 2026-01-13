# 🧠 WHAT I UNDERSTAND (1.1): The Nuances

I have reviewed the additional clarifications in `IMPROVEDLAYER1POINT1.MD`. Here is my specific understanding of the four critical pillars you outlined.

## 1. The "Dependency" Mechanism
You confirmed that **Team Strength = ∑(Individual Player Contributions)**.
- This is the mathematical foundation for the Injury Matrix.
- **Why it matters**: It allows us to say *"Man City is 88 points strong normally, but 76 points without KDB,*" rather than just guessing.
- **My Action**: Ensure the `ImprovedPlayerLayer` outputs atomic ratings that can be summed.

## 2. Temporal Discipline (The Time Machine)
For the 2026 validation to be real, we must **simulate the calendar**.
- **Rule**: If predicting Jan 15, 2026, we can ONLY use data from *before* Jan 15, 2026.
- **The Rolling Window**: It moves forward in time. We cannot peek at the future.
- **My Action**: The `validate_2026.py` script must strictly enforce this date cutoff for feature engineering.

## 3. The "New Guy" Bias (Cold Start)
You highlighted a major flaw: A new signing (e.g., Mbappe joining Liverpool) has no history *at Liverpool*.
- **The Problem**: Using his PSG history might over/under-estimate his fit. Using zero history assumes he is useless.
- **The Fixes**:
    1.  **League Factors**: Discount stats from weaker leagues.
    2.  **Integration Decay**: Weight recent games less until they settle in? (Actually usually the inverse, but "Integration Decay" suggests trusting history *more* initially until new data arrives?).
        - *Correction based on text*: "weight increasing to full strength after they reach a certain minute threshold". Yes.
    3.  **Positional Average**: Use the league average for "Generic Center Back" as a baseline if data is missing.

## 4. The Data Supply Chain
- **Primary**: **FBref** (24-48h latency) -> Good for xG, Pressing.
- **Secondary**: StatsBomb (Events) -> Too sporadic? 
- **Validation**: Football-Data.co.uk -> Good for Odds/Results.
- **My Action**: I will treat **FBref** as the source of truth for the `PlayerMatchPerformance` dataclass.

**Verification**:
I am aligned. The system is not just "Calculating Ratings", it is **Simulating a Season** day-by-day to ensure no future leakage.
