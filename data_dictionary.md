# The "Blink-Level" Data Dictionary

Welcome to the scouting room! To reach the granularity you requested, we track these "Super Quality" metrics. Here is what they mean and why they matter for our Hierarchical Engine.

## 1. Offensive Metrics (Level 1)
| Metric | Meaning | Why it's a "Blink" detail |
| :--- | :--- | :--- |
| **xG (Expected Goals)** | The probability that a shot will result in a goal based on its location and type. | Tells us if a player is getting into good positions, regardless of if they scored today. |
| **Progressive Carries** | Runs that move the ball at least 10 yards towards the opponent's goal. | Shows us who the "Ball Progressors" are—the players who drive the team forward. |
| **Touches in Box** | How many times a player touches the ball inside the opponent's penalty area. | The ultimate indicator of danger. If this is high, goals are coming. |

## 2. Defensive Metrics (Level 2)
| Metric | Meaning | Why it's a "Blink" detail |
| :--- | :--- | :--- |
| **PPDA** | Passes Per Defensive Action. | Measures how many passes an opponent makes before our team tries to tackle them. **Lower is faster pressing.** |
| **Ball Recoveries** | When a player wins back a "loose ball" that neither team controlled. | The secret sauce of ball-winning midfielders like Rodri. |
| **Pressures Applied** | How many times a player closed down an opponent during a pass or shot. | Shows the true defensive 'engine' of a player, even if they don't get the tackle. |

## 3. The "Blink" Dynamics (Level 3 & 4)
| Metric | Meaning | Why it's a "Blink" detail |
| :--- | :--- | :--- |
| **Momentum (-1 to 1)** | A live score of which team is currently dominant. | Used in our **Markov Chain** to predict the next 5 minutes of a match. |
| **Energy/Fatigue (1.0 to 0)** | A decay score based on minutes played and intensity. | Essential for predicting late-game goals (the "Substitution Effect"). |

---

> [!TIP]
> **Why do we need 5 Seasons (from 2020)?**
> A single season can be an outlier. 5 seasons allow us to track **Career Trajectories** (is a player peaking or declining?) and see how they perform under different managers over time.
