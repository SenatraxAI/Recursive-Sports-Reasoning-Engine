import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.predict_recursive import RecursivePredictor
from src.lineup_parser import LineupParser

# Page Config
st.set_page_config(
    page_title="Profit Engine 2.0",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark/Green Theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #4CAF50;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engines():
    """
    Load TWO engines:
    1. Static (The Veteran): Uses long-term averages, ignores recent noise.
    2. Dynamic (The Scout): Updates ratings daily, catches streaks.
    """
    # 1. Static Engine (Never updates state)
    static_pred = RecursivePredictor()
    # (No update loop)

    # 2. Dynamic Engine (Replays history)
    dynamic_pred = RecursivePredictor()
    matches = pd.read_csv("data/raw/jan2026_matches.csv")
    if 'date' in matches.columns: 
        matches['date'] = pd.to_datetime(matches['date'])
        matches = matches.sort_values('date').reset_index(drop=True)
    
    for i in range(len(matches)):
        m = matches.iloc[i]
        dynamic_pred.state.update(
            str(m['home_team']).lower().strip(), 
            str(m['away_team']).lower().strip(), 
            m['home_goals'], m['away_goals'], 
            m.get('home_xg', np.nan), m.get('away_xg', np.nan)
        )
    return static_pred, dynamic_pred, matches

@st.cache_resource
def load_parser_v4():
    return LineupParser()

engine_static, engine_dynamic, history = load_engines()
parser = load_parser_v4()

# --- SIDEBAR: INPUTS ---
st.sidebar.markdown("## 🎮 Match Inputs")

teams = sorted(list(engine_dynamic.state.ratings.keys()))
teams = [t.title() for t in teams]

home_team = st.sidebar.selectbox("Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away_team = st.sidebar.selectbox("Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

st.sidebar.markdown("### 📊 Live Odds (Bookie)")
col1, col2, col3 = st.sidebar.columns(3)
odds_h = col1.number_input("Home", value=2.00, step=0.01)
odds_d = col2.number_input("Draw", value=3.20, step=0.01)
odds_a = col3.number_input("Away", value=3.50, step=0.01)

st.sidebar.markdown("### 📝 Lineup Context")
formations = ["4-3-3", "4-2-3-1", "3-4-3", "3-5-2", "4-4-2", "5-3-2"]
h_fmt = st.sidebar.selectbox(f"{home_team} Shape", formations, index=0)
a_fmt = st.sidebar.selectbox(f"{away_team} Shape", formations, index=0)

with st.sidebar.expander("Paste Starting XI (Text/URL)"):
    h_text = st.text_area(f"{home_team} Lineup", height=70, placeholder="Paste player names...")
    a_text = st.text_area(f"{away_team} Lineup", height=70, placeholder="Paste player names...")
    
    calc_h = 100
    calc_a = 100
    if h_text:
        calc_h, msg_h = parser.calculate_strength(home_team, h_text)
        st.caption(f"Score: {calc_h}% ({msg_h})")
    if a_text:
        calc_a, msg_a = parser.calculate_strength(away_team, a_text)
        st.caption(f"Score: {calc_a}% ({msg_a})")

st.sidebar.markdown("### 🩹 Final Strength Adjustment")
h_strength = st.sidebar.slider(f"{home_team} Strength %", 50, 100, calc_h, help="100% = Best Starting XI. Reduce this if key players are injured (e.g. Rice out -> 90%).")
a_strength = st.sidebar.slider(f"{away_team} Strength %", 50, 100, calc_a, help="100% = Best Starting XI. Reduce this if key players are injured (e.g. Palmer out -> 90%).")

col_btn1, col_btn2 = st.sidebar.columns(2)
analyze_btn = col_btn1.button("🚀 PREDICT")
refresh_btn = col_btn2.button("🔄 REFRESH")

if refresh_btn:
    st.cache_resource.clear()
    st.rerun()

# --- MAIN DASHBOARD ---
st.title("💰 Profit Engine: The Council")
st.markdown("### Dual-Model Consensus System")
st.divider()

if analyze_btn:
    try:
        m = {
            'home_team': home_team.lower(),
            'away_team': away_team.lower(),
            'home_odds': odds_h, 
            'draw_odds': odds_d, 
            'away_odds': odds_a,
            'home_formation': h_fmt, 'away_formation': a_fmt
        }
        
        # ASK ADVISERS
        probs_s, hp_s, ap_s = engine_static.get_prediction(m)
        probs_d, hp_d, ap_d = engine_dynamic.get_prediction(m)
        
        # Apply Lineup Penalties (Simple Power Reduction)
        # Rule of thumb: 10% Strength drop = 5% Win Prob drop roughly.
        # We adjust the probs manually for the display/advice.
        
        h_factor = h_strength / 100.0
        a_factor = a_strength / 100.0
        
        def adjust_probs(probs, h_f, a_f):
            # Normalize boost/penalty
            # If Home is 90%, Away relative str increases.
            p_h = probs['HOME'] * h_f
            p_a = probs['AWAY'] * a_f
            p_d = probs['DRAW']
            tot = p_h + p_a + p_d
            return {'HOME': p_h/tot, 'DRAW': p_d/tot, 'AWAY': p_a/tot}

        if h_strength < 100 or a_strength < 100:
            probs_s = adjust_probs(probs_s, h_factor, a_factor)
            probs_d = adjust_probs(probs_d, h_factor, a_factor)
            st.toast(f"⚠️ Adjusted for Injuries: {home_team} {h_strength}%, {away_team} {a_strength}%")
        
        # Calculate Edges
        def get_best_pick(probs, odds_h, odds_d, odds_a):
            choices = [('HOME', probs['HOME'], odds_h), ('DRAW', probs['DRAW'], odds_d), ('AWAY', probs['AWAY'], odds_a)]
            best = "SKIP"; max_edge = 0.0
            for l, p, o in choices:
                e = (p * o) - 1.0
                if e > 0.15 and e > max_edge: max_edge = e; best = l
            return best, max_edge

        pick_s, edge_s = get_best_pick(probs_s, odds_h, odds_d, odds_a)
        pick_d, edge_d = get_best_pick(probs_d, odds_h, odds_d, odds_a)
        
        # CONSENSUS LOGIC
        final_verdict = "SKIP"
        verdict_color = "#F44336" # Red
        verdict_reason = "No Value Found"
        
        if pick_s == pick_d and pick_s != "SKIP":
            final_verdict = f"STRONG {pick_s}"
            verdict_color = "#4CAF50" # Green
            verdict_reason = "✅ Both Advisers Agree (High Confidence)"
        elif pick_d != "SKIP" and pick_s == "SKIP":
            final_verdict = f"SPECULATIVE {pick_d}"
            verdict_color = "#FFC107" # Amber
            verdict_reason = "⚠️ Scout detects Form Streak (Veteran disagrees)"
        elif pick_s != "SKIP" and pick_d == "SKIP":
             # This is rare: History says yes, but Recent Form says "Stay Away"
             final_verdict = "SKIP" 
             verdict_color = "#F44336"
             verdict_reason = "🛑 Scout warns against it (Recent Slump)"
        elif pick_s != pick_d and pick_s != "SKIP" and pick_d != "SKIP":
            final_verdict = "CONFLICT"
            verdict_reason = f"⚔️ Civil War: Veteran wants {pick_s}, Scout wants {pick_d}"

        # DISPLAY
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.markdown("### 👴 The Veteran")
            st.caption("Static (Long-Term)")
            st.metric(f"{home_team}", f"{hp_s:.2f}")
            st.bar_chart({"Home": probs_s['HOME'], "Draw": probs_s['DRAW'], "Away": probs_s['AWAY']}, height=150)
            if pick_s != "SKIP": st.success(f"Bet {pick_s} ({edge_s*100:.0f}%)")
            else: st.error("No Bet")

        with c2:
            st.markdown("### 🕵️ The Scout")
            st.caption("Dynamic (Short-Term)")
            st.metric(f"{home_team}", f"{hp_d:.2f}", delta=f"{hp_d-hp_s:.2f}")
            st.bar_chart({"Home": probs_d['HOME'], "Draw": probs_d['DRAW'], "Away": probs_d['AWAY']}, height=150)
            if pick_d != "SKIP": st.success(f"Bet {pick_d} ({edge_d*100:.0f}%)")
            else: st.error("No Bet")
            
        with c3:
            st.markdown("### ⚖️ The Council")
            st.caption("Final Verdict")
            st.markdown(f"""
            <div style="background-color: {verdict_color}20; border: 2px solid {verdict_color}; border-radius: 10px; padding: 15px;">
                <h2 style="color: {verdict_color}; text-align: center;">{final_verdict}</h2>
                <p style="text-align: center;">{verdict_reason}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Market Check")
            st.write(f"**Implied Odds:** {1.0/odds_h:.1%} (Bookie)")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👈 Ask the advisers for a prediction.")
