# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# ================================
# 🚀 Load trained model & preprocessors
# ================================
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ================================
# 🎨 Streamlit Page Config
# ================================
st.set_page_config(
    page_title="F1 Lap Time Predictor",
    page_icon="🏎️",
    layout="wide",
)

# --- Custom CSS for non-vanilla look ---
st.markdown("""
    <style>
        .main {
            background-color: #0b0c10;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 {
            color: #f0131e;
            text-shadow: 0 0 8px rgba(255, 0, 0, 0.5);
        }
        .stButton button {
            background-color: #f0131e;
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: bold;
            transition: 0.2s;
        }
        .stButton button:hover {
            background-color: #ff4545;
            color: black;
        }
        .css-1v0mbdj {
            background-color: #0b0c10;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ================================
# 🏁 Top Navigation Bar
# ================================
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=90)
with col2:
    st.title("Formula 1 Lap Time Predictor")

st.markdown("---")

# ================================
# 🧠 Sidebar Inputs
# ================================
st.sidebar.header("Driver & Race Inputs")

# Track dropdowns
selected_circuit = st.sidebar.selectbox(
    "Select Circuit",
    ["Italian Grand Prix", "Azerbaijan Grand Prix", "British Grand Prix", "Belgian Grand Prix"]
)

selected_compound = st.sidebar.selectbox(
    "Tyre Compound",
    ["SOFT", "MEDIUM", "HARD"]
)

tyre_life = st.sidebar.slider("Tyre Life (laps completed on current tyre)", 0, 60, 10)
lap_number = st.sidebar.slider("Lap Number", 1, 70, 25)
stint = st.sidebar.slider("Stint Number", 1, 5, 2)
stint_lap = st.sidebar.slider("Lap Number within Stint", 1, 30, 10)

# Speed traps
st.sidebar.subheader("Speed Data (km/h)")
speed_i1 = st.sidebar.slider("Speed at I1", 200, 350, 290)
speed_i2 = st.sidebar.slider("Speed at I2", 200, 350, 305)
speed_fl = st.sidebar.slider("Speed at Finish Line", 200, 350, 315)

# Derived features
total_laps = 70
lap_progress = lap_number / total_laps
stint_total = 20
stint_progress = stint_lap / stint_total

tyre_phase = (
    "Fresh" if tyre_life <= 5 else
    "Mid" if tyre_life <= 15 else
    "Worn" if tyre_life <= 30 else
    "VeryWorn"
)

# ================================
# 🧩 Build Input Vector (One-Hot Encoding)
# ================================
user_input = {
    "LapNumber": lap_number,
    "TyreLife": tyre_life,
    "SpeedI1": speed_i1,
    "SpeedI2": speed_i2,
    "SpeedFL": speed_fl,
    "Stint": stint,
    "LapProgress": lap_progress,
    "StintLap": stint_lap,
    "StintProgress": stint_progress,
}

# Create one-hot features dynamically
for col in model_columns:
    if col.startswith("Circuit_"):
        user_input[col] = 1 if col == f"Circuit_{selected_circuit}" else 0
    elif col.startswith("Compound_"):
        user_input[col] = 1 if col == f"Compound_{selected_compound}" else 0
    elif col.startswith("TyrePhase_"):
        user_input[col] = 1 if col == f"TyrePhase_{tyre_phase}" else 0
    elif col not in user_input:
        user_input[col] = 0

# Ensure all model columns are present and ordered
df_input = pd.DataFrame([user_input])[model_columns]

# ================================
# ⚙️ Preprocess & Predict
# ================================
df_imputed = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)
df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_input.columns)
pred_laptime = model.predict(df_scaled)[0]

# ================================
# 🏆 Output Display
# ================================
st.markdown("## 🏁 Predicted Lap Time")

st.metric(
    label="Predicted Lap Time (seconds)",
    value=f"{pred_laptime:.3f} s"
)

# Add a nice note
st.markdown(
    f"<p style='color:#aaa;'>Circuit: <b>{selected_circuit}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Tyre: <b>{selected_compound}</b> ({tyre_phase})</p>",
    unsafe_allow_html=True
)

# Optional: show dataframe (debug mode)
with st.expander("Show model input data"):
    st.dataframe(df_input)

# Footer
st.markdown("<hr><center>🏎️ Built for F1 Lap Time Analysis (2023 Season)</center>", unsafe_allow_html=True)
