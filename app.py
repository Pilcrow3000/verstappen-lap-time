import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import base64
import os

# ----------------------------------------------------------
#  Load pre-trained model and preprocessing artifacts
# ----------------------------------------------------------
MODEL_DIR = "models"

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
with open(os.path.join(MODEL_DIR, "model_columns.pkl"), "rb") as f:
    model_columns = pickle.load(f)

# ----------------------------------------------------------
#  Streamlit Page Setup
# ----------------------------------------------------------
st.set_page_config(page_title="F1 Lap Time Predictor", page_icon="🏎️", layout="wide")

# Add custom CSS for navbar & layout
st.markdown("""
    <style>
        /* General page background */
        body {
            background-color: #0a0a0a;
            color: #f1f1f1;
        }

        /* Top navbar styling */
        .navbar {
            background-color: #111;
            padding: 0.7rem 1.5rem;
            border-bottom: 2px solid #d90000;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
        }

        /* Padding for main content to avoid overlap */
        .main-content {
            padding-top: 5rem;
        }

        /* F1 logo */
        .navbar img {
            height: 40px;
        }

        /* App title */
        .navbar h1 {
            font-size: 1.4rem;
            color: #f5f5f5;
            margin: 0;
            font-weight: 600;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Prediction card */
        .result-card {
            background-color: #151515;
            border: 1px solid #d90000;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
        }

        .result-value {
            font-size: 2.4rem;
            font-weight: 700;
            color: #ff4747;
        }

        /* Streamlit button tweaks */
        button[kind="primary"] {
            background-color: #d90000 !important;
            color: white !important;
            border: none !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
#  Navbar with F1 Logo
# ----------------------------------------------------------
st.markdown("""
<div class="navbar">
    <img src="https://upload.wikimedia.org/wikipedia/en/3/33/F1.svg" alt="F1 Logo">
    <h1>Max Verstappen Lap Time Predictor (2023)</h1>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ----------------------------------------------------------
#  Sidebar Input Fields
# ----------------------------------------------------------
st.sidebar.header("Input Parameters")

# User inputs
circuit = st.sidebar.selectbox("Circuit", [
    "Italian Grand Prix", "Azerbaijan Grand Prix", "British Grand Prix", "Belgian Grand Prix"
])

compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
tyre_life = st.sidebar.slider("Tyre Life (laps)", 0, 60, 10)
stint = st.sidebar.number_input("Stint Number", 1, 5, 2)
lap_number = st.sidebar.number_input("Lap Number", 1, 60, 20)
speed_i1 = st.sidebar.slider("Speed at Sector 1 (km/h)", 250, 360, 310)
speed_i2 = st.sidebar.slider("Speed at Sector 2 (km/h)", 250, 360, 315)
speed_fl = st.sidebar.slider("Speed at Finish Line (km/h)", 250, 360, 320)

# Derived features (like in training)
lap_progress = lap_number / 60
stint_lap = lap_number % 20 if lap_number % 20 != 0 else 20
stint_progress = stint_lap / 20
tyre_phase = pd.cut([tyre_life], bins=[-1, 5, 15, 30, 999],
                    labels=["Fresh", "Mid", "Worn", "VeryWorn"]).astype(str)[0]

# ----------------------------------------------------------
#  Create Input DataFrame
# ----------------------------------------------------------
input_dict = {
    'LapNumber': [lap_number],
    'TyreLife': [tyre_life],
    'Stint': [stint],
    'SpeedI1': [speed_i1],
    'SpeedI2': [speed_i2],
    'SpeedFL': [speed_fl],
    'LapProgress': [lap_progress],
    'StintLap': [stint_lap],
    'StintProgress': [stint_progress],
    'Circuit_' + circuit: [1],
    'Compound_' + compound: [1],
    'TyrePhase_' + tyre_phase: [1]
}

df_input = pd.DataFrame(0, index=[0], columns=model_columns)  # Ensure all model columns present
for col, val in input_dict.items():
    if col in df_input.columns:
        df_input[col] = val

# ----------------------------------------------------------
#  Apply preprocessing (imputer + scaler)
# ----------------------------------------------------------
df_imputed = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)
df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_input.columns)

# ----------------------------------------------------------
#  Make Prediction
# ----------------------------------------------------------
predicted_time = model.predict(df_scaled)[0]

# ----------------------------------------------------------
#  Display Result
# ----------------------------------------------------------
st.markdown("""
<div class="result-card">
    <h2>Predicted Lap Time</h2>
    <p class="result-value">{:.3f} seconds</p>
</div>
""".format(predicted_time), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
