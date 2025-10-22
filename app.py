import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from PIL import Image

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="🏎️ Verstappen Lap Time Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================
# LOAD ASSETS
# ==============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
    with open("models/model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    return model, scaler, imputer, model_columns

model, scaler, imputer, model_columns = load_artifacts()

# ==============================================
# STYLING
# ==============================================
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #ff1801;
        text-align: left;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ff1801;
    }
    .subheader {
        font-size: 18px;
        color: #333;
        margin-top: -10px;
        padding-bottom: 10px;
    }
    .stApp {
        background-color: #f9f9f9;
    }
    .css-18e3th9 {
        padding-top: 4rem !important;  /* Fix Streamlit toolbar overlap */
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# HEADER
# ==============================================
col1, col2 = st.columns([0.08, 0.92])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/33/F1.svg", width=80)
with col2:
    st.markdown("<div class='main-title'>Verstappen Lap Time Predictor (2023)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Predict lap time based on tyre, stint, circuit, and speed data</div>", unsafe_allow_html=True)

# ==============================================
# CIRCUIT CONFIGURATION
# ==============================================
st.sidebar.header("🏁 Circuit Selection")

circuit_length_map = {
    "Italian Grand Prix": 5793,
    "Azerbaijan Grand Prix": 6003,
    "British Grand Prix": 5891,
    "Belgian Grand Prix": 7004,
}

circuit = st.sidebar.selectbox("Select Circuit", list(circuit_length_map.keys()))
circuit_length = circuit_length_map[circuit]

# ==============================================
# LAP & TYRE CONFIGURATION
# ==============================================
st.sidebar.header("⚙️ Lap Setup")

compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
tyre_life = st.sidebar.slider("Tyre Life (laps)", 0, 60, 10)
stint = st.sidebar.selectbox("Stint Number", [1, 2, 3, 4])
lap_number = st.sidebar.slider("Lap Number", 1, 60, 10)

st.sidebar.header("💨 Speed Inputs")
speed_i1 = st.sidebar.slider("Speed at I1 (km/h)", 200, 340, 300)
speed_i2 = st.sidebar.slider("Speed at I2 (km/h)", 200, 340, 310)
speed_fl = st.sidebar.slider("Speed at Finish Line (km/h)", 200, 340, 320)

# ==============================================
# FEATURE PREPARATION
# ==============================================
lap_progress = lap_number / 60
stint_lap = lap_number if stint == 1 else lap_number % 20
stint_progress = stint_lap / 20

tyre_phase = "Fresh" if tyre_life <= 5 else "Mid" if tyre_life <= 15 else "Worn" if tyre_life <= 30 else "VeryWorn"

# Construct input row
input_data = {
    "LapNumber": lap_number,
    "TyreLife": tyre_life,
    "Stint": stint,
    "SpeedI1": speed_i1,
    "SpeedI2": speed_i2,
    "SpeedFL": speed_fl,
    "LapProgress": lap_progress,
    "StintLap": stint_lap,
    "StintProgress": stint_progress,
    "CircuitLength": circuit_length,
    "Circuit_" + circuit: 1,
    "Compound_" + compound: 1,
    "TyrePhase_" + tyre_phase: 1,
}

# Create DataFrame from model columns
df_input = pd.DataFrame(columns=model_columns)
for col in df_input.columns:
    df_input[col] = 0

for key, val in input_data.items():
    if key in df_input.columns:
        df_input[key] = val

# ==============================================
# PREDICTION
# ==============================================
if st.button("🔮 Predict Lap Time"):
    df_imputed = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

    prediction = model.predict(df_scaled)[0]
    pred_min, pred_sec = divmod(prediction, 60)

    st.markdown("## 🏎️ Predicted Lap Time")
    c1, c2 = st.columns([0.4, 0.6])
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color:#ff1801'>{pred_min:.0f} min {pred_sec:.3f} s</h3>
            <p style='color:#555;'>Predicted lap time</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>{circuit}</h3>
            <p style='color:#555;'>Circuit Length: <b>{circuit_length} m</b></p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================
# FOOTER
# ==============================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Built by the <b>Media Intelligence Team</b> | Powered by Streamlit & Scikit-Learn</p>",
    unsafe_allow_html=True
)
