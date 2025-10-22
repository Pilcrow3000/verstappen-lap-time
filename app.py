import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="F1 Lap Time Predictor",
    page_icon="🏎️",
    layout="wide"
)

# -------------------- Navigation Bar with F1 Logo --------------------
st.markdown(
    """
    <style>
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #0e0e0e;
        padding: 0.6rem 2rem;
        border-bottom: 1px solid #d90429;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .navbar img {
        height: 40px;
    }
    .navbar-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f5f5f5;
        margin-left: 10px;
        letter-spacing: 0.5px;
    }
    .navbar-sub {
        font-size: 0.9rem;
        color: #aaa;
        margin-left: 10px;
    }
    .nav-container {
        display: flex;
        align-items: center;
    }
    </style>

    <div class="navbar">
        <div class="nav-container">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg">
            <div>
                <div class="navbar-title">Lap Time Predictor</div>
                <div class="navbar-sub">Max Verstappen • 2023 Data • Machine Learning</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------- Custom Styling --------------------
st.markdown("""
    <style>
    body {
        background-color: #0e0e0e;
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stButton>button {
        background-color: #d90429;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #ef233c;
        color: white;
        border: 1px solid #ffffff20;
    }
    .prediction-box {
        background-color: #1a1a1a;
        border: 1px solid #d90429;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #aaa;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Load Model & Tools --------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

# Try loading feature names from scaler (or use manual fallback)
expected_features = getattr(scaler, "feature_names_in_", None)
if expected_features is None:
    # Manual fallback if feature names not saved
    try:
        expected_features = joblib.load("feature_names.pkl")
    except:
        expected_features = None

# -------------------- Header --------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/33/F1.svg", width=90)
with col2:
    st.title("F1 Lap Time Predictor")
    st.caption("Max Verstappen | 2023 Data | Machine Learning Regression")

st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.header("Input Parameters")
st.sidebar.markdown("### Configure your lap scenario")

circuit = st.sidebar.selectbox(
    "Circuit",
    ["Italian Grand Prix", "Azerbaijan Grand Prix", "British Grand Prix", "Belgian Grand Prix"]
)
compound = st.sidebar.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
tyre_life = st.sidebar.slider("Tyre Life (laps)", 0, 60, 12)
stint = st.sidebar.number_input("Stint Number", min_value=1, max_value=5, value=1)
lap_number = st.sidebar.number_input("Lap Number", min_value=1, max_value=80, value=30)
speed_i1 = st.sidebar.slider("Speed at I1 (km/h)", 200, 350, 305)
speed_i2 = st.sidebar.slider("Speed at I2 (km/h)", 200, 350, 310)
speed_fl = st.sidebar.slider("Speed at Finish Line (km/h)", 200, 350, 315)

# Derived metrics
lap_progress = lap_number / 80
stint_lap = lap_number % 20
stint_progress = stint_lap / 20
tyre_phase = (
    "Fresh" if tyre_life < 5 else
    "Mid" if tyre_life < 15 else
    "Worn" if tyre_life < 30 else
    "VeryWorn"
)

# -------------------- Input Assembly --------------------
input_dict = {
    "LapNumber": [lap_number],
    "TyreLife": [tyre_life],
    "SpeedI1": [speed_i1],
    "SpeedI2": [speed_i2],
    "SpeedFL": [speed_fl],
    "LapProgress": [lap_progress],
    "StintLap": [stint_lap],
    "StintProgress": [stint_progress],
}

# Encode one-hots manually
for c in ["Italian Grand Prix", "Azerbaijan Grand Prix", "British Grand Prix", "Belgian Grand Prix"]:
    input_dict[f"Circuit_{c}"] = [1 if circuit == c else 0]
for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
    input_dict[f"Compound_{c}"] = [1 if compound == c else 0]
for phase in ["Fresh", "Mid", "Worn", "VeryWorn"]:
    input_dict[f"TyrePhase_{phase}"] = [1 if tyre_phase == phase else 0]

df_input = pd.DataFrame(input_dict)

# -------------------- Align Columns --------------------
if expected_features is not None:
    # Add missing columns as 0
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
    # Reorder to match training
    df_input = df_input.reindex(columns=expected_features)
else:
    st.warning("⚠️ Warning: Feature names not found. Using current input columns order.")

# -------------------- Apply Imputer & Scaler --------------------
df_imputed = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)
df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_input.columns)

# -------------------- Layout: Inputs + Prediction --------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Lap Context Summary")
    st.dataframe(df_input.style.highlight_max(axis=0, color="#d9042915"))

with col_right:
    st.subheader("Prediction Output")

    if st.button("🏁 Predict Lap Time"):
        prediction = model.predict(df_scaled)[0]
        st.markdown(f"""
        <div class="prediction-box">
            <div class="metric-label">Estimated Lap Time</div>
            <div class="metric-value">{prediction:.2f} s</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Prediction based on 2023 race data and trained regression model.")

# -------------------- Example Preset --------------------
with st.expander("💡 Try Example Lap"):
    st.markdown("""
    **Example:**  
    - Circuit: Baku (Azerbaijan Grand Prix)  
    - Tyre: SOFT, Tyre Life = 8 laps  
    - Speeds ~305 / 310 / 320 km/h  
    → Expected Lap Time ≈ 96–98 seconds
    """)
