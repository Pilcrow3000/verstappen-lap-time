import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# --------------------------
# 🎨 PAGE CONFIGURATION
# --------------------------
st.set_page_config(page_title="F1 Lap Time Predictor", page_icon="🏎️", layout="centered")

# --------------------------
# 🏎️ LOAD LOCAL LOGO
# --------------------------
# Make sure your logo is saved as: deploy/f1_logo.png
with open("deploy/f1_logo.png", "rb") as f:
    f1_logo_base64 = base64.b64encode(f.read()).decode()

# --------------------------
# 💅 CUSTOM DARK THEME STYLING
# --------------------------
st.markdown(f"""
    <style>
        body {{
            background-color: #0d0d0d;
            color: #f2f2f2;
            font-family: 'Inter', sans-serif;
        }}
        .navbar {{
            background-color: #1a1a1a;
            padding: 1rem 2rem 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            border-bottom: 1px solid #333;
            margin-bottom: 2rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            position: sticky;
            top: 0;
            z-index: 999;
        }}
        .navbar img {{
            height: 42px;
            margin-right: 14px;
        }}
        .title-text {{
            font-size: 1.7rem;
            font-weight: 600;
            color: #f1f1f1;
            letter-spacing: 0.5px;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #e10600, #ff1e00);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 1.4rem;
            font-weight: 600;
            transition: 0.2s ease-in-out;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #ff2a00, #ff4b00);
            transform: translateY(-2px);
        }}
    </style>

    <div class="navbar">
        <img src="data:image/png;base64,{f1_logo_base64}" alt="F1 Logo">
        <span class="title-text">F1 2023 Lap Time Predictor</span>
    </div>
""", unsafe_allow_html=True)

# --------------------------
# 📦 LOAD MODEL ARTIFACTS
# --------------------------
model = joblib.load("deploy/model.pkl")
scaler = joblib.load("deploy/scaler.pkl")
imputer = joblib.load("deploy/imputer.pkl")
model_columns = joblib.load("deploy/model_columns.pkl")

st.write("### Predict Verstappen’s lap time based on circuit, tyre, stint, and speed data")

# --------------------------
# 🏁 USER INPUTS
# --------------------------
circuits = ['Italian Grand Prix', 'Azerbaijan Grand Prix', 'British Grand Prix', 'Belgian Grand Prix']
compounds = ['SOFT', 'MEDIUM', 'HARD']
stints = [1, 2, 3, 4, 5]

col1, col2 = st.columns(2)
with col1:
    circuit = st.selectbox("🏟️ Circuit", circuits)
    compound = st.selectbox("🛞 Tyre Compound", compounds)
    stint = st.selectbox("🔁 Stint Number", stints)
    tyre_life = st.slider("Tyre Age (laps)", 0, 35, 5)
with col2:
    lap_number = st.slider("Lap Number", 1, 55, 10)
    speed_i1 = st.slider("Speed at Sector 1 (km/h)", 150, 350, 250)
    speed_i2 = st.slider("Speed at Sector 2 (km/h)", 150, 350, 250)
    speed_fl = st.slider("Finish Line Speed (km/h)", 150, 350, 250)

# --------------------------
# 🧮 BUILD INPUT
# --------------------------
track_attributes = {
    'Italian Grand Prix':   {'Length': 5.79, 'Corners': 11, 'Elevation': 2,  'SpeedProfile': 1.10},
    'Azerbaijan Grand Prix':{'Length': 6.00, 'Corners': 20, 'Elevation': 4,  'SpeedProfile': 0.90},
    'British Grand Prix':   {'Length': 5.89, 'Corners': 18, 'Elevation': 11, 'SpeedProfile': 1.00},
    'Belgian Grand Prix':   {'Length': 7.00, 'Corners': 19, 'Elevation': 40, 'SpeedProfile': 0.95},
}

track = track_attributes[circuit]
lap_progress = lap_number / 55
stint_lap = lap_number % 15 if lap_number > 15 else lap_number
stint_progress = stint_lap / 15
stint_phase = "Early" if stint_progress <= 0.33 else ("Mid" if stint_progress <= 0.66 else "Late")

tyre_wear_factor = np.log1p(tyre_life)
avg_speed = np.mean([speed_i1, speed_i2, speed_fl])
speed_std = np.std([speed_i1, speed_i2, speed_fl])
relative_speed = avg_speed / track["Length"]
compound_factor = {"SOFT": 0.98, "MEDIUM": 1.00, "HARD": 1.03}[compound]
laps_since_pit = min(tyre_life + 1, 20)
relative_pit_progress = laps_since_pit / 20

# Construct DataFrame
input_dict = {
    'CircuitLength': track["Length"],
    'CornerDensity': track["Corners"] / track["Length"],
    'ElevationFactor': track["Elevation"] / track["Length"],
    'SpeedProfile': track["SpeedProfile"],
    'LapNumber': lap_number,
    'LapProgress': lap_progress,
    'Stint': stint,
    'StintLap': stint_lap,
    'StintProgress': stint_progress,
    'TyreLife': tyre_life,
    'TyreWearFactor': tyre_wear_factor,
    'CompoundFactor': compound_factor,
    'SpeedI1': speed_i1,
    'SpeedI2': speed_i2,
    'SpeedFL': speed_fl,
    'AvgSpeed': avg_speed,
    'SpeedStd': speed_std,
    'RelativeSpeed': relative_speed,
    'LapsSincePit': laps_since_pit,
    'RelativePitProgress': relative_pit_progress,
}

for phase in ['Early', 'Mid', 'Late']:
    input_dict[f'StintPhase_{phase}'] = 1 if stint_phase == phase else 0
for phase in ['Fresh', 'Mid', 'Worn', 'VeryWorn']:
    input_dict[f'TyrePhase_{phase}'] = 1 if (
        (phase == 'Fresh' and tyre_life <= 5) or
        (phase == 'Mid' and 5 < tyre_life <= 15) or
        (phase == 'Worn' and 15 < tyre_life <= 30) or
        (phase == 'VeryWorn' and tyre_life > 30)
    ) else 0
for c in circuits[1:]:
    input_dict[f'Circuit_{c}'] = 1 if circuit == c else 0
for comp in compounds[1:]:
    input_dict[f'Compound_{comp}'] = 1 if compound == comp else 0

df_input = pd.DataFrame([input_dict])
df_input = df_input.reindex(columns=model_columns, fill_value=0)

# Impute and scale
df_imputed = pd.DataFrame(imputer.transform(df_input), columns=model_columns)
df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=model_columns)

# --------------------------
# 🚀 PREDICTION
# --------------------------
if st.button("🏁 Predict Lap Time"):
    pred_time = model.predict(df_scaled)[0]
    mins, secs = divmod(pred_time, 60)
    st.markdown(f"### 🕒 Predicted Lap Time: **{int(mins)}m {secs:.3f}s**")
    st.caption(f"Based on {circuit}, {compound} tyres, stint {stint}, avg speed {avg_speed:.1f} km/h.")

# --------------------------
# 📊 FOOTER
# --------------------------
st.markdown("""
<hr style='border: 0.5px solid #333; margin-top: 2rem;'>
<div style='text-align: center; color: #777; font-size: 0.9rem;'>
Developed by <b>Media Intelligence Team</b> | Data: FastF1, FIA Circuits 2023
</div>
""", unsafe_allow_html=True)
