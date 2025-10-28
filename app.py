import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="F1 Race Pace Prediction - Max Verstappen",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    .stApp { background-color: #15151E; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #1E1E2E; border-right: 2px solid #E10600; }
    [data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: 700; color: #E10600; }
    [data-testid="stMetricLabel"] { font-size: 1rem; color: #FFFFFF; text-transform: uppercase; letter-spacing: 1px; }
    h1 { color: #E10600; font-weight: 800; border-bottom: 3px solid #E10600; padding-bottom: 10px; letter-spacing: 1px; }
    h2 { color: #00D2BE; font-weight: 700; margin-top: 2rem; }
    h3 { color: #FFFFFF; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; background-color: #1E1E2E; border-radius: 8px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { color: #FFFFFF; background-color: #38383F; border-radius: 4px; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #E10600; color: #FFFFFF; }
    .stButton>button { background-color: #E10600; color: #FFFFFF; border: none; padding: 10px 25px; font-weight: 700; border-radius: 4px; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s; }
    .stButton>button:hover { background-color: #C10500; box-shadow: 0 4px 12px rgba(225,6,0,0.4); }
    .racing-stripe { height: 3px; background: #E10600; margin: 20px 0; box-shadow: 0 2px 4px rgba(225,6,0,0.3); }
    .stat-card { background-color: #1E1E2E; padding: 20px; border-radius: 8px; border-left: 4px solid #E10600; margin: 10px 0; }
    .stButton>button { background-color: #1E1E2E; color: #FFFFFF; border: 1px solid rgba(94,82,64,0.3); margin-bottom: 8px; text-align: left; padding-left: 16px; }
    .stButton>button:hover { background-color: #E10600; border-color: #E10600; color: #FFFFFF; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Constants and helpers
# ----------------------------

AVG_R2 = 0.9065
AVG_MAE = 0.1388  # seconds

def normalize_model_name_for_distribution(name: str) -> str:
    """Group XGBoost under GradientBoosting for family chart; drop ' (Tuned)'."""
    clean = name.replace(' (Tuned)', '').strip()
    if 'XGBoost' in clean:
        clean = 'GradientBoosting'
    return clean

def get_stacking_components_from_pkl_2025():
    """
    Try to load exact top-5 unique base models used in 2025 stacking.
    Looks for list keys or infers from StackingRegressor.estimators_/estimators.
    Returns list[str] or None.
    """
    try:
        import joblib
        d = joblib.load('./models/model_2025.pkl')
        for key in ['stack_selected_models', 'top_models', 'selected_models']:
            if key in d and isinstance(d[key], list) and len(d[key]) >= 3:
                return d[key][:5]
        mdl = d.get('model', None)
        if mdl is not None:
            ests = getattr(mdl, 'estimators_', None) or getattr(mdl, 'estimators', None)
            names = []
            if ests:
                for e in ests:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        names.append(e[0])
                    else:
                        names.append(type(e).__name__)
                if len(names) >= 3:
                    return names[:5]
        return None
    except Exception:
        return None

STACKING_2025_COMPONENTS_FALLBACK = [
    "ExtraTrees",
    "RandomForest (Tuned)",
    "Ridge",
    "CatBoost (Tuned)",
    "GradientBoosting",
]

def render_stacking_components_2025():
    exact = get_stacking_components_from_pkl_2025()
    if exact and isinstance(exact, list):
        items = exact
        heading = "Exact 2025 Stacking Ensemble (Top‑5 unique)"
        note = "Meta‑learner: Ridge(alpha=1.0), 3‑fold CV"
    else:
        items = STACKING_2025_COMPONENTS_FALLBACK
        heading = "2025 Stacking Ensemble (Top‑5 unique)"
        note = "Exact list not stored; showing representative set. Meta‑learner: Ridge(alpha=1.0)."
    st.markdown(f"### {heading}")
    st.markdown(
        "<div class='stat-card'><ul>" +
        "".join([f"<li><strong>{m}</strong></li>" for m in items]) +
        f"</ul><p style='opacity:0.8;'>{note}</p></div>",
        unsafe_allow_html=True
    )

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🏎️ F1 RACE PACE")
st.sidebar.markdown("### Max Verstappen")
st.sidebar.markdown("**Monza 2022-2025**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Stats**")
st.sidebar.metric("Years Analyzed", "4")
st.sidebar.metric("Total Laps", "189")
st.sidebar.metric("Avg R²", f"{AVG_R2:.4f}")
st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation")
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"
pages = ["🏠 Home", "📊 Multi-Year Analysis", "🔍 Year Deep Dive", "🔮 Live Prediction", "⚙️ Methodology", "👤 About"]
for page_name in pages:
    if st.sidebar.button(page_name, use_container_width=True, key=page_name):
        st.session_state.page = page_name
        st.rerun()
page = st.session_state.page

# ============================================================================
# HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1:
        st.title("F1 RACE PACE PREDICTION")
        st.markdown("### Physics-Informed Machine Learning for Monza")
        st.markdown("""
        Predicting **Max Verstappen's race lap times** using practice session data from Friday/Saturday 
        to forecast Sunday performance. Validated across **4 consecutive years (2022–2025)** with 
        **~90.65% average accuracy (R²)**.
        """)
    with c2:
        st.metric("Driver", "Max Verstappen 🇳🇱", delta="#1")
        st.metric("Status", "Production Ready", delta="4 Years Validated")

    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    st.markdown("## 🏆 Key Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Average R²", f"{AVG_R2:.4f}", delta=f"{AVG_R2*100:.2f}% Accuracy")
    with c2:
        st.metric("Best Year", "2025", delta="R² = 0.9797")
    with c3:
        st.metric("Average MAE", f"{AVG_MAE:.4f}s", delta="Per Lap Error", delta_color="inverse")
    with c4:
        st.metric("Success Rate", "100%", delta="4/4 Years ≥ 0.80")

    st.markdown("---")

    try:
        summary_df = pd.read_csv('./data/summary_results.csv')
        st.markdown("## 📈 Year-by-Year Summary")
        df = summary_df[['year','model_name','test_r2','mae','dataset_size']].copy()
        df.columns = ['Year','Model','Test R²','MAE (s)','Laps']
        df['Test R²'] = df['Test R²'].round(4)
        df['MAE (s)'] = df['MAE (s)'].round(4)
        df['Year'] = df['Year'].astype(int)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Summary data not found: {e}")

    st.markdown("---")

    L1, L2 = st.columns(2)
    with L1:
        st.markdown("""
        <div class="stat-card">
        <h3>🎯 What We Predict</h3>
        <ul>
            <li>Max Verstappen's race lap times</li>
            <li>12 physics-informed features</li>
            <li>Tire degradation modeling</li>
            <li>Fuel load dynamics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with L2:
        st.markdown("""
        <div class="stat-card">
        <h3>🚀 Technology Stack</h3>
        <ul>
            <li>FastF1 API for telemetry</li>
            <li>Ensemble ML (CatBoost, GradientBoosting, Stacking)</li>
            <li>Hyperparameter tuning via RandomizedSearchCV</li>
            <li>Production filtering & composite scoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MULTI-YEAR ANALYSIS
# ============================================================================
elif page == "📊 Multi-Year Analysis":
    st.title("📊 Multi-Year Comparison")
    st.markdown("Analyzing Max Verstappen's consistency and trends across 2022–2025 Monza races")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    try:
        with open('./data/cross_year_data.json','r') as f:
            cross_data = json.load(f)

        tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "🔧 Model Selection", "🌡️ Physics Features"])

        with tab1:
            st.markdown("### R² Score Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cross_data['years'],
                y=cross_data['r2_scores'],
                mode='lines+markers',
                name='Test R²',
                line=dict(color='#E10600', width=3),
                marker=dict(size=12, color='#E10600')
            ))
            fig.add_hline(y=0.80, line_dash="dash", line_color="#00D2BE", annotation_text="Target (0.80)")
            fig.update_layout(
                title="Model Accuracy Over Time",
                xaxis_title="Year",
                yaxis_title="R² Score",
                template="plotly_dark",
                height=400,
                hovermode='x unified',
                xaxis=dict(
                    tickmode='array',
                    tickvals=cross_data['years'],
                    ticktext=[str(y) for y in cross_data['years']]
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### MAE Comparison")
                fig2 = go.Figure(data=[go.Bar(
                    x=cross_data['years'],
                    y=cross_data['mae_scores'],
                    marker_color='#00D2BE',
                    text=[f"{mae:.3f}s" for mae in cross_data['mae_scores']],
                    textposition='outside'
                )])
                fig2.update_layout(
                    title="Mean Absolute Error by Year",
                    xaxis_title="Year",
                    yaxis_title="MAE (seconds)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(t=60,b=40,l=40,r=40),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=cross_data['years'],
                        ticktext=[str(y) for y in cross_data['years']]
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)

            with c2:
                st.markdown("### Dataset Size")
                fig3 = go.Figure(data=[go.Bar(
                    x=cross_data['years'],
                    y=cross_data['dataset_sizes'],
                    marker_color='#E10600',
                    text=cross_data['dataset_sizes'],
                    textposition='outside'
                )])
                fig3.update_layout(
                    title="Race Laps Analyzed",
                    xaxis_title="Year",
                    yaxis_title="Number of Laps",
                    template="plotly_dark",
                    height=400,
                    margin=dict(t=60,b=40,l=40,r=40),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=cross_data['years'],
                        ticktext=[str(y) for y in cross_data['years']]
                    )
                )
                st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            st.markdown("### Model Selection Pattern")
            model_counts = {}
            for model in cross_data['model_names']:
                clean = normalize_model_name_for_distribution(model)
                model_counts[clean] = model_counts.get(clean, 0) + 1
            fig4 = go.Figure(data=[go.Pie(
                labels=list(model_counts.keys()),
                values=list(model_counts.values()),
                hole=0.4,
                marker_colors=['#E10600','#00D2BE','#FFD700','#38383F']
            )])
            fig4.update_layout(title="Model Distribution (2022–2025)", template="plotly_dark", height=400)
            st.plotly_chart(fig4, use_container_width=True)

            # Show top‑5 unique stack components for 2025
            render_stacking_components_2025()

        with tab3:
            st.markdown("### FP3 Baseline Pace Evolution")
            years = [int(y) for y in cross_data['years']]
            paces = cross_data['baseline_paces']

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=years,
                y=paces,
                mode='lines+markers',
                name='FP3 Baseline',
                line=dict(color='#00D2BE', width=3),
                marker=dict(size=12),
                fill='tozeroy',
                fillcolor='rgba(0,210,190,0.1)'
            ))

            xmin, xmax = min(years), max(years)
            fig5.update_layout(
                title="Fastest Practice Lap Time Trend",
                xaxis_title="Year",
                yaxis_title="Lap Time (seconds)",
                template="plotly_dark",
                height=400,
                yaxis=dict(range=[75, 85]),
                xaxis=dict(
                    type='linear',
                    tickmode='array',
                    tickvals=years,                  # Only exact years as ticks
                    ticktext=[str(y) for y in years],
                    range=[xmin - 1, xmax + 1]       # ±1 year padding
                )
            )

            st.plotly_chart(fig5, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                best_idx = paces.index(min(paces))
                st.metric("Fastest Baseline", f"{min(paces):.3f}s", delta=f"{years[best_idx]}")
            with c2:
                improvement = max(paces) - min(paces)
                st.metric("4-Year Improvement", f"{improvement:.3f}s", delta="Faster")

    except Exception as e:
        st.error(f"Error loading multi-year data: {e}")
        st.info("Make sure `data/cross_year_data.json` exists")

# ============================================================================
# YEAR DEEP DIVE
# ============================================================================
elif page == "🔍 Year Deep Dive":
    st.title("🔍 Year-Specific Deep Dive")
    st.markdown("Detailed analysis of Max Verstappen's performance by year")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    selected_year = st.radio("Select Year", [2022, 2023, 2024, 2025], horizontal=True)

    try:
        summary_df = pd.read_csv('./data/summary_results.csv')
        year_data = summary_df[summary_df['year'] == selected_year].iloc[0]

        with open(f'./results/{selected_year}/predictions.json','r') as f:
            predictions = json.load(f)
        with open(f'./results/{selected_year}/feature_importance.json','r') as f:
            features = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Model", year_data['model_name'])
        with c2: st.metric("Test R²", f"{year_data['test_r2']:.4f}")
        with c3: st.metric("MAE", f"{year_data['mae']:.4f}s")
        with c4: st.metric("Laps", int(year_data['dataset_size']))

        st.markdown("---")

        if selected_year == 2025:
            render_stacking_components_2025()
            st.markdown("---")

        st.markdown("### Actual vs Predicted Lap Times")
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=predictions['actual'], y=predictions['predicted'],
            mode='markers', name='Predictions',
            marker=dict(size=8, color='#E10600', opacity=0.6)
        ))
        min_val = min(min(predictions['actual']), min(predictions['predicted']))
        max_val = max(max(predictions['actual']), max(predictions['predicted']))
        fig6.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='#00D2BE', dash='dash')
        ))
        fig6.update_layout(
            title=f"{selected_year} Prediction Accuracy",
            xaxis_title="Actual Lap Time (s)",
            yaxis_title="Predicted Lap Time (s)",
            template="plotly_dark", height=500
        )
        st.plotly_chart(fig6, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Prediction Error Distribution")
            fig7 = go.Figure(data=[go.Histogram(
                x=predictions['residuals'], nbinsx=20,
                marker_color='#00D2BE', opacity=0.7
            )])
            fig7.update_layout(
                title="Error Distribution",
                xaxis_title="Residual (s)", yaxis_title="Frequency",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig7, use_container_width=True)
        with c2:
            st.markdown("### Feature Importance")
            top_n = 8
            top_features = sorted(zip(features['features'], features['importance']), key=lambda x: x[1], reverse=True)[:top_n]
            fig8 = go.Figure(data=[go.Bar(
                y=[f[0] for f in top_features],
                x=[f[1] for f in top_features],
                orientation='h', marker_color='#E10600',
                text=[f"{f[1]:.3f}" for f in top_features],
                textposition='outside'
            )])
            fig8.update_layout(
                title="Top Features",
                xaxis_title="Importance", yaxis_title="Feature",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig8, use_container_width=True)

    except FileNotFoundError:
        st.error(f"❌ Data for {selected_year} not found!")
        st.info(f"Make sure `results/{selected_year}/` contains predictions.json and feature_importance.json")
    except Exception as e:
        st.error(f"❌ Error loading {selected_year} data: {e}")
        st.info("Check that the files are valid and all features are provided")

# ============================================================================
# LIVE PREDICTION
# ============================================================================
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction Engine")
    st.markdown("Test the model by entering custom feature values")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    preset_values = {}
    if 'preset' in st.session_state:
        if st.session_state.preset == "start":
            preset_values = {'lap': 1, 'tire_age': 0, 'stint': 1, 'fuel': 110.0, 'compound': 0}
        elif st.session_state.preset == "mid":
            preset_values = {'lap': 25, 'tire_age': 10, 'stint': 1, 'fuel': 70.0, 'compound': 0}
        elif st.session_state.preset == "end":
            preset_values = {'lap': 53, 'tire_age': 35, 'stint': 2, 'fuel': 15.0, 'compound': 1}
        del st.session_state.preset

    selected_year = st.radio("Select Model Year", [2022, 2023, 2024, 2025], horizontal=True)

    try:
        import joblib
        model_data = joblib.load(f'./models/model_{selected_year}.pkl')
        model = model_data['model']
        model_name = model_data.get('model_name', model_data.get('name', 'Unknown'))
        st.success(f"✅ Model {selected_year} loaded successfully! ({model_name})")

        if selected_year == 2025:
            render_stacking_components_2025()

        summary_df = pd.read_csv('./data/summary_results.csv')
        year_info = summary_df[summary_df['year'] == selected_year].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Model", year_info['model_name'])
        with c2: st.metric("Test R²", f"{year_info['test_r2']:.4f}")
        with c3: st.metric("Expected Error", f"±{year_info['mae']:.4f}s")

        st.markdown("---")
        st.markdown("## 🎛️ Input Features")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Practice Session Features")
            fp3_baseline = st.number_input("FP3 Baseline (seconds)", min_value=75.0, max_value=85.0, value=79.5, step=0.1)
            medium_deg = st.number_input("Medium Tire Degradation (s/lap)", min_value=-5.0, max_value=1.0, value=-0.5, step=0.1)
            hard_deg = st.number_input("Hard Tire Degradation (s/lap)", min_value=-1.0, max_value=2.0, value=0.06, step=0.01)
            track_temp = st.slider("Race Track Temperature (°C)", min_value=20.0, max_value=60.0, value=43.0, step=0.5)
            air_temp = st.slider("Race Air Temperature (°C)", min_value=15.0, max_value=40.0, value=26.0, step=0.5)
        with c2:
            st.markdown("### Lap Context Features")
            lap_number = st.number_input("Lap Number", min_value=1, max_value=53, value=preset_values.get('lap', 25), step=1)
            tire_age = st.number_input("Tire Age (laps)", min_value=0, max_value=40, value=preset_values.get('tire_age', 10), step=1)
            stint_number = st.selectbox("Stint Number", [1, 2, 3], index=preset_values.get('stint', 1) - 1)
            fuel_remaining = st.slider("Fuel Remaining (kg)", min_value=0.0, max_value=110.0, value=preset_values.get('fuel', 70.0), step=1.0)
            compound = st.selectbox("Tire Compound", ["MEDIUM", "HARD"], index=preset_values.get('compound', 0))

        st.markdown("---")
        st.markdown("### 📝 Quick Presets")
        st.caption("Click to populate lap context fields with typical scenarios")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🏁 Race Start (Lap 1)", use_container_width=True):
                st.session_state.preset = "start"; st.rerun()
        with b2:
            if st.button("⏱️ Mid-Race (Lap 25)", use_container_width=True):
                st.session_state.preset = "mid"; st.rerun()
        with b3:
            if st.button("🏆 Final Lap (Lap 53)", use_container_width=True):
                st.session_state.preset = "end"; st.rerun()

        st.markdown("---")
        if st.button("🏎️ PREDICT LAP TIME", use_container_width=True):
            lap_progress = lap_number / 53.0
            tire_age_squared = tire_age ** 2
            compound_encoded = 0 if compound == "MEDIUM" else 1
            input_features = [[
                fp3_baseline, medium_deg, hard_deg, track_temp, air_temp,
                lap_number, tire_age, tire_age_squared, stint_number,
                fuel_remaining, lap_progress, compound_encoded
            ]]
            prediction = model.predict(input_features)[0]

            st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
            a1, a2 = st.columns([1,1])
            with a1:
                st.markdown("### 🏁 Predicted Lap Time")
                st.markdown(f"<h1 style='color: #E10600;'>{prediction:.3f}s</h1>", unsafe_allow_html=True)
            with a2:
                diff = prediction - fp3_baseline
                diff_text = f"+{diff:.3f}s" if diff > 0 else f"{diff:.3f}s"
                st.markdown("### ⏱️ Delta vs FP3")
                st.markdown(f"<h2 style='color: #00D2BE;'>{diff_text}</h2>", unsafe_allow_html=True)

            st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Prediction Breakdown")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                <h4>Input Summary</h4>
                <ul>
                    <li><strong>Base Pace:</strong> {fp3_baseline:.3f}s (FP3)</li>
                    <li><strong>Tire Condition:</strong> {tire_age} laps old</li>
                    <li><strong>Fuel Load:</strong> {fuel_remaining:.1f}kg</li>
                    <li><strong>Track Temp:</strong> {track_temp:.1f}°C</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="stat-card">
                <h4>Factors Affecting Time</h4>
                <ul>
                    <li><strong>Fuel:</strong> Lighter = Faster</li>
                    <li><strong>Tires:</strong> Older = Slower</li>
                    <li><strong>Temperature:</strong> Affects grip</li>
                    <li><strong>Lap Position:</strong> Race pace vs sprint</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"❌ Model for {selected_year} not found!")
        st.info("Make sure model files are in the `models/` folder")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Check that the model file is valid and all features are provided")

# ============================================================================
# METHODOLOGY
# ============================================================================
elif page == "⚙️ Methodology":
    st.title("⚙️ Methodology")
    st.markdown("How the prediction system works")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    st.markdown("## 🎯 Feature Engineering")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="stat-card">
        <h3>Practice Session Features (5)</h3>
        <ul>
            <li><strong>FP3 Baseline</strong> — Fastest practice lap</li>
            <li><strong>Medium Degradation</strong> — Rate from FP2</li>
            <li><strong>Hard Degradation</strong> — Rate from FP2</li>
            <li><strong>Track Temperature</strong> — Race conditions</li>
            <li><strong>Air Temperature</strong> — Race conditions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
        <h3>Lap Context Features (7)</h3>
        <ul>
            <li><strong>Lap Number</strong> — Position in race</li>
            <li><strong>Tire Age</strong> — Laps on current tires</li>
            <li><strong>Tire Age²</strong> — Non-linear degradation</li>
            <li><strong>Stint Number</strong> — Which tire stint</li>
            <li><strong>Fuel Remaining</strong> — Dynamic weight</li>
            <li><strong>Lap Progress</strong> — Fraction of race</li>
            <li><strong>Compound</strong> — MEDIUM or HARD</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🤖 Model Pipeline")
    st.markdown("""
    - 11 models tested per year; tuning via RandomizedSearchCV (≈30 iters, 3-fold CV)
    - Composite scoring: Test R² − (0.3 × Overfit Gap)
    - Stacking: Top-5 unique models (tuned preferred) as base learners; meta-learner Ridge(alpha=1.0), 3-fold CV
    - Production filtering: DecisionTree excluded; ensembles preferred
    - Validation: 70/30 split, 3-fold CV, cross-year reproducibility (2022–2025)
    """)

    st.markdown("---")
    st.markdown("## ✨ Why This Works")
    st.markdown("""
    <div class="stat-card">
    <h3>Physics-Informed Approach</h3>
    <p>Domain features (fuel, tires, temperature) guide learning toward physically meaningful patterns.</p>
    <ul>
        <li><strong>Generalization</strong> — Works across seasons and conditions</li>
        <li><strong>Data efficiency</strong> — 40–50 laps/year</li>
        <li><strong>Interpretability</strong> — Feature importance maps to F1 strategy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ABOUT
# ============================================================================
elif page == "👤 About":
    st.title("👤 About This Project")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("""
        ## 🎓 Academic Project
        **Institution**: RCOEM, Nagpur  
        **Branch**: CSE (AI/ML)  
        **Year**: 2025
        ---
        ## 🏎️ Project Overview
        Physics-informed ML predicts Max Verstappen's race lap times at Monza using ensemble models.
        - **~90.65% average R² (4 years)**
        - **Production-ready dashboard**
        - **Reproducible methodology (2022–2025)**
        ---
        ## 📊 Results Highlights
        ✅ 4/4 years met R² ≥ 0.80  
        ✅ Best: 2025 (R² = 0.9797)  
        ✅ Average MAE: 0.1388s  
        ✅ 189 laps analyzed
        """)
    with c2:
        st.markdown("""
        <div class="stat-card">
        <h3>👨‍💻 Developer</h3>
        <p><strong>Daksh Badhoniya</strong></p>
        <p>CSE AI/ML — RCOEM</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit**")
st.sidebar.markdown("F1 Race Pace Prediction © 2025")
