import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="F1 Race Pace Prediction - Max Verstappen",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #15151E;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1E1E2E;
        border-right: 2px solid #E10600;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #E10600;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    h1 {
        color: #E10600;
        font-weight: 800;
        border-bottom: 3px solid #E10600;
        padding-bottom: 10px;
        letter-spacing: 1px;
    }
    
    h2 {
        color: #00D2BE;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    h3 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        background-color: #38383F;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E10600;
        color: #FFFFFF;
    }
    
    .stButton>button {
        background-color: #E10600;
        color: #FFFFFF;
        border: none;
        padding: 10px 25px;
        font-weight: 700;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #C10500;
        box-shadow: 0 4px 12px rgba(225, 6, 0, 0.4);
    }
    
    .racing-stripe {
        height: 3px;
        background: #E10600;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(225, 6, 0, 0.3);
    }
    
    .stat-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #E10600;
        margin: 10px 0;
    }
    
    .stButton>button {
        background-color: #1E1E2E;
        color: #FFFFFF;
        border: 1px solid rgba(94, 82, 64, 0.3);
        margin-bottom: 8px;
        text-align: left;
        padding-left: 16px;
    }
    
    .stButton>button:hover {
        background-color: #E10600;
        border-color: #E10600;
        color: #FFFFFF;
    }

    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏎️ F1 RACE PACE")
st.sidebar.markdown("### Max Verstappen")
st.sidebar.markdown("**Monza 2022-2025**")
st.sidebar.markdown("---")

st.sidebar.markdown("**Project Stats**")
st.sidebar.metric("Years Analyzed", "4")
st.sidebar.metric("Total Laps", "189")
st.sidebar.metric("Avg R²", "0.8851")
st.sidebar.markdown("---")

# Navigation with buttons
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
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("F1 RACE PACE PREDICTION")
        st.markdown("### Physics-Informed Machine Learning for Monza")
        st.markdown("""
        Predicting **Max Verstappen's race lap times** using practice session data from Friday/Saturday 
        to forecast Sunday performance. Validated across **4 consecutive years (2022-2025)** with 
        **88.51% average accuracy**.
        """)
    
    with col2:
        st.metric("Driver", "Max Verstappen 🇳🇱", delta="#1")
        st.metric("Status", "Production Ready", delta="4 Years Validated")
    
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("## 🏆 Key Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average R²", "0.8851", delta="88.51% Accuracy")
    
    with col2:
        st.metric("Best Year", "2025", delta="R² = 0.9800")
    
    with col3:
        st.metric("Average MAE", "0.156s", delta="Per Lap Error", delta_color="inverse")
    
    with col4:
        st.metric("Success Rate", "75%", delta="3/4 Years ≥ 0.80")
    
    st.markdown("---")
    
    # Load and display summary
    try:
        summary_df = pd.read_csv('./data/summary_results.csv')
        
        st.markdown("## 📈 Year-by-Year Summary")
        
        display_df = summary_df[['year', 'model_name', 'test_r2', 'mae', 'dataset_size']].copy()
        display_df.columns = ['Year', 'Model', 'Test R²', 'MAE (s)', 'Laps']
        display_df['Test R²'] = display_df['Test R²'].round(4)
        display_df['MAE (s)'] = display_df['MAE (s)'].round(3)
        display_df['Year'] = display_df['Year'].astype(int)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.warning(f"Summary data not found: {e}")
    
    st.markdown("---")
    
    # Highlights
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        st.markdown("""
        <div class="stat-card">
        <h3>🚀 Technology Stack</h3>
        <ul>
            <li>FastF1 API for telemetry</li>
            <li>Ensemble ML (CatBoost, XGBoost, RF)</li>
            <li>Hyperparameter tuning via RandomizedSearchCV</li>
            <li>Production filtering & composite scoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: MULTI-YEAR ANALYSIS
# ============================================================================
elif page == "📊 Multi-Year Analysis":
    st.title("📊 Multi-Year Comparison")
    st.markdown("Analyzing Max Verstappen's consistency and trends across 2022-2025 Monza races")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    try:
        # Load data
        with open('./data/cross_year_data.json', 'r') as f:
            cross_data = json.load(f)
        
        # Create tabs
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
            
            fig.add_hline(y=0.80, line_dash="dash", line_color="#00D2BE", 
                         annotation_text="Target (0.80)")
            
            fig.update_layout(
                title="Model Accuracy Over Time",
                xaxis_title="Year",
                yaxis_title="R² Score",
                template="plotly_dark",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### MAE Comparison")
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=cross_data['years'],
                        y=cross_data['mae_scores'],
                        marker_color='#00D2BE',
                        text=[f"{mae:.3f}s" for mae in cross_data['mae_scores']],
                        textposition='outside'
                    )
                ])
                
                fig2.update_layout(
                    title="Mean Absolute Error by Year",
                    xaxis_title="Year",
                    yaxis_title="MAE (seconds)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(t=60, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.markdown("### Dataset Size")
                fig3 = go.Figure(data=[
                    go.Bar(
                        x=cross_data['years'],
                        y=cross_data['dataset_sizes'],
                        marker_color='#E10600',
                        text=cross_data['dataset_sizes'],
                        textposition='outside'
                    )
                ])
                
                fig3.update_layout(
                    title="Race Laps Analyzed",
                    xaxis_title="Year",
                    yaxis_title="Number of Laps",
                    template="plotly_dark",
                    height=400,
                    margin=dict(t=60, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            st.markdown("### Model Selection Pattern")
            
            # Model frequency
            # Model frequency - normalize names (remove "Tuned" suffix)
            model_counts = {}
            for model in cross_data['model_names']:
                clean_name = model.replace(' (Tuned)', '').strip()
                model_counts[clean_name] = model_counts.get(clean_name, 0) + 1

            
            fig4 = go.Figure(data=[
                go.Pie(
                    labels=list(model_counts.keys()),
                    values=list(model_counts.values()),
                    hole=0.4,
                    marker_colors=['#E10600', '#00D2BE', '#FFD700', '#38383F']
                )
            ])
            
            fig4.update_layout(
                title="Model Distribution (2022-2025)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
            st.markdown("### Why Ensemble Methods?")
            st.markdown("""
            <div class="stat-card">
            <ul>
                <li><strong>CatBoost/XGBoost/RandomForest</strong> selected 100% of time</li>
                <li><strong>DecisionTree excluded</strong> - High CV variance, poor generalization</li>
                <li><strong>Production stability</strong> - Ensemble methods provide consistent performance</li>
                <li><strong>Composite scoring</strong> - Balanced accuracy with overfitting penalty</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### FP3 Baseline Pace Evolution")
            
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=cross_data['years'],
                y=cross_data['baseline_paces'],
                mode='lines+markers',
                name='FP3 Baseline',
                line=dict(color='#00D2BE', width=3),
                marker=dict(size=12),
                fill='tozeroy',
                fillcolor='rgba(0, 210, 190, 0.1)'
            ))
            
            fig5.update_layout(
                title="Fastest Practice Lap Time Trend",
                xaxis_title="Year",
                yaxis_title="Lap Time (seconds)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Fastest Baseline",
                    f"{min(cross_data['baseline_paces']):.3f}s",
                    delta=f"2025"
                )
            
            with col2:
                improvement = max(cross_data['baseline_paces']) - min(cross_data['baseline_paces'])
                st.metric(
                    "4-Year Improvement",
                    f"{improvement:.3f}s",
                    delta="Faster"
                )
    
    except Exception as e:
        st.error(f"Error loading multi-year data: {e}")
        st.info("Make sure `data/cross_year_data.json` exists")

# ============================================================================
# PAGE: YEAR DEEP DIVE
# ============================================================================
elif page == "🔍 Year Deep Dive":
    st.title("🔍 Year-Specific Deep Dive")
    st.markdown("Detailed analysis of Max Verstappen's performance by year")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    # Year selector
    selected_year = st.radio(
        "Select Year",
        [2022, 2023, 2024, 2025],
        horizontal=True
    )

    
    try:
        # Load summary for metrics
        summary_df = pd.read_csv('./data/summary_results.csv')
        year_data = summary_df[summary_df['year'] == selected_year].iloc[0]
        
        # Load year-specific data
        with open(f'./results/{selected_year}/predictions.json', 'r') as f:
            predictions = json.load(f)
        
        with open(f'./results/{selected_year}/feature_importance.json', 'r') as f:
            features = json.load(f)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", year_data['model_name'])
        
        with col2:
            st.metric("Test R²", f"{year_data['test_r2']:.4f}")
        
        with col3:
            st.metric("MAE", f"{year_data['mae']:.3f}s")
        
        with col4:
            st.metric("Laps", int(year_data['dataset_size']))
        
        st.markdown("---")
        
        # Add 2023 explanation
        if selected_year == 2023:
            st.info("""
            **📊 2023 Analysis Note:**  
            The 2023 season presented unique predictive challenges. Max Verstappen's Red Bull RB19 showed extreme dominance 
            (0.7s ahead in qualifying) that wasn't fully revealed in conservative practice running. Teams strategically sandbag 
            to hide competitive advantages, creating a practice-race correlation gap. Additionally, extended one-stop strategies 
            with 35+ lap stints entered thermal degradation phases not captured in shorter FP2 runs.
            
            **Result:** R² = 0.7854 demonstrates the real-world limitation that models perform best when race conditions mirror 
            practice running. This validates our multi-year approach - honest reporting of both successes and challenges.
            """)
            st.markdown("---")
        
        # Predictions scatter
        st.markdown("### Actual vs Predicted Lap Times")
        
                
        fig6 = go.Figure()
        
        fig6.add_trace(go.Scatter(
            x=predictions['actual'],
            y=predictions['predicted'],
            mode='markers',
            name='Predictions',
            marker=dict(size=8, color='#E10600', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(min(predictions['actual']), min(predictions['predicted']))
        max_val = max(max(predictions['actual']), max(predictions['predicted']))
        fig6.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='#00D2BE', dash='dash')
        ))
        
        fig6.update_layout(
            title=f"{selected_year} Prediction Accuracy",
            xaxis_title="Actual Lap Time (s)",
            yaxis_title="Predicted Lap Time (s)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig6, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prediction Error Distribution")
            
            fig7 = go.Figure(data=[
                go.Histogram(
                    x=predictions['residuals'],
                    nbinsx=20,
                    marker_color='#00D2BE',
                    opacity=0.7
                )
            ])
            
            fig7.update_layout(
                title="Error Distribution",
                xaxis_title="Residual (s)",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            st.markdown("### Feature Importance")
            
            # Top 8 features
            top_n = 8
            top_features = sorted(zip(features['features'], features['importance']), 
                                key=lambda x: x[1], reverse=True)[:top_n]
            
            fig8 = go.Figure(data=[
                go.Bar(
                    y=[f[0] for f in top_features],
                    x=[f[1] for f in top_features],
                    orientation='h',
                    marker_color='#E10600',
                    text=[f"{f[1]:.3f}" for f in top_features],
                    textposition='outside'
                )
            ])
            
            fig8.update_layout(
                title="Top Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig8, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading {selected_year} data: {e}")
        st.info(f"Make sure `results/{selected_year}/` contains predictions.json and feature_importance.json")

# ============================================================================
# PAGE: LIVE PREDICTION
# ============================================================================
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction Engine")
    st.markdown("Test the model by entering custom feature values")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    # Check for preset selection
    preset_values = {}
    if 'preset' in st.session_state:
        if st.session_state.preset == "start":
            preset_values = {
                'lap': 1, 'tire_age': 0, 'stint': 1, 
                'fuel': 110.0, 'compound': 0
            }
        elif st.session_state.preset == "mid":
            preset_values = {
                'lap': 25, 'tire_age': 10, 'stint': 1,
                'fuel': 70.0, 'compound': 0
            }
        elif st.session_state.preset == "end":
            preset_values = {
                'lap': 53, 'tire_age': 35, 'stint': 2,
                'fuel': 15.0, 'compound': 1
            }
        del st.session_state.preset
    
    # Year selector
    selected_year = st.radio(
        "Select Model Year",
        [2022, 2023, 2024, 2025],
        horizontal=True
    )
    
    try:
        # Load the trained model
        import joblib
        model_data = joblib.load(f'./models/model_{selected_year}.pkl')
        model = model_data['model']
        model_name = model_data.get('model_name', model_data.get('name', 'Unknown'))
        
        st.success(f"✅ Model {selected_year} loaded successfully! ({model_name})")
        
        # Load summary to show model info
        summary_df = pd.read_csv('./data/summary_results.csv')
        year_info = summary_df[summary_df['year'] == selected_year].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", year_info['model_name'])
        with col2:
            st.metric("Training R²", f"{year_info['test_r2']:.4f}")
        with col3:
            st.metric("Expected Error", f"±{year_info['mae']:.3f}s")
        
        st.markdown("---")
        st.markdown("## 🎛️ Input Features")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Practice Session Features")
            
            fp3_baseline = st.number_input(
                "FP3 Baseline (seconds)",
                min_value=75.0,
                max_value=85.0,
                value=79.5,
                step=0.1,
                help="Fastest lap time from FP3 practice session"
            )
            
            medium_deg = st.number_input(
                "Medium Tire Degradation (s/lap)",
                min_value=-5.0,
                max_value=1.0,
                value=-0.5,
                step=0.1,
                help="Tire wear rate for MEDIUM compound from FP2"
            )
            
            hard_deg = st.number_input(
                "Hard Tire Degradation (s/lap)",
                min_value=-1.0,
                max_value=2.0,
                value=0.06,
                step=0.01,
                help="Tire wear rate for HARD compound from FP2"
            )
            
            track_temp = st.slider(
                "Race Track Temperature (°C)",
                min_value=20.0,
                max_value=60.0,
                value=43.0,
                step=0.5
            )
            
            air_temp = st.slider(
                "Race Air Temperature (°C)",
                min_value=15.0,
                max_value=40.0,
                value=26.0,
                step=0.5
            )
        
        with col2:
            st.markdown("### Lap Context Features")
            
            lap_number = st.number_input(
                "Lap Number",
                min_value=1,
                max_value=53,
                value=preset_values.get('lap', 25),
                step=1,
                help="Which lap in the race (1-53)"
            )
            
            tire_age = st.number_input(
                "Tire Age (laps)",
                min_value=0,
                max_value=40,
                value=preset_values.get('tire_age', 10),
                step=1,
                help="How many laps on current tire set"
            )
            
            stint_number = st.selectbox(
                "Stint Number",
                [1, 2, 3],
                index=preset_values.get('stint', 1) - 1,
                help="Which tire stint (usually 1-3)"
            )
            
            fuel_remaining = st.slider(
                "Fuel Remaining (kg)",
                min_value=0.0,
                max_value=110.0,
                value=preset_values.get('fuel', 70.0),
                step=1.0,
                help="Estimated fuel load"
            )
            
            compound = st.selectbox(
                "Tire Compound",
                ["MEDIUM", "HARD"],
                index=preset_values.get('compound', 0),
                help="Current tire compound"
            )
        
        st.markdown("---")
        
        # Add preset buttons BEFORE predict button
        st.markdown("### 📝 Quick Presets")
        st.caption("Click to populate lap context fields with typical scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏁 Race Start (Lap 1)", use_container_width=True):
                st.session_state.preset = "start"
                st.rerun()
        
        with col2:
            if st.button("⏱️ Mid-Race (Lap 25)", use_container_width=True):
                st.session_state.preset = "mid"
                st.rerun()
        
        with col3:
            if st.button("🏆 Final Lap (Lap 53)", use_container_width=True):
                st.session_state.preset = "end"
                st.rerun()
        
        st.markdown("---")
        
        # Predict button
        if st.button("🏎️ PREDICT LAP TIME", use_container_width=True):
            # Prepare input data
            lap_progress = lap_number / 53.0
            tire_age_squared = tire_age ** 2
            compound_encoded = 0 if compound == "MEDIUM" else 1
            
            input_features = [[
                fp3_baseline,
                medium_deg,
                hard_deg,
                track_temp,
                air_temp,
                lap_number,
                tire_age,
                tire_age_squared,
                stint_number,
                fuel_remaining,
                lap_progress,
                compound_encoded
            ]]
            
            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Display result
            st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### 🏁 Predicted Lap Time")
                st.markdown(f"<h1 style='text-align: center; color: #E10600;'>{prediction:.3f}s</h1>", 
                           unsafe_allow_html=True)
                
                diff = prediction - fp3_baseline
                diff_text = f"+{diff:.3f}s" if diff > 0 else f"{diff:.3f}s"
                st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>Difference from FP3: {diff_text}</p>", 
                           unsafe_allow_html=True)
            
            st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
            
            # Show feature interpretation
            st.markdown("### 📊 Prediction Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
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
# PAGE: METHODOLOGY
# ============================================================================
elif page == "⚙️ Methodology":
    st.title("⚙️ Methodology")
    st.markdown("How the prediction system works")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    # Feature Engineering
    st.markdown("## 🎯 Feature Engineering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
        <h3>Practice Session Features (5)</h3>
        <ul>
            <li><strong>FP3 Baseline</strong> - Fastest practice lap</li>
            <li><strong>Medium Degradation</strong> - Rate from FP2</li>
            <li><strong>Hard Degradation</strong> - Rate from FP2</li>
            <li><strong>Track Temperature</strong> - Race conditions</li>
            <li><strong>Air Temperature</strong> - Race conditions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
        <h3>Lap Context Features (7)</h3>
        <ul>
            <li><strong>Lap Number</strong> - Position in race</li>
            <li><strong>Tire Age</strong> - Laps on current tires</li>
            <li><strong>Tire Age²</strong> - Non-linear degradation</li>
            <li><strong>Stint Number</strong> - Which tire stint</li>
            <li><strong>Fuel Remaining</strong> - Dynamic weight</li>
            <li><strong>Lap Progress</strong> - Fraction of race</li>
            <li><strong>Compound</strong> - MEDIUM or HARD</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Pipeline
    st.markdown("## 🤖 Model Pipeline")
    
    st.markdown("""
    ### 1. Model Training
    - **11 models tested** per year: LinearRegression → CatBoost
    - **Hyperparameter tuning**: RandomizedSearchCV (30 iterations, 3-fold CV)
    - **Composite scoring**: `Test R² - (0.3 × Overfit Gap)`
    
    ### 2. Production Filtering
    - **DecisionTree excluded** - High variance, poor CV stability
    - **Ensemble methods only** - CatBoost, XGBoost, RandomForest, GradientBoosting
    - **Top 5 selection** - Based on composite score
    
    ### 3. Validation
    - **Train/Test split**: 70/30 stratified by lap number
    - **Cross-validation**: 3-fold for stability check
    - **Multi-year validation**: Same pipeline across 2022-2025
    """)
    
    st.markdown("---")
    
    # Why it works
    st.markdown("## ✨ Why This Works")
    
    st.markdown("""
    <div class="stat-card">
    <h3>Physics-Informed Approach</h3>
    <p>By encoding F1 domain knowledge (fuel consumption, tire degradation, temperature effects) 
    into features, the model learns <strong>physically meaningful patterns</strong> rather than 
    spurious correlations. This produces:</p>
    <ul>
        <li><strong>Better generalization</strong> - Works across different years/conditions</li>
        <li><strong>Lower data requirements</strong> - 40-50 laps sufficient per year</li>
        <li><strong>Interpretability</strong> - Feature importance aligns with F1 strategy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "👤 About":
    st.title("👤 About This Project")
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎓 Academic Project
        
        **Course**: Machine Learning / Applied Data Science  
        **Institution**: Shri Ramdeobaba College of Engineering and Management (RCOEM)  
        **Branch**: Computer Science & Engineering (AI/ML)  
        **Year**: 2025
        
        ---
        
        ## 🏎️ Project Overview
        
        This system demonstrates **physics-informed machine learning** for predicting **Max Verstappen's race lap times** at Monza. 
        By encoding F1 domain knowledge into features and using ensemble algorithms, the model achieves:
        
        - **88.51% average R² score** across 4 consecutive years
        - **Production-ready deployment** with interactive dashboard
        - **Reproducible methodology** validated on 2022-2025 seasons
        
        ---
        
        ## 🛠️ Technical Implementation
        
        **Data Source**: FastF1 API (official F1 telemetry)  
        **ML Framework**: scikit-learn, XGBoost, LightGBM, CatBoost  
        **Visualization**: Plotly, Streamlit  
        **Deployment**: 108KB lightweight package, Docker-ready
        
        **Key Features:**
        - 12 physics-informed features (tire deg, fuel load, temps)
        - Ensemble model selection (CatBoost, XGBoost, RandomForest)
        - Hyperparameter optimization via RandomizedSearchCV
        - Multi-year validation for generalization proof
        
        ---
        
        ## 📊 Results Highlights
        
        ✅ **3/4 years met R² ≥ 0.80 target**  
        ✅ **Best performance: 2025 (R² = 0.9800)**  
        ✅ **Average MAE: 0.156s per lap**  
        ✅ **189 total race laps analyzed**  
        
        The model proves that domain-driven feature engineering combined with modern ensemble 
        methods can achieve reliable race pace predictions using limited practice session data.
        """)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
        <h3>👨‍💻 Developer</h3>
        <p><strong>Daksh Badhoniya</strong></p>
        <p>CSE AI/ML Engineering<br>RCOEM, Nagpur</p>
        <br>
        <h4>🔗 Connect</h4>
        <p>
        <a href="https://github.com/Pilcrow3000" target="_blank" style="color: #00D2BE;">
        <strong>GitHub</strong> →
        </a>
        </p>
        <p>
        <a href="https://www.linkedin.com/in/daksh-badhoniya/" target="_blank" style="color: #00D2BE;">
        <strong>LinkedIn</strong> →
        </a>
        </p>
        <p>
        <a href="mailto:dakshbadhoniya1@gmail.com" style="color: #00D2BE;">
        <strong>Email</strong> →
        </a>
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-card">
        <h3>📦 Repository</h3>
        <p>Full source code, notebooks, and documentation available on GitHub.</p>
        <br>
        <p>⭐ Star the repo if you find this useful!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🚀 Future Enhancements
    
    - **Expand to all 23 F1 circuits** (current: Monza only)
    - **Multi-driver analysis** (current: Max Verstappen only)
    - **Real-time predictions** during race weekends
    - **Strategy optimization** (pit stop timing, tire choices)
    - **Weather impact modeling** (rain scenarios)
    
    ---
    
    ## 🙏 Acknowledgments
    
    - **FastF1 Team** - Open-source F1 telemetry library
    - **Streamlit** - Rapid dashboard framework
    - **F1 Community** - Domain knowledge and inspiration
    - **RCOEM Faculty** - Academic guidance and support
    
    ---
    
    ## 📜 License
    
    This project is open source and available under the MIT License.
    """)
    
    st.markdown('<div class="racing-stripe"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Code Lines", "~500", delta="Python")
    
    with col2:
        st.metric("Models Trained", "44", delta="4 Years × 11 Models")
    
    with col3:
        st.metric("Project Status", "Complete", delta="Production Ready")
    
    st.markdown("---")
    st.markdown("### 🏆 Built with passion for F1 and Machine Learning")
    st.markdown("*Predicting Max Verstappen's race pace, one lap at a time.* 🏎️💨")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit**")
st.sidebar.markdown("F1 Race Pace Prediction © 2025")
