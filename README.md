# 🏎️ F1 Race Pace Prediction - Max Verstappen

> Physics-informed machine learning for predicting Max Verstappen's race lap times at Monza using practice session data.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌐 Live Dashboard

**Try it now:** [https://verstappen.streamlit.app/](https://verstappen.streamlit.app/)

Interactive dashboard with multi-year analysis, predictions, and model insights.

## 📊 Results

| Year | Model | Test R² | MAE |
|------|-------|---------|-----|
| 2022 | CatBoost | 0.8519 | 0.126s |
| 2023 | XGBoost (Tuned) | 0.7854 | 0.171s |
| 2024 | CatBoost (Tuned) | 0.9230 | 0.199s |
| 2025 | RandomForest (Tuned) | **0.9800** | **0.086s** |

**Average Accuracy:** 88.51% R² across 4 years | **189 total laps analyzed**

## 🎯 Features

- **12 physics-informed features** (tire degradation, fuel load, temperature)
- **Ensemble model selection** (CatBoost, XGBoost, RandomForest)
- **Multi-year validation** (2022-2025 Monza)
- **Interactive dashboard** with F1-themed design
- **Production-ready deployment** (108KB package)

## 🚀 Quick Start

```
# Clone repository
git clone https://github.com/Pilcrow3000/verstappen-lap-time
cd f1-race-pace-prediction

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

## 📁 Project Structure

```
f1-race-pace-prediction/
├── data/
│   ├── cross_year_data.json
│   ├── summary_results.csv
│   └── summary_results.json
├── models/
│   ├── model_2022.pkl
│   ├── model_2023.pkl
│   ├── model_2024.pkl
│   └── model_2025.pkl
├── results/
│   ├── 2022/
│   │   ├── predictions.json
│   │   ├── feature_importance.json
│   │   └── summary.json
│   ├── 2023/
│   │   ├── predictions.json
│   │   ├── feature_importance.json
│   │   └── summary.json
│   ├── 2024/
│   │   ├── predictions.json
│   │   ├── feature_importance.json
│   │   └── summary.json
│   └── 2025/
│       ├── predictions.json
│       ├── feature_importance.json
│       └── summary.json
├── app.py
├── main.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔧 How It Works

1. **Data Collection**: FastF1 API extracts practice session telemetry
2. **Feature Engineering**: 12 domain features (FP3 baseline, tire deg, fuel, temps)
3. **Model Training**: 11 models tested per year with hyperparameter tuning
4. **Production Filtering**: DecisionTree excluded, ensemble methods only
5. **Validation**: 70/30 train-test split, 3-fold cross-validation

## 📈 Dashboard

Interactive Streamlit app with:
- Multi-year performance comparison
- Year-specific deep dive (scatter plots, feature importance)
- Methodology explanation
- F1-themed dark mode design

## 🛠️ Tech Stack

- **Data**: FastF1, Pandas, NumPy
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Plotly, Streamlit
- **Deployment**: Streamlit Cloud / Docker-ready

## 🎓 Academic Context

**Course:** Machine Learning / Data Science  
**Institution:** RCOEM (Shri Ramdeobaba College of Engineering)  
**Branch:** CSE (AI/ML)  
**Year:** 2025

## 🤝 Connect

**Daksh Badhoniya**  
📧 dakshbadhoniya1@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/daksh-badhoniya/)  
💻 [GitHub](https://github.com/Pilcrow3000)

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - F1 telemetry API
- [Streamlit](https://streamlit.io/) - Dashboard framework
- F1 Community - Domain knowledge

---

⭐ **Star this repo if you find it useful!**
```
