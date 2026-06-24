# Stock Price Trend Prediction

**A systematic time-series machine learning project to predict short-term stock price trends using rolling statistical features.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![pandas](https://img.shields.io/badge/pandas-2.0+-blue)

## Objective

Build and evaluate models to classify whether a stock will experience a significant upward price movement (2% or 5% above its recent average) on the next trading day, using only **price skewness** and **relative volume** derived from rolling windows (14 and 60 days).

This project emphasizes **time-series validation**, systematic experimentation, and clear reporting on challenging financial data.

## Key Results

- Best mean accuracy: **XX.X%** (5-fold TimeSeriesSplit CV)
- Random Forest and Naive Bayes perform evenly, with each outperforming the other in 4 out of 8 assessments
- 60-day window + 5% threshold performed the strongest
- Full results available in the [`reports/`](reports/) folder

## Project Structure
```bash
stock-price-trend-prediction/
├── data/raw/                  # Downloaded stock data (auto-fetched)
├── notebooks/
│   └── stock_trend_analysis.ipynb    # Main analysis notebook
├── src/
│   └── features.py            # Feature engineering functions
├── reports/
│   ├── figures/               # Saved bar charts
│   ├── model_summary_timeseries_cv.csv
│   └── model_detailed_report_cv.txt
├── requirements.txt
└── README.md
```

## Methodology

- **Features**: Skewness of normalized prices + relative volume over rolling windows
- **Target**: Binary (1 = next day return ≥ threshold, 0 = otherwise)
- **Models**: Random Forest & Gaussian Naive Bayes
- **Validation**: `TimeSeriesSplit` (5-fold expanding window CV)
- **Experiment Grid**: 2 stocks × 2 windows × 2 thresholds

## Results Visualizations

**FSM - 5% Threshold**  
![FSM 5%](reports/figures/FSM_1_05.png)

**RIG - 5% Threshold**  
![RIG 5%](reports/figures/RIG_1_05.png)

*(Additional charts for other combinations are also saved in `reports/figures/`)*

## How to Reproduce

1. Clone the repo:
   ```bash
   git clone https://github.com/d-toups/stock-price-trend-prediction.git
   cd stock-price-trend-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook notebooks/stock_trend_analysis.ipynb
   ```
All data is automatically downloaded via yfinance

### **Colab Notebook Link**

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13mDccMvxfU-q6eJRyUqPKNgLm5Y_GNrd?usp=sharing)

## Conclusions & Learnings
- Short-term (14-day) windows capture more predictable structure than medium-term ones.
- Stronger price movements (5% threshold) are easier to classify than milder ones.
- Simple, well-engineered features can be competitive on noisy datasets.

**Key Lesson:** Thoughtful feature engineering and rigorous validation often matter more than model complexity in financial prediction tasks.

## Future Work

- Expand to 10–20 stocks across sectors
- Add technical indicators (RSI, MACD, ATR, etc.)
- Experiment with XGBoost/LightGBM and ensemble methods
- Full walk-forward backtesting with transaction costs
- Interactive Streamlit dashboard for live predictions
