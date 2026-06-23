# Stock Price Trend Prediction

**Predicting upward stock trends using skewness and relative volume**  
**FSM vs RIG** — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---

## Objective

This project explores whether **two simple engineered features** — *skewness of normalized prices* and *relative trading volume* — can predict short-term upward price trends in stocks.

We analyze two stocks (**FSM** and **RIG**) over ~5 years of daily data and compare **Random Forest** vs **Gaussian Naive Bayes** across different rolling windows and trend thresholds.

---

## Key Results

- The **5% threshold** (last close > 5% above rolling average) produced the strongest predictive signal.
- **14-day windows** performed better when targeting stronger trends (5%).
- **Random Forest and Naive Bayes performed very similarly** — each model won in 4 out of 8 total scenarios.
- Best overall performance came from the **5% threshold + 14-day window**.
- **Key Insight**: With only two features, a simple model (Naive Bayes) can be just as effective as a more complex ensemble.

This highlights an important lesson in applied ML: **more complex ≠ always better**, especially on noisy financial data with limited features.

---

## Repository Structure

```text
stock-price-trend-prediction/
├── notebooks/
│   └── stock_price_trend_analysis.ipynb     ← Main analysis (Colab)
├── src/
│   └── stock_analysis.py                    ← Clean Python scripts
├── data/
│   └── raw/
│       ├── FSM_daily.pkl
│       └── RIG_daily.pkl
├── reports/
│   └── FULL_MODEL_REPORTS.txt               ← Full classification reports
├── models/                                  ← (Optional) Saved models
├── requirements.txt
├── README.md
└── .gitignore
```

### **Colab Notebook Link**

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13mDccMvxfU-q6eJRyUqPKNgLm5Y_GNrd?usp=sharing)

## Features & Methodology

- **Features Engineered**:
  - Skewness of normalized closing prices in the rolling window
  - Relative trading volume on the last day

- **Experiment Design**:
  - Rolling windows: **14-day** and **60-day**
  - Trend thresholds: **2%** and **5%** above rolling average
  - Models: Random Forest + Gaussian Naive Bayes
  - Evaluation: Accuracy, Precision, Recall, F1-score

## Technologies

- **Python** • **pandas** • **NumPy** • **scikit-learn**
- **Matplotlib** / **Seaborn** • **yfinance**
- **Jupyter / Google Colab**

## How to Reproduce
```bash
# Clone the repository
   git clone https://github.com/d-toups/stock-price-trend-prediction.git
   cd stock-price-trend-prediction

# Install dependencies:
  pip install -r requirements.txt
```
4. Open and run the notebook:
- notebooks/stock_price_analysis.ipynb
- data will be automatically downloaded if the  pickle files are missing

4. View full results in reports/FULL_MODEL_REPORTS.txt

## Learnings & Reflections

- Stock price prediction is extremely challenging due to market noise and efficiency.
- Feature engineering often matters more than model complexity.
- Simple models can compete with ensembles when the feature set is small.

## Future Improvements

- Add more technical indicators (RSI, ATR, MACD, volatility, etc.)
- Implement proper time-series cross-validation (walk-forward)
- Test XGBoost / LightGBM
- Build and backtest a full trading strategy with risk management
