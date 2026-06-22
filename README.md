# Stock Price Trend Prediction Model – Python

Machine Learning project that classifies whether a stock is trending upward using technical features and two classifiers (Decision Tree and Naïve Bayes).

## Project Overview
Built and compared **Naïve Bayes** and **Decision Tree** classifiers to predict if a stock will trend upward based on historical daily price data. Focused on feature engineering and model evaluation across different time windows and price movement thresholds.

## Key Features & Methodology
- Engineered two technical features: **skewness of returns** and **relative trading volume**
- Tested multiple evaluation windows (14-day and 60-day) and price movement thresholds (5% and 10%)
- Compared model performance using accuracy, precision, recall, and F1-score

## Results
- **Naïve Bayes** outperformed the Decision Tree in most configurations
- Best performance: **70% accuracy** (Naïve Bayes) on 60-day window with 5% price movement threshold
- Demonstrated the difficulty and limitations of short-term stock price prediction

## Repository Structure

```text
stock-price-trend-prediction/
├── notebooks/                  ← Main analysis (Colab/Jupyter)
│   └── stock_price_analysis.ipynb
├── src/                        ← Clean Python code
│   └── stock_analysis.py
├── data/                       ← Raw and processed data
│   ├── raw/
│   │   ├── FSM_daily.csv
│   │   └── RIG_daily.csv
│   └── processed/
├── reports/
│   └── figures/                ← Saved plots & visualizations
├── models/                     ← Saved trained models (optional)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
## Technologies
- Python
- pandas
- scikit-learn (DecisionTreeClassifier, GaussianNB)
- matplotlib / seaborn

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/d-toups/stock-price-trend-prediction.git
cd stock-price-trend-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python src/stock_analysis.py
```

## Key Learnings
- Stock price prediction is extremely challenging due to market efficiency
- Feature engineering and proper time-series validation are critical
- Naïve BAyes can sometimes outperform more complex models on small, noisy financial datasets
